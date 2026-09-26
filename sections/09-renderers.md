# Renderers — prime-rl @ b944873

> Scope: the `renderers` package (submodule `deps/renderers` @ `6b8da3f`) — what a renderer does to messages and tokens (render, parse, bridge), how one is selected and configured, the token-in/token-out `generate()` client, multimodal sidecars — plus every place prime-rl and verifiers v1 construct or call a renderer. Seam S4 (env/harness → model calls) is owned by F; this doc owns *what the renderer does to messages/tokens*.
>
> Files read in full: `deps/renderers/renderers/__init__.py` (233), `base.py` (2122), `client.py` (637), `configs.py` (1140), `parsing.py` (2098), `parsers.py` (483), `reasoning.py` (199), `default.py` (286), `qwen3.py` (615), `qwen35.py` (1232), `qwen3_vl.py` (995), `gpt_oss.py` (806); `README.md` (216), `docs/renderer-config.md` (235), `examples/README.md` (90), `examples/vllm/multiturn_generate_vllm.py` (208); tests `conftest.py` (59), `parity.py` (786), `test_parity.py` (222), `test_bridge.py` (338), `test_incremental.py` (173), `reference_rendering.py` (lines 400–668), `test_roundtrip.py` (1–120), `test_disabled_thinking_stability.py` (1–90); verifiers `deps/verifiers/verifiers/v1/clients/train.py` (459), `configs/client.py` (104), `graph.py` (1–56, 376–855); prime-rl `src/prime_rl/utils/chat_template.py` (50), `orchestrator/algo/opsd.py` (78), `orchestrator/generation_source.py` (49), `orchestrator/algo/base.py` (1–40), relevant slices of `orchestrator/orchestrator.py`, `orchestrator/clients.py`, `orchestrator/envs.py`, `orchestrator/trajectories.py`, `trainer/sft/data.py`, `configs/{orchestrator,sft,algorithm}.py`; `docs/algorithms.md` §Multi-Turn Trajectories (484–556). The remaining model renderers (`deepseek_{r1,v3,v4}.py`, `gemma4.py`, `glm45.py`, `glm5.py`, `hy3.py`, `inkling.py`, `kimi_k2.py`, `kimi_k25.py`, `laguna_xs2.py`, `laguna_s21.py`, `llama_3.py`, `minimax_m2.py`, `nemotron3.py`, `prime_qwen3.py`, `qwen36.py`, `qwen38.py`) were read in full by two helper sub-agents that reported with `path:line` cites; their findings populate the per-family table in §7.2 and are marked where I did not re-verify them myself.
>
> Related docs: `07-verifiers-core.md` (F — owns S4: interception, dialects, the trace graph that consumes renderer attribution), `06-inference-and-transports.md` (E — the vLLM `/inference/v1/generate` route and routed-experts wire), `03-orchestrator.md` (B — client construction), `04-algorithms-loss-data-path.md` (C — how branches become samples), `05-trainer.md` (D — SFT data path, VLM `mm_token_type_ids`), `08-harnesses-runtimes-serve.md` (G — which process hosts the train client).

Paths below: `renderers/...` and `tests/...` are relative to `deps/renderers/`; `verifiers/...` is `deps/verifiers/verifiers/...`; everything else is relative to the prime-rl root.

---

## 1. Mental model

A **renderer** is a model family's chat template rewritten as a Python object. It owns the message ↔ token boundary in both directions: `render()` turns OpenAI-style messages into the exact token ids the model sees (plus per-token attribution metadata), `parse_response()` turns sampled completion ids back into `(content, reasoning_content, tool_calls)`, and `bridge_to_next_turn()` produces the next turn's prompt by *appending* to the previous turn's `prompt_ids + completion_ids` instead of re-rendering history (`renderers/base.py:716-845`, `README.md:1-3`).

prime-rl training is **renderer-only**: the orchestrator's policy client is always a token-in/token-out "renderer" client (`src/prime_rl/orchestrator/orchestrator.py:199-205`, `train_client_type="renderer"`), and the orchestrator config field `renderer` is documented as "required — training is renderer-only" (`packages/prime-rl-configs/src/prime_rl/configs/orchestrator.py:535-539`). The engine never applies a chat template on the training path: the client POSTs raw `token_ids` to vLLM's `/inference/v1/generate` (`renderers/client.py:315-339`). SFT tokenization is also renderer-only (`build_training_sample` in `src/prime_rl/trainer/sft/data.py:359-367`).

Why not HF `apply_chat_template`? RL needs the trainer to see *exactly* the token ids the sampler saw, and every successive turn's prompt to extend the previous turn's `prompt + completion` byte-for-byte (the **extension property**, `docs/algorithms.md:488-498`). Re-rendering history through a Jinja template breaks that in many ways (`README.md:131-148`): (a) `false` parsed to Python `False` re-renders as `"False"`; (b) BPE retokenization drift when neighbouring bytes change; (c) tool-call XML normalization (dropped empty `</parameter>`); (d) client-side max-seq-len truncation zeroing the anchor; (e) templates that **strip past `<think>` blocks** once a new user query arrives (`docs/algorithms.md:526-543`). A renderer sidesteps (a)–(d) by never re-tokenizing sampled tokens: the bridge keeps them verbatim and renders only the *new* environment messages. (e) is a *policy* question (should history keep thinking the template would drop?) which the renderer answers with `thinking_retention` (§3.6). Scaffold-level rewrites (e.g. an agent SDK "repairing" tool calls before sending history back) happen before rendering and cannot be fixed by a renderer (`README.md:139`).

How to think about it: a renderer is a **deterministic, attribution-carrying tokenizer for conversations with a proof obligation**. Every token it emits is tagged with the message that produced it (`message_indices`), whether the model would have sampled it (`sampled_mask`), and whether it is caller/model body vs template scaffold (`is_content`). `bridge_to_next_turn` either proves the exact-prefix invariant holds or returns `None`, and the caller falls back to a full render — which, downstream, starts a new training sample ("best-effort interleaving", `docs/algorithms.md:500-514`, §3.8). Parity against the model's reference encoder (HF Jinja, Harmony for gpt-oss, a Python encoder for DeepSeek V4) is enforced by a test matrix (§6).

---

## 2. Where it runs

Renderers are pure in-process Python objects (a tokenizer + cached special-token ids + an optional image cache); there is no renderer server. They are instantiated in four places:

| Process | Who builds it | How | Purpose |
|---|---|---|---|
| Env-server worker processes (the verifiers `TrainClient` is owned by the in-process interception server, one client per distinct `ClientConfig` JSON, `verifiers/v1/interception/server.py:356-366`; see `08-harnesses-runtimes-serve.md`) | `ElasticRendererPool.grow()` | `create_renderer(load_tokenizer(renderer_model_name), config, chat_template_kwargs=...)` on a thread (`verifiers/v1/clients/train.py:278-288`) | Render / bridge every policy (and frozen-generator) turn, call `generate()`, parse the completion |
| Orchestrator (only for `opsd` algorithm) | `OPSDAlgorithm.setup()` | `create_renderer(load_tokenizer(self.clients.model_name), self.renderer_config)` (`src/prime_rl/orchestrator/algo/opsd.py:47-54`) | Tokenize the demonstration "hint block" system message that is prepended before prefill-scoring (`opsd.py:73-78`) |
| SFT trainer | `RendererResolver.__call__` | `create_renderer(self.tokenizer, config)`, cached per frozen config; the trainer's processor is injected into `renderer._processor` (`src/prime_rl/trainer/sft/data.py:224-249`) | `build_training_sample` per dataset row |
| Orchestrator config validation (no instance) | `validate_renderer_auto_resolves` | imports `MODEL_RENDERER_MAP` (`configs/orchestrator.py:698-730`) | Fail at config time if `name="auto"` would fall back to `DefaultRenderer` |

Lifecycle in the train client:

- **Start.** `TrainClient.__init__` calls `ElasticRendererPool(...).warm()` when `renderer_model_name` is pinned (prime-rl always pins it: `src/prime_rl/orchestrator/clients.py:85-91`, `270-280`), which schedules `grow()` on the running loop so the tokenizer loads while the rollout provisions (`train.py:253-264`, `317-327`).
- **Steady state.** Renderers are process-global (a `ClassVar` dict on `ElasticRendererPool`, so every `TrainClient` in one env-server worker shares them; each worker process builds its own), keyed by `(renderer_model, config.model_dump_json(), sorted chat_template_kwargs)` (`train.py:222-251`). Each rollout turn `acquire()`s a `RendererSlot` for the whole turn (render + `generate()` + parse); `grow()` returns the *first* slot with `load < multiplex` (first-fit, not least-loaded), so a slot carries up to `multiplex` in-flight turns (default 256; prime-rl never overrides it, `verifiers/v1/configs/client.py:80-84`, `orchestrator/clients.py:259-280`), and the pool grows (single-flight per key via an `asyncio.Lock` re-minted per event loop, `train.py:266-295`) when all slots are at capacity. **Encode-side work (render, bridge) runs in `asyncio.to_thread` under a per-slot `threading.Lock`** because encoding mutates fast-tokenizer state; decode-side work (`parse_response`, stop ids) is called bare (`train.py:199-214`, `414-416`).
- **Memory.** ~75–95 MB per renderer per the config docstring (`configs/client.py:81`). Multimodal renderers also keep a FIFO image-processor cache bounded by `image_cache_max` (default 256, `renderers/configs.py:232-236`; eviction at `renderers/qwen3_vl.py:451-455`).
- **Shutdown.** Nothing to tear down ("a renderer is pure memory", `train.py:222-226`).
- **Crash modes.** Construction asserts on missing special tokens (`renderers/qwen3.py:83-88`); auto-resolution raises for unregistered VLMs (`renderers/base.py:1534-1542`); `generate()` raises `OverlongPromptError` pre-flight and `MalformedGenerateResponseError` on unusable logprobs (§3.7). The train client maps only `OverlongPromptError` → `ProviderError(status_code=400)` and `OpenAIError` → `model_error` (`train.py:444-449`); both are `RolloutError`s, which the interception server stashes on `session.error` (re-raised after the harness returns) and answers with their status. `MalformedGenerateResponseError` is a plain `ValueError` (`client.py:59-60`), as is the "does not support tools" guard — these hit the server's generic `except Exception` and go back to the harness as **HTTP 502 without setting `session.error`** (`verifiers/v1/interception/server.py:847-870`), so a harness SDK that retries 5xx simply re-issues the turn.

Tokenizer source: renderers in the train client and OPSD load their **own** tokenizer via `renderers.base.load_tokenizer(model_name)` (HF hub, `trust_remote_code=False` except pinned Kimi revisions, gated Llama-3.2 redirected to `unsloth` mirrors; `renderers/base.py:1171-1313`). They do **not** use the orchestrator's `TokenizerConfig` (neither its `name` nor its `chat_template` override): the orchestrator's `setup_tokenizer(config.tokenizer)` result is only handed to `TrainSink` (`orchestrator/orchestrator.py:192`, `371`), while the renderer is built in the env-server worker from `TrainClientConfig.renderer_model_name = config.model.name` (`orchestrator.py:199-205`, `clients.py:85-91`). The SFT trainer, by contrast, passes its own `setup_tokenizer(config.tokenizer)` tokenizer (`trainer/sft/train.py:152`, `sft/data.py:232-249`), so there `tokenizer.name` does drive auto-resolution (its `chat_template` override is still ignored by every typed renderer).

---

## 3. Mechanics

### 3.1 Data model (`renderers/base.py`)

**Messages** are OpenAI-shaped `TypedDict`s: `Message{role, content, tool_calls, tool_call_id, name, reasoning, reasoning_content}` (`base.py:106-119`); `content` is `str | list[ContentPart]` with parts `TextPart{type:"text",text}`, `ThinkingPart{type:"thinking",thinking}`, `ImagePart{type:"image"|"image_url", image|url|path|image_url}`, `VideoPart` (`base.py:33-80`). Tool calls are `ToolCall{type,id,function:{name, arguments: dict|str}}` (`base.py:83-95`). Tools are passed either flat `{name, description, parameters}` or in the OpenAI `{"type":"function","function":{...}}` envelope; parsers accept both (`renderers/parsing.py:43-66`).

**`RenderedTokens`** (`base.py:223-315`) — the output of `render()` and `bridge_to_next_turn()`:

| Field | Type | Meaning |
|---|---|---|
| `token_ids` | `list[int]` | The sequence. |
| `message_indices` | `list[int]` | Per token: index into the caller's message list (for a bridge: into `new_messages`); `-1` = scaffolding outside any message (generation prompt, a tools-only system block, the whole bridged prior). |
| `sampled_mask` | `list[bool]` | Per token: would the model emit this at inference? True on assistant body + its stop token; False on role openers, inter-turn `\n`, all user/system/tool tokens, all bridge output. `[]` = not provided (`DefaultRenderer`). |
| `is_content` | `list[bool]` | Per token: produced from message-body bytes (caller content / tool_calls / reasoning, or the model's emission) vs template scaffold. Equals `sampled_mask` on assistant tokens by construction; carries new info on user/tool/system. `[]` when the renderer can't attribute (Default renderer, or tokenizer without `return_offsets_mapping`). |
| `message_roles` | `list[str]` | Role per caller-relative message (length = number of messages). |
| `message_tool_names` | `list[str\|None]` | Per message; tool function name for `role="tool"` (from `name` or joined via `tool_call_id` against an earlier assistant's `tool_calls`, `base.py:122-181`); metadata only, never changes tokens. |
| `multi_modal_data` | `MultiModalData\|None` | `{mm_hashes: {modality: [str]}, mm_placeholders: {modality: [PlaceholderRange(offset,length)]}, mm_items: {modality: [dict of processor arrays]}}` (`base.py:189-220`). |

Helper views: `tokens_per_message(n, sampled_only=)` (`base.py:317-373`), `message_token_spans()` (half-open per-message spans; tolerant of non-contiguous attribution by returning the outer span, `base.py:375-423`), `role_token_spans()`, `tokens_by_role()`, `content_token_spans_by_role()`, `content_mask_for_roles(roles)` (`base.py:425-556`).

**`ParsedResponse`** (`base.py:620-641`): `content: str`, `reasoning_content: str|None`, `tool_calls: list[ParsedToolCall]`, `reasoning_complete: bool|None` (False = generation ended inside reasoning; None = undeterminable). **`ParsedToolCall`** (`base.py:591-617`): `raw` (decoded block text, always set), `name`, `arguments: dict|str|None`, `token_span: (start,end)|None` into the *stop-stripped* completion, `status: ToolCallParseStatus`, `id` (only formats that carry a native id, Kimi). Statuses: `OK`, `INVALID_JSON`, `UNCLOSED_BLOCK`, `MISSING_NAME`, `MALFORMED_STRUCTURE`, `UNKNOWN_TOOL` (`base.py:559-588`). Every *attempt* is recorded, in order — a deliberate divergence from vLLM/SGLang parsers, which collapse failures (`base.py:574-580`).

**Tokenizer protocols** (`base.py:674-713`): `Tokenizer` needs `name_or_path, unk_token_id, eos_token_id, encode, decode, convert_tokens_to_ids`; `OffsetTokenizer` additionally supports `__call__(..., return_offsets_mapping=True)` (probed at runtime by encoding `"a"`, `base.py:1812-1831`); `ChatTemplateTokenizer` adds `apply_chat_template` (needed only by `DefaultRenderer`).

### 3.2 The `Renderer` protocol — full contract

`Renderer` is a `runtime_checkable` `Protocol`; concrete renderers do not inherit from a base class (`base.py:716-845`).

| Method | Contract |
|---|---|
| `render(messages, *, tools=None, add_generation_prompt=False) -> RenderedTokens` | Token-for-token equal to the model's reference encoder (modulo documented deviations, §7). History-reasoning behaviour is fixed at construction by the config, never per call (`base.py:727-735`). Multimodal renderers accept an extra `process_multimodal: bool = True` kwarg (`renderers/qwen35.py:353-360`). Raises on empty `messages` (e.g. `qwen3.py:133-134`). |
| `render_ids(...) -> list[int]` | `render(...).token_ids` (hand-coded renderers literally delegate, `qwen3.py:279-290`). |
| `parse_response(token_ids, *, tools=None, prompt_ids=None) -> ParsedResponse` | Truncate at the first stop token, determine whether reasoning was open at the start (from `prompt_ids`), split reasoning / content / tool calls by **token id** of atomic markers. `prompt_ids` omitted, `None`, or `[]` all mean "self-contained completion" (`base.py:755-771`, `README.md:50-58`). `tools` enables schema-aware argument typing for XML-ish formats (`base.py:763-770`). |
| `get_stop_token_ids() -> list[int]` | Ids that end a turn; the client forces them into `sampling_params.stop_token_ids` (`client.py:310-312`). |
| `bridge_to_next_turn(previous_prompt_ids, previous_completion_ids, new_messages, *, tools=None) -> RenderedTokens\|None` | If not `None`, `B.token_ids[:len(prev_prompt)+len(prev_completion)] == prev_prompt + prev_completion` and `B` ends where the next assistant turn starts generating (i.e. equivalent to a full render with `add_generation_prompt=True`, except prior sampled tokens are kept verbatim). Prior portion: `message_indices=-1`, `sampled_mask=False`, `is_content=False`; new portion indexed relative to `new_messages`; `sampled_mask` uniformly False. Must return `None` whenever the contract can't be proven — in particular for any assistant message in `new_messages`, and whenever the resolved thinking-retention policy demands a re-render (`base.py:778-845`). |
| (multimodal) `mm_token_type_id_map` property; `bridge_to_next_turn(..., previous_multi_modal_data=None)` | `MultimodalRenderer` protocol: `{image_pad_id: 1, video_pad_id: 2}`; bridge merges prior `MultiModalData` so placeholder counts stay consistent (`base.py:848-892`). `is_multimodal(r)` caches the protocol check per type (`base.py:895-917`). |

Optional duck-typed attributes consulted by the client: `supports_tools` (only `DefaultRenderer` defines it, False unless a `tool_parser` is configured, `renderers/default.py:123-125`; `generate()` raises if tools are passed to a renderer that lacks support, `client.py:261-265`) and `supports_process_multimodal` (`client.py:266-271`; set on `Qwen35Renderer`, `Qwen3VLRenderer`, `KimiK25Renderer`).

Every hand-coded renderer also exposes `self.config` (the frozen typed config) and `self.effective_thinking_retention` (§3.6).

### 3.3 Selecting a renderer: `create_renderer` and configs

```
create_renderer(tokenizer, config=None, *, chat_template_kwargs=None)      base.py:1386
  _populate_registry()   # lazy: imports every renderer module once, fills RENDERER_REGISTRY  base.py:1316-1383
  _resolve_renderer_config(...)                                                 base.py:1468-1487
    config None  -> AutoRendererConfig()
    Auto  -> _resolve_auto_config                                               base.py:1490-1560
               name = MODEL_RENDERER_MAP.get(tokenizer.name_or_path)   # EXACT match only
               hit  -> cfg_cls(thinking_retention=auto.thinking_retention if set), then merge kwargs
               miss + chat_template_kwargs -> ValueError
               miss + (in MULTIMODAL_MODELS or AutoConfig has vision_config/vision_tower/image_token_id) -> ValueError
               miss + auto.thinking_retention set -> NotImplementedError
               miss -> DefaultRendererConfig()  (INFO log)
    typed -> _merge_chat_template_kwargs(config, kwargs)                        base.py:1433-1465
  cls = RENDERER_REGISTRY[config.name]; return cls(tokenizer, config)
```

- **Exact-match auto-detection by `tokenizer.name_or_path`** against `MODEL_RENDERER_MAP` (`base.py:930-1047`). Prefix matching is intentionally off because base/instruct/fine-tunes of one architecture can ship different templates (`base.py:922-929`). A local path or a fine-tune name therefore misses.
- **Chat-template kwargs** are validated against the concrete config's `_template_fields` allowlist; renderer-internal fields (e.g. `image_cache_max`) must come through the typed config; `DefaultRendererConfig` accepts opaque Jinja kwargs except its reserved fields (`base.py:1441-1465`). The merged config is rebuilt with `model_validate` so validators re-run.
- **VLM guard**: an unregistered model is probed with `transformers.AutoConfig.from_pretrained(..., trust_remote_code=False)`; any exception → treated as text-only (`base.py:1116-1157`). Without `transformers` installed, auto-resolution of an unregistered name raises `ImportError` (`base.py:1132-1141`).
- **Config classes** (`renderers/configs.py`): every variant subclasses `BaseRendererConfig(pydantic_config.BaseConfig)` with `frozen=True` (hashable — used as a dict key by the SFT `RendererResolver`, `sft/data.py:236-248`) and inherits `extra="forbid"` (`configs.py:60-95`). `__pydantic_init_subclass__` enforces that every renderer-specific field is classified exactly once as `_template_fields` (mirrors a Jinja kwarg) or `_internal_fields` (renderer-only), raising `TypeError` at import otherwise (`configs.py:97-110`). `RendererConfig` is a discriminated union on `name` (`configs.py:997-1031`) — this is the type prime-rl uses for `[orchestrator.renderer]`, `[renderer]` (SFT), `algo.renderer` (OPSD), and verifiers `TrainClientConfig.renderer`. `config_from_name(name)` returns a default config (`"auto"` → `None`, `configs.py:1089-1100`). Renaming a discriminator is a breaking change (`docs/renderer-config.md:231-235`).
- **Registered names** (`base.py:1351-1382`): `default, qwen3, prime-qwen3, qwen3-vl, gemma4, qwen3.5, qwen3.6, qwen3.8, glm-5, glm-5.1, glm-5.3, glm-4.5, minimax-m2, deepseek-v3, deepseek-r1, deepseek-v4, hy3, inkling, kimi-k2, kimi-k2.5, laguna-xs.2, laguna-m.1, laguna-xs-2.1, laguna-s-2.1, llama-3, nemotron-3, nemotron-3-ultra, nemotron-3.5, gpt-oss`. Concrete classes are lazily importable from the package root via PEP 562 `__getattr__` (`renderers/__init__.py:87-132`).

### 3.4 The render path (Qwen3 as the reference walk-through)

`Qwen3Renderer.render` (`renderers/qwen3.py:126-277`) is the simplest hand-coded ChatML renderer; all others follow the same skeleton.

1. **Construction** resolves special-token ids from the tokenizer (never hard-coded): `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`, `<tool_call>`, `</tool_call>`, `<tool_response>`, `</tool_response>`, `</think>` (`qwen3.py:74-81`), asserting each exists and is not `unk` (`qwen3.py:83-88`).
2. **Four parallel lists** `tokens / indices / sampled / content_mask` are appended by three emitters (`qwen3.py:136-174`):
   - `emit_special(id, msg_idx, is_sampled, is_content)` — one atomic special token.
   - `emit_text(text, ...)` — `tokenizer.encode(text, add_special_tokens=False)`, one attribution for all resulting tokens.
   - `emit_text_segments([(text, is_content), ...], ...)` — **joins the segments and encodes them in one BPE pass**, then attributes each resulting token back to its source segment via `offset_mapping` (`attribute_text_segments`, `base.py:1909-2019`). This is the core trick: wrap + body (e.g. `"user\n"` + content) must be encoded together to reproduce the template's BPE merges, yet the body must be attributable. A token is attributed to the segment containing its first character; `overlap_is_content=True` widens to "any character" for templates that glue wrap and body without whitespace (`base.py:1925-1940`). Without an offset tokenizer the ids are still identical but `is_content` is returned as `[]` (`base.py:1958-1964`, `_content_mask_or_empty` at `base.py:1900-1906`).
   Splitting an encode call at an arbitrary boundary is **never** allowed — the renderer only splits at special tokens or at boundaries where the tokenizer already breaks (e.g. the `\n` after `assistant`, `qwen3.py:498-505`).
3. **System + tools block** (`qwen3.py:176-208`): with tools, one `<|im_start|>system\n{system?}\n\n# Tools ... <tools>\n{json.dumps(tool)}...</tools> ... <|im_end|>\n` block; the system text is body, everything else scaffold; attribution index is `0` if the first message is a system message else `-1`. `json.dumps(tool, ensure_ascii=False)` must match the template's `tojson` byte-for-byte.
4. **Per-message loop** (`qwen3.py:210-259`). `last_query_index` = last user message that is not a `<tool_response>…</tool_response>` wrapper (`qwen3.py:109-124`). Assistant rendering (`qwen3.py:464-569`):
   - `reasoning_content` from the field, else extracted from inline `<think>…</think>` in `content` (`qwen3.py:476-486`).
   - Thinking block emitted iff `msg_idx > last_query_index and (is_last or reasoning_content)` (template rule), **or** (`enable_thinking=False` and no `reasoning_content`) — the documented deviation that re-emits the empty `<think>\n\n</think>\n\n` wrapper on history so sampled streams stay prefixes of later re-renders (`qwen3.py:1-17`, `508-520`).
   - `<|im_start|>assistant\n` is `sampled=False`; body, tool calls (`<tool_call>\n{"name": "...", "arguments": ...}\n</tool_call>`, `qwen3.py:534-562`) and `<|im_end|>` are `sampled=True, is_content=True`; the trailing `\n` is scaffold (`qwen3.py:564-569`).
   - Tool messages: consecutive tool messages share one `<|im_start|>user … <|im_end|>` envelope, each wrapped `\n<tool_response>\n{content}\n</tool_response>` (`qwen3.py:571-615`).
5. **Generation prompt** (`qwen3.py:262-268`): `<|im_start|>assistant\n`, plus `<think>\n\n</think>\n\n` when `enable_thinking=False`; all `-1`, unsampled.

Qwen3.5 (`renderers/qwen35.py`) is the same skeleton with: XML tool calls `<tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>\n</function>\n</tool_call>` (`qwen35.py:1081-1127`); `<think>` / `</think>` as special tokens; the **system text placed after the tool instructions** (`qwen35.py:519-544`, unlike Qwen3's before-tools placement at `qwen3.py:187-196`); a per-model `enable_thinking` default table (0.8B/2B default False; others True; unknown names default True) because the family ships two template polarities (`qwen35.py:82-122`); gen prompt `<think>\n` (thinking) or `<think>\n\n</think>\n\n` (`qwen35.py:627-637`); argument values rendered as `json.dumps` for dict/list else `str(v)` — so a bool renders as `True`/`False`, matching the Qwen3.5 template (`qwen35.py:1006-1018`; Qwen3.6 overrides this); string `arguments` are `json.loads`-ed and **silently become `{}` if not valid JSON** (`qwen35.py:1103-1109`); image parts in user/tool content (§3.9); hooks `_reasoning_instructions`, `_omit_empty_system_message`, `_extract_assistant_parts`, `_should_render_thinking`, `_render_arg_value` for subclasses (`qwen35.py:314-347`, `997-1018`).

### 3.5 The parse path

Shared machinery (`renderers/parsing.py`, `renderers/reasoning.py`):

1. `_strip_stop_tokens(ids, stop_ids)` truncates at the **first** stop token (`parsing.py:148-153`). (gpt-oss is special: it truncates only at `<|return|>`, because `<|call|>` ends individual tool-call blocks, `parsing.py:1793-1800`.)
2. **Initial reasoning state.** `prompt_ends_in_reasoning(tokenizer, prompt_ids, stop_ids=...)` looks at the prompt after its last stop token, skips the assistant header, and runs `scan_reasoning` on that tail to decide whether the generation prompt left a `<think>` open (`reasoning.py:44-77`). This is how "thinking prefilled by the gen prompt" is detected — from the actual `prompt_ids`, never from config.
3. `scan_reasoning(tokenizer, ids, prefilled=..., open_id, close_id, tool_start_id, assistant_prefix)` (`reasoning.py:80-199`) returns `ReasoningBoundary(is_open, text)`. In the default `initial_only` mode reasoning exists only if prefilled or if the body's **first** token is the opener; the first closer permanently switches to content, so `Before<think>x</think>After` is all content (`reasoning.py:122-143`, `README.md:60-64`). Atomic markers are matched by id when the tokenizer has single-token `<think>`/`</think>`, else by decoded text.
4. If `is_open`, the parser returns `ParsedResponse(content="", reasoning_content=text, reasoning_complete=False)` — **no tool calls are extracted from an unclosed reasoning region**, even if the engine reported `stop` (`parsing.py:218-221`; `README.md:50-56`).
5. Tool calls are searched only **after** the reasoning close (regression #78: drafts inside `<think>` must not become calls, `parsing.py:1057-1062`). Multi-token `</think>` (DeepSeek-V3, Kimi) is located by binary search over prefix decodes (`parsing.py:163-193`).
6. Tool-call extraction is per format (`parse_qwen3` JSON hermes `parsing.py:199-306`; `parse_qwen35` XML `312-457`; `parse_glm` `<arg_key>/<arg_value>` by token id with vLLM-parity `UNKNOWN_TOOL` name validation when `tools` is passed `463-639`; `parse_hy3` `649-843`; `parse_laguna_xs2` `852-1018`; `parse_deepseek_v3` `1024-1206`; `parse_deepseek_v4` DSML with explicit `string="true|false"` typing `1212-1405`; `parse_minimax` `1411-1534`; `parse_kimi_k2` section tokens + `functions.name:idx` ids `1540-1730`; `parse_gpt_oss` Harmony channels `1736-1887`; `parse_llama_3` bare JSON body `1893-1954`; `parse_inkling` segment markers `1957-2098`).
7. **Schema-aware argument coercion** for XML-ish formats (values are unquoted on the wire, so `true` vs `"true"` is ambiguous): with `tools`, a param declared `"type":"string"` is kept verbatim; otherwise `json.loads` with raw-text fallback; `INVALID_JSON` is flagged only when the fallback fired *and* the schema does not permit a string (`parsing.py:89-124`).
8. `reasoning_content` convention: `None` = no think block; `""` = an empty-but-present block. `parse_qwen35` keeps `""` so `<think></think>` round-trips for renderers with key-presence semantics (PrimeQwen3) (`parsing.py:326-332`, `374-383`); most other parsers collapse `""` to `None` (`reasoning or None`).

`parsers.py` is a separate, smaller registry of **pluggable** parsers used only by `DefaultRenderer` (`tool_parser` ∈ {`qwen3`, `qwen3.5`, `glm`, `deepseek-v3`}; `reasoning_parser` ∈ {`think`}, `parsers.py:458-483`). Note the `Qwen35ToolParser`/`GlmToolParser` there have no schema-aware coercion (`parsers.py:214-218`, `318-322`).

### 3.6 The bridge path (`bridge_to_next_turn`)

Canonical algorithm (Qwen3, `qwen3.py:316-462`; Qwen3.5 `qwen35.py:696-991`; Qwen3-VL `qwen3_vl.py:677-912`; gpt-oss §3.11):

```
if not prev_prompt or not new_messages or any(m.role=="assistant" for m in new_messages): return None
b = scan_reasoning(prev_completion, prompt_ids=prev_prompt, stop_ids, tool_start_id)
if b.is_open:                                     # completion ended inside <think>
    if any stop token in prev_completion: return None   # can't insert </think> before a sampled stop
    prev_completion += [</think>]                        # synthesized, non-loss prompt context
if should_rerender_for_thinking_retention(policy, new_messages): return None
prev = trim_to_turn_close(prev_prompt, prev_completion, {<|im_end|>,<|endoftext|>}, synthesize_close=<|im_end|>)
ext = "\n"                                        # the inter-turn \n vLLM never returns (it stops at <|im_end|>)
for i, m in enumerate(new_messages): render m exactly like render() does (user/system/tool only; else return None)
ext += generation prompt
return RenderedTokens(prev + ext, indices=[-1]*len(prev)+ext_idx, sampled=[False]*..., is_content=[False]*len(prev)+ext_content)
```

- `trim_to_turn_close` scans **only the completion** (a close token in the prompt is scaffolding) for the last close token and cuts after it — dropping anything sampled past a stop; on truncation (no close) it appends `synthesize_close` if given, else returns `None` (`base.py:1764-1796`, tests `tests/test_incremental.py:39-69`). The synthesized close and the synthesized `</think>` become prompt tokens (`sampled_mask=False`) of the next turn, so they are never trained on.
- **Thinking-retention policy.** Each renderer resolves `effective_thinking_retention ∈ {"template","tool_cycle","all"}` at construction: `resolve_thinking_retention(config, implied)` returns the explicit `config.thinking_retention` if set, else the renderer's template-derived default (`base.py:2050-2063`). `should_rerender_for_thinking_retention`: `"template"` → always re-render; `"all"` → never for this reason; `"tool_cycle"` → re-render iff `new_messages` contains a user query (renderer-specific predicate; Qwen excludes `<tool_response>`-wrapped user messages) (`base.py:2036-2077`). Qwen3 default: `"all"` if `enable_thinking=False` else `"tool_cycle"` (`qwen3.py:69-72`); Qwen3.5: `preserve_thinking` → all, `enable_thinking=False` → all, else tool_cycle (`qwen35.py:150-159`); Qwen3-VL: always `all` (`qwen3_vl.py:327-330`); gpt-oss: `auto_drop_analysis` → tool_cycle else all (`gpt_oss.py:138-141`); DefaultRenderer: `"template"` (`default.py:112-115`). The full table is in `docs/renderer-config.md:131-154` and §4.4.
  - Rationale: with a thinking template, a new user query makes the *template* drop prior reasoning. Bridging verbatim would feed the model a context shape (history with thinking) it wasn't trained on; re-rendering gives the faithful context but breaks extension → new training sample. Concretely (Qwen3-0.6B, default config): a `[tool]` tail bridges and is token-identical to a full render; a `[user]` tail returns `None`, and the full re-render diverges from `prev_prompt + prev_completion` at the prior assistant turn, where `<think>…</think>\n\n` is dropped (`qwen3.py:508-510`). The divergence exists even when the sampled reasoning was empty, because the parser collapses `""` to `None` and the empty wrapper is then not re-emitted. `AutoRendererConfig(thinking_retention="all")` carries through auto-resolution (`base.py:1509-1518`) and makes the same step bridge. `tool_cycle` picks faithfulness; `thinking_retention="all"` picks linearity (`tests/test_bridge.py:208-260`). Config validators reject contradictions such as `GLM5RendererConfig(clear_thinking=False, thinking_retention="tool_cycle")` (`configs.py:30-50`, `379-387`).
- **Bridge attribution**: `message_indices` relative to `new_messages`; `message_roles`/`message_tool_names` only for `new_messages` (so tool names whose issuing assistant is in the prior come back `None`, `base.py:132-135`).
- `DefaultRenderer.bridge_to_next_turn` always returns `None` (`default.py:255-269`).
- Verified by `tests/test_bridge.py` for the `bridge` suite models (`tests/parity.py:89-271`): verbatim prefix, assistant rejection, empty-input rejection, synth-close on truncation, new-message text present, thinking guard declines across user queries but extends across tool responses.

### 3.7 The `generate()` client (`renderers/client.py`)

`async generate(*, client: AsyncOpenAI, renderer, messages, model, prompt_ids=None, multi_modal_data=None, prompt_attribution=None, tools=None, sampling_params=None, cache_salt=None, priority=None, extra_headers=None, max_prompt_len=None, process_multimodal=True) -> dict` (`client.py:191-423`). Yes — it talks to vLLM's **`/inference/v1/generate`** (token-in/token-out), not chat completions:

1. Guards: tools on a renderer with `supports_tools=False` → `ValueError`; `process_multimodal=False` on a renderer without `supports_process_multimodal` → `NotImplementedError` (`client.py:261-271`).
2. Prompt: use caller `prompt_ids` (+ `multi_modal_data`, `prompt_attribution`) if given; else `renderer.render(messages, tools, add_generation_prompt=True)` (`client.py:273-300`).
3. **Pre-flight overlong check** (only with `process_multimodal=True`): if `max_prompt_len` is None, discover it once per `(base_url, model)` via `GET /v1/models` → the card whose `id == model` → `max_model_len` (vLLM extension), cached (incl. `None`, which also covers lookup exceptions or no matching card — the check is then silently off) under a module-level `asyncio.Lock`; `len(prompt_ids) > cap` (prompt alone, not prompt + `max_tokens`) → `OverlongPromptError` without contacting the engine (`client.py:63-109`, `302-308`).
4. **Sampling params**: caller dict copied, then **forced** `stop_token_ids = renderer.get_stop_token_ids()`, `logprobs = 1`, and `skip_special_tokens` defaulted to False (`client.py:310-313`).
5. **POST body** to `{base_url without trailing /v1}/inference/v1/generate` (built as an absolute URL so the OpenAI SDK doesn't prepend `/v1`, `client.py:335-339`):

| Field | When | Value |
|---|---|---|
| `model` | always | request model name |
| `token_ids` | always | prompt ids |
| `sampling_params` | always | as above (+ `routed_experts_prompt_start` injected by the verifiers train client on a bridge, §3.8) |
| `features` | mm data present & `process_multimodal` | `{mm_hashes:{image:[..]}, mm_placeholders:{image:[{offset,length}]}, kwargs_data:{image:[base64 MultiModalKwargsItem]} or null}` (`client.py:463-637`) |
| `content_parts` | `process_multimodal=False` | `[{"type":"image_url","url":...}, ...]` flattened from messages (`client.py:436-460`) |
| `cache_salt`, `priority` | if given | passthrough |
| headers | if given | `extra_headers` (verifiers passes the session-id header, `train.py:440-442`) |

6. **Response parsing**: `parse_generate_response` splices the potentially huge base64 `routed_experts.data` out of the raw JSON before `json.loads` and re-inserts it as a `memoryview` (`client.py:112-134`). `choices[0].token_ids` = completion ids; `prompt_token_ids` / `mm_placeholders` are required only in the `content_parts` mode (`client.py:355-366`). **Logprob validation** is strict and runs before parsing: `choice.logprobs` must be an object, `.content` a list with exactly one entry per completion token, each entry an object with `token == f"token_id:{id}"` and a numeric (non-bool), finite `logprob` that is not vLLM's `-9999.0` (used by vLLM both as missing-evidence marker and as lower clamp) — otherwise `MalformedGenerateResponseError` (`client.py:30-33`, `137-188`, `368`). The same error is raised when `content_parts` mode lacks `prompt_token_ids`/`mm_placeholders`.
7. `parse_response(completion_ids, prompt_ids=effective_prompt_ids, tools=tools)` (`client.py:370-374`).
8. **Finish-reason promotion**: `/inference/v1/generate` never returns `tool_calls`; if ≥1 parsed call has `status == OK` and the engine said `stop`, the client reports `finish_reason="tool_calls"` so OpenAI-style agent loops continue (`client.py:381-394`).
9. Returns `{request_id, usage, prompt_ids, renderer_prompt_ids, mm_placeholders, completion_ids, completion_logprobs, content, reasoning_content, tool_calls (list[ParsedToolCall]), finish_reason, reasoning_complete, routed_experts, sampling_mask, multi_modal_data, prompt_attribution}` (`client.py:396-423`).

The engine side of this route (prime-rl's `PrimeRlServingTokens` subclass that emits compact `{data, shape, start, dtype}` routed-experts objects "the renderers parse", `src/prime_rl/inference/vllm/serving_tokens.py:1-15`) is documented in `06-inference-and-transports.md`.

### 3.8 How verifiers v1 and prime-rl consume renderers

**Train client turn** (`verifiers/v1/clients/train.py:329-456`; S4 owned by F — summarized here for the renderer-facing parts):

1. Only the chat-completions dialect is accepted (`NotImplementedError` otherwise, `train.py:342-350`); tools must be un-namespaced function tools (`train.py:40-52`).
2. `chat_template_kwargs` and `cache_salt` are popped from the sampling wire args; the kwargs become part of the renderer pool key and are applied via `create_renderer(..., chat_template_kwargs=...)` (`train.py:368-376`, `281-286`).
3. **Bridge attempt** only if: a graph-resolved `PendingTurn` exists, the prompt has **no `image_url` parts**, and the un-committed tail is `[tool*]` or `[tool*, user]` (`train.py:176-196`, `386-390`). The anchor comes from `PendingTurn.previous_token_ids()`: the reused prefix's node tokens, split at the first sampled token of the last (assistant) node — so `previous_completion_ids` is exactly the sampled completion and `previous_prompt_ids` includes its generation prompt; no anchor (→ full render) if the last reused node is not sampled or its mask is not all-True after the first sampled token (`verifiers/v1/graph.py:504-531`). On success, `sampling_params["routed_experts_prompt_start"] = max(len(prev_prompt)+len(prev_completion)-1, 0)` — computed from the *untrimmed* prior, so it always indexes inside the verbatim prefix (`train.py:409-412`).
4. Otherwise a full `render(..., add_generation_prompt=True)` (`train.py:417-426`).
5. `generate(..., prompt_ids=..., prompt_attribution=...)`; the result becomes a `Response` whose `TurnTokens` carry `prompt_ids`, `completion_ids`, `completion_logprobs`, `message_spans` (from `attribution.message_token_spans()`, or re-offset bridge-tail spans via `PendingTurn.prompt_message_spans`), `is_content`, `multi_modal_data`, `mm_token_type_id_map`, `routed_experts`, `sampling_mask` (`train.py:106-173`, `graph.py:533-546`). Parsed tool calls with `status == UNKNOWN_TOOL` or no name are dropped from the `AssistantMessage` (other non-OK statuses with a name *are* surfaced) (`train.py:119-131`). **`is_content` lands on nodes** at commit: only if `len(is_content) == len(prompt_ids)`, each new input node gets `is_content[start:end]` of its slice (which includes leading glue such as the bridge's `\n`, False), the assistant node gets `[False]*len(gen_prompt) + [True]*len(completion)` (stop token included); otherwise nodes get `[]` (`graph.py:784-786`, `876-909`). Reused prefix nodes keep the value from their first commit, so the bridge's all-False prior portion is never written.
6. At commit, the trace graph re-checks **token identity** of the reused prefix against the new `prompt_ids` node by node and forks at the first divergence — this is where a full re-render that drifted (dropped `<think>`, BPE drift) becomes a new branch = new training sample (`graph.py:788-813`). A successful bridge matches fully and stays linear.

**prime-rl wiring:**

| Consumer | Config field | Notes |
|---|---|---|
| Policy rollouts | `[orchestrator.renderer]: RendererConfig = AutoRendererConfig()` (`configs/orchestrator.py:535`) | Forwarded into `TrainClientConfig(renderer=..., renderer_model_name=config.model.name)` (`orchestrator/clients.py:83-91`, `259-280`). Validator rejects only `name="auto"` when `tokenizer.name or model.name` ∉ `MODEL_RENDERER_MAP`; `auto_setup_tokenizer` (declared earlier, `:609-615`) has already filled `tokenizer.name = model.name` if unset, so the check effectively keys on `tokenizer.name`. An explicit `name="default"` passes (the error message even recommends it for fine-tunes) (`configs/orchestrator.py:698-730`). |
| Frozen generation source (`sampling.source` = external model) | **same** `orchestrator.renderer` | `GenerationSource` → `connect_frozen_client(..., renderer_config=renderer_config)` builds a renderer client with `renderer_model_name` = the frozen model's `name` (`generation_source.py:25-37`, `algo/base.py:18-32`, `envs.py:257`). No validator looks at the frozen name or forbids an explicit policy renderer here. `GenerationSource.sampling_args` pops `logprobs` for frozen sources (`generation_source.py:43-49`), but `generate()` re-forces `logprobs=1` and validates them, so frozen-sampled nodes carry the frozen engine's real sampled logprobs — and the frozen endpoint must serve vLLM's `/inference/v1/generate` with `token_id:N` logprobs. |
| OPSD hint block | `algo.renderer: RendererConfig = AutoRendererConfig()` (`configs/algorithm.py:338-342`) | `renderer.render_ids([{"role":"system","content":hint}], add_generation_prompt=False)` prepended to each branch's `token_ids` before prefill scoring (`opsd.py:68-78`). |
| ECHO | — | Uses the per-token `is_content` recorded on trace nodes to CE-train tool/user bodies; falls back to the whole non-sampled span when a renderer doesn't attribute content (`configs/algorithm.py:212-219`; `orchestrator/algo/echo.py:52-70`). |
| SFT | `[renderer]` (`configs/sft.py:201`) | Validator requires a *typed* renderer: rejects `auto` on unregistered models **and** `name="default"`, except for `data.type="fake"` with no `val` (`configs/sft.py:495-515`). `build_training_sample(renderer, messages, role_to_mask=None if loss_mask.assistant else should_mask, tools, content_sft_roles={user/system/tool opted in}, ensure_final_stop=True)` (`sft/data.py:350-367`). A per-row `reasoning_effort` column re-validates a config copy (`sft/data.py:212-221`). Messages pass through `normalize_messages` + `_drop_null_fields` + `deserialize_tool_calls` (JSON-string arguments → dict) first (`sft/data.py:286-307`, `src/prime_rl/utils/chat_template.py:5-50`). |
| Trainer VLM path | — | `mm_token_type_ids` come from the renderer's `mm_token_type_id_map` (stamped on the trace at commit, `graph.py:779-781`) and are forwarded to the model (`orchestrator/trajectories.py:10-13`, `144-165`; `src/prime_rl/trainer/model.py:1095-1115`). |

`build_training_sample` (`base.py:1604-1753`): one `render()`; per token: `-1` → False; roles in `content_sft_roles` → `is_content[k]`; else `sampled_mask[k]` AND (`role_to_mask(msg)` if given). With no `sampled_mask` (`DefaultRenderer`) `role_to_mask` is mandatory. `ensure_final_stop` appends `get_stop_token_ids()[0]` (trainable) only when `sampled_mask` is populated, the last message is an assistant the role filter trains, and the last trainable token is not a stop — needed for templates like GLM that close an assistant turn only with the *next* message's role token (`base.py:1668-1735`). Multimodal: returns `multi_modal_data` and `mm_token_type_ids` (0 text / 1 image / 2 video from placeholder ranges, `base.py:1568-1601`, `1737-1753`). `build_trajectory_step` (`base.py:2080-2122`) is a helper that splits a full render at the common prefix with the gen-prompt render; it is not called anywhere in prime-rl or verifiers v1 at the pin (grep of `src/` and `deps/verifiers/verifiers/v1/` finds no caller).

### 3.9 Multimodal (Qwen3-VL / Qwen3.5 family)

Image parts are recognized by `type ∈ {"image","image_url"}`, or (untyped) a *truthy* `image`/`image_url` key — truthiness matters because HF Arrow schema unification fills missing keys with `None` (`qwen3_vl.py:75-99`). `_load_pil_image` accepts PIL, bytes, `data:` URIs, `http(s)://` (fetched with `urllib.request.urlopen` — a network call inside render), `file://` and bare paths; converts to RGB (`qwen3_vl.py:102-156`). The image hash is `sha256(pil.tobytes() + str(size))[:32]` (`qwen3_vl.py:159-168`).

Per image (`qwen3_vl.py:433-456`, `480-516`; `qwen35.py:204-226`, `423-464`):
- Cache lookup by hash; on miss `processor.image_processor(images=[pil], return_tensors="np")` (processor lazily loaded with `AutoProcessor.from_pretrained(tokenizer.name_or_path)` unless injected, `qwen3_vl.py:380-393`); `N = prod(image_grid_thw[0]) // merge_size**2`.
- Tokens: optional `"Picture k: "` (if `add_vision_id`; counter runs over the whole conversation) → `<|vision_start|>` → **N × `<|image_pad|>`** (`is_content=True`, unsampled) → `<|vision_end|>`.
- Sidecar: `mm_hashes["image"] += [h]`, `mm_placeholders["image"] += [PlaceholderRange(offset_of_first_pad, N)]`, `mm_items["image"] += [{"pixel_values": np.ndarray, "image_grid_thw": np.ndarray}]`.
- With `process_multimodal=False`: a single `<|image_pad|>` per image, no sidecar; the engine expands and returns `prompt_token_ids` (`qwen3_vl.py:488-493`, `client.py:320-366`). Not used by prime-rl/verifiers at the pin.
- Text around images is buffered and flushed at special-token boundaries so BPE matches the processor's template (`_Emitter`, `qwen3_vl.py:171-293`). Video parts raise `NotImplementedError` (`qwen3_vl.py:544-547`, `qwen35.py:500-503`).

What the server receives: `generate()` serializes the sidecar via `_build_mm_features` — for `Qwen3VLRenderer`/`Qwen35Renderer` (and subclasses Qwen3.6/3.8) it concatenates per-image arrays into a `BatchFeature`, applies vLLM's `_create_qwen2vl_field_factory(spatial_merge_size=2)`, builds `MultiModalKwargsItems`, and base64-encodes each item with `vllm...mm_serde.encode_mm_kwargs_item` (`client.py:572-637`); Gemma 4 has its own branch that renames `image_position_ids` → `pixel_position_ids` (`client.py:508-569`); **any other multimodal renderer raises `NotImplementedError`** (`client.py:502-505`). This requires `vllm` and `torch` importable in the *client* process (`client.py:585-599`).

Bridge: prior `MultiModalData` is merged (lists copied, never mutated) with new items; offsets of prior images are unchanged because they precede the synthesized close (`qwen3_vl.py:871-902`, `qwen35.py:941-991`). With `add_vision_id=True`, the bridge refuses if prior images exist but `previous_multi_modal_data` was not passed (it can't recover the picture counter) (`qwen3_vl.py:731-747`). **verifiers never bridges a prompt containing `image_url` parts** (`train.py:386-390`), so in prime-rl multimodal rollouts always take the full-render path and `previous_multi_modal_data` is unused.

Training side: `mm_token_type_id_map = {image_pad: 1, video_pad: 2}` (`qwen3_vl.py:362-373`); the trace graph attaches each image's item to the node whose message introduced it, in prompt order (`graph.py:647-686`), and `trajectories.py` concatenates items per kwarg key along dim 0 (`orchestrator/trajectories.py:35-49`). SFT truncation never cuts inside a placeholder run and drops items past the cut (`sft/data.py:184-209`, `385-402`).

Qwen3-VL has **no reasoning handling**: `reasoning_content` is ignored in history and the gen prompt is plain `<|im_start|>assistant\n` (`qwen3_vl.py:615-618`, `914-961`), though `parse_response` still runs `parse_qwen3` with `</think>` (`qwen3_vl.py:653-672`).

### 3.10 `DefaultRenderer` (fallback)

`DefaultRenderer` (`renderers/default.py:90-269`) wraps `tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, add_generation_prompt=..., tools=..., **config.model_extra)` (`default.py:160-169`) after JSON-decoding string tool-call arguments (GLM-style templates iterate `arguments.items()`, `default.py:38-87`).
- `render()` is **O(N) template calls**: it renders prefixes `messages[:i+1]` and attributes the new suffix to message `i` (`default.py:127-158`) — only correct if the template is prefix-stable; `sampled_mask`/`is_content` are empty.
- `parse_response` uses the generic `<think>` scanner plus the optional `tool_parser`/`reasoning_parser`; stop ids are just `[eos_token_id]` (`default.py:182-253`).
- Bridge always `None`; explicit `thinking_retention` is rejected (`default.py:106-111`); `supports_tools` is False without a `tool_parser` (`default.py:123-125`).

### 3.11 gpt-oss (Harmony) specifics

`GptOssRenderer` (`renderers/gpt_oss.py`) is an adapter over the `openai_harmony` reference encoder, which vLLM also uses (`gpt_oss.py:1-32`):
- **Prefix**: a `SystemContent` (reasoning effort, `conversation_start_date` — defaults to *today at construction*, `gpt_oss.py:147-152` — optional knowledge cutoff / model identity) plus a `DeveloperContent` holding the caller's first system message as instructions and the function tools, rendered with `render_conversation` so the "Calls to these tools must go to the commentary channel" line appears (`gpt_oss.py:354-421`). The whole prefix is attributed to the first system message (or `-1`); `is_content` for the system body is recovered by diffing against a render with empty instructions (`gpt_oss.py:177-259`).
- **Messages** are converted to Harmony messages and rendered one at a time via `enc.render(m)` for per-message attribution (`gpt_oss.py:423-433`). Assistant → optional `analysis` (reasoning, only when `_should_emit_analysis`: always if `auto_drop_analysis=False`, else only for tool-calling turns not followed by a later final answer, `gpt_oss.py:668-686`), `final` (content), one `commentary` message per tool call with recipient `functions.<name>` (`gpt_oss.py:741-806`). Tool results → author `functions.<name>` (from `msg["name"]`, **else `functions.unknown`**), recipient `assistant`, channel `commentary` (`gpt_oss.py:718-730`).
- A trailing final assistant message's `<|end|>` is patched to `<|return|>` when not generating (`gpt_oss.py:442-450`).
- **Generation prompt is `<|start|>assistant<|channel|>analysis<|message|>`** — it prefills the analysis channel (`gpt_oss.py:454-459`); the parity oracle does the same (`tests/reference_rendering.py:595-601`).
- Stop ids `[<|return|>, <|call|>]` (`gpt_oss.py:531-532`). `parse_response` re-attaches the Harmony header from the prompt tail when the completion doesn't start with `<|start|>` (`gpt_oss.py:483-529`); parse maps `analysis`→reasoning, `final`/recipient-less `commentary`→content, `to=functions.X`→tool call (`parsing.py:1770-1874`).
- Bridge: refuses if the completion ends inside an unfinished `analysis` block after a stop (`gpt_oss.py:557-581`); trims to the last `{<|return|>, <|call|>}` and synthesizes `<|end|>` on truncation (`gpt_oss.py:589-594`); accepts only `tool/user/system/developer` new messages (`gpt_oss.py:611-614`).

---

## 4. Interfaces & contracts

### 4.1 Package surface

`from renderers import ...` exports (`renderers/__init__.py:135-233`): data types (`Message`, `ContentPart`, `TextPart`, `ThinkingPart`, `ImagePart`, `VideoPart`, `ToolCall`, `ToolCallFunction`, `ToolSpec`, `RenderedTokens`, `RenderedConversation`, `RenderedTrainingSample`, `ParsedResponse`, `ParsedToolCall`, `ToolCallParseStatus`, `MultiModalData`, `PlaceholderRange`), protocols (`Renderer`, `MultimodalRenderer`, `Tokenizer`, `OffsetTokenizer`, `ChatTemplateTokenizer`), functions (`create_renderer`, `build_training_sample`, `build_trajectory_step`, `trim_to_turn_close`, `reject_assistant_in_extension`, `attribute_text_segments`, `extract_message_tool_names`, `is_multimodal`, `config_from_name`), errors (`OverlongPromptError`, `MalformedGenerateResponseError`), `MULTIMODAL_MODELS`, all config classes and `RendererConfig`, and all renderer classes (lazy). `load_tokenizer`, `MODEL_RENDERER_MAP`, `RENDERER_REGISTRY` live in `renderers.base`.

Install extras: base (BYO tokenizer), `renderers[transformers]` (tokenizer loading, auto-resolution of unknown models), `renderers[multimodal]` (Pillow + HF processors) (`README.md:9-28`). gpt-oss needs `openai_harmony` at import (`gpt_oss.py:40-51`).

### 4.2 `generate()` wire (client side)

Request: `POST {base}/inference/v1/generate`, JSON `{model, token_ids, sampling_params{..., stop_token_ids, logprobs:1, skip_special_tokens:false[, routed_experts_prompt_start]}, [features], [content_parts], [cache_salt], [priority]}`; `GET {base}/v1/models` once per `(base_url, model)` for `max_model_len`. Response fields read: `request_id`, `usage`, `prompt_token_ids`, `mm_placeholders`, `choices[0].{token_ids, logprobs.content[i].{token,logprob}, finish_reason, routed_experts{data,shape,start,dtype}, sampling_mask}` (`client.py:353-423`). Server semantics: see `06-inference-and-transports.md`.

### 4.3 Config fields

| Where | Field | Type / default | Effect |
|---|---|---|---|
| Orchestrator | `renderer` | `RendererConfig = AutoRendererConfig()` | Policy + frozen-source renderer (`configs/orchestrator.py:535`) |
| OPSD algo | `renderer` | `RendererConfig = AutoRendererConfig()` | Hint-block renderer (`configs/algorithm.py:338`) |
| SFT | `renderer` | `RendererConfig = AutoRendererConfig()` | Must resolve to a typed, non-default renderer (`configs/sft.py:201`, `495-515`) |
| verifiers `TrainClientConfig` | `renderer` | `RendererConfig \| None = None` | `None` ⇒ auto (`verifiers/v1/configs/client.py:71-75`) |
| | `renderer_model_name` | `str \| None = None` | Tokenizer/renderer model; prime-rl pins it to `model.name` (`configs/client.py:76-79`) |
| | `multiplex` | `int = 256, ≥1` | Rollouts per renderer slot (`configs/client.py:80-84`) |
| every `*RendererConfig` | `thinking_retention` | `Literal["tool_cycle","all"] \| None = None` | Bridge-policy override (`configs.py:78-89`) |
| sampling wire | `chat_template_kwargs` | mapping | Popped by the train client; validated against the renderer's template allowlist (`train.py:369`; `base.py:1433-1465`) |

TOML example (`docs/renderer-config.md:209-217`):
```toml
[orchestrator.renderer]
name = "qwen3.5"
enable_thinking = false
thinking_retention = "all"
```

### 4.4 Per-renderer config fields and derived bridge policy

From `docs/renderer-config.md:27-55` and `131-154`, verified against `configs.py` for the rows marked ✓ (others per docs + sub-agent reads):

| Renderer (`name`) | Template fields (defaults) | Renderer-only fields | Default bridge policy |
|---|---|---|---|
| `qwen3` ✓ | `enable_thinking=True` | — | `enable_thinking=False`→all else tool_cycle |
| `prime-qwen3` ✓ | — | — | all |
| `qwen3.5` ✓ | `enable_thinking=None` (per-model table), `add_vision_id=False` | `image_cache_max=256` | as qwen3 |
| `qwen3.6` ✓ | + `preserve_thinking=False` | `image_cache_max` | preserve→all; no-think→all; else tool_cycle |
| `qwen3.8` ✓ | + `preserve_thinking=True`, `reasoning_effort="xhigh"` (`xhigh/medium/low`) | `image_cache_max` | same as 3.6 (so default all) |
| `qwen3-vl` ✓ | `add_vision_id=False` | `image_cache_max` | all |
| `gemma4` ✓ | `enable_thinking=False`, `preserve_thinking=False` | `image_cache_max` | preserve→all; no-think→all; else tool_cycle |
| `glm-5` / `glm-5.1` ✓ | `enable_thinking=True`, `clear_thinking=True` | — | clear=False→all; no-think→all; else tool_cycle |
| `glm-5.3` ✓ | `clear_thinking=False`, `reasoning_effort="max"` | — | clear=False→all else tool_cycle |
| `glm-4.5` ✓ | `enable_thinking=True` | — | no-think→all else tool_cycle |
| `gpt-oss` ✓ | `reasoning_effort="medium"`, `conversation_start_date=None` | `use_system_prompt=True`, `knowledge_cutoff`, `model_identity`, `auto_drop_analysis=True` | auto_drop→tool_cycle else all |
| `hy3` ✓ | `reasoning_effort="no_think"`, `preserved_thinking=None`, `is_training=False`, `raw_last_assistant=False`, `fallback_strategy=None` | — | preserved (resolved; default True iff tools)→all else tool_cycle |
| `kimi-k2` ✓ | — | `enable_thinking=True` (no-op) | all |
| `kimi-k2.5` ✓ | `thinking=True` | `image_cache_max` | thinking=False→all else tool_cycle |
| `inkling` ✓ | `reasoning_effort=0.9` (label or float ∈[0,0.99]) | `image_cache_max`, `audio_cache_max` | all |
| `laguna-xs.2` / `laguna-m.1` ✓ | `enable_thinking=False`, `render_assistant_messages_raw=False` | — | all |
| `laguna-xs-2.1` ✓ | `enable_thinking=False` | — | all |
| `laguna-s-2.1` ✓ | `enable_thinking=True`, `preserve_thinking=False` | — | all |
| `llama-3` ✓ | `date_string="26 Jul 2024"`, `tools_in_user_message=True` | — | all |
| `minimax-m2` ✓ | `model_identity="You are a helpful assistant. Your name is MiniMax-M2.5 …"` | — | tool_cycle |
| `nemotron-3` ✓ | `enable_thinking=True`, `truncate_history_thinking=True`, `low_effort=False` | — | truncate=False→all; no-think→all; else tool_cycle |
| `nemotron-3-ultra` ✓ | … `medium_effort=False` instead of `low_effort` | — | same |
| `nemotron-3.5` ✓ | `enable_thinking=True`, `truncate_history_thinking=True` | — | same |
| `deepseek-v3` ✓ | — | — | all |
| `deepseek-r1` ✓ | — | — | **template** (never bridges unless overridden) |
| `deepseek-v4` ✓ | `enable_thinking=False`, `drop_thinking=True`, `reasoning_effort="low"` | — | no-think or drop=False→all else tool_cycle |
| `default` ✓ | opaque Jinja kwargs (`extra="allow"`) | `tool_parser`, `reasoning_parser` | template (bridge always None) |

### 4.5 Files / env

Renderers write no files. They read the HF hub/cache (tokenizer, `AutoConfig` probe for unknown models, `AutoProcessor` for VLMs) and may fetch `http(s)` image URLs during render (`qwen3_vl.py:145-150`). No renderer-specific env vars; `HF_*` env vars govern hub access.

---

## 5. Invariants & assumptions

1. **Exact-prefix (extension) contract** of every non-`None` bridge result (`base.py:786-795`); prior tokens are never re-tokenized. Tested per bridge-suite model (`tests/test_bridge.py:121-205`).
2. **Reference parity** of `render()` against the model's reference encoder for every catalogued `(model × scenario × template-kwarg combination)` cell, except declaratively excluded cells (`tests/test_parity.py:48-58`, `191-222`; exclusions in `tests/parity.py:734-786`).
3. **`is_content == sampled_mask` on assistant tokens** (`base.py:259-265`); assistant role openers are unsampled; the assistant stop token is sampled; inter-turn `\n` is not.
4. **Nothing the bridge emits is sampled** — `sampled_mask` all False — so synthesized `</think>` / close tokens and the generation prompt are never loss targets (`base.py:808-814`).
5. **Stop-token assumption**: the engine stops *on* a stop id and returns it as the last completion token, and does *not* return the template's trailing scaffold (`\n` after `<|im_end|>`); the bridge re-adds that scaffold (`qwen3.py:411-414`). `generate()` guarantees the stop ids are the renderer's (`client.py:311`).
6. **Tokenizer identity**: renderer tokenizer == engine tokenizer == trainer tokenizer. Nothing checks this; prime-rl loads the renderer tokenizer from `model.name` via `load_tokenizer` (§2).
7. **BPE discipline**: text is only split at special tokens or tokenizer-guaranteed boundaries; wrap+body joined encodes preserve merges (`base.py:1915-1924`).
8. **Single initial reasoning region** for tagged formats; tool calls inside open or closed reasoning are not calls (§3.5).
9. **Configs are frozen value objects**; behaviour is fixed per renderer instance (`configs.py:76`, `base.py:727-735`).
10. **Exact-name resolution**: auto mode is only safe for names in `MODEL_RENDERER_MAP`. prime-rl enforces this at config time for the RL *policy* (auto only, keyed on `tokenizer.name`; explicit `default` allowed, `configs/orchestrator.py:698-730`) and for SFT (auto and `default` both refused, `configs/sft.py:495-515`). Frozen generation sources and OPSD's `algo.renderer` are not checked.
11. **Encoding is not thread-safe** on a shared fast tokenizer — the verifiers pool serializes encode-side calls per slot (`train.py:199-214`). Any other multi-threaded caller must do the same.

---

## 6. Extension points

### 6.1 Recipe: add a renderer for a new model family

1. **Study the reference encoder** (the model's Jinja `chat_template`, or its Python encoder / Harmony). Enumerate: role wrappers, special tokens and which are atomic, system/tools block, tool-call wire format, tool-response wrapping (and whether consecutive tool messages share an envelope), how history reasoning is kept/dropped and on which boundary, generation-prompt variants per kwarg, final-turn terminator, whitespace stripping of content.
2. **Config** — add `class FooRendererConfig(BaseRendererConfig)` in `renderers/configs.py` with `name: Literal["foo"] = "foo"`, one field per template kwarg (listed in `_template_fields`) and per renderer-only knob (`_internal_fields`); the subclass hook raises `TypeError` if any field is unclassified (`configs.py:97-110`). If a template knob determines history retention, add a `model_validator` calling `_reject_thinking_retention_conflict` (`configs.py:30-50`). Add the class to the `RendererConfig` union (`configs.py:997-1031`), to `_CONFIG_BY_NAME` (`configs.py:1046-1077`), and to `__all__` in `configs.py` and `renderers/__init__.py`.
3. **Renderer** — new module `renderers/foo.py` with `class FooRenderer` implementing `render`, `render_ids`, `parse_response(token_ids, *, tools=None, prompt_ids=None)`, `get_stop_token_ids`, `bridge_to_next_turn`. Copy the `Qwen3Renderer` skeleton: resolve special ids in `__init__` with the asserting `_token_id`; set `self.config` and `self.effective_thinking_retention = resolve_thinking_retention(cfg, <template-derived default>)`; use `emit_special` / `emit_text` / `emit_text_segments` with correct `(msg_idx, is_sampled, is_content)` on every token; finish with `_content_mask_or_empty`, `message_roles`, `extract_message_tool_names`. For the parser, reuse or add a `parse_foo` in `renderers/parsing.py` built on `_strip_stop_tokens`, `prompt_ends_in_reasoning` + `scan_reasoning`, token-id markers, `ParsedToolCall` with statuses and `token_span`s, and `_coerce_arg_value` if values are unquoted. For the bridge: guards → reasoning-open handling → `should_rerender_for_thinking_retention` (with your template's user-query predicate) → `trim_to_turn_close(close_ids, synthesize_close=canonical_close)` → the inter-turn glue your engine does not return → new messages rendered with the same helpers as `render()` → generation prompt. Multimodal: add `mm_token_type_id_map`, the `previous_multi_modal_data` / `process_multimodal` kwargs, `supports_process_multimodal = True`, and a `_build_mm_features` branch in `renderers/client.py:488-505` (the dispatch is by class).
4. **Register** — `RENDERER_REGISTRY` entry in `_populate_registry` (`base.py:1316-1383`), lazy entry in `_LAZY_RENDERERS` (`renderers/__init__.py:87-117`), exact HF ids in `MODEL_RENDERER_MAP` (`base.py:930-1047`), and `MULTIMODAL_MODELS` for VLMs (`base.py:1059-1094`). If the tokenizer needs remote code, pin it in `TRUSTED_REVISIONS` (`base.py:1171-1175`); for gated repos with an audited mirror use `TOKENIZER_SOURCE_OVERRIDES` (`base.py:1183-1186`).
5. **Tests** — add a `_model(...)` entry to `MODEL_CATALOG` in `tests/parity.py:89-271` (flags: `shared`, `roundtrip`, `bridge`, `excluded` scenarios, `oracle_defaults`, `extra_suites` such as `tool-arg-types`, `disabled-thinking`); every template field needs representative values in `KWARG_VALUES` (`tests/parity.py:673-708`; `test_catalog_covers_every_declared_kwarg` enforces it, `tests/test_parity.py:108-135`). If the reference isn't HF Jinja, add an oracle to `REFERENCE_ORACLES` and route it in `RENDERER_ORACLE_ROUTES` (`tests/reference_rendering.py:605-622`). Deliberate deviations go into `scenario_is_valid` carve-outs plus a dedicated stability test (pattern: `tests/test_disabled_thinking_stability.py`).
6. **Verify token-exactness** — the matrix test does exactly this: `renderer.render_ids(messages, tools=..., add_generation_prompt=...) == tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, **kwargs)` for every cell (`tests/test_parity.py:191-222`, oracle `tests/reference_rendering.py:404-419`). Also run `test_roundtrip.py` (render → slice assistant tokens → parse → equivalent message), `test_bridge.py` (contract), `test_multimodal.py` for VLMs (byte parity vs `processor.apply_chat_template` + `processor(...)["input_ids"]`, placeholder anchoring, bridge parity; `tests/test_multimodal.py:1-22`). For prime-rl: set `[orchestrator.renderer] name = "foo"` (or rely on auto once mapped).

### 6.2 Other seams

- **Pluggable parsers for `DefaultRenderer`**: register a class with `extract(...)` in `TOOL_PARSERS` / `REASONING_PARSERS` (`parsers.py:458-467`); select via `DefaultRendererConfig(tool_parser=..., reasoning_parser=...)`. `_resolve_parser` also accepts a parser *instance* instead of a name (`default.py:272-277`), though the frozen config types them as `str`.
- **Template knobs**: new fields on the relevant `*RendererConfig` + consumption in the renderer (`docs/training.md:166`).
- **Subclass hooks** in `Qwen35Renderer` for sibling templates (`_reasoning_instructions`, `_omit_empty_system_message`, `_extract_assistant_parts`, `_should_render_thinking`, `_render_arg_value`, `_config_cls`) — Qwen3.6/3.8 use them.
- **Engine pluggability**: `_build_mm_features` is vLLM-specific; the code comments propose either per-engine encode methods on renderers or an `Encoder` strategy at `generate()` (`client.py:477-486`). SGLang/Transformers/Tinker work via offline examples only (`examples/README.md`).

---

## 7. Gotchas & limitations

### 7.1 General

1. **Auto-detection is exact-match on `name_or_path`.** A local checkpoint path, fine-tune, or LoRA-merged model misses (`load_tokenizer(<local snapshot dir>)` keeps the path as `name_or_path` → `DefaultRenderer`, `supports_tools=False`, bridge always `None`). prime-rl refuses `auto` in that case for the RL policy and SFT, but everything else — verifiers `TrainClientConfig(renderer=None)`, frozen sources, OPSD — silently falls back to `DefaultRenderer` (INFO log only) (`base.py:1544-1560`). An RL policy with explicit `name="default"` is accepted; with tools and no `tool_parser` every turn then fails with the `ValueError` guard (`client.py:261-265`) → HTTP 502 to the harness (§2).
2. **Validator/runtime name mismatch (both directions).** The orchestrator validator keys on `tokenizer.name` (effectively; `configs/orchestrator.py:609-615`, `715`), but the renderer tokenizer is `load_tokenizer(model.name)` in the env-server worker (`orchestrator.py:199-205` → `clients.py:85-91` → `train.py:281-286`); the orchestrator's `TokenizerConfig` tokenizer never reaches a renderer (§2). So `[orchestrator.tokenizer] name = <mapped id>` with `model.name = <local path>` **passes validation and runs on `DefaultRenderer`**; conversely an unmapped `tokenizer.name` with a mapped `model.name` is rejected although it would resolve correctly. `[tokenizer].chat_template` has no effect on any renderer. Fix for local checkpoints: set `[orchestrator.renderer] name = "<family>"` explicitly.
3. **Frozen generation sources reuse the policy's renderer config** (`envs.py:257`, `generation_source.py:36`) and nothing validates the combination. With an explicit (non-auto) `orchestrator.renderer`, a frozen generator of a different family gets the wrong renderer (wrong special ids → construction assert, or silently wrong template for a same-token family); with `auto`, an unmapped frozen `name` silently gets `DefaultRenderer`.
4. **Default thinking policy breaks extension at every new user query** for thinking templates (`tool_cycle`) — multi-user-turn envs produce one training sample per user turn unless you set `thinking_retention="all"` (which then deviates from the template's context shape) (§3.6). A `<tool_response>`-wrapped `role="user"` message does not count as a query for the Qwen family (`qwen3.py:109-117`), but does for renderers using the generic predicate (e.g. GLM, Laguna). `deepseek-r1`'s `template` default refuses *every* bridge, tool turns included.
5. **DefaultRenderer is quadratic and attribution-poor**: O(N) template calls per render, no `sampled_mask` (so `role_to_mask` required), no bridge, no tools without a `tool_parser`, stop ids only `eos` (`default.py:127-158`, `249-253`).
6. **Unclosed reasoning ⇒ empty content, no tool calls**, even with `finish_reason="stop"`; `reasoning_complete=False` is the signal (`README.md:50-77`).
7. **`finish_reason` is rewritten** `stop → tool_calls` when ≥1 OK tool call is parsed (`client.py:389-394`); other non-OK attempts stay visible in `tool_calls` but verifiers drops only `UNKNOWN_TOOL`/nameless ones (`train.py:119-131`), so `INVALID_JSON` calls with a name *are* passed to the harness as tool calls with raw-string arguments.
8. **Strict logprob validation**: a single `-9999.0` logprob (vLLM's clamp/sentinel) fails the whole request with `MalformedGenerateResponseError` (`client.py:31-33`, `183-186`). Because that error is a `ValueError`, not a `RolloutError`, the harness sees an HTTP 502 and the rollout is not marked errored by it (§2) — a retrying SDK resamples the turn.
9. **Overlong pre-flight** caches `max_model_len` per process forever; if the engine is restarted with a different `max_model_len`, the client keeps the stale cap (`client.py:63-109`).
10. **Tokenizer encode is not thread-safe**; don't share a renderer across threads without the slot lock (`train.py:199-214`).
11. **Qwen3.5 silently drops non-JSON string tool arguments** to `{}` when re-rendering history (`qwen35.py:1105-1109`) — only matters on the full re-render path.
12. **Multimodal**: never bridged in verifiers (§3.9); `generate()` feature serialization only exists for Qwen3-VL/Qwen3.5-family and Gemma 4 — other VLM renderers (Kimi K2.5/K2.6, Inkling) raise `NotImplementedError` if media is sent through `generate()` with `process_multimodal=True` (`client.py:497-505`). Kimi K2.5 could use `process_multimodal=False` (it sets `supports_process_multimodal`), but the verifiers train client never passes that flag, and Inkling supports neither path — so **multimodal RL through prime-rl works only for Qwen3-VL, the Qwen3.5 family, and Gemma 4** at this pin; video unsupported (`qwen3_vl.py:544-547`); `http(s)` images are fetched synchronously during render (`qwen3_vl.py:146-150`); client needs `vllm`+`torch` importable to build features (`client.py:585-599`).
13. **gpt-oss**: `conversation_start_date` defaults to the construction date, so renders differ across days/processes (`gpt_oss.py:147-152`); pin it for reproducibility. Tool results without `name` render as `functions.unknown` (`gpt_oss.py:722-724`); verifiers only sets `name` if the harness supplied it (`verifiers/v1/dialects/chat.py:242-250`). The gen prompt prefills the `analysis` channel (`gpt_oss.py:454-459`).
14. **Stale docs**: `docs/algorithms.md:545` and `docs/training.md:168` list an older, shorter renderer set; `examples/README.md:87-90` says "Renderers are text-only today", contradicted by the code. Trust `MODEL_RENDERER_MAP`.
15. **Documented intentional deviations from `apply_chat_template`** (excluded from parity, covered by stability tests): Qwen3/3.5/3.6/3.8 and Gemma 4 26B/31B re-emit the empty disabled-thinking wrapper on history (`tests/parity.py:745-785`); gpt-oss gen prompt includes the analysis channel.
16. **`build_trajectory_step`** exists but is unused by prime-rl/verifiers at the pin; don't assume it's on the hot path.
17. **Not every bridge uses `trim_to_turn_close`.** GLM, Hy3 and Laguna XS.2/M.1 hand-roll the close check against the *last* token only (`glm5.py:398-405`, `laguna_xs2.py:417-424`), so a stop token in the middle of a completion is not trimmed, and for Laguna XS.2/M.1 a clean stop produces a prefix that differs from a full render by one `\n` (§7.2). Bridge coverage in the test catalog is per-model (`bridge=True/False`, `tests/parity.py:89-271`); check the flag before trusting a family's bridge.
18. **GLM-family turn ends are the next role's marker** (`<|user|>`/`<|observation|>`), sampled by the model and attributed to the following message; role-based loss filters drop them — use `sampled_mask` (prime-rl SFT does, `sft/data.py:350-353`).
19. **Kimi tool-call ids carry the function name** (`functions.{name}:{idx}`); harness-generated ids like `call_0` (verifiers' fallback, `train.py:121`) render into history but the parser would then recover the wrong name if that history were ever re-parsed.
20. **`warm()` pre-builds under the wrong key when `chat_template_kwargs` are used.** `TrainClient.__init__` warms `(model, config, None)` (`train.py:322-327`) but turns key on the popped `sampling.extra_body.chat_template_kwargs` (`train.py:369-376`), so the warmed renderer (~75–95 MB) is never used and the first turn pays a cold tokenizer load. Prefer typed config fields over `chat_template_kwargs`.

### 7.2 Per-family table

Rows marked † were read in full by a helper sub-agent; the cited lines marked ✓ I re-read myself. "Default retention" is the effective bridge policy with a default-constructed config. Paths are `renderers/<file>.py`.

| Renderer | Stop ids | Generation prompt | History reasoning | Tool call wire / parser | Bridge close & glue | Default retention | Multimodal | Gotchas |
|---|---|---|---|---|---|---|---|---|
| `qwen3` ✓ | `<\|im_end\|>`, `<\|endoftext\|>` | `<\|im_start\|>assistant\n` (+ `<think>\n\n</think>\n\n` if thinking off) | kept iff after last real user query and (last msg or has reasoning); empty wrapper re-emitted when thinking off (deviation) | hermes JSON `<tool_call>\n{"name","arguments"}\n</tool_call>` / `parse_qwen3` | `trim_to_turn_close({im_end,eot}, synth im_end)`; glue `\n` | tool_cycle | — | inline `<think>` in `content` is extracted into reasoning (`qwen3.py:476-486`) |
| `qwen3.5` ✓ | same | `…assistant\n<think>\n` / `<think>\n\n</think>\n\n`; per-model default polarity | kept iff after last query (or thinking-off wrapper) | XML `<function=..><parameter=..>`; bools `True/False` via `str()` / `parse_qwen35` (schema-aware) | as qwen3 | tool_cycle (all on 0.8B/2B: thinking off) | images (Qwen-VL pads) | system text *after* tool instructions; invalid-JSON string args → `{}` on re-render |
| `qwen3.6` † | same | same | + `preserve_thinking` | XML; non-str values `json.dumps` (`true`/`null`) (`qwen36.py:32-36`) | as qwen3 | tool_cycle | images | only change vs 3.5 is `_render_arg_value` (fixes bool-case drift) |
| `qwen3.8` †✓ | same | same; reasoning-effort instruction injected into (or as synthetic) system block when thinking on (`qwen38.py:23-46`) | `preserve_thinking=True` default | as 3.6 | as qwen3 | **all** | images | raises `ValueError("No user query found")` without a user message (`qwen38.py:52-56`) ✓; inline `<think>` stays visible content (`qwen38.py:58-63`) ✓ |
| `qwen3-vl` ✓ | same | `…assistant\n` (no think) | none (ignores `reasoning_content`) | hermes JSON / `parse_qwen3` | as qwen3 | all | images; video raises | `add_vision_id` bridge refusal without prior mm data |
| `prime-qwen3` †✓ | `im_end`, `endoftext` | `…assistant\n` | **key-presence**: `"reasoning_content" in msg` (any value) → `<think>{r}</think>`; absent → content verbatim (`prime_qwen3.py:377-415`) ✓; never dropped | Qwen3-coder XML / `parse_qwen35` | `{im_end,eot}` synth im_end; glue `\n` | all | — | tool-call turns never render reasoning (`_render_assistant_tool_calls`, `prime_qwen3.py:430+`) ✓ → reasoning before a call does not round-trip; bridge renders unknown roles instead of refusing |
| `gemma4` † | `<turn\|>`, `<\|tool_response>`, `<eos>` | `<\|turn>model\n` (+ empty `<\|channel>thought\n<channel\|>` on 26B/31B when thinking off); after a tool response, `<\|channel>thought\n` if thinking on | after last user, or `preserve_thinking` on tool-call turns; empty-thought re-emit deviation on 26B/31B (`gemma4.py:13-18`) | `<\|tool_call>call:N{k:<\|"\|>v<\|"\|>}<tool_call\|>`, keys **dictsorted**; parsed inline (`gemma4.py:1106-1234`) | `trim_to_turn_close({turn,tool_response,eos}, synth <turn\|>)`; refuses if prefix ends in `<eos>`; tool responses continue the same model turn (`gemma4.py:1315-1371`) | all (thinking off by default) | images: `<\|image>` + N×`<\|image\|>` + `<image\|>`, N=`num_soft_tokens_per_image`; items `{pixel_values, image_position_ids}`; client renames to `pixel_position_ids` | small-template detection is a narrow string probe; tool-call arg key order not preserved; `<\|tool_response>` is both a stop id and the response opener |
| `glm-5` / `glm-5.1` †✓ | `<\|endoftext\|>`, `<\|user\|>`, `<\|observation\|>` | `<\|assistant\|><think>` / `<\|assistant\|></think>` | after last user or `clear_thinking=False` | `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>` / `parse_glm` (`UNKNOWN_TOOL` validation) | **no `trim_to_turn_close`**: if last token ∉ stops, append `<\|endoftext\|>`; dedups the next role marker if the model already sampled it (`glm5.py:398-496`) ✓ | tool_cycle | — | next-role marker is the assistant's stop token and is attributed to the *next* message but `sampled=True` (use `sampled_mask`, not role filters; `ensure_final_stop` for SFT); a model that stopped on `<\|user\|>` followed by a tool message yields `<\|user\|><\|observation\|>` [inferred from ✓ code]; `_visible_text(None)` → `"None"` |
| `glm-5.3` † | same | `<\|assistant\|><think>` always; `<\|system\|>Reasoning Effort: …` preamble | kept unless `clear_thinking=True` and before last user | as GLM-5; full render reorders tool results by call id | inherited GLM-5 bridge (does not reorder) | all | — | `json.loads(arguments)` without try — bad JSON raises |
| `glm-4.5` † | same | `<\|assistant\|>` (thinking on) / `<\|assistant\|>\n<think></think>` + `/nothink` appended to user (off) | only after last user | `<tool_call>name\n<arg_key>…` / `parse_glm` | GLM custom close | tool_cycle | — | history `sampled_mask` matches neither gen-prompt mode: `\n` after `<\|assistant\|>` is marked unsampled though the thinking-on gen prompt (`<\|assistant\|>` only) leaves it to the model, while `<think></think>` is marked sampled though the thinking-off gen prompt prefills it (`glm45.py:256-261`, `522-531`) — SFT-only effect (RL masks come from real completions) |
| `minimax-m2` † | `[e~[` | `]~b]ai\n<think>\n` always | only after last user | `<minimax:tool_call><invoke name=..><parameter name=..>` / `parse_minimax` | `{eos}` synth eos; glue `\n` | tool_cycle | — | `content=None` renders literal `"None"` |
| `deepseek-v3` † | EOS | `<｜Assistant｜>` (omitted after tool outputs) | no reasoning channel | `<｜tool▁calls▁begin｜>…function<｜tool▁sep｜>name\n```json…```` / `parse_deepseek_v3` | `{EOS}` synth EOS | all | none (images dropped) | **`tools` schema is never rendered** into the prompt |
| `deepseek-r1` † | EOS | `<｜Assistant｜><think>\n` | always stripped (`split("</think>")[-1]`) | as V3 | as V3 | **template** → never bridges by default | — | every turn is a full re-render; for R1-Distill use qwen3/llama-3 |
| `deepseek-v4` † | EOS | `<｜Assistant｜></think>` (chat, default) / `<｜Assistant｜><think>` + effort preamble | kept if thinking and (tools present, `drop_thinking=False`, or after latest query) | DSML `<｜DSML｜invoke>`/`parameter string="true\|false"`; tool results folded into the user turn, sorted by call id / `parse_deepseek_v4` | `{EOS}` synth EOS; **refuses >1 tool message** | all | none | reference is a Python encoder (no Jinja); possible query-boundary mismatch with tool results when thinking+drop and no tools [UNVERIFIED] |
| `hy3` † | `<｜hy_eos｜>` | `<｜hy_Assistant｜><think></think>` (`no_think`) / `<think>` | `is_training` or preserved (default = tools present) or after last user | `<tool_calls>\n<tool_call>n<tool_sep>…</tool_calls><eos>` / `parse_hy3` | custom: append EOS unless last is EOS or `</tool_responses>`; refuses system messages | **per call**: all with tools, tool_cycle without | — | all system messages hoisted into the header; `:opensource`-suffixed tokens; Hy3-preview unsupported |
| `kimi-k2` † | `<\|im_end\|>` | `<\|im_assistant\|>assistant<\|im_middle\|>` | `reasoning_content` dropped; content verbatim | section tokens; call id `functions.{name}:{idx}` carries the name / `parse_kimi_k2` | `{im_end}` synth im_end; open-think appends *text* `</think>` | all | — | tool-call ids must follow `functions.name:idx` to round-trip; module docstring contradicts code; `trust_remote_code` tokenizer pinned by revision |
| `kimi-k2.5` (K2.5/K2.6) † | `im_end`, `endoftext` | `…<\|im_middle\|><think>` or `<think></think>` (text) | `<think></think>` up to last non-tool-call assistant, full after | TypeScript tool declaration; section tokens | `{im_end,eot}` synth im_end | tool_cycle | images: one `<\|media_pad\|>` per image; items `{pixel_values, grid_thws}`; `supports_process_multimodal=True` | **no `_build_mm_features` branch** → media via `generate(process_multimodal=True)` raises (`client.py:502-505`) ✓ |
| `laguna-xs.2` / `laguna-m.1` †✓ | `</assistant>`, `〈\|EOS\|〉` | `<assistant>\n<think>` / `<assistant>\n</think>` | every turn | `<tool_call>name\n<arg_key>…` (plain-text arg tags) / `parse_laguna_xs2` | custom: on truncation append `</assistant>\n`; **on a clean stop no `\n` is added** before the next `<user>`, whereas `render()` emits `</assistant>\n` (`laguna_xs2.py:417-424` vs `600-601`) ✓ — a divergence from full render; these two are `bridge=False` in the test catalog (`tests/parity.py:233-238`) | all | — | M.1 prefers `reasoning` over `reasoning_content`; XS.2 injects a default system prompt |
| `laguna-xs-2.1` / `laguna-s-2.1` † | same | `<assistant><think>` / `<assistant></think>` | kept iff `enable_thinking` (S-2.1: or `preserve_thinking`) | packed, no newlines | truncation appends `</assistant>`; glue `\n`; `bridge=False` in the test catalog (`tests/parity.py:239-240`) | all | — | S-2.1 defaults thinking **on**, XS-2.1 off |
| `llama-3` † | `<\|eot_id\|>`, `<\|end_of_text\|>`, `<\|eom_id\|>` | `<\|start_header_id\|>assistant<\|end_header_id\|>\n\n` | n/a | single bare JSON `{"name","parameters"}` / `parse_llama_3` | `{eot,eot_text,eom}` synth eot | all | — | >1 tool call raises; `tools=[]` still switches to tool mode; `date_string` pinned to 26 Jul 2024 |
| `nemotron-3` / `-ultra` / `-3.5` †✓ | `im_end`, `endoftext?` | `…assistant\n<think>\n` / `<think></think>` | collapsed to `<think></think>` before last user when `truncate_history_thinking` | Qwen3.5 XML, `str()` values / `parse_qwen35` | `{im_end,eot}` synth im_end; glue `\n`; **refuses when an effort hint is active** | tool_cycle (`nemotron3.py:129-138`) ✓ | — | Super detected by `"super"` in `name_or_path`; Ultra/3.5 glue `</think>` without `\n` |
| `inkling` † | `<\|content_model_end_sampling\|>`, `endoftext?` | `<\|message_model\|>` (effort in a system message) | always kept | `<\|message_model\|>{name}<\|content_invoke_tool_json\|>{"name","args"}<\|end_message\|>` / `parse_inkling` | `{end_sampling,eot}` synth end_sampling; refuses tool messages with `tool_call_id` but no `name` | all | images **and audio** (pre-decoded waveform only); no `supports_process_multimodal` | media through `generate()` raises either way ✓ (`client.py:266-271`, `502-505`); `content=""` vs `None` render differently |
| `gpt-oss` ✓ | `<\|return\|>`, `<\|call\|>` | `<\|start\|>assistant<\|channel\|>analysis<\|message\|>` | analysis kept only for unfinished tool cycles (`auto_drop_analysis`) | commentary to `functions.N` JSON / `parse_gpt_oss` | `trim({return,call}, synth <\|end\|>)`; no glue | tool_cycle | — | date defaults to today; tool results need `name` (else `functions.unknown`) |
| `default` ✓ | `[eos_token_id]` | template's | template's | optional `tool_parser` | always `None` | template | **VLMs rejected at auto-resolution** | O(N) renders, no masks |

---

## 8. For a custom framework

**Essential design (keep):**
- **Token ids are the source of truth; messages are a view.** Own tokenization client-side and call the engine token-in/token-out. This is the single decision that makes RL on multi-turn agents correct; everything else in this package serves it.
- **The bridge with a proof obligation + safe fallback.** `bridge(prev_prompt, prev_completion, new_msgs) -> tokens | None` with an exact-prefix contract, synthesized-close-as-prompt on truncation, and refusal on anything it can't prove. Pair it with a *token-level* prefix check at trajectory-assembly time (verifiers does this in `graph.py:788-813`) so a failed bridge degrades to "new sample", never to corrupted data.
- **Per-token attribution emitted at render time** (`message_indices`, `sampled_mask`, `is_content`). It makes SFT masks, ECHO-style tool-body CE, length penalties, and per-turn stats O(1) renders, and removes role-filter bugs (GLM closers attributed to the next message).
- **Parse by atomic token ids, record every attempt with status + span.** Loss masking of malformed tool calls and schema-adherence rewards need this; engine parsers throw it away.
- **Differential parity testing against the reference encoder** over a model × scenario × kwarg matrix, with declared (not skipped) exclusions.

**Incidental / simplify:**
- ~30 hand-written renderers each re-implement emitters, bridge scaffolding and tool rendering. A framework could factor a declarative "template spec" (role wrappers, glue, close ids, tool codec, reasoning policy) + a shared engine, keeping hand-code only for exotic formats (Harmony, DSML, Inkling segments). The duplicated `bridge_to_next_turn` bodies (e.g. Qwen3 vs Qwen3.5 vs Qwen3-VL) are ~80% identical.
- The typed-config zoo (`_template_fields` / `_internal_fields` classification) is good hygiene but heavy; keep the discriminated union, keep the allowlist for pass-through kwargs.
- `DefaultRenderer`'s incremental O(N) attribution is a stopgap — in a custom framework, refuse to train on models without a real renderer (prime-rl's SFT validator already does).

**Coupling points to watch:**
- Engine behaviour at stop tokens (returns the stop id, not trailing scaffold) is baked into every bridge's glue.
- The generate route's response schema (logprob `token_id:N` strings, routed-experts object) is vLLM-specific; multimodal feature encoding imports vLLM internals client-side.
- The tokenizer must be identical across renderer, engine, and trainer — make it a single config value, not three.
- Thinking-retention policy is a *modeling* decision (train/infer context shape) disguised as a tokenization knob; expose it at the algorithm level and log extension-break rates per env.

**What we'd do differently:** decouple the "bridge" from the "renderer" object so bridging is a generic operation over (close ids, glue, per-message encoder); make the engine client pluggable (vLLM / SGLang `/generate`) with the multimodal encoder behind an interface; add a runtime assertion (cheap: compare first/last k ids) that the renderer tokenizer and engine agree on special-token ids at startup.

---

## 9. Open questions

1. How many env-server worker processes (hence renderer pools; each builds ≥1 renderer per `(model, config, kwargs)` key, +1 per 256 concurrent turns, ~75–95 MB each) a prime-rl run spawns per node — G's pool sizing (`08-harnesses-runtimes-serve.md`).
2. How vLLM's `/inference/v1/generate` handles `features.kwargs_data = null` (hash-only cache path) under prime-rl's router — relevant if multimodal ever bridges. → E.
3. Kimi K2.5/K2.6 and Inkling media cannot go through `generate()` as prime-rl calls it (§7.1 #12). Is multimodal RL on these families simply unsupported, or is there an intended SGLang/other path? → renderers upstream.
4. The Prime Intellect renderers blog post referenced by `docs/algorithms.md:552` is external and not verified here.
5. Laguna XS.2/M.1 bridge `\n` divergence on clean stops (§7.2) — confirmed by reading, not by running; is it a real bug or does the engine return the trailing `\n`? (Engines stop *on* `</assistant>`, so the former is likely.) → run `tests/test_bridge.py` with those models flipped to `bridge=True`.
