# verifiers v1 core: rollout lifecycle, interception, token-in train client, trace graph — prime-rl @ b944873

> **Scope.** The `verifiers.v1` object model (Env / Taskset / Task / Agent / Harness / Rollout / Episode / Trace), the rollout lifecycle, and **seam S4 in full**: how a harness's model calls reach vLLM during training (interception server, dialects, eval vs train clients, renderer bridging, trace/graph → training samples), plus the prime-rl ↔ verifiers API surface.
>
> **Pin.** prime-rl `b944873`; verifiers submodule `69cc0f9` (`deps/verifiers`); renderers `6b8da3f` (`deps/renderers`).
>
> **Files read in full** (all under `deps/verifiers/verifiers/v1/` unless noted):
> `__init__.py` (382), `env.py` (411), `taskset.py` (111), `task.py` (323), `state.py` (20), `types.py` (291), `errors.py` (118), `rollout.py` (554), `episode.py` (168), `session.py` (605), `trace.py` (806), `graph.py` (956), `agent.py` (752), `harness.py` (381), `semantic.py` (132), `judge.py` (229), `judges/{__init__,reference,rubric}.py` + `reference.txt`, `rubric.txt`;
> `clients/{__init__,base,client,eval,train}.py` (21/41/89/218/459); `dialects/{__init__,base,chat,responses,anthropic}.py` (25/387/597/775/679); `interception/{__init__,base,pool,server}.py` (118/66/151/1037), `interception/tunnel/{__init__,base,custom,prime}.py`;
> `configs/{__init__,client,agent,retries,taskset,task,env,harness,judge,serve,runtime}.py`, `configs/cli/{__init__,env,init,eval,debug,replay,validate}.py`;
> `utils/{decorators,retries,compile,loaders,score,artifacts,generic}.py`;
> `serve/{server,client,delta,types,encoding,__init__}.py` (boundary only — S3 is G's), `envs/single_agent/env.py`, `envs/best_of_n/env.py`, `envs/user_sim/env.py`, `harnesses/bash/harness.py`, `harnesses/utils/launch.py`, `acp/__init__.py` (lines 1–160);
> `deps/renderers/renderers/client.py` (637);
> docs: `deps/verifiers/{AGENTS.md,README.md,docs/overview.md,docs/v1/*.md}`, `skills/*/SKILL.md`;
> tests: `tests/v1/test_graph.py` (565), `test_trace.py` (735), `test_e2e.py` (927), `test_scoring.py` (164);
> prime-rl side: `src/prime_rl/orchestrator/{envs,trajectories,clients,dispatcher,train_sink,generation_source}.py`, `algo/{base,routing,grpo}.py` (relevant parts), `annotations.py`, `entrypoints/env_server.py`, `packages/prime-rl-configs/src/prime_rl/configs/{env_server,orchestrator}.py` (lines 1–200 + renderer field).
>
> **Related docs:** 03-orchestrator.md (B: dispatcher/sinks), 04-algorithms-loss-data-path.md (C: advantages, samples), 06-inference-and-transports.md (E: `/inference/v1/generate` server side, router), 08-harnesses-runtimes-serve.md (G: env server pool, runtimes, harness programs, ACP, MCP), 09-renderers.md (H: `render`/`bridge_to_next_turn`), 10-envs-and-tasks.md (I).

---

## 1. Mental model

**verifiers v1 is an execution substrate for "an agent program solving a task", whose only non-negotiable contract is: every model call the program makes goes through a verifiers-owned HTTP server.** The program (the *harness*: a tiny in-house chat loop, Codex, Claude Code, mini-swe-agent, …) runs inside a *runtime* (host subprocess, docker, a Prime/Modal sandbox). It is pointed at an **interception server** (`OPENAI_BASE_URL`-style) and never talks to the provider. The interception server parses each request in the program's native wire format (*dialect*: OpenAI chat, OpenAI Responses, Anthropic Messages), decides whether the turn is allowed (limits, `@stop` hooks), rewrites it if task hooks want (`@intercept`), forwards it to a *client*, records the exchange into a per-rollout **`Trace`**, and returns a native-format response to the program. Everything the trainer needs is reconstructed from that recording — the harness never reports anything.

**Two clients split eval from training.** The `EvalClient` relays the (overridden) native JSON to any OpenAI/Anthropic-compatible endpoint and parses a copy for the trace: no token ids. The `TrainClient` does *not* forward the request: it parses it into typed messages, **tokenizes client-side with a `renderers.Renderer`**, calls vLLM's token-in/token-out `/inference/v1/generate`, gets back exact completion token ids + sampled-token logprobs (+ optional MoE routing and sampling masks), parses the completion back into a message with the same renderer, and serializes an OpenAI `chat.completion` for the program. To keep tokens exact across turns, it *bridges*: when the new prompt is "previous prompt + previous completion + new tool/user messages", it asks the renderer to extend the previous turn's **exact** token ids instead of re-rendering the whole conversation (the "extension property"). Training is therefore chat-completions-dialect only (`clients/train.py:342-350`). This is enforced per model call at runtime (a 502 to the harness), never at config validation; §7.1 has the per-harness table.

**The trace is a message graph, not a list.** Each distinct message is stored once as a `MessageNode` holding only the tokens it *adds* (`graph.py:1-19`). A new request is matched against the graph by message hash (then tightened by token identity); the matched prefix is reused and the unmatched tail + the sampled assistant are appended. Whenever the harness rewrites history (compaction, subagents, dropped `<think>`, re-tokenization drift), the prefix breaks and a **new branch** is born. Each root→leaf path (`Trace.branches`) is one exact token sequence = one training sample with a loss mask (`sampled` tokens only) and per-token logprobs. prime-rl receives, per dispatched task, one `Episode` holding ≥1 `Trace` (one per agent run); it turns each trainable branch of each trainable trace into one `TrainingSample` (`orchestrator/trajectories.py:136-180`).

---

## 2. Where it runs

```
 orchestrator process (prime-rl)                  env-server process (verifiers, one per train/eval source; pool of workers)
 ─────────────────────────────                    ─────────────────────────────────────────────────────────────────────────
 Env.start: vf.load_taskset (client-side tasks)    EnvServer (ZMQ ROUTER)  ── per worker ──►  Env (SingleAgentEnv / custom)
 Dispatcher ── EnvClient.run(RunRequest) ──ZMQ──►   _run → Env.run_slot → run_episode → Agent.run → Rollout
    client = Train/EvalClientConfig (JSON)            Env.serving(): shared tool servers + ONE Interception (elastic pool)
    model, sampling(+cache_salt), task_data           │
 ◄── delta frames (trace diffs) + reply(head) ──────  │  Interception server (aiohttp, host side of env-server worker)
 WireEpisode ─► TrainSink ─► trace_to_samples         │     routes /v1/chat/completions, /v1/responses, /v1/messages, ...
                                                      │     owns one Client per distinct client config (TrainClient/EvalClient)
                                                      │         TrainClient ──HTTP──► vLLM/router  POST {base}/inference/v1/generate
                                                      │                              (X-Session-ID: trace.id)
                                                      └── Runtime (subprocess / docker / prime / modal sandbox)
                                                             harness program ──HTTP(S) via loopback / docker proxy / prime tunnel──► interception
```

- **Process placement.** The interception server lives in the **env-server worker process** (host side), created in `Env.serving()` (`env.py:360-386`) and held for the worker's life (`serve/server.py:193`). The harness program lives in the runtime; it reaches the interception at `runtime.host_url(f"{base_url}/v1")` (`rollout.py:260`). Base runtime `host_url` is identity (`runtimes/base.py:370-372`); docker rewrites loopback to a proxy callback URL (`runtimes/docker/__init__.py:257-263`); remote runtimes get a **tunnel** URL (§3.6). Spawning of env-server processes and addresses is S1/S3 — see 01 (A) and 08 (G).
- **Default is remote.** `AgentConfig.runtime` defaults to `PrimeConfig()` (`configs/agent.py:30`); `PrimeRuntime.is_local = False` (`runtimes/prime.py:144`), so by default `requires_tunnel` is true and the elastic pool mints Prime tunnels (frpc) (`interception/pool.py:124`). A local training run must set `--env.agent.runtime.type subprocess|docker` (or similar) to stay on loopback. With `requires_tunnel=False` the pool still constructs servers with `PrimeTunnelConfig`, but no tunnel object is made and they bind `127.0.0.1:0` (`server.py:343-345, 443-455`). Shipped prime-rl configs train with `harness.id = "null"` (72), `"bash"` (17) or `"rlm"` (12), e.g. `intellect-3.1/rl.toml:62-68` (count per 10/08 reviews).
- **Lifecycle.** `Env.serving()` enters shared tool servers, then `make_interception(...)` (`env.py:364-376`); `ElasticInterceptionPool.start` warms one server in the background (`interception/pool.py:105`); each rollout `acquire`s a slot (registers a `RolloutSession` under fresh secrets), releases it on exit (`interception/server.py:389-395`). On worker shutdown `Env.stop()` then the pool's `AsyncExitStack` tears down every server + tunnel + client (`interception/base.py:48-50`, `server.py:365`).
- **Renderers** (tokenizers) are process-global per env-server worker, cached on the class (`clients/train.py:222-230`), warmed when the `TrainClient` is constructed if `renderer_model_name` is set (`train.py:322-327`; prime-rl always sets it, `orchestrator/clients.py:85`).

---

## 3. Mechanics

### 3.1 Object model (who owns what)

| Object | File | What it is | Owns / lifetime |
|---|---|---|---|
| `Taskset[TaskT, TasksetConfigT]` | `taskset.py:38` | Loader: `load()` yields typed `Task`s; `INFINITE` flag; `head`/`shuffle` views; `toolsets()` = shared MCP servers | Loaded **client-side** by prime-rl (`orchestrator/envs.py:105`) and **also** in every env-server worker (`env.py:104`), but the worker never calls `load()` — it rebuilds tasks from wire data (`serve/server.py:80-87`). |
| `TaskData` | `task.py:81-129` | Frozen pydantic row: `idx, name, description, prompt (str\|Messages\|None), system_prompt, image, workdir, skills, network_allow ["*"], network_block, artifacts, artifact_max_bytes (32 MiB), timeout: TaskTimeout, resources: TaskResources` | The **only** task state on the trace (`trace.task.data`). `WireTaskData` (`task.py:132`) = `extra="allow"` for consumers without the package. |
| `Task[DataT, StateT, ConfigT]` | `task.py:142` | Behavior around one row: `setup/finalize/stage_verifier/validate`, `@reward/@metric/@stop/@intercept` methods, `toolsets()` (per-rollout MCP), `score()` | Built per dispatched task. `hash` = sha256 of `data.model_dump(json, exclude_none)` (`task.py:150-154`); `key` defaults to hash (`:156-164`). |
| `State` | `state.py:9` | Mutable per-rollout shared state (`artifacts: dict[str, bytes\|None]`), synced with tool servers over `/state` | On `Trace.state`, **excluded from serialization** (`trace.py:435`) — never crosses the env-server wire. |
| `Harness[ConfigT]` | `harness.py:34` | The program: `setup(runtime)`, `launch(...)`, optional `resume`/`session`, `@metric`s, capability flags | Stateless value, one loaded object per distinct harness config per env (`env.py:116-124`). |
| `HarnessSession` | `harness.py:309` | Rollout-scoped handle; default calls `launch` once then `resume` per user turn | Per rollout. ACP harnesses override with a live-process session (`acp/__init__.py:103-136`). |
| `AgentConfig` | `configs/agent.py:27-51` | Seat config: `harness, runtime (PrimeConfig()), model, client, sampling, max_turns, max_{input,output,total}_tokens, timeout, retries` | Field of an `EnvConfig`; field name = agent name. |
| `Agent` / `_EpisodeAgent` | `agent.py:262`, `:596` | Harness × model × runtime-policy; `run(task) → Trace`; `interaction(task)` for turn-by-turn user control | `_EpisodeAgent` is minted fresh per episode, borrows the env's interception + shared tools (`env.py:192-240`). |
| `Env[ConfigT]` | `env.py:76` | Control flow between agents: `run(task, agents)`, optional `setup/finalize/complete/start/stop` | One per env-server worker; `SingleAgentEnv.run` = `await agents.agent.run(task)` (`envs/single_agent/env.py:28`). |
| `Rollout` | `rollout.py:55` | One agent run's lifecycle: `open → step* → close` (or `abort`) | Per agent run; creates the `Trace` and the `RolloutSession`. |
| `RolloutSession` | `session.py:137` | The per-rollout unit the interception server serves: ctx, trace, limits, stops, interceptors, tool-gate state, idempotency cache, assigned `Client` | Registered on a server under a model secret + state secret (`server.py:368-377`); sealed by `release()` (`session.py:581-585`). |
| `Trace` | `trace.py:399` | Per-agent-run record: `task, agent, tools, nodes (graph), calls (ModelCall[]), rewards, metrics, info, root_reply, state, extra_usage, is_completed, ok, stop_condition, errors, timing, mm_token_type_id_map, request/response_rewrites` | Built live by the interception server; streamed as deltas. |
| `MessageNode` / `Branch` | `graph.py:83`, `trace.py:194` | Graph node (message + delta tokens + mask + logprobs + annotations); root→leaf view = one training sample | Nodes append-only; branches are computed views (`trace.py:532-570`). |
| `Episode` | `episode.py:87` | One dispatched task's result: `id, env, task, group, run, ok, errors, traces` | The wire atom returned by the env server (`WireEpisode`, `episode.py:166`). |

`Agent` identity: an unpinned `harness` becomes the taskset's bundled harness else `bash` (`configs/env.py:159-165`, `utils/loaders.py:153-161`); in an env, unpinned `model/client/sampling` inherit the run's `ModelContext`, and a seat's `sampling` is **deep-merged over** the run's (`env.py:206-225`).

### 3.2 Episode path: env server → Env → Agent → Rollout

1. `EnvServer._run(req)` builds `ModelContext(client=req.client, model=req.model, sampling=req.sampling)` (`serve/server.py:89-100`) and one `RunSlot` from `_build_task(req.task_data)` = `data_cls.model_validate(task_data)` wrapped in `task_cls(data, env.config.taskset.task)` (`:80-87`). A `TaskData` field with `exclude=True` is refused at server construction (`:43-50`).
2. `Env.run_slot(slot, ctx, self._gate, on_trace=streamer.watch)` (`env.py:313-358`): holds one `--max-concurrent` permit per attempt; wraps `run_episode` in `run_episode_with_retry(attempt, config.retries)` (`utils/retries.py:107-141`; off by default: `RetryConfig.max_retries=0`, `configs/retries.py:11`).
3. `Env.run_episode` (`env.py:242-305`): mints `Episode(env=EnvInfo(id=env_id), task=TraceTask(...))`, builds per-episode `Agents`, then under `asyncio.timeout(config.timeout.episode)` runs `setup(agents)` and `run(task, agents)` inside `boundary(EnvError, ...)`. A `run()` that minted no trace raises (`:274-278`). Exceptions become `episode.errors` and return early (partial traces kept, `ok=False`). Then `finalize(task, episode)` under `timeout.finalize` (cross-trace judgement: `record_reward/record_metric` on sibling traces, e.g. best-of-n `envs/best_of_n/env.py:34-44`). Finally `episode.ok = all(t.ok for t in episode.traces)` (`:304`).
4. `_EpisodeAgent.run(task)` (`agent.py:655-673`) acquires the episode's agent semaphore (`EnvConfig.max_concurrent_agents`, default 1, `configs/env.py:50`), calls `Agent.run`, and appends the final trace to `episode.traces`. Its `_watch` stamps `trace.agent.name` and `trace.agent.trainable` at mint and discards retried attempts from the live view (`agent.py:633-653`).
5. `Agent.run` (`agent.py:387-436`): per-agent whole-rollout retries (`AgentConfig.retries`, retry when any `trace.errors[*].type` matches include/exclude, `utils/retries.py:76-91`, backoff `min(0.5·2^a,30)·U(0.5,1.5)` `:39-41`); never retries into a borrowed runtime. Each attempt = `_run_once` (`:438-471`): `Rollout(...)`; `if await run.open(): await run.step(); if run.ok: trace.stop("agent_completed")`; `await run.close()`.
6. `_rollout_params` (`agent.py:538-579`) resolves: runtime config per task (`utils/compile.py:21-73`: injects `task.data.image` — refuses subprocess; applies task `workdir`, network policy intersection, resources where still default), pairing checks (`compile.py:76-112`: MCP support, skills, `NEEDS_CONTAINER`), stage timeouts (agent timeout = `timeout.rollout` > task `timeout.agent` > 4 h default; `0` = unbounded, `agent.py:57-83`; capped at 24 h on non-Prime remote runtimes `compile.py:115-134`), and which interception to ride (`_interception_for`, `agent.py:348-368`: an injected one always — in an env it is the env's pool).

### 3.3 Rollout lifecycle (one agent run)

```
Rollout.__init__  → Trace minted (task, state=State subclass, agent=AgentInfo(config)) → on_trace(trace)
                    RolloutSession(ctx, trace, network_policy, stops, limits, interceptors, gates_tools)
open()            timing.boot  : make_runtime(...).start(); prepare_setup()
                  timing.setup : task.setup(trace, runtime)   [TaskError]   ┐ one shared setup deadline
                                 harness.setup(runtime)       [HarnessError]┘
                                 task.toolsets(config)        [ToolsetError]
                                 serve_interception → Slot(base_url, model_secret, state_secret)
                                 endpoint = runtime.host_url(base_url + "/v1")
                                 serve_tools(...) → mcp_urls ; runtime.prepare_execution([endpoint, *mcp_urls])
                                 (request interceptors → prepare_users rewrites the prompt)
                                 harness.session(ctx, trace, runtime, endpoint, secret, mcp_urls, data[, tool_interception_url])
step(messages?)   timing.agent : HarnessSession.turn → launch() / resume() → program runs to exit
                                 (every model call → interception server → §3.5)
close()           harness_session.close(); stack.aclose() (interception slot released → session sealed; tool servers down)
                  if not failed: timing.finalize: task.finalize(trace, runtime) [+ collect artifacts]
                                 timing.scoring : gather(task.score(trace, runtime), harness.score(trace, runtime))
                  trace.is_completed=True; trace.ok = not failed; split_agent_time(); harness.cleanup; runtime.stop()
```

Mechanical details:

- **Trace creation** (`rollout.py:87-98`): `Trace(task=TraceTask(type, data, key, hash), state=state_cls(type(task))(), agent=AgentInfo(config=agent_config))`. `on_trace` fires immediately (this is how the env server starts streaming a trace before any I/O).
- **Hooks classification** (`rollout.py:101-128`): `@vf.intercept` methods are routed by their annotated parameter type: `Request` → request interceptor, `Response` → response interceptor (`session.py:50-64`); `@vf.stop` hooks (`task.hooks("stop")`, decorated + config-plugged, `task.py:289-307`) may take `Request`, `Response` or `Trace`.
- **`open()` failure** is captured onto the trace via `fail(e)` (`rollout.py:343-345`) → `trace.record_error` stops the trace with `stop_condition="error"`, `ok=False` (`trace.py:773-788`). A cancellation mid-setup calls `abort()` (frees runtime/servers, no scoring) and re-raises (`rollout.py:346-351`).
- **`step()`** (`rollout.py:358-431`): computes the segment deadline from the remaining agent budget (a cumulative budget across interaction segments, `:374-378`). On `TimeoutError` with the deadline passed → `HarnessError("agent timeout ...")` (`:392-405`). On any other exception: if the session is `stopped` (a `@stop`/limit fired), the exit is clean; else if the interception stashed a real model-call error on `session.error` and the harness raised a `RolloutError`, **the stashed error is recorded instead of the secondary `HarnessError`** (`:406-415`). A segment that committed no turn returns `False` (`:431`).
- **`HarnessSession.turn`** (`harness.py:340-348`) runs the program under `boundary(HarnessError, ...)`, then `_check_result`: a nonzero exit is OK if a stop condition is set; else `SandboxError` if the runtime died, else `HarnessError` with the stderr tail (`harness.py:167-183`).
- **`close()`** (`rollout.py:451-554`): finalize and scoring run **only if not failed** (`:479`) — a *stopped* run (limit/@stop) is complete and scores its partial trajectory. Scoring is `asyncio.wait_for(gather(task.score, harness.score), timeouts.scoring)` inside `boundary(TaskError, "scoring")` (`:497-505`). The runtime is still alive during scoring (rewards may exec in it); it is stopped afterwards unless borrowed (`:539-545`).
- **Stop conditions** written by the framework: `"agent_completed"` (`agent.py:462`), `"user_closed"` (`agent.py:258`), limits `"max_turns" | "max_input_tokens" | "max_output_tokens" | "max_total_tokens"` (`session.py:96-115`), a `@stop` function's `__name__`, `"error"` (`trace.py:788`). `Trace.is_truncated` is true for the limit names, `"compaction_failed"`, or when the last non-error call had `finish_reason == "length"` (`trace.py:659-671`).

### 3.4 Scoring: where reward is computed

- **Task scoring** `Task.score(trace, runtime)` (`task.py:203-274`): skipped if `scoring_deferred` (isolated-verifier env). Collects `hooks("metric")`, `hooks("reward")`, and `plugged_judges()` (`task.py:284-287`: `load_judge(cfg)` for each `TaskConfig.judges` entry). With `runtime=None` (offline replay), hooks whose `runtime` param has no default are skipped (`:224-242`). **Seeds** every expected key to `None` first (`utils/decorators.py:48-54`) so a crash leaves visible unscored keys. Hooks run in three **sequential phases**: metrics, then rewards, then judges. Within a phase they are `asyncio.gather`ed, with arguments injected by *parameter name* (`task`, `trace`, `runtime`; `utils/decorators.py:29-45`; `task.py:243-273`). Hooks must be `async`: `gather` over a sync hook's float raises `TypeError`, which becomes a `TaskError` (`utils/loaders.py:242` "the imported async function"). A hook may return a float or a `Mapping[str, float]` (multiple keys). Rewards are recorded as `Reward(score, weight)` with weight from `@vf.reward(weight=...)` (`_vf_weight`, `decorators.py:130`) or config (`RewardFunctionConfig.weight`); judges use `JudgeConfig.weight` (`configs/judge.py:19`).
- **`trace.reward`** = `Σ score·weight` over non-None rewards (`trace.py:489-491`). prime-rl's GRPO reads exactly this (`orchestrator/algo/grpo.py:28`).
- **A scoring failure is an error, not a zero.** Any exception or timeout escaping `Task.score` becomes `TaskError` (`boundary`, `errors.py:76-89`; `rollout.py:497-505`). `Rollout.close` then calls `fail(e)`, so `trace.ok=False` (`rollout.py:507-509`). prime-rl drops that trace from training (`algo/base.py:35-42`). It does not train it with reward 0.
- **Harness metrics** `Harness.score` records `@metric`s on the harness object (`harness.py:214-228`) — never rewards.
- **Cross-trace judgement** = `Env.finalize(task, episode)` after all agent runs, no live runtime (`env.py:163-170`).
- **Judges** (`judge.py`): `Judge.complete` opens a fresh `AsyncOpenAI` per call (`build_async_openai(self.config)`, `judge.py:184`; `JudgeConfig` *is* a `BaseClientConfig`, default model `"openai/gpt-5.6-luna"`, `configs/judge.py:14-22`) and records the exchange to `trace.info["judge_calls"]` + `trace.extra_usage` (`trace.py:735-756`) — judge usage never enters `trace.usage`. Built-ins: `ReferenceJudge` (yes/no vs `answer_field`; empty reply → 0.0 without a call; unparsable verdict **raises** so the rollout errors rather than scores 0, `judges/reference.py:63-92`) and `RubricJudge` (weighted criteria file, per-criterion metrics `"<judge>/<criterion>"`, batches via `max_criteria`, plain-text JSON by default, `judges/rubric.py:186-272`). The judge client has `max_retries=0` and no read timeout (`clients/base.py:12-31`). A missing key resolves to `"EMPTY"`, after a fallback to the Prime CLI config for pinference (`configs/client.py:93-104`). The provider's 401 then raises out of the reward, so the **trace fails** (it is not scored 0). The scoring timeout defaults to `None`, so a hung judge hangs its rollout (`agent.py:78-82`; task `timeout.scoring` also defaults to none).

### 3.5 THE generation path (seam S4): one model call during training

```mermaid
sequenceDiagram
    participant H as harness program (in runtime)
    participant I as InterceptionServer (env-server worker)
    participant S as RolloutSession / Trace graph
    participant C as TrainClient (+ElasticRendererPool)
    participant V as vLLM /inference/v1/generate
    H->>I: POST {endpoint}/chat/completions  Authorization: Bearer <model_secret>  (stream=true)
    I->>I: dialect by route; session by secret; body = apply_overrides(body, ctx.model, ctx.sampling)
    I->>I: idempotency / retry coalescing (x-stainless-retry-count, Idempotency-Key)
    I->>S: refused()? (limits, Trace-@stops)  → 400 "rollout stopped"
    I->>S: rewrite_request (Request-@intercept, Request-@stops, pinned tool results)
    I->>S: prepare_turn(trace, messages, tools) → PendingTurn(prefix_node_ids, path_len, tail)
    I->>C: get_response(dialect, body, ctx.sampling, session_id=trace.id, turn=PendingTurn)
    C->>C: acquire renderer; bridge_to_next_turn(prev_prompt_ids, prev_completion_ids, tail) or render(full)
    C->>V: POST {base}/inference/v1/generate {model, token_ids, sampling_params(+stop_token_ids, logprobs=1), cache_salt}  X-Session-ID
    V-->>C: choices[0].token_ids, logprobs.content, [routed_experts], [sampling_mask], finish_reason
    C->>C: renderer.parse_response → content/reasoning/tool_calls; Response(tokens=TurnTokens(...)); raw = serialize_completion
    C-->>I: Response
    I->>S: rewrite_response (Response-@intercept / @stops)
    I->>S: turn.commit(response) → nodes appended; ModelCall recorded; gate_tool_calls(node)
    I-->>H: SSE frames from Response.raw (one chunk + [DONE]); keepalives if > 60 s
```

#### 3.5.1 Routes, auth, session lookup

- `InterceptionServer.start` (`interception/server.py:423-459`) registers, for every dialect in `DIALECTS = (ChatDialect(), ResponsesDialect(), AnthropicDialect())` (`dialects/__init__.py:11`), a POST handler per `dialect.routes` and per `aux_routes`: `/v1/chat/completions` (`chat.py:425`), `/v1/responses` (`responses.py:372`), `/v1/messages` + aux `/v1/messages/count_tokens` (`anthropic.py:435-436`). Plus `GET /v1/models` (relayed from the session's endpoint, `server.py:935-968`), `GET|PUT /state`, `POST /tool`, `GET /task`. **The wire format is chosen by the route the SDK posted to**; harnesses declare nothing.
- Bind: `127.0.0.1:0` without a tunnel; the tunnel's `bind_host/bind_port` otherwise (`server.py:443-459`). `client_max_size = MAX_REQUEST_BODY = 1 GiB` (`:78`).
- Auth: the request's per-rollout **model secret** is read from the dialect's carrier — `Authorization: Bearer` (default, `dialects/base.py:299-302`) or Anthropic `x-api-key` (`anthropic.py:505-507`) — and looked up in `self.sessions` (`server.py:586-589`). The state/tool servers use a *separate* state secret (`server.py:368-377`), so a model bearer cannot read `/state`.
- How the harness is pointed at it: `Harness.launch(ctx, trace, runtime, endpoint, secret, mcp_urls, data)` receives `endpoint = runtime.host_url(base_url + "/v1")` and `secret = model_secret` (`rollout.py:260-261, 333-342`). Each harness wires them its own way: the bash/null chat program gets `--base-url={endpoint} --api-key={secret} --model={ctx.model}` over argv (`harnesses/utils/launch.py:52-57`); Claude Code gets `ANTHROPIC_BASE_URL=endpoint.removesuffix("/v1")`, `ANTHROPIC_API_KEY=secret`, `ANTHROPIC_MODEL=ctx.model` (`harnesses/claude_code/harness.py:93-95`). Per the docs, Codex speaks Responses and Claude Code Anthropic Messages (`docs/v1/architecture.md`). Full per-harness wiring is G's (08).

#### 3.5.2 `handle_request` step by step (`server.py:583-899`)

1. **Parse + override.** Body parsed with `pydantic_core.from_json` (fallback `json.loads`) (`:591-595`); `body = dialect.apply_overrides(body, session.ctx.model, session.ctx.sampling)` (`:596`) imposes the run's model and sampling in the dialect's shape:
   - chat: keep the program's fields, overlay `sampling.wire_args()` and `model` (and drop the program's `max_tokens`/`max_completion_tokens` if the run sets either) (`chat.py:583-597`);
   - responses: drops program `temperature/top_p/max_output_tokens/max_tokens`, maps `max_tokens→max_output_tokens`, `reasoning_effort→reasoning.effort`, adds `include: ["reasoning.encrypted_content"]` + `reasoning.summary="auto"` for OpenAI reasoning models (`responses.py:737-775`);
   - anthropic: always drops program `temperature/top_p`, keeps program `max_tokens` unless the run sets one, maps `reasoning_effort → output_config.effort` (`anthropic.py:657-679`).
   `SamplingConfig.wire_args()` flattens `extra_body` into top-level keys (`types.py:274-284`).
2. **Streaming decision.** `streaming = dialect.streaming(body)` (`body["stream"]`); `relay = streaming and not isinstance(ctx.client, TrainClientConfig)` (`:597-600`). **Training never relays** — the train client generates the whole response first.
3. **Retry atomicity.** `req_hash = blake2b(raw body)` (inline ≤1 MiB else on a thread, `:103-113`). Replay key = `"explicit:<Idempotency-Key>"` if that header is present (and the header is stripped from upstream headers), else `"retry:<path>:<hash>"` — but the retry key is only *consulted* when `x-stainless-retry-count > 0` (`:621-654`). A completed replay returns the cached bytes; an in-flight one **coalesces** onto the same future (`:673-691`). Cache: TTL 600 s, ≤64 completed per session (`:92-93`, `_prune_idempotent_requests :181-206`). Net effect: an SDK retry after a dropped connection never commits a second turn.
4. **ACP header.** `X-ACP-Model-Request-ID` is validated and stripped (`semantic.py:112-132`; `server.py:606-609`); it is stored on the `ModelCall.acp` so harness-published semantic edges can later resolve request ids → nodes.
5. **Parse, sealed/stopped checks.** `model_request = dialect.parse_request(body)` → typed `Request(messages, tools)` (`:656-659`; chat refuses `n != 1`, `chat.py:512-514`). Released session → 409; stopped → 400 (`:660-668`).
6. **Refusal** `session.refused()` (`session.py:587-605`): limits first (`RolloutLimits.reached`), then `Trace`-typed `@stop`s; sets `trace.stop(name)` and the handler returns 400 `"rollout stopped: <name>"` (`server.py:704-708`). The harness's SDK does not retry 4xx; its program exits; `_check_result` treats the exit as clean because a stop condition exists.
7. **Request rewriting** `session.rewrite_request` (`session.py:232-359`): re-imposes pinned tool results (rewrites the harness keeps re-sending), runs `Request`-typed `@intercept` hooks segment-by-segment over the **uncommitted tail** (only new user/tool messages may change; count/tools must not change; tool-call ids/names immutable), then `Request`-typed `@stop`s. If changed, patched back into the native body (`dialect.rewrite_request`, `server.py:715-716`). A request `@stop` records the prompt (`turn.commit_prompt()`, graph nodes with no tokens) and stops (`:729-738`).
8. **Capability mediation.** Under a restricted network policy, `dialect.mediate_external_capabilities` removes provider-side tools/URLs the sandbox policy cannot enforce and appends a user notice (`server.py:475-491`, `dialects/base.py:332-341`); body re-parsed if touched (`:740-744`).
9. **Graph prefix** `turn = graph.prepare_turn(trace, messages, tools)` (`server.py:745-747`, §3.7); `trace.preview(turn, turn.tail)` shows the pending tool results to live watchers (`:754`).
10. **`sample()`** (`:773-895`): resets `session.error`; calls `session.client.get_response(dialect, body, session.ctx.sampling, headers=upstream_headers, session_id=trace.id, turn=turn)` (`:793-800`) — or `relay` + `_collect_stream` for streamed eval. If the session was released meanwhile → 409 (`:806-809`). Response interceptors/stops (`session.rewrite_response`, `session.py:385-438`; interceptors may only replace the assistant message with **inert text**, and force `finish_reason="stop"`). Then **`node = turn.commit(call_response)`** (`:836`), `session.consume_prepared`, and `gate_tool_calls(node)` (`session.py:440-474`: runs request hooks against each proposed tool call with an empty probe result; a rewrite becomes that call's verdict served to the harness's `/tool` gate, or — for harnesses without `SUPPORTS_TOOL_INTERCEPTION` — ends the rollout).
11. **Errors.** A `RolloutError` (e.g. `ProviderError`) is stashed on `session.error` and returned as the dialect's error body with the error's `status_code` (5xx/429 so the SDK retries transient faults, 4xx otherwise) (`:847-861`). Any other exception becomes a 502 and is **not** stashed (`:862-870`). That covers the TrainClient's dialect `NotImplementedError`, the default renderer's "does not support tools" `ValueError` (`renderers/client.py:261-265`), and `MalformedGenerateResponseError`. The SDK retries the 502, and if the program then dies, `trace.errors` shows only a `HarnessError`. The real cause survives only in `trace.calls[*].error`. **Every real exchange** (success or failure, even cancellation) is recorded by `record_call` in `finally` (`:876-894`): `ModelCall(node, model, sampling=dialect.parse_sampling(body), endpoint=dialect.upstream_path, finish_reason, usage, time, error, policy, acp)` (`:518-581`). Replayed/coalesced retries never reach `record_call`.
12. **Serving.** `serve(response, events)` (`:756-771`): non-streaming → JSON of `Response.raw`; streaming → the provider's own SSE bytes when relayed unchanged, else `dialect.stream_events(response.raw)` (chat: **one** `chat.completion.chunk` carrying the whole message as its delta + `[DONE]`, `chat.py:559-578`). Streaming requests go through `_buffered_stream` (`:238-310`): if the turn finishes within `KEEPALIVE_GRACE_SECONDS = 60` it is served with its real HTTP status; otherwise a 200 SSE stream is committed and the dialect's keepalive is written every `KEEPALIVE_INTERVAL_SECONDS = 3` (chat: `": keepalive\n"` comment; Responses: placeholder `response.created/in_progress` events; Anthropic: `ping` events), and a late failure is framed as the dialect's SSE error. A reader disconnect leaves the turn running so its retry can coalesce.
13. **Eval streaming** reads the provider stream **to the end** before serving (`_collect_stream`, `:209-235`; terminal event required, per-dialect `is_terminal_event`). The program therefore never sees partial tokens; "streaming" is framing only.

#### 3.5.3 The TrainClient (`clients/train.py:310-459`)

Constructed by the interception server, **one per distinct client config per server** (key = `config.model_dump_json()`), shared by all rollouts multiplexed on that server (`server.py:356-366`); its `AsyncOpenAI` has connect timeout 5 s, no read timeout, `max_retries=0`, pool 1000/100 keepalive (`clients/base.py:12-31`).

`get_response(dialect, body, sampling, session_id, turn, headers)`:

1. **Chat only.** A non-`ChatDialect` request raises `NotImplementedError` (`train.py:342-350`). Nothing checks this earlier: prime-rl has no config-time harness or dialect check. Each Responses or Anthropic turn therefore 502s (§3.5.2 step 11) and the trace fails. See §7.1 for the harness table.
2. **Prompt source.** `turn.prompt` / `turn.tools` (the graph-resolved view of the parsed request) (`:351-357`). **The program's sampling fields in `body` are ignored** — only `body["model"]` and `sampling = session.ctx.sampling` are used (`:367-370`). `sampling.wire_args()` becomes vLLM `sampling_params`, minus `chat_template_kwargs` (which goes into the renderer build key) and `cache_salt` (`:368-370`). `wire_args` flattens `extra_body` into top-level keys, and typed fields win over them (`types.py:274-284`). So prime-rl's `extra_body.top_k` and other extras land **inside** `sampling_params`, while `cache_salt` is popped out and sent as a **top-level** `/generate` body field (`renderers/client.py:330-331`). The full path is dispatcher, then `Env._sampling` (`envs.py:121-125`), then `RunRequest.sampling.extra_body` (msgpack JSON), then the worker's `ctx.sampling`, `wire_args()`, the `pop` here, and `generate(cache_salt=...)`. On the eval path, `ChatDialect.apply_overrides` overlays the same flattened dict, so `cache_salt` is top-level in the relayed chat body (`chat.py:589-597`). `logprobs=1` is always forced (`renderers/client.py:312`), so a frozen generation source still records the frozen model's sampled-token logprobs even though `GenerationSource.sampling_args` pops `logprobs` (`generation_source.py:43-49`). An engine that omits them fails the turn (`MalformedGenerateResponseError`). Tools are converted with `tool_to_wire`, which rejects namespaced/non-function tools (`:40-52`).
3. **Renderer slot.** `ElasticRendererPool(renderer_model_name or model, config.renderer, chat_template_kwargs, multiplex)` keyed by `(model, renderer_config_json, chat_template_kwargs_json)` (`:232-251`); `acquire()` returns a slot with `load < multiplex` (default 256, `configs/client.py:80`) or builds one more tokenizer on a thread, single-flight per key (`:266-307`). Encode-side work (render, bridge) runs under the slot's `threading.Lock` on a thread (`RendererSlot.run`, `:209-214`).
4. **Bridge (the extension property).** `can_bridge = turn and no image parts and wire tail matches [tool*, user?]` (`_is_valid_incremental_tail`, `:176-186`; `:386-390`). Then `turn.previous_token_ids()` returns `(prompt_ids, completion_ids)` of the prefix **only if the prefix ends at a sampled assistant node whose mask is a contiguous sampled suffix** (`graph.py:504-531`). If so, `renderer.bridge_to_next_turn(prev_prompt, prev_completion, tail_wire_messages, tools)` (`:395-403`) extends the previous turn's exact tokens with the new messages; `prompt_ids = bridged.token_ids`, attribution = the bridged `RenderedTokens`, and `sampling_params["routed_experts_prompt_start"] = max(len(prev_prompt)+len(prev_completion)-1, 0)` (`:404-412`) so the engine returns routing only for the new region (plus the one previously-unforwarded position).
5. **Fallback full render.** If no bridge (or the renderer returned `None`): `renderer.render(message_to_wire(prompt), tools, add_generation_prompt=True)` (`:417-426`). A full render may tokenize prior assistant turns differently from how they were sampled (chat templates dropping `<think>`, BPE boundary drift) — the graph then **forks** (§3.7).
6. **Generate.** `renderers.client.generate(client, renderer, messages, model, prompt_ids, multi_modal_data, prompt_attribution, tools, sampling_params, cache_salt, extra_headers={"X-Session-ID": trace.id})` (`:428-443`). Inside (`deps/renderers/renderers/client.py:191-423`): pre-flight overflow check against `max_model_len` discovered once per `(base_url, model)` from `GET /v1/models` (`:71-109`, `:302-308`) → `OverlongPromptError` → mapped to `ProviderError(status_code=400)` so the SDK doesn't retry (`train.py:444-447`); forces `sampling_params.stop_token_ids = renderer.get_stop_token_ids()`, `logprobs = 1`, default `skip_special_tokens=False` (`client.py:310-313`); POSTs `{base without /v1}/inference/v1/generate` with body `{model, token_ids, sampling_params, [features], [cache_salt], [priority]}` (`:315-352`); validates that `logprobs.content[i].token == "token_id:<completion_ids[i]>"` and rejects the vLLM sentinel `-9999.0` (`:137-188`); parses completion ids with `renderer.parse_response` into content / reasoning / tool calls; promotes `finish_reason "stop" → "tool_calls"` if a well-formed tool call was parsed (`:389-394`).
7. **Response.** `response_from_generate` (`train.py:106-173`): drops tool calls with parse status `UNKNOWN_TOOL`; message spans from the attribution (bridged: shifted by `path_len` via `PendingTurn.prompt_message_spans`, `graph.py:533-546`); `Usage(prompt_tokens=len(prompt_ids), completion_tokens=len(completion_ids))`; `tokens = TurnTokens(prompt_ids, completion_ids, completion_logprobs, message_spans, is_content, multi_modal_data, mm_token_type_id_map, routed_experts, sampling_mask)` (transient fields excluded from serialization, `types.py:220-248`). `response.raw = serialize_completion(response, model)` — an OpenAI `chat.completion` dict with `content`, `reasoning_content`, `tool_calls`, `usage` (`train.py:55-103`) — which is what the program receives.

**Prefix caching / affinity.** The request carries exact token ids, so vLLM's prefix cache applies to the literal token prefix. prime-rl salts it per policy version: `cache_salt = str(group.policy_version_at_start)` for live-policy rollouts and `None` for frozen ones (`orchestrator/dispatcher.py:559-565`). The salt is fixed when the **group opens**. A group that straddles a weight swap, and every later turn of an in-flight episode, keep the old salt on new weights, and the prefix cache is never reset (06 §3.6.1). `X-Session-ID = trace.id` lets a session-affinity router pin a rollout's turns to one engine (`clients/client.py:16-18`). After each episode, prime-rl POSTs `/finish_session?session_id=<trace id>` to the router (5 s timeout, errors swallowed) for every trace id seen in deltas or the reply. It does this only when `admin_base_url` is set, i.e. managed routed deployments (`orchestrator/clients.py:96-100, 113-129`; `dispatcher.py:581-610`). The router side is E's.

#### 3.5.4 The EvalClient (`clients/eval.py:62-218`)

Relays: builds headers from the intercepted request minus localhost/framing/auth/digest headers and the ACP header (`:21-55`), applies endpoint `headers`, sets `X-Session-ID`, then the dialect's provider auth (`:103-122`); POSTs `join_url(base_url, dialect.upstream_path)` (dedups a repeated `/v1`, `clients/base.py:33-41`); transport faults → `ProviderError` 504/503, HTTP errors keep their status (`:139-164`); a malformed JSON / schema-invalid body → retryable 502 (`:91-98`). `Response.raw` = the provider's native object (returned verbatim to the program); the trace gets a parsed copy with **no tokens** (`tokens=None`). Streaming relays complete SSE events (split on blank lines, `:166-200`). Only the eval client supports `relay_aux` (Anthropic `count_tokens`, `:202-215`).

| | EvalClient (`type="eval"`) | TrainClient (`type="train"`) |
|---|---|---|
| Upstream | `{base_url}{dialect.upstream_path}` (any provider) | `{base_url − /v1}/inference/v1/generate` (vLLM + prime-rl extension) |
| Dialects | chat, responses, anthropic | chat only |
| Program's own sampling keys | kept unless the run overrides (per dialect rules) | **ignored**; only `ctx.sampling` |
| Streaming | provider SSE buffered whole, then relayed | response generated whole, framed as one SSE chunk |
| Tokens / logprobs on trace | none | exact prompt/completion ids, sampled logprobs, optional routing/sampling masks, `is_content` |
| Usage | provider's (cache split) | `len(prompt_ids)`, `len(completion_ids)` |
| prime-rl use | eval rollouts (`clients.eval_client`), frozen generation sources without renderer | all train rollouts (`clients.train_client`) |

### 3.6 Interception shapes and tunnels

- `EnvConfig.interception: InterceptionConfig = ElasticInterceptionPoolConfig()` (`configs/env.py:60`); union discriminated on `type` ∈ `server | static | elastic` (`interception/__init__.py:26-31`).
- `requires_tunnel(harness_is_local, server_configs, shared)` (`interception/__init__.py:34-55`) is true if any agent runtime is non-local, or a live shared tool server / a non-colocated, non-URL tool server config runs in a remote runtime (tool servers must reach `/state` + `/task`). The env computes it over all declared roles (`env.py:388-402`).
- `ElasticInterceptionPool` (default, `interception/pool.py:86-150`): warms one server; `acquire` (under a lock) reuses a server with `load < multiplex` (default 32, `:80`), else brings up a new `InterceptionServer(InterceptionServerConfig(tunnel=PrimeTunnelConfig()))` (`:115-135`). N concurrent rollouts ≈ N/32 servers/tunnels. `StaticInterceptionPool` = fixed list, least-loaded (`:44-72`). `InterceptionServer` alone = one server (`server.py:313-349`).
- Tunnels: `PrimeTunnel.expose(port)` starts a `prime_tunnel.Tunnel(local_port=port)` (frpc), retried 3× with a run-wide limiter of 512 starts/min (`tunnel/prime.py:20-23, 40-65`), failure → `TunnelError`. `CustomTunnel` opens nothing: binds `0.0.0.0:<port>` and yields your `url` (`tunnel/custom.py:15-42`) — the plaintext port is protected only by the per-rollout secret.
- Outside an env: an entered `Agent` owns one `InterceptionServer`; an un-entered agent creates a per-rollout server sized to the task (`interception/__init__.py:74-101`, `agent.py:321-368`).

### 3.7 Trace graph: recording turns (`graph.py`)

**Node identity.** `message_hash(message)` (`graph.py:278-352`): blake2b over role type, content (text or parts incl. image URLs; `None` ≡ `""`), assistant `reasoning_content`, opaque `provider_state` fields (`encrypted_content, signature, data, phase`), tool calls (type, id, name, namespace, canonicalized JSON args), tool `tool_call_id`. `_node_key = (parent, tools_hash if root else None, message_hash)` (`:361-368`) — roots are distinguished by their tool set. `trace._head_index` maps keys → the *latest* node id, rebuilt lazily after deserialization (`:371-378`).

**`prepare_turn(trace, prompt, tools)`** (`:594-639`) walks the prompt following the head index and returns `PendingTurn(prefix_node_ids, path_len = Σ len(token_ids) of the prefix)`. It does not mutate the trace (inference happens between prepare and commit, and concurrent requests of one trace may interleave).

**`_commit_turn(turn, response)`** (`:774-935`), synchronous:

1. If the response has tokens, **tighten the message-hash prefix to a token prefix**: keep prefix nodes while `prompt_ids[off:off+len(node.token_ids)] == node.token_ids` (`:799-813`). The first divergent node ends the reused prefix → the rest becomes a new branch carrying this turn's real tokens.
2. **Reconcile** tail messages that a concurrent request committed while this one was in flight: per message, match an existing child by exact token span (or, for a renderer-unattributed assistant span `None`, the longest existing variant fitting before the next attributed message, `_matching_prefix_node :437-478`) (`:820-858`).
3. **Append new input nodes** for the remaining tail: node `i` gets `prompt_ids[cursor : span_i.end]` (its leading template scaffold + its own tokens), `mask = [False]*n`, `is_content` slice (`:870-893`).
4. **Append the assistant node**: `token_ids = prompt_ids[cursor:] + completion_ids` (the trailing generation-prompt scaffold + sampled tokens), `mask = [False]*len(gen_prompt) + [True]*len(completion)`, `logprobs = completion_logprobs` (compact: one per True mask entry), `sampled=True` (`:896-913`). `is_content` is filled only when the renderer's attribution has exactly `len(prompt_ids)` entries (`:785`). Input nodes get their slice. The assistant node gets `False` over the generation prompt and `True` over **every** completion token, reasoning and stop token included. Tokenless (eval) turns get `[]`, and so do mismatched lengths (e.g. `process_multimodal=False`). Registered in the head index so the next request (which restates it) reuses it (`:915-917`).
5. Attribute multimodal items per introducing node (`_attribute_mm :647-686`), MoE routing rows per node (`_attribute_routed_experts :721-757`, including repairing the previous turn's placeholder last row from this turn's prefill, `:689-718`), and the sampling mask on the assistant node (`:760-771`).

**Invariant:** concatenating `token_ids` along any root→leaf path reproduces exactly `prompt_ids + completion_ids` the engine saw on the leaf's turn (`graph.py:13-18`; tested `tests/v1/test_graph.py:477-539`).

**When branches appear** (each leaf = one branch, `graph.leaves :941-945`): (a) the harness rewrote an earlier message (compaction, summarization, subagent context, a different tool result) → message-hash mismatch; (b) same messages but different tokens (retokenization after a failed bridge) → token-prefix mismatch (`test_graph.py:477-539`); (c) a harness dropped/changed `reasoning_content` when re-sending an assistant turn (reasoning participates in the hash, `test_graph.py:233-269`); (d) parallel requests (subagents) from one trace with different continuations (`test_graph.py:272-319`); (e) ACP compaction attempts (dangling leaves, `trace.py:554-568`). Eval traces (no tokens) use message-hash identity only.

**Semantic edges** (`semantic.py`, `trace.py:572-636`): ACP harnesses publish `SemanticEdgeSet`s over logical request ids (`continuation`, `compaction_attempt`, `compaction`, `subagent_call`, `subagent_return`, …) in response metadata key `"ai.prime.acp/semantic-edges-v1"`; `Trace.add_semantic_edges` resolves them to nodes via `ModelCall.acp.request_id` (last committed retry wins) and appends `ParentLink`s (cycle-checked, idempotent). They never affect physical branches, **except** trainability: a leaf with a `compaction_attempt` link is trainable only if some node later declares it as its `compaction` parent (`trace.py:540-566`; `test_trace.py:564-689`).

### 3.8 Trace → training data

`Trace.branches` (`trace.py:532-570`) → `Branch(index, nodes, calls, trainable, mm_token_type_id_map)` with flat views:

| Branch property | Definition | cite |
|---|---|---|
| `token_ids` | concat of node `token_ids` | `trace.py:213-220` |
| `sampled_mask` | concat of node `mask` | `:222-229` |
| `logprobs` | node compact logprobs spread onto sampled positions, 0.0 elsewhere | `:231-258` |
| `advantages` / `reference_logprobs` / `trainer_logprobs` / `entropies` | same spread; `None` if no node on the path has any | `:260-289` |
| `loss_weights(name)` | per-token streams aligned to `token_ids` (not compact) | `:291-310` |
| `multi_modal_data`, `mm_token_type_ids` | concatenated renderer mm items; token → modality marker | `:312-336` |
| `routed_experts` | `[tokens, layers, top_k]` in the engine payload's dtype (uint8, or uint16 with >256 experts; `routed_experts.py:18-25`) over token-carrying nodes; `None` if any of them lacks it or the rows ≠ tokens | `:338-346` |
| `sampling_mask` | per-token kept-vocab ids (`ids` flat + `counts`), zero-count on non-sampled tokens | `:348-367` |

prime-rl (`orchestrator/trajectories.py`):
- `iter_trainable_branches(trace)` (`:91-114`) skips `branch.trainable == False` and **trains each shared sampled node only once**. The owner is the first branch in `trace.branches` order (leaf node id ascending), keyed on object `id(node)`. A later branch carries that node's tokens with mask `False`. `advantages`/`logprobs` stay unmasked and the mask does the exclusion. Branches with no trainable token are skipped, so `branch_index` can be non-contiguous. On a retokenization fork, the re-rendered earlier assistant turns become new **input** nodes (mask False) on the new branch, so they never train twice. Verified on a synthetic trace: a fork after a shared `a1` masks `a1` in branch 1, and a re-rendered `a1` roots a new branch whose only trainable span is the new completion.
- `trace_to_samples(trace, env_name)` (`:136-180`) → one `TrainingSample(token_ids, mask, logprobs, temperatures=[] (filled later), env_name, ref_logprobs, mm_kwargs, mm_token_type_ids, routed_experts (padded/truncated to len), rl_weights/ce_weights/ref_kl_weights (from node.loss_weights, each shared node once), advantages, sampling_mask, trace_id, branch_index)` per surviving branch.
- Advantages are written **onto graph nodes** by the algorithm (`algo/routing.py:11-35`: scalar broadcast or list over all sampled tokens of trainable nodes in node order), then read back through `branch.advantages`. So credit is per node, and a node shared by two branches carries one credit.
- Only traces with `not trace.has_error and trace.agent.trainable and any(node.mask)` are trainable (`algo/base.py:35-42`); `TrainSink.process_group` stamps `sample.temperatures = [env.sampling_args["temperature"]] * len` (`train_sink.py:279-283`). That value is the per-source `TrainSamplingConfig.temperature` (`configs/orchestrator.py:44`; `envs.py:170`), never what vLLM used. It also **raises `RuntimeError`**, which is fatal to the orchestrator, when truncated sampling needs masks and a sample lacks one (`:284-292`). `trace_to_samples` reads these `Branch` fields: `nodes` (`sampled`, `mask`, `loss_weights`, `token_ids`), `trainable`, `token_ids`, `logprobs`, `reference_logprobs`, `advantages`, `multi_modal_data`, `mm_token_type_ids`, `routed_experts`, `sampling_mask`, `index`. It also reads `trace.id`.

### 3.9 Streaming the trace to prime-rl (summary; protocol is G's)

The env server attaches a `DeltaStreamer` to each minted trace (`serve/server.py:111-114`); `Trace.notify()` (fired at phase changes and after every `record_call`, `trace.py:462-465`, `server.py:581`) schedules a diff: header once (`version,id,verifiers,task`), appended `nodes/calls/errors/extra_usage/request_rewrites/response_rewrites`, changed scalars (`agent, tools, mm_token_type_id_map, rewards, metrics, info, root_reply, is_completed, ok, stop_condition, timing, num_*_tokens`), new semantic links, routing-row repairs, and the `pending` preview (`serve/delta.py:44-72, 192-246`). Nodes are sent once (append-only), dumped `mode="python"` so numpy arrays ride as raw bytes and floats keep full precision (`graph.py:57-80`, `delta.py:85-88`). `EnvClient.run` assembles deltas into dicts and validates one `WireEpisode` off the event loop when the reply lands, checking node/call counts (`serve/client.py:186-221`, `delta.py:302-321`).

---

## 4. Interfaces & contracts

### 4.1 prime-rl ↔ verifiers API surface (exhaustive)

| prime-rl call site | verifiers API | Arguments → returns |
|---|---|---|
| `orchestrator/clients.py:259-280` `setup_client` | `TrainClientConfig(base_url, api_key_var, headers, renderer, renderer_model_name)` / `EvalClientConfig(base_url, api_key_var, headers)` | `client_type == "renderer"` → Train; else Eval. `renderer_model_name = model_name` (policy base name) only for renderer (`:85`). |
| `orchestrator/orchestrator.py:199-205` | — | policy `InferenceClient(train_client_type="renderer", eval_client_type="openai_chat_completions", renderer_config=config.renderer)` |
| `orchestrator/envs.py:95-119` `Env.start` | `verifiers.v1.serve.EnvClient(address)`, `.wait_for_server_startup(timeout=600)`, `vf.load_taskset(config.env.taskset)`, `type(taskset).INFINITE`, `iter(taskset)` | Tasks held client-side as `vf.Task` objects (materialized on a thread if finite; shuffled with seed 42 if `shuffle`). |
| `orchestrator/envs.py:121-125` | `vf.SamplingConfig(**sampling_args)` | train: `{temperature, top_p, logprobs: True, [max_completion_tokens], [extra_body{top_k,...}]}` (`configs/orchestrator.py:91-111`); `+ extra_body.cache_salt`. |
| `orchestrator/envs.py:127-153` `Env.run` | `EnvClient.run(task_data=task.data.model_dump(mode="json"), client, model, sampling, on_delta)` → `vf.WireEpisode` | On `not episode.ok`, prime-rl marks every `ok` sibling trace failed (`errors += [episode.last_error or Error("EpisodeFailed")]`, `ok=False`). |
| `orchestrator/dispatcher.py:116-120` | `episode.task.key/hash` vs `task.key/hash` | provenance must match or `ValueError`. |
| `dispatcher.py:668-679` | `vf.Error`, `trace.has_error`, `trace.num_turns` | a `ok` trace with 0 turns → `EmptyTrajectory` error; no traces → `EmptyEpisode`. |
| `dispatcher.py:715-728` | `episode.env.name`, `vf.GroupInfo(id)`, `vf.PolicySpan(start, end)`, `vf.TrainWorkInfo/EvalWorkInfo(step, policy)`, `vf.TrainRunInfo(id, name, work)`, `episode.record_run(run)` | consumer stamping. |
| `trajectories.py`, `algo/*`, `train_sink.py`, `metrics.py` | `vf.Trace` (`branches, nodes, reward, num_*_tokens, num_turns, has_error, agent.trainable, agent.name, id, info`), `vf.Branch` (all §3.8 props), `MessageNode.{mask, advantages, reference_logprobs, loss_weights, is_content, token_ids}`, `vf.SamplingMask` | read + write node annotations. |
| `entrypoints/env_server.py:44-51` | `serve_env(**pool_serve_kwargs(config.serve.pool), address, address_queue, log_setup, config_data=env_config_data(config.env), max_concurrent)`, `set_base_sandbox_labels` | env server process (G). |
| `configs/orchestrator.py`, `configs/env_server.py` | `vf.EnvConfig`, `vf.SingleAgentEnvConfig`, `vf.ServeConfig`, `vf.resolve_env_field`, `vf.narrowed_env_annotation` | config narrowing (A/I). |

`RunRequest` wire (`serve/types.py:42-50`; msgpack of `model_dump(mode="json")`, `serve/client.py:109`), as prime-rl fills it:

| Field | prime-rl train value |
|---|---|
| `task_data` | `task.data.model_dump(mode="json")` |
| `client` | `{type: "train", base_url, api_key_var, headers, renderer: RendererConfig\|null, renderer_model_name: <policy name>, multiplex: 256}`, or `{type: "eval", base_url, api_key_var, headers}`. The **key value never crosses**: the worker reads `$<api_key_var>` from its own env (`configs/client.py:93-104`). |
| `model` | `config.model.name`, or a frozen source's name |
| `sampling` | `{temperature, top_p, max_tokens (from alias max_completion_tokens), logprobs: true (extra; omitted for frozen), extra_body: {top_k?, cache_salt?, ...}}` |

The reply is preceded by `delta` frames, then `RunResponse{success, error, head: Episode-without-traces dict, traces: [TraceSummary{id,nodes,calls}]}`. The client rebuilds a `WireEpisode = Episode[WireTaskData, State, WireAgentConfig]` (`episode.py:166`) with `id, env{id,name}, task{type,data,key,hash}, group, run, ok, errors, traces[WireTrace]`. prime-rl stamps `env.name`, `group`, and `run` after `_validate_episode_task` (`dispatcher.py:710-729`).

### 4.2 Harness ↔ interception HTTP contract

| Route | Method | Auth | Body → Response |
|---|---|---|---|
| `/v1/chat/completions` | POST | `Authorization: Bearer <model_secret>` | OpenAI chat request (`n=1`) → `chat.completion` JSON, or SSE (`stream=true`) |
| `/v1/responses` | POST | Bearer | OpenAI Responses → response object / SSE (eval only) |
| `/v1/messages` | POST | `x-api-key` or Bearer | Anthropic Messages → Message / SSE (eval only) |
| `/v1/messages/count_tokens` | POST | as above | relayed JSON (eval only), not recorded |
| `/v1/models` | GET | any dialect's carrier | relayed from `ctx.client.base_url/v1/models` (30 s timeout), not recorded |
| `/tool` | POST | Bearer model secret | `{tool_call_id, name, arguments}` → `{"action": "allow"}` \| `{"action":"deny","message":ToolMessage}` \| `{"action":"stop","reason"}` |
| `/state` | GET / PUT | Bearer state secret (or shared-server secret + `X-Verifiers-State-Route: <trace.id>`) | `trace.state` JSON / replace (validated into the `State` subclass) |
| `/task` | GET | Bearer state secret | `{"cls": "module:Qualname", "task": TaskData JSON}` |

Status codes surfaced to the program: 400 stop/refusal/bad request/overlong prompt, 401 unknown secret, 409 rollout concluded, provider status or 502/503/504 for upstream faults. Request headers consumed: `x-stainless-retry-count`, `Idempotency-Key`, `X-ACP-Model-Request-ID`.

### 4.3 Client → inference contract (train)

`POST {base_url.rstrip("/").removesuffix("/v1")}/inference/v1/generate`, headers: auth from `api_key_var`, endpoint `headers`, `X-Session-ID: <trace.id>`. Body: `{model, token_ids, sampling_params: {...ctx.sampling.wire_args() minus chat_template_kwargs/cache_salt, stop_token_ids, logprobs: 1, skip_special_tokens: False (default), [routed_experts_prompt_start]}, [cache_salt], [features], [content_parts]}`. Response fields read: `request_id, usage, prompt_token_ids, choices[0].{token_ids, logprobs.content[{token:"token_id:N", logprob}], finish_reason, routed_experts{data(base64),shape,start}, sampling_mask}, mm_placeholders` (`renderers/client.py:352-423`). Also `GET {base}/v1/models` for `max_model_len`. Server side is E's (06).

### 4.4 Config fields (verifiers side, S4-relevant)

| Field | Type / default | Effect | cite |
|---|---|---|---|
| `BaseClientConfig.base_url` | `str`, `"https://api.pinference.ai/api/v1"` (Prime CLI/env override) | upstream endpoint | `configs/client.py:23-38` |
| `.api_key_var` | `"PRIME_API_KEY"` | env var for the key; `"EMPTY"` if unset | `:33, 93-104` |
| `.headers` | `dict` | extra headers (Prime team header auto-added for pinference) | `:34, 54-61` |
| `TrainClientConfig.renderer` | `RendererConfig \| None` | renderer; `None` = auto from model (default renderer has no tool support) | `:71-75` |
| `.renderer_model_name` | `str \| None` | tokenizer model (pin base model under LoRA) | `:76-79` |
| `.multiplex` | `int = 256` | rollouts per renderer instance | `:80-84` |
| `EnvConfig.interception` | `elastic` (`multiplex=32`) \| `server` (`tunnel: prime\|custom`) \| `static` | interception shape | `configs/env.py:60`, `pool.py:75-80`, `server.py:313-321` |
| `EnvConfig.max_concurrent_agents` | `int\|None = 1` | agent runs active per episode | `configs/env.py:50` |
| `EnvConfig.timeout.{episode,finalize}` | `None` | env hook deadlines | `:18-23` |
| `EnvConfig.retries` / `AgentConfig.retries` | `RetryConfig(max_retries=0, include=[], exclude=[])` | whole-episode / whole-rollout retries by error type name | `configs/retries.py:7-17` |
| `AgentConfig.max_turns / max_{input,output,total}_tokens` | `None` | refusal before the next call (soft by one turn) | `configs/agent.py:41-48`, `session.py:82-115` |
| `AgentConfig.timeout.{setup,rollout,finalize,scoring}` | `None` (rollout → task → 4 h; `0` = none) | stage timeouts | `configs/agent.py:13-24`, `agent.py:57-83` |
| `SamplingConfig` | `temperature, top_p, reasoning_effort, max_tokens (alias max_completion_tokens)`, `extra="allow"` | run sampling; extras pass through; `extra_body` flattened | `types.py:263-284` |
| `TaskConfig.{judges, stops, metrics, rewards}` | plug judges / override hooks by name | scoring composition | `configs/task.py:31-63` |

Env vars: `VF_RUN_ID` (run scope for creation limiters; prime-rl sets it from `PRL_RUN_ID`, `entrypoints/env_server.py:60`), `PRIME_API_KEY`, `PRIME_INFERENCE_URL`, `PRIME_TEAM_ID` (`configs/client.py:47-58`).

---

## 5. Invariants & assumptions

1. **Every model call goes through the interception endpoint.** A harness that calls a provider directly produces no trace nodes; prime-rl then sees an `EmptyTrajectory` or a trace with no trainable tokens (`skills/create-environments/SKILL.md` "Custom harnesses"; `dispatcher.py:672-679`).
2. **Training is chat-completions only.** `TrainClient` refuses other dialects (`train.py:342-350`) and namespaced/non-function tools (`:40-44`).
3. **The rollout's sampling is authoritative in training.** The TrainClient ignores the program's sampling fields; prime-rl assumes one temperature per env when it stamps `TrainingSample.temperatures` (`train_sink.py:279-283`). Per-agent `sampling` in an `EnvConfig` is deep-merged over the run's (`env.py:208-215`) — a trainable seat with a pinned temperature silently violates prime-rl's temperature assumption.
4. **One logical turn commits at most one node set.** Coalescing/replay of SDK retries (`server.py:617-691`) and the `released` seal (`session.py:164-169, 581-585`) guarantee no duplicated or late turns after the rollout concluded.
5. **Path-concat exactness.** `concat(node.token_ids)` along a branch equals what the engine consumed + produced on that branch's last turn; token-prefix tightening forks rather than reusing stale tokens (`graph.py:788-813`).
6. **Compact logprobs.** `len(node.logprobs) == sum(node.mask)` and the sampled tokens form a contiguous suffix of an assistant node; `previous_token_ids` and `Branch.spread` fast paths rely on it (`graph.py:515-521`, `trace.py:239-244`).
7. **`n == 1`.** One request = one completion (`chat.py:512-514`).
8. **Task identity round-trips.** `TaskData.model_dump(mode="json")` on the orchestrator must re-validate on the server to the same `hash` (sha256 of `model_dump(json, exclude_none)`), or dispatch raises (`dispatcher.py:116-120`); fields with `exclude=True` are forbidden (`serve/server.py:43-50`).
9. **`trace.state` never leaves the worker.** Anything needed downstream must be in `trace.info`, rewards, or metrics (`trace.py:435`; `test_trace.py:150-176`).
10. **Errors are data.** Rollout/episode failures are recorded on `trace.errors`/`episode.errors`; the only exception crossing the Env boundary is cancellation or lifetime bugs (`errors.py:1-22`). prime-rl drops errored traces from training and marks siblings of a failed episode as failed (`envs.py:146-152`).
11. **Scoring runs only on non-failed traces** and while the runtime is alive (`rollout.py:479-507`); a stopped (limit/@stop) trace is scored.
12. **Session affinity id = trace id** on every turn (`clients/client.py:16-18`); prime-rl calls `/finish_session` with the trace ids it saw in deltas (`dispatcher.py:583-605`).

---

## 6. Extension points

**6.1 Custom taskset (the 95% case).** Package exports one `vf.Taskset` subclass via `__all__` (loader: `utils/loaders.py:99-126`; module `verifiers.v1.tasksets.<id>` or top-level `<id>` with `-`→`_`, `:70-96`). Recipe (`docs/v1/tasksets.md`): subclass `vf.TaskData` (row fields), `vf.Task[Data, State, TaskConfig]` with `@vf.reward` / `@vf.metric` / `@vf.stop` / `@vf.intercept` methods and optional `setup/finalize/validate/toolsets`, `vf.TasksetConfig` for load-time knobs, and `load()` (may be a generator; set `INFINITE = True` for endless). Don't override `Taskset.__init__`. `uv run vf-init <name> [-T] [-H]` scaffolds. In prime-rl: `[orchestrator.train.source.env.taskset] id = "<id>"`; the package must be importable **both** in the orchestrator (it loads tasks) and in env-server workers.

**6.2 Rewards without code changes.** `--env.taskset.task.rewards.<name> = {fn = "file.py:func", weight=...}` (replace or add), `.metrics`, `.stops` (`configs/task.py:11-63`, `utils/loaders.py:236-271`); fn-less entries only override metadata (weight/priority).

**6.3 Judges.** Call a `vf.Judge` from a reward (`judge.evaluate(trace=trace, **fields)` records usage on the trace), or plug one via `--env.taskset.task.judges = [{id="reference"|"rubric"|<pkg>, ...}]` (`configs/judge.py:26-61`). A plugin judge = package exporting one `Judge` subclass with `score(task, trace)` and an `id`-pinned `JudgeConfig` subclass. Judges run after task rewards and their result keys are `name or id`; duplicates refused (`configs/judge.py:49-61`).

**6.4 Request/response rewriting & tool policing.** `@vf.intercept` methods taking `Request` (rewrite new user/tool messages; tool rewrites are pinned for the rest of the rollout) or `Response` (replace the assistant message with inert text); `@vf.stop` on `Request/Response/Trace`. Enforcing a tool-call verdict *before execution* needs a harness with `SUPPORTS_TOOL_INTERCEPTION` (it asks `/tool`); otherwise a rewrite ends the rollout (`session.py:469-472`).

**6.5 Custom harness.** Subclass `vf.Harness[Config]`, implement `launch(ctx, trace, runtime, endpoint, secret, mcp_urls, data) -> ProgramResult` routing every model call to `endpoint`/`secret`; declare `APPENDS_SYSTEM_PROMPT, SUPPORTS_MCP, SUPPORTS_RESUME, SUPPORTS_TOOL_INTERCEPTION, SUPPORTS_SKILLS, NEEDS_CONTAINER, EXECUTES_CODE` accurately (`harness.py:35-59`); optional `setup`, `resume`, `session` (stateful handle), `cleanup`, `@metric`s. An in-process loop is allowed as long as calls go through the endpoint (`harness.py:302-306`). **For training**, the program must use chat completions, `n=1`, un-namespaced function tools, and should re-send assistant turns verbatim (including `reasoning_content`) to keep one branch. ACP harnesses: subclass `vf.ACPHarness`, implement `prepare_acp`; publish semantic edges under `"ai.prime.acp/semantic-edges-v1"` and send `X-ACP-Model-Request-ID` per model request (G owns ACP details).

**6.6 Multi-agent env.** Subclass `vf.EnvConfig` with `AgentConfig` fields (default instances; name = role) and `vf.Env[YourConfig]` with `run(task, agents)` (+ `setup` to mark `agents.x.trainable = False`, `finalize` for sibling rewards); export via the taskset package or as `verifiers.v1.envs.<id>` for `--env.id` (`utils/loaders.py:164-187`). Use `agents.x.interaction(task)` for user/turn-driven control flow (`agent.py:473-536`). prime-rl trains every `trainable` trace with tokens; non-trainable seats (judges, users) can use a different `client` (e.g. an external `EvalClientConfig`) and `model`.

**6.7 New dialect.** Implement `Dialect[RespT]` (`dialects/base.py:266-387`: `routes`, `upstream_path`, `response_type`, `sampling_fields`, `parse_request`, `parse_response`, `validate_response`, `apply_overrides`, `rewrite_request/response`, `stream_events`, `stream_parser`, `mediate_external_capabilities`, optional `auth_headers/secret/error_body/stream_keepalive/stream_error/is_terminal_event`) and append it to `DIALECTS` (`dialects/__init__.py:11`) — a module-level tuple, so this is a source edit. It works with the EvalClient immediately; the TrainClient additionally needs a renderer path for the dialect (lockstep change in `clients/train.py:342-350` + renderers).

**6.8 New client.** Subclass `Client` (`clients/client.py:30-71`: `get_response(dialect, body, sampling, session_id, turn, headers) -> Response`, optional `relay/relay_aux/close`). Selection is hard-coded in `resolve_client` by config type (`:73-80`) and the client config union is `EvalClientConfig | TrainClientConfig` (`configs/client.py:88`); adding one means extending that union (it crosses the env-server wire) and `resolve_client`. To train, the returned `Response.tokens` must carry `prompt_ids/completion_ids/completion_logprobs/message_spans` consistent with the §3.7 invariants, and `Response.raw` must be a native object the dialect can serialize.

---

## 7. Gotchas & limitations

1. **Only chat-completions harnesses train.** Each Responses or Anthropic turn fails at runtime with a 502 (§3.5.3 step 1). The table gives each built-in's transport under a `TrainClientConfig`:

| Harness | Transport it speaks | Trainable? | Evidence |
|---|---|---|---|
| `null`, `bash` | chat (`client.chat.completions.create`) | **yes** (with `rlm`, the harnesses shipped prime-rl configs train: `null` 72×, `bash` 17×, `rlm` 12×) | `harnesses/utils/core.py:266`, `utils/launch.py:52-55` |
| `browser_use` | chat | yes | `browser_use/program.py:182` |
| `mini_swe_agent`, `terminus_2` | chat (litellm `custom_llm_provider=openai`, `api_base=endpoint`) | yes | `mini_swe_agent/harness.py:59-63`, `terminus_2/program.py:69-70` |
| `prime_agent` (ACP) | chat (`api: "openai-completions"`) | yes | `prime_agent/harness.py:217-218` |
| `pi`, `kimi_code` (ACP) | `transport` knob, default `chat_completions`; `responses`/`anthropic_messages` are eval-only | yes at default | `pi/harness.py:52-54, 109-113`; `kimi_code/harness.py:43-45, 82-86` |
| `hermes_agent` (ACP) | train leaves `providers.openai.transport` unset (set only when `ctx.client.type == "eval"`), so Hermes' default for provider `openai` applies | yes (inferred: train path leaves provider `openai` with no vendor API override; Hermes default outside the pin) | `hermes_agent/harness.py:88-89`, `program.py:12-14` |
| `openclaw` (ACP) | provider = `ctx.model` prefix, `baseUrl=endpoint`, no `api` field, so OpenClaw's default applies | depends on the model-name prefix; treat as eval-only until checked | `openclaw/harness.py:161, 192-199` |
| `rlm` (ACP) | nano-rlm `provider.base_url=endpoint` | **yes**: trained through the renderer client in 12 shipped configs (nano-rlm source outside the pin) | `rlm/harness.py:223-229`; `intellect-3.1/rl.toml:62-68` |
| `codex` (ACP) | OpenAI **Responses** via the codex-acp gateway (`baseUrl=endpoint`; the dialect is Codex's own default, documented in `docs/v1/architecture.md:21`), and its MCP tools arrive namespaced, which `tool_to_wire` refuses; either alone rules out training | **no** | `codex/harness.py:239-248`; `docs/v1/architecture.md:21`; `tests/v1/test_e2e.py:415-423`; `clients/train.py:40-44` |
| `claude_code` (ACP) | **Anthropic Messages** (`ANTHROPIC_BASE_URL`) | **no** | `claude_code/harness.py:93-95` |

A chat harness must also send only un-namespaced function tools (`train.py:40-44`) and `n=1`. To stay on one branch it must re-send assistant turns verbatim, including `reasoning_content`.
2. **Default runtime is a remote Prime sandbox** (`configs/agent.py:30`), which means Prime tunnels get minted (Prime credentials required, `ensure_prime_auth`). The 512 starts/min cap (`tunnel/prime.py:20-24`) is a flock'd leaky bucket file `~/.cache/verifiers/limiter/prime-tunnel-<VF_RUN_ID>.bucket`. It is shared by every process on a host with the same run id, and without `VF_RUN_ID` it is per process (`runtimes/limiters.py:1-60`, `utils/scope.py`). Each start is retried 3× before `TunnelError`. The elastic pool only grows: idle servers and their tunnels live until the worker stops (`pool.py:116-136`). Local clusters must override the runtime.
3. **Program sampling is silently ignored in training** (temperature, `max_tokens`, `stop`, `tool_choice`), yet `ModelCall.sampling` records the *post-override request body* and `ModelCall.endpoint` records `/chat/completions` even though the call went to `/inference/v1/generate` (`server.py:543-556`). Don't trust `trace.calls[*].sampling` as "what vLLM used" in training runs.
4. **Bridge failures fork branches.** Any tail that is not `[tool*, user?]` after a sampled assistant (e.g. the harness injects a system message, edits a tool result after the fact, or sends images), or a renderer returning `None`, forces a full re-render; if the template re-tokenizes prior turns differently, the prompt breaks the token prefix and a new branch starts. Each branch is a separate training sample repeating the prefix; shared sampled nodes train only once (`trajectories.py:91-114`), but the context is recomputed per branch. Branch explosion = trainer token blow-up.
5. **Dropping reasoning on re-send forks.** `reasoning_content` is in the message hash; harnesses that strip thinking when replaying history will branch every turn.
6. **Streaming is fake for the program.** Training generates whole; eval buffers the provider stream whole. Long turns (> 60 s) switch to a committed 200 SSE with keepalives, so a late error arrives as an SSE error event, not an HTTP status (`server.py:238-310`) — SDK retry semantics differ past the grace period.
7. **Limits are soft by one turn** (checked before each call, `session.py:82-89`); `max_output_tokens` does not bound a single completion (that's `sampling.max_tokens`).
8. **No client-side retries anywhere** (`MAX_RETRIES = 0`, `clients/base.py:14`): transient faults must be retried by the harness SDK (status codes are chosen for that) or by opt-in whole-rollout/episode retries. A harness without retry logic turns one 503 into a failed trace.
9. **Overlong prompts** are caught client-side (`OverlongPromptError` → 400) only if vLLM reports `max_model_len` in `/v1/models`; the cap is cached forever per `(base_url, model)` and matched on `card.id == model` (`renderers/client.py:67-109`). If the harness then **exits cleanly**, the failed call is only recorded on `trace.calls`. The trace stays `ok` and trains its partial trajectory (`rollout.py:421-424`). If the harness dies instead, the stashed `ProviderError(400)` becomes the trace error.
10. **Tool calls the renderer can't parse** (`UNKNOWN_TOOL`) are dropped from the message (`train.py:119-131`) — the program sees plain text; malformed tool-call attempts still train as sampled tokens.
11. **Eval traces carry no tokens** (EvalClient). prime-rl evals always go through `clients.eval_client` (chat-completions, `dispatcher.py:547-553`), so eval metrics can't use token-level fields and eval may differ subtly from train decoding (no renderer; server's chat template).
12. **A frozen generation source without `renderer_config`** builds an `InferenceClient` with an eval-type train client (`algo/base.py:25-30`) → no tokens → nothing trains (warning only, `trajectories.py:176-179`). prime-rl's `GenerationSource` passes the renderer config (`generation_source.py:25-37`).
13. **Per-rollout `Task.toolsets` servers + remote runtimes force tunnels** even for local harnesses, because tool servers must reach `/state` (`interception/__init__.py:46-55`).
14. **A `Response`-typed `@vf.intercept` rewrite drops the turn's tokens (confirmed).** `session.rewrite_response` keeps `Response.tokens`, because interceptors may change only `message`/`finish_reason` (`session.py:402-420`). The server then rebuilds the response via `dialect.parse_response(dialect.validate_response(raw))` (`server.py:818-828`), and `ChatDialect.parse_response` is `response_from_wire`, which carries no token ids (`chat.py:254-256, 537-538`). So the rewritten turn commits two kinds of tokenless node (`graph.py:782, 896-911`). Its new input nodes get `token_ids=[]`. Its assistant node is `sampled=True` with `token_ids=mask=logprobs=[]` and carries the *rewritten* text as if the model sampled it. Replaying this on a synthetic trace confirmed four consequences. (a) The sampled tokens and routing of that turn never train. (b) The next turn cannot bridge (`previous_token_ids` sees no sampled mask), so it does a full render. (c) Concatenation stays exact: the empty nodes pass token tightening trivially, and the next new input node absorbs their re-rendered tokens as mask-False context. (d) `Branch.routed_experts` skips the empty nodes, and the placeholder-row repair targets the last *token-carrying* prefix node. Nothing in code or tests marks this as intended. A `Response`-typed `@stop` alone does not trigger it: that path commits with tokens, then stops (`server.py:812-846`).
15. **`Env.run` with `max_concurrent_agents=1` serializes agents**, including best-of-n attempts (`configs/env.py:50-59`); raising it multiplies live runs per episode.
16. The chat `apply_overrides` docstring in the base says "program's sampling keys are dropped", but `ChatDialect.apply_overrides` actually keeps them unless overridden (`dialects/base.py:381-387` vs `chat.py:583-597`) — matters for eval only.
17. **Usage-derived token counts are lower bounds, and prime-rl consumes them.** `Trace.num_{input,output,total}_tokens` come from per-call `usage`, and input is undercounted under branching (`trace.py:180-191`). Four places read them. GRPO's `length_penalty` reads `num_output_tokens`/`num_total_tokens` (`algo/grpo.py:33-36`). Adaptive concurrency growth reads `episode.num_total_tokens` (`orchestrator.py:350`, `concurrency.py:166-180`). The zero-output tally reads it (`train_sink.py:340-341`), and `payload_tokens` falls back to it when there are no samples (`train_sink.py:31`). Trainer batching uses sample lengths.
18. **The renderer warm-up can be wasted.** `TrainClient.__init__` warms a renderer keyed without `chat_template_kwargs` (`train.py:322-327`). If the run's sampling carries `chat_template_kwargs`, the first turn builds a second tokenizer under a different key (`:369-376`).

---

## 8. For a custom framework

**Essential design (keep):**
- **Interception as the sole contract with agent programs.** It decouples "agent scaffolding" from "training data capture" and lets arbitrary third-party CLIs generate RL data. The session-secret multiplexing (one server/tunnel per N rollouts) and retry coalescing are cheap and necessary at scale.
- **Token-in/token-out generation with client-side rendering + bridging.** This is the actual fix for multi-turn RL: sampled tokens are never re-tokenized, prompts are extensions of exact prior ids, and the engine prefix-caches on literal tokens. Keep the invariant "a branch's tokens are exactly what the engine saw".
- **Message graph with delta tokens per node.** Linear storage, natural handling of compaction/subagents/retokenization, each shared sampled token trained once. The "prepare (hash) → commit (token-tightened)" split is what makes concurrent requests from one trace safe.
- **Errors as data with boundary typing** (`boundary()`), and stashing the real upstream error behind the harness's HTTP failure.
- **Rewards as named, weighted, seeded records** + judges recording their own usage separately.

**Incidental / simplify:**
- The three-way dialect stack is justified for eval breadth, but for a training framework only chat completions matter today; a minimal framework can ship one dialect and treat others as eval-only.
- The Env/Agent/Interaction/`_EpisodeAgent` layering (semaphores, per-episode agent minting, deep-merged seat configs) is heavy for single-agent RL; keep `Episode` as the wire atom but collapse the rest until multi-agent is needed.
- Streaming framing/keepalive logic exists to keep third-party CLIs and tunnels happy; with in-house harnesses a plain non-streaming endpoint suffices.
- Network-policy capability mediation, artifacts, skills, and tunnels are sandbox-product features, not RL mechanics.

**Coupling points to design explicitly:**
- **Sampling ownership.** Make "who owns sampling" (run vs program vs seat) a single explicit field, and record the *effective* engine parameters per call (the current `ModelCall.sampling` is misleading in training).
- **Trainer temperature / sampling masks** must come from the same source as generation (today prime-rl re-derives temperature from its own config).
- **Renderer ↔ engine**: stop tokens, `skip_special_tokens`, logprob format and routed-expert offsets are a tight contract spread over three repos (verifiers, renderers, prime-rl's vLLM extension). Put it in one versioned schema.
- **Task identity hash** across processes (orchestrator loads tasks, server rebuilds them) — a subtle cross-process invariant; consider shipping an opaque task id instead.

---

## 9. Open questions

1. **`renderer.bridge_to_next_turn` semantics** — when exactly does it return `None` (length-truncated previous completion without EOS? tool results with images? templates that re-emit the previous turn's close tag)? Where the previous completion ended without the turn-close token, does the bridge append it (and is that token then unmasked context)? → H (`deps/renderers/renderers/base.py` + model renderers). [UNVERIFIED]
2. **Engine-side semantics of `routed_experts_prompt_start` and `sampling_mask` row shape** are verified here only from the client side. The `/generate` field list and top-level `cache_salt` agree with 06 (E). → E (`src/prime_rl/inference/vllm/serving_tokens.py`).
3. **Transports of `hermes_agent` (train path) and `openclaw`** (§7.1; `rlm` is trained in shipped configs) depend on third-party defaults: Hermes' `determine_api_mode`/provider default, OpenClaw's default provider `api`, and nano-rlm's client. Settling them needs those sources or a live probe. [UNVERIFIED]
4. **Multiple trainable agents per episode** (e.g. self-play `kuhn-poker`, both seats trainable): how does GRPO group them — per episode or per agent name? `rae.py` keys baselines by `trace.agent.name`; GRPO flattens all trainable traces of the group. → C.
5. Env-server worker count × `ElasticInterceptionPool` multiplex × `ElasticRendererPool` multiplex sizing guidance at prime-rl scale is not documented; defaults are 32 rollouts/server and 256 rollouts/renderer. → G/J.
