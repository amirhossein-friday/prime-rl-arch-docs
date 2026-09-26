# Harnesses, Runtimes & Env Serving — prime-rl @ b944873

> Scope: seam **S3** (env-server process, ZMQ protocol, delta stream, worker pool) and seam **S5** (harness ↔ sandbox runtime), plus the harness catalog, MCP tool placement, ACP, tasksets adapters (Harbor / NeMo-Gym / OpenEnv / TextArena) and the reusable envs.
> Pin: prime-rl `b944873`, verifiers `69cc0f9`. Paths under `deps/verifiers/verifiers/v1/` are written `v1/…` below for brevity (= `deps/verifiers/verifiers/v1/…` from the repo root).
>
> **Files read in full.**
> serve: `v1/serve/{__init__ 23, types 63, encoding 68, server 224, client 239, pool 371, delta 329}`.
> runtimes: `v1/runtimes/{__init__ 134, base 410, limiters 71, container 249, subprocess 227, apptainer 214, modal 420, prime 446, docker/__init__ 490, docker/egress 566}`.
> harness: `v1/harness.py 381`, `v1/harnesses/{__init__ 57, node 49, utils/{__init__ 7, core 506, compaction 256, install 65, launch 86, mcp 201}, bash 127, null 54, browser_use/{harness 122, program 287}, claude_code 125, codex/{harness 253, gate.mjs 34}, hermes_agent/{harness 123, program 19}, kimi_code 144, mini_swe_agent/{harness 73, program 8}, openclaw 244, pi/{harness 221, gate.mjs 11}, prime_agent 292, rlm 305, terminus_2/{harness 78, program 94}}` + every `__init__`.
> acp/mcp: `v1/acp/{__init__ 373, runner 401}`, `v1/mcp/{__init__ 19, launch 629, server 358, toolset 34}`.
> tasksets/envs: `v1/tasksets/{__init__ 23, harbor/{__init__ 16, taskset 819, env 86, toolset 86}, nemo_gym/{__init__ 17, taskset 165, server 72, response 119, toolset 100}, openenv/{__init__ 17, taskset 154}, textarena/{__init__ 17, taskset 117}}`, `v1/envs/{agentic_judge 362+17, best_of_n 44+3, isolated_verifier 182+7, single_agent 29+3, user_sim 99+3, shared_agentic_judge/__init__ 6}`.
> configs: `v1/configs/{runtime 126, harness 73, serve 65, agent 80}`, `v1/configs/env.py` (lines 1–80).
> callers read to pin the seams: `v1/rollout.py 554`, `v1/agent.py 752`, `v1/env.py 411`, `v1/utils/{compile 134, loaders 327, scope 27, aio 31, paths 13}`, `v1/interception/{__init__ 118, base 66 (partial), pool 76–151, tunnel/* 185}`, `v1/interception/server.py 310–469`.
> prime-rl: `src/prime_rl/entrypoints/env_server.py 65`, `packages/prime-rl-configs/src/prime_rl/configs/env_server.py 45`, `src/prime_rl/orchestrator/envs.py 273`, `src/prime_rl/orchestrator/dispatcher.py 867`, `src/prime_rl/entrypoints/rl.py 40–340`, `src/prime_rl/utils/pathing.py 200–259`, `src/prime_rl/templates/multi_node_rl.sbatch.j2 150–175, 530–575`, `orchestrator.py 80–99, 212–246`, `orchestrator/utils.py 60–95`, `configs/orchestrator.py 150–195, 780–804`.
> docs: `deps/verifiers/docs/v1/{harnesses 59, harbor 144, tasksets 264}.md`, `deps/prime-envs/HARBOR.md 45`.
> tests: `deps/verifiers/tests/v1/{test_serve_delta 279, conftest 275}`, `test_e2e.py 120–280, 810–928`.
>
> Related docs: 01 (launcher/topology, S1), 03 (orchestrator call site of S3), 06 (inference/transports), 07 (verifiers core: interception, trace, dialects — S4), 10 (envs & tasks: env resolution/loading), 11 (live traces/monitors).

---

## 1. Mental model

**The orchestrator never runs an environment.** For every train/eval *source* (one `[[orchestrator.train.source]]` / eval source), the launcher starts one **env-server** process (`uv run env-server @ envs/<split>/<name>.json`). The orchestrator owns the *taskset* (it loads and samples tasks client-side) and ships each episode's `TaskData` over a **ZMQ ROUTER/DEALER** socket as a `run` request; the env server rebuilds the task, runs the **episode** (1..n agent runs under an `Env`), and streams the growing traces back as **msgpack deltas**, closing with a reply that carries the episode head and per-trace counts (`src/prime_rl/orchestrator/envs.py:1-17`, `v1/serve/server.py:80-123`). An env server is either one in-process `EnvServer` or a **broker** (`EnvServerPool`) in front of N `spawn`ed worker processes, each an ordinary `EnvServer` on an `ipc://` socket; the wire protocol is identical so the client can't tell (`v1/serve/pool.py:1-21`).

**Inside a worker, one agent run = one Rollout = one fresh runtime (sandbox).** `Rollout.open` provisions a runtime named after the trace id, runs task setup and *harness setup* (install the agent program) with the network open, brings up an **interception** slot (the OpenAI/Anthropic-compatible proxy that records every model call into the trace — F's area) and the task's **MCP tool servers**, then — *only if the resolved runtime policy is restricted* — cuts egress to the framework routes, and launches the harness program (`v1/rollout.py:177-356`). The default policy is `allow=["*"]`, under which the cut is a no-op on every runtime (`v1/configs/runtime.py:60-80`). A **harness** is a thin adapter that knows how to install an agent program in a runtime, point it at `endpoint`+`secret` (the interception), hand it the prompt and the MCP URLs, and run it to completion (`v1/harness.py:284-306`). The transcript is **not scraped from the program**: the interception is the contract — every model call must go through `endpoint`, and the trace *is* the record; the program's own output is only used for its exit code / error tail (`v1/harness.py:167-183`) and, for ACP agents, the final visible reply (`v1/acp/__init__.py:305-306`).

**A runtime is "where commands run"**: `subprocess` (host), `docker`/`podman` (local containers driven through the CLI), `apptainer` (unprivileged HPC containers on the host network), `prime` (remote Prime VM sandboxes) and `modal` (remote Modal sandboxes) (`v1/runtimes/__init__.py:41-70`). Its interface is small: `start/stop`, `run(argv, env) -> ProgramResult`, `read/write` files, optional `open_process` (live stdio — required by ACP harnesses), `run_background` (tool servers), `expose` (publish an in-sandbox port), `host_url` (translate a host-loopback URL into one reachable from inside), and `prepare_setup/prepare_execution` (network policy phases) (`v1/runtimes/base.py:134-410`). **Locality is the one bit that decides networking**: `is_local=True` runtimes reach the host-loopback interception directly or through a runtime translation (Docker's egress proxy); `is_local=False` runtimes (prime/modal) force the interception to be published through a **tunnel** (prime_tunnel/frpc by default) (`v1/interception/__init__.py:34-55`, `v1/interception/tunnel/prime.py:1-65`). **Note the default:** `AgentConfig.runtime = PrimeConfig()` — out of the box every agent seat provisions a *remote Prime sandbox* and the interception is tunneled (`v1/configs/agent.py:30`).

---

## 2. Where it runs

```
 node running the orchestrator (single-node: the launch box; SLURM: trainer rank0's node, or the
 last inference node with orchestrator_on_inference — multi_node_rl.sbatch.j2:153-160)
 ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
 │ orchestrator proc ── EnvClient (DEALER) ──tcp://127.0.0.1:<os-port>──┐  (one per source)      │
 │                                                                      ▼                        │
 │ env-server proc  = broker (EnvServerPool ROUTER)  ──ipc:///tmp/vf-pool-XXXX/<i>──┐            │
 │    └─ worker i (spawned, EnvServer ROUTER):                                       │            │
 │         Env (taskset obj, harness objs), shared MCP servers, interception pool   ◄┘            │
 │         (aiohttp on 127.0.0.1 or tunnel bind), per-rollout Runtime objects,                    │
 │         Docker EgressProxy (per docker runtime), CreationLimiter files ~/.cache/verifiers      │
 │    └─ local runtimes: subprocess children, docker/podman containers, apptainer instances       │
 └──────────────────────────────────────────────────────────────────────────────────────────────┘
          │ ClientConfig.base_url (from the RunRequest)                 │ prime/modal SDK calls
          ▼                                                             ▼
   inference server(s) (vLLM / router)                      remote sandboxes (Prime VMs, Modal)
          ▲                                                             │ agent → tunnel URL /v1
          └──────── interception (inside the env-server worker) ◄───────┘ (prime_tunnel / frpc)
```

**Who launches it.**
- Single node: `rl_local` writes one `EnvServerConfig` JSON per launcher-managed source (`src/prime_rl/entrypoints/rl.py:109-113`, `src/prime_rl/utils/pathing.py:230-246`) and `Popen`s `env-server @ <config_dir>/envs/<split>/<name>.json` with logs to `<log_dir>/envs/<split>/<name>.log`, env = `os.environ + DEFAULT_COMMON_ENV_VARS + config.env_vars + orchestrator.env_vars`, and a `monitor_process` thread per server (`rl.py:259-292`). Servers start **concurrently** with the orchestrator; there is no ordering.
- Multi-node SLURM: on the orchestrator's node only, the sbatch body backgrounds `uv run env-server @ $CONFIG_DIR/envs/<split>/<name>.json > $LOG_DIR/envs/<split>/<name>.log 2>&1 &` for every train/eval source right before starting the orchestrator (`src/prime_rl/templates/multi_node_rl.sbatch.j2:542-560`). The comment says each server "publishes the loopback address it bound".
- "Launcher-managed" = `source.serve.address is None`. A source with `serve.address` set is *externally managed*: no config written, no process spawned, the orchestrator connects to that address (`rl.py:55-61`, `configs/orchestrator.py:163-164`). The docstring mentions "a k8s deployment running env servers in their own pods", but the shipped chart defines no env-server resource (`k8s/prime-rl/templates/` = deployment, pvc, service), and only the `rl`/`eval`/`sft` launchers ever spawn env servers — a bare `orchestrator` with launcher-managed sources waits 600s for the address file and fails. A k8s run must deploy env servers itself and set `serve.address` per source (which also drops the loopback/same-filesystem assumptions of §5.3).
- `eval` and `sft` (online evals) entrypoints do the same for eval sources (`src/prime_rl/entrypoints/eval.py:147-174`, `sft.py:132-136, 397-402`).

**Process tree of one env server.**
- `main()` sets proc title `EnvServer`, defaults `VF_RUN_ID` to `$PRL_RUN_ID` (else a fresh uuid) — this keys run-wide creation limiters (`src/prime_rl/entrypoints/env_server.py:54-61`).
- `run_server` computes `sandbox_labels = [$PRL_RUN_NAME]`, optionally starts a thread that writes the bound address atomically to `address_file` (`env_server.py:24-41`), then calls `serve_env(**pool_serve_kwargs(serve.pool), address=serve.address or "tcp://127.0.0.1:0", address_queue, log_setup=partial(setup_worker,…), config_data=env_config_data(env), max_concurrent=serve.max_concurrent)` (`env_server.py:44-51`).
- `serve_env`: if `max_workers is None or > 1` → broker `EnvServerPool` in this process + spawned workers; else a single in-process `EnvServer` (`v1/serve/pool.py:334-366`). **Default `ServeConfig.pool` is `ElasticPoolConfig(max_workers=None, multiplex=128)`** (`v1/configs/serve.py:21-45`) → default is always broker + ≥1 worker process, scaling without cap. The in-process path is reached by `static num_workers=1` **or** `elastic max_workers=1` (`pool_serve_kwargs`, `serve.py:57-65`).
- In pool mode the broker publishes its address (→ address file) **before any worker has loaded the env** (`pool.py:343-353`; workers spawn inside `run()`, `:171-172`); in single mode only after `load_environment` + bind (`server.py:66-71`).
- Each worker process runs `serve_env(max_workers=1, address="ipc://…/<i>", death_pipe=…, log_setup=…, **server_kwargs)` (`pool.py:113-128`), i.e. an `EnvServer` whose `log_setup` (`setup_env_server_logging` + `set_base_sandbox_labels`) runs in-process (`env_server.py:19-21`, `orchestrator/utils.py:73-79`).

**Lifecycle.**
- *Start*: worker constructs `EnvServer` → `load_environment(config)` (imports taskset/harness/env plugins, builds harness objects; the taskset object is constructed but never iterated — `server.py:36-39, 80-87`) → binds ROUTER → `run()` enters `env.serving()` (shared MCP tool servers + interception pool, then `Env.start()`) (`server.py:183-194`, `v1/env.py:360-386`).
- *Steady state*: one asyncio task per request (`server.py:195-216`).
- *Shutdown*: SIGTERM is converted to `KeyboardInterrupt` (`_arm_teardown`) so `asyncio.run` runs the `finally`s — interception/tunnel teardown, `runtime.stop()` of live rollouts (`pool.py:49-71`). The broker terminates workers (`terminate` → `join(10)` → `kill`) and unlinks the ipc files (`pool.py:266-286`). Workers also watch a *death pipe* from the broker and self-SIGTERM on EOF, so a SIGKILLed broker doesn't orphan workers (`pool.py:63-71`). Runtimes additionally have a synchronous `atexit` backstop (`v1/runtimes/base.py:98-124`). **SIGKILL of a worker runs none of this** — its sandboxes leak until provider idle/lifetime limits (Prime `idle_timeout=3600s`, Modal 24h) (`v1/runtimes/prime.py:98-99`, `v1/runtimes/modal.py:234`).
- *Crash*: see §7 — the pool has **no per-worker restart or death detection** (`pool.py:18-20` TODO).

**How the orchestrator addresses it.** `Env.start()` (one per source) waits up to `ENV_SERVER_STARTUP_TIMEOUT=600s` for `<config_dir>/envs/<split>/<name>.address` (polling 0.5s), builds `EnvClient(address)`, polls `health` until it answers (a second, separate 600s budget), then loads the taskset client-side (`src/prime_rl/orchestrator/envs.py:39-119`, `src/prime_rl/utils/pathing.py:222-227`). All episodes of a source multiplex over that one DEALER socket. With the default pool, health proves only that the *broker* bound — it answers inline, before and regardless of any worker (§7.2). Only local `rl`/`sft` attach `monitor_process` (broker PID only, `rl.py:284-291`, `sft.py:372`); multi-node SLURM backgrounds env servers as shell jobs, so a non-zero broker exit is caught by the node's `wait -n` loop and tears the run down (`multi_node_rl.sbatch.j2:550-559, 590-596`); on either path only the broker is watched, never pool workers and `eval` only reaps them at exit (`eval.py:196-199`).

---

## 3. Mechanics

### 3.1 One `run` request end-to-end (orchestrator → worker → sandbox → back)

```mermaid
sequenceDiagram
  participant D as Dispatcher (orch)
  participant C as EnvClient (DEALER)
  participant B as Broker (ROUTER)
  participant W as Worker EnvServer (ROUTER)
  participant R as Rollout (in W)
  participant S as Runtime/sandbox
  D->>C: env.run(client, model, cache_salt, task_data, on_delta)
  C->>B: [req_id, "run", msgpack(RunRequest)]
  B->>W: [req_id, "run", payload] (least-busy worker DEALER)
  W->>R: env.run_slot(slot, ctx, gate, on_trace=streamer.watch)
  R->>S: make_runtime + start + prepare_setup + task.setup + harness.setup
  R->>R: serve_interception, serve_tools, prepare_execution(routes)
  R->>S: harness.session(...).turn() -> run_program / ACP prompt
  S-->>R: model calls via interception (recorded into Trace; Trace.notify)
  W-->>B: [id, req_id, "delta", msgpack(delta)] (per flush)
  B-->>C: [client_id, req_id, "delta", data]
  C-->>D: on_delta(dict) (live view)
  R->>S: finalize, score, harness.cleanup, runtime.stop()
  W-->>B: [id, req_id, "reply", msgpack(RunResponse)]
  B-->>C: reply → EpisodeAssembly.finish → WireEpisode (validated off-loop)
```

1. **Dispatch** (orchestrator, B's area): `schedule_group_episode` picks the client (`train_client` from the source's generation source, `eval_client` for evals), a `cache_salt = str(policy_version_at_start)` for live-policy work, and creates `run_episode()` which calls `env.run(client, model_name, cache_salt, task_data=group.task.data.model_dump(mode="json"), on_delta)` (`src/prime_rl/orchestrator/dispatcher.py:536-615`). `Env._sampling` injects `cache_salt` into `SamplingConfig.extra_body` (`envs.py:121-125`). After the episode, a shielded `clients.finish_sessions(session_ids)` runs for every trace id seen in deltas or the episode (`dispatcher.py:602-610`).
2. **Client send** (`v1/serve/client.py:93-135`): `request_id = uuid4().hex`; register a future in `_pending` and the delta callback in `_deltas`; payload = `msgpack.packb(request.model_dump(mode="json"), use_bin_type=True)`; send `[request_id, method, payload]`. `timeout=None` for runs ("rollouts run untimed"), only health uses a timeout.
3. **Broker route** (`pool.py:185-238`): `health` answered inline with a pre-packed `HealthResponse` (never touches a worker); `cancel` routed to the worker holding the target (`pending[target]`) or answered inline with `CancelResponse(cancelled=False)`; everything else goes to `min(workers, key=active)`, incrementing `active` and `in_flight`, recording `pending[request_id] = {client_id, worker}`, forwarding `[request_id, method, payload]` over that worker's DEALER. With `elastic`, `_maybe_scale_up(in_flight)` runs after each forward.
4. **Worker receive** (`server.py:183-216`): poll 100ms; a 4-frame message `[client_id(=broker DEALER identity), request_id, method, payload]` becomes `asyncio.create_task(_handle(...))`; for `run` the task is registered in `_running[request_id]` *at dispatch* so a cancel that arrives before the handler first runs still finds it (ZMQ preserves per-peer order).
5. **Rebuild & run** (`server.py:97-123`): `RunRequest.model_validate` → `ModelContext(client, model, sampling)` (no client object built here; each rollout builds/closes its own) → `_build_task(task_data)` = `data_cls.model_validate(task_data)` wrapped in `task_cls(data, env.config.taskset.task)` → `(slot,) = env.slots(task)` → inside `DeltaStreamer(slot, send_delta)`: `env.run_slot(slot, ctx, self._gate, on_trace=streamer.watch)`. `_gate` is `Semaphore(max_concurrent)` spanning requests; `run_slot` holds one permit per episode *attempt* (`v1/env.py:313-358`).
6. **Episode** (`v1/env.py:242-305`): `Env.setup(agents)` → `Env.run(task, agents)` (e.g. `SingleAgentEnv` → `agents.agent.run(task)`) under `timeout.episode`, then `finalize` under `timeout.finalize`; whole-episode retries via `run_episode_with_retry(..., config.retries)`. Each agent run acquires the episode's `max_concurrent_agents` semaphore (default 1) (`v1/configs/env.py:49-59`, `v1/agent.py:664-673`).
7. **Rollout** (§3.4) produces a `Trace`; every `Trace.notify()` (phase changes, and the interception after each recorded turn) schedules a delta flush (`v1/serve/delta.py:1-21`).
8. **Reply** (`server.py:115-123, 161-181`): `RunResponse(head=dump(episode, exclude={"traces"}), traces=[TraceSummary(id, nodes, calls)])`, packed with `msgpack_encoder`, sent as `[client_id, request_id, b"reply", data]`. The `DeltaStreamer.__aexit__` flushes the final state *before* the reply is built (`delta.py:124-134`), so all deltas precede the reply on the wire.
9. **Broker relay back** (`pool.py:239-260`): `delta` frames are relayed as-is (routing key only is copied); the `reply` pops `pending`, decrements `active`/`in_flight`, and is relayed.
10. **Client assemble** (`client.py:66-91, 186-221`): one receive loop per client. `delta` → `unpack` → `EpisodeAssembly.apply` → user `on_delta(dict)`. `reply` → future. Then `assembly.finish(head, summaries)` checks per-trace node/call counts and fails loudly on a gap (`delta.py:302-321`); the record is validated into `WireEpisode` in a thread, one at a time per client (`client.py:137-152`, `_decode_slots = BoundedSemaphore(1)`).
11. **Orchestrator post-processing** (`envs.py:127-153`): a failed multi-trace episode marks its otherwise-clean traces failed ("partial episodes never train"). The dispatcher then validates task provenance `(key, hash)` against the dispatched task (`dispatcher.py:116-120, 712`), promotes empty episodes/trajectories to errors (`dispatcher.py:668-679`), stamps group/run/policy info and enqueues (`dispatcher.py:705-729`).

### 3.2 Cancellation path

- Orchestrator cancels an episode task (stale drop, overload shed, eval superseded, shutdown — `dispatcher.py:240-267, 399-437, 731-853`) → `EnvClient._request` catches `CancelledError`, pops the pending future and fires a **fire-and-forget** `CancelRequest(request_id=<run id>)` with a fresh id, kept alive in `_cancel_tasks` (`client.py:118-129, 154-163`). `close()` drains those tasks before closing the socket (`client.py:223-239`; regression test `test_env_client_close_drains_finished_cancels`, `tests/v1/test_e2e.py:851-858`).
- Broker routes the cancel to the target's worker (`pool.py:198-223`); worker cancels `_running[target]` and replies `CancelResponse(cancelled=True|False)` (`server.py:141-147`). The cancelled run handler still replies `BaseResponse(success=False, error="Cancelled: rollout aborted")` so broker accounting stays exact (`server.py:152-155`); the client already dropped it.
- In the worker, cancellation propagates into the Rollout: `Agent._run_once` / `Rollout.open` call `abort()` on `BaseException` → harness session close, interception/tool stack unwind, `harness.cleanup`, `runtime.stop()` (shielded) (`v1/agent.py:458-468`, `v1/rollout.py:343-351, 433-449`). `DeltaStreamer.__aexit__` with an exception cancels in-flight flushes and sends nothing more (`delta.py:132-134`).

### 3.3 Delta streaming (`v1/serve/delta.py`)

- **Field classes** (`delta.py:44-73`): `HEADER_FIELDS = (version, id, verifiers, task)` sent once in `open`; `LIST_FIELDS = (nodes, calls, errors, extra_usage, request_rewrites, response_rewrites)` append-only, each delta carries items past the sent count; `SCALAR_FIELDS = (agent, tools, mm_token_type_id_map, rewards, metrics, info, root_reply, is_completed, ok, stop_condition, timing, num_input_tokens, num_output_tokens, num_total_tokens)` re-sent whole when their packed value changes. A unit test asserts the three sets exactly cover every serialized `Trace` field — add a Trace field and you must route it here (`tests/v1/test_serve_delta.py:74-82`).
- **Two in-place mutations are allowed** on already-sent nodes: new `semantic_parents` links (`links: {node_idx: [ParentLink…]}`) and a **repair of the last routed-experts row** of a node (`routing_repairs: {node_idx: encoded_row}`) because the next prefill can correct the previous turn's final MoE routing row, possibly widening dtype (`delta.py:167-190, 208-218`; client side rebuilds the array `delta.py:256-268`; test with uint8→uint16 `test_serve_delta.py:180-245`).
- **Pending preview**: `pending` = messages of the in-flight request no node holds yet (e.g. a tool result before the model answers) — sent when it changes, cleared implicitly when nodes land (`delta.py:237-243, 296-300`; test `test_serve_delta.py:117-176`).
- **Discard**: a trace id the streamer has a cursor for but that is no longer in `slot.traces` (per-agent or episode retry) → `{trace, discard: True}` (`delta.py:199-201`; `v1/env.py:328-340`).
- **Flush discipline**: `Trace.notify` fires at rollout phase changes (`rollout.py:184, 224, 355, 478-507`), after every call the interception records (`interception/server.py:581`) and on each pending-preview update (`trace.py:473-477`); → `call_soon(_start_flush)` coalesces changes within one loop iteration; `flush` is serialized by a lock so deltas leave in diff order; the cursor advances only after a successful send, so a failed send is re-diffed next time (`delta.py:136-165`; test `test_serve_delta.py:85-114`). Diffs deep-copy the cursor and re-pack every scalar each flush — CPU cost is O(scalar size) per flush per trace.
- **Encoding** (`v1/serve/encoding.py`): msgpack with a `default=` hook only for non-native types: Path/UUID→str, set→list, Enum→value, datetime→iso, numpy scalars→item, ndarray/torch tensor → `{"__torch_tensor__": True, dtype, shape, data: bytes}`, pydantic → `model_dump()`, dataclass → `asdict`, else `to_jsonable_python`. Responses/deltas use `model_dump(mode="python", serialize_as_any=True)` so subclass fields (env-specific TaskData, harness configs) survive (`delta.py:85-88`). Requests use `model_dump(mode="json")`. `unpack` uses `strict_map_key=False` because node-index keys are ints (`delta.py:80-82`).

### 3.4 One Rollout (`v1/rollout.py`), the S5 seam in order

`Agent._rollout_params` first resolves the per-task runtime config (`resolve_runtime_config`), validates the harness/task/runtime pairing, resolves timeouts and caps remote timeouts at 24h for non-Prime remote runtimes, and picks the interception (`v1/agent.py:538-579`, `v1/utils/compile.py:21-134`). Then `Rollout.open()` (`rollout.py:177-356`):

| # | Step | Code | Network state |
|---|---|---|---|
| 1 | `make_runtime(runtime_config, name=trace.id)` + `atexit` register (unless borrowed) | `rollout.py:185-186`, `runtimes/__init__.py:73-76` | – |
| 2 | `runtime.env = task.runtime_env()` (borrowed: `with_env` view sharing the physical box) | `rollout.py:206-211`, `base.py:206-212` | – |
| 3 | `runtime.start()` (provision) | `rollout.py:218-219` | setup-open |
| 4 | `runtime.prepare_setup()` (re-opens egress if this box was already cut once) | `rollout.py:220`, `base.py:374-380` | open |
| 5 | `task.setup(trace, runtime)` under `timeout.setup` | `rollout.py:231-235` | open |
| 6 | `harness.setup(runtime)` — install the agent program — same deadline | `rollout.py:236-240` | open |
| 7 | `serve_interception(...)` → `(base_url, model_secret, state_secret)` | `rollout.py:247-259` | open |
| 8 | `endpoint = runtime.host_url(f"{base_url}/v1")` | `rollout.py:260` | – |
| 9 | `serve_tools(toolsets, runtime, shared, state_secret, state_route=trace.id, state_base=base_url)` → `{name: mcp_url}` | `rollout.py:262-271` | open |
| 10 | `runtime.prepare_execution([endpoint, *mcp_urls])` — **egress cut iff `network_restricted`** | `rollout.py:274` | restricted |
| 11 | optional request-interceptor rewrite of the initial prompt | `rollout.py:279-318` | – |
| 12 | `harness.session(ctx, trace, runtime, endpoint, secret, mcp_urls, data, tool_interception_url=host_url(base_url+"/tool") if SUPPORTS_TOOL_INTERCEPTION and interceptors/stops exist)` | `rollout.py:319-342` | restricted |

"Network state" applies only when the task-intersected runtime policy is restricted (`network_restricted = "*" ∉ allow or block≠[]`, `configs/runtime.py:78-80`); with the default `allow=["*"]`, docker/prime/modal return early from `prepare_setup`/`prepare_execution` (`base.py:374-380`, `docker/__init__.py:335-336`, `prime.py:243-244`, `modal.py:244-245`) and the agent keeps full egress. `subprocess`/`apptainer` configs aren't `NetworkPolicyConfig`, so they never cut and a policy-bearing task is rejected (`compile.py:45-53`). **Deadlines:** steps 5–6 (+11–12) share `timeout.setup` (default `None`); the agent segment has the only default bound (4h, `agent.py:57-69`); steps 1–4 and 7–10 run under **no deadline**; `finalize`/`scoring`/`env.timeout.episode` default to `None` (`configs/agent.py:13-24`, `configs/env.py:18-24`).

`step()` runs **one segment** under the cumulative agent deadline (`asyncio.timeout_at`): `HarnessSession.turn(messages)` → `launch()` first, `resume()` afterwards; an expired deadline becomes `HarnessError("agent timeout: …")` (`rollout.py:358-431`, `v1/harness.py:340-377`). `close()` closes the session and the interception/tool stack, then (if not failed) task `finalize` (+ optional artifact collection) under `timeout.finalize`, then `task.score ‖ harness.score` under `timeout.scoring`, then `harness.cleanup(trace, runtime)` and `runtime.stop()` for owned runtimes (`rollout.py:451-554`).

**Setup is trusted and online; a restricted agent is not.** Everything that needs the internet (apt, npm, uv, git clone, image pulls) must happen in steps 3–6/9. Under a restricted policy, after step 10 only the interception and MCP routes (plus the configured allowlist) are reachable. What "cut" means per runtime: **docker/podman** — a one-shot `NET_ADMIN` helper installs iptables (only `lo` + gateway replies out) and every later `exec` gets `HTTP(S)_PROXY` pointing at the in-process egress proxy, which enforces allow/block per host and TLS SNI (§3.6); **prime** — `set_network(allow=[route hosts + allow] | deny=block)` polled until applied (60s) or the rollout fails; **modal** — outbound domain allowlist (SNI, :443 only) with the setup-time `0.0.0.0/0` CIDR grant removed; **subprocess/apptainer** — nothing. Note that the ACP runner and chat program are **prepared** (uv sync) in `setup()` and only **launched** after the cut, reusing the per-runtime digest cache (`v1/acp/__init__.py:67-70, 235-245`, `base.py:250-302`).

### 3.5 Model-endpoint reachability from inside the box

`serve_interception` either borrows the env-level interception (`Env.serving` builds one per worker: default `ElasticInterceptionPool(multiplex=32)`, `v1/env.py:360-386`, `v1/configs/env.py:60`, `v1/interception/pool.py:76-151`) or, for a bare `Agent.run`, brings up a per-rollout `InterceptionServer` (`v1/interception/__init__.py:74-101`). `requires_tunnel` is true iff any consumer is off the host network: a non-local harness runtime, a live shared tool server in a remote runtime, or a task tool server config placing one remotely (`interception/__init__.py:34-55`; env-level decision `v1/env.py:388-402`).

| Harness runtime | Interception bind | URL handed to harness (`endpoint`) | Path of a model call |
|---|---|---|---|
| `subprocess` | `127.0.0.1:0` | `http://127.0.0.1:<p>/v1` | direct loopback |
| `apptainer` | `127.0.0.1:0` | same (host network; `host_url` = identity) | direct loopback |
| `docker`/`podman` (Linux) | `127.0.0.1:0` | `http://<userinfo@>127.0.0.1:<proxy_port>/.vf-host/<cap_token>/v1` via `EgressProxy.callback_url` | container → a proxy **listener socket created inside the container netns** (fd passed over `SCM_RIGHTS`) but *serviced by the worker process* → worker dials host `127.0.0.1:<p>` (`docker/__init__.py:237-263, 272-331`, `egress.py:139-158, 206-275`) |
| `docker` (macOS) | same | `http://host.docker.internal:<proxy_port>/.vf-host/<token>/v1` | container → host proxy → loopback |
| `prime` / `modal` (`is_local=False`) | tunnel's `bind_host/bind_port` (prime: `127.0.0.1:0`; custom: `0.0.0.0:<port>`) | public tunnel URL + `/v1` (`host_url` = identity) | sandbox → prime_tunnel (frpc) public URL → worker's interception (`interception/server.py:441-459`, `tunnel/prime.py:39-65`, `tunnel/custom.py:33-42`) |

Then the interception forwards to the inference server using the `ClientConfig` shipped in the `RunRequest` (one server-owned client per distinct client config, `interception/server.py:356-366`) — i.e. the worker process, not the sandbox, needs a route to vLLM. Prime tunnel creation is rate-capped at 512/min per API token via a run-scoped file-lock bucket (`tunnel/prime.py:17-24`, 3 retries `:52-56`); the scope is `VF_RUN_ID`, so two runs sharing one token on one host each get 512/min. The elastic *interception* pool exists precisely so `multiplex=32` rollouts share one server+tunnel (`interception/pool.py:76-84`) — and it **hard-codes `PrimeTunnelConfig()`** for every server it mints (`interception/pool.py:123-125`), so a `custom` (bring-your-own) tunnel is only reachable via `interception.type = server | static`. Likewise a host-local tool server consumed by a remote harness is always bridged with `PrimeTunnel()` (`mcp/launch.py:381-384`). Remote runtimes without Prime credentials therefore need `interception.type=server` + `tunnel.type=custom` and colocated tools.

### 3.6 Runtime providers — mechanics per provider

**Common base (`v1/runtimes/base.py`).** The interface, by obligation:

| Member | Status (default) | Who needs it |
|---|---|---|
| `start()`, `run(argv, env)`, `_read(path, max_bytes)`, `write(path, data)` | **abstract** | every rollout (`prepare_uv_script` = `write`+`run`; `read` for scoring/artifacts/Harbor rewards) |
| `cleanup()` (sync) / `teardown()` (async) | optional (no-op / `to_thread(cleanup)`) | `stop()`, atexit backstop |
| `open_process(argv, env) -> RuntimeProcess` | optional (raises ⇒ `supports_live_processes=False`) | **all 8 ACP harnesses** (`ACPHarness.session` raises `HarnessError`, `acp/__init__.py:119-122`) |
| `run_background(argv, env, log)` | optional (raises) | tool servers placed in this runtime (`serve_in_runtime`, `mcp/launch.py:346`) |
| `published_port` / `expose(port)` | optional (`None` / `http://127.0.0.1:<port>`) | a non-colocated tool server hosted here |
| `host_url(url)` | optional (identity) | local runtimes not on the host network (Docker) |
| `prepare_setup()` / `prepare_execution(routes)` | optional (policy no-ops) | `NetworkPolicyConfig` runtimes |
| `is_local`, `scripts_dir` ClassVars | `True`, `/tmp/vf-scripts` | tunnel decision; uv script cache |
| `stop`, `read`, `run_program`, `alive`, `with_env`, `prepare_uv_script` | framework — don't override | |

Process-style harnesses need only the abstract four (`run_program` = `run`).
- `Runtime.__init__(name)`: `name = name or "vf-<12hex>"`, per-run `env: dict`, uv interpreter cache + per-digest locks, MCP install bookkeeping, `_setup_claimed`, `stopped` (`base.py:153-166`).
- `stop()` sets `stopped=True` then `run_shielded(teardown())` — cancellation cannot interrupt teardown; `teardown()` defaults to `to_thread(cleanup)`; `cleanup()` is the sync, idempotent source of truth usable from `atexit` (`base.py:176-196`; `run_shielded` semantics `v1/utils/aio.py:10-31`).
- `run_program` = `run` but documented as "never replayed by any framework layer" (the rollout's main program) (`base.py:224-230`).
- `prepare_uv_script(script)`: content-addressed path `scripts_dir/<sha256>.py`, atomic write via temp+`mv`, then `_ENSURE_UV` (keep existing uv, else `pip install --user uv`, else curl/wget the installer, installing curl via apt/apk if needed) + `uv sync --script` + `uv python find --script`; returns an argv that activates the script venv (or the bare interpreter with `activate=False` so the venv doesn't leak into child processes) (`base.py:27-46, 250-317`).
- `read(path, max_bytes)`: cap enforced at the source by reading `max_bytes+1` and raising `SandboxError` past the cap; default capped `_read` is `head -c | base64` through a temp file (so a missing file still fails) (`base.py:319-364`).
- `alive()` = `run(["true"])` succeeded — used on non-zero harness exit to distinguish `SandboxError` (box died) from `HarnessError` (`base.py:214-222`, `harness.py:167-183`).
- `SERVICE_PORT = 8000` is the one port a sandbox forwards out for an in-box service (`base.py:48-51`).

**`subprocess`** (`runtimes/subprocess.py`): `start` makes `~/.cache/verifiers/runtimes/subprocess/<name>` as cwd (`:95-98`); every `run` inherits the host env **minus any var containing `API_KEY`**, plus `process_env(env)`; `start_new_session=True` and a SIGKILL of the process group if the await is cancelled (timeouts reap hung children) (`:100-126`). `open_process`/`run_background` supported; teardown SIGTERMs background groups, waits 5s, SIGKILLs, rmtree's the workdir (`:128-227`). uv script envs are shared class-wide across a worker's runtimes (`:81-89`). No isolation, no network policy, `is_local=True`.

**`docker` / `podman`** (`runtimes/docker/__init__.py`, shared exec machinery `runtimes/container.py`):
- `start`: check `docker version` (hint about the `docker` group); if restricted, build/cache `localhost/verifiers-network:1` (alpine + iptables) once; name `vf-<uuid>`; `docker run --detach --network bridge --publish 127.0.0.1::8000 [--cpus] [--memory Ng] [--gpus N | --device nvidia.com/gpu=i] [--http-proxy=false (podman)] [restricted: --cap-drop NET_ADMIN --cap-drop NET_RAW --security-opt no-new-privileges --sysctl net.ipv6.conf.all.disable_ipv6=1] [--add-host host.docker.internal:host-gateway (non-Linux)] --env K=V… --entrypoint sleep --name vf-… <image> infinity` (`:83-174`). Docker GPU type is verified by `nvidia-smi` inside (`:176-199`). Reads the image `Config.Env`, creates the workdir (default `/app`) owned by the image user, reads the published host port, starts the `EgressProxy` (`:200-255`).
- **Unrestricted (the default)**: no `--cap-drop`, no iptables, no `HTTP(S)_PROXY` — the container keeps full bridge egress for the whole rollout; the proxy still starts (framework-only policy) but serves only `/.vf-host/` host callbacks (`:239-249, 335-336`).
- Every operation is a `docker exec [-i] --env … --workdir <wd> <ctr> …` (`_exec`, `:427-453`); when unrestricted, `NO_PROXY` gets `localhost,127.0.0.1,host.docker.internal` appended; when cut, `HTTP(S)_PROXY=http://verifiers:<token>@127.0.0.1:<proxy_port>` is injected (`:409-420`). `write` = `sh -c 'mkdir -p parent && cat > path'` over stdin; `_read` = `cat`/`head -c` (`container.py:230-249`).
- `open_process`: wraps argv in `sh -c` that writes its (post-`setsid`) PID to `/tmp/vf-process-<uuid>.pid`, polls for the pidfile for up to 5s, signals the process group via a second `exec … kill` (the CLI doesn't forward signals) (`container.py:69-216`).
- `run_background`: a `docker exec --detach … sh -c 'exec argv > log 2>&1'`, proxy env included when restricted (`:455-469`).
- `expose(8000)` → `http://127.0.0.1:<host_port>`; any other port raises (`:265-270`).
- `cleanup`: `docker rm --force <ctr>` (30s timeout), idempotent (`:471-485`). `PodmanRuntime` only changes `engine`/`info_cls` (`:488-490`).

**Docker egress enforcement** (restricted policy only; `prepare_execution`, `docker/__init__.py:333-407`; proxy `egress.py`). On Linux the proxy's listening socket is created by a throwaway `python3` helper *inside the container netns* (`--network container:<ctr>`, image's own python or `python:3.11-alpine`) and handed back over `SCM_RIGHTS`; the worker's asyncio server accepts on it, so the agent reaches `127.0.0.1:<proxy_port>` on its own loopback while the bytes are served by the worker (`docker/__init__.py:50-58, 272-331`). The model endpoint is a capability URL on that port — `127.0.0.1` is in `NO_PROXY`, so the harness talks to it directly and the `/.vf-host/<token>/` path (not `Proxy-Authorization`) authenticates it; the proxy then dials the interception on host loopback (`egress.py:139-158, 206-212, 255-263`).
- `routes=None` → proxy policy "allow all incl. non-global" (trusted re-setup); otherwise `NetworkPolicy(config, framework_routes)`.
- First cut runs a throwaway `--network container:<ctr> --cap-add NET_ADMIN` helper that: blackholes Docker's embedded DNS `127.0.0.11`; INPUT accepts local-source, gateway, established-reply, rejects rest; OUTPUT accepts `lo`, established replies to the gateway, and (non-Linux) the host proxy port, rejects rest. After this the proxy is the only way out.
- The proxy (h11-based) requires `Proxy-Authorization: Basic verifiers:<token>` except for `/.vf-host/<cap>/…` callback paths (`egress.py:206-231`); `CONNECT` tunnels are checked against the policy on the port *and* again on the TLS ClientHello **SNI** before bytes flow (`egress.py:311-327, 79-114`); plain HTTP gets exactly one request per connection; loopback targets are refused (host callbacks must use the capability path); resolved addresses must be globally routable unless the destination is a framework route or setup is trusted (`egress.py:51-76, 255-291`). Callback responses have cookies re-scoped under the capability path and same-origin redirects re-wrapped (`egress.py:435-512`). Timeouts: 10s header, 300s I/O; callbacks have no read timeout (long streaming model calls) (`egress.py:19-21, 329`).

**`apptainer`** (`runtimes/apptainer.py`): no network policy (config doesn't inherit `NetworkPolicyConfig`) and shares the host network (`:27-31`). `start`: `mkdtemp` backing dir with `workspace/` and `session/`; image = local SIF, or `docker://ref` pulled once to `~/.cache/verifiers/runtimes/apptainer/images/<sha256>.sif` under a per-(loop,ref) lock with atomic publish; refuses images with a non-trivial startscript; seeds the writable workspace from the image's workdir; `apptainer instance start --cleanenv --containall --no-mount cwd,hostfs,bind-paths --no-eval --workdir <session> --writable-tmpfs --bind <workspace>:<workdir> [--cpus] [--memory] [--nv] <sif> vf-<uuid>` (`:79-199`). Exec via `apptainer exec --cleanenv --no-eval --cwd <wd> instance://… env K=V… argv` (env passed through `env`, not `--env`, because `--env` is comma-separated) (`:65-77`). `cleanup`: `instance stop --force` + rmtree (`:201-214`). `is_local=True`.

**`modal`** (`runtimes/modal.py`), `is_local=False`, `published_port=8000`:
- `start`: `modal.App.lookup("verifiers-v1", create_if_missing=True)`; inside a run-scoped `CreationLimiter("modal-sandbox", VF_RUN_ID, creates_per_sec=40)`, a **shielded** `Sandbox.create("sleep","infinity", image=Image.from_registry(image).entrypoint([]), workdir, env=self.env, cpu, memory MB, gpu, region, block_network=not network_access, outbound_domain_allowlist=["*"] / outbound_cidr_allowlist=["0.0.0.0/0"] if restricted, timeout=24h, encrypted_ports=[8000])` that assigns `_sandbox` before the cancellation is delivered (else the sandbox bills invisibly for 24h) (`:174-236`). Any provisioning error → `SandboxError`.
- `prepare_execution`: `_experimental_set_outbound_network_policy(domain_allowlist, cidr_allowlist=[])` within 60s; only HTTPS/443 DNS names (leading `*.` only) are translatable; loopback framework routes are dropped (colocated); **no deny lists** (`allow=["*"]` with `block` is rejected at config time) (`:43-114, 238-268`).
- `run` = `sandbox.exec` draining stdout/stderr concurrently; `open_process` with pidfile and signal-by-exec; on a live process without a readable PID the whole sandbox is terminated (fail closed) (`:283-357`). `write` via filesystem API (mkdir parent first), relative paths resolved against workdir (`:371-395`). `expose` → `sandbox.tunnels()[port].url` (`:270-281`). Teardown `terminate.aio()`; atexit uses the sync `terminate()` (`:397-420`). Agent timeout is capped at 24h (`v1/utils/compile.py:115-134`).

**`prime`** (`runtimes/prime.py`), `is_local=False`, `supports_live_processes=True`:
- Requires Prime auth at construction (`ensure_prime_auth`) (`:146-151`). One `AsyncSandboxClient(background_job_output_concurrency=100)` per event loop, leased by all runtimes on that loop so status polls batch (≤100 ids/request) (`:53-65, 157-167`).
- `start`: map `cpu_cores, memory_gb, disk_size_gb, gpu_count/gpu_type (parse_gpu), timeout_minutes=-1, idle_timeout_minutes=ceil(idle_timeout/60) (default 60), region`; inside a run-scoped limiter `creates_per_min` (default None = off), a shielded `create(CreateSandboxRequest(name, labels=BASE_LABELS+config.labels, docker_image, environment_vars=self.env, …))` capturing `info.id`; if `pending_image_build_id` is set the platform auto-builds a VM image (~10 min, warns); `wait_for_creation(id, max_attempts=180)`; `mkdir -p workdir` (`:157-239`). `BASE_LABELS` = `[$PRL_RUN_NAME]` set by the env server's worker setup and the orchestrator (`env_server.py:19-21, 36-37`, `orchestrator.py:220-224`).
- `run` = `start_background_job` + poll `get_background_job` with exponential backoff (0.1→3s); a finished job whose output read expired (`stdout_error/stderr_error`) is re-read up to 10 times (output deadline 300s) — tuned for thousands of concurrent setups (`:35-47, 284-324`).
- `open_process` = SDK live process; `run_background` = background job with `exec … > log 2>&1`; `expose` **unsupported** (raises) — tool servers must be colocated or live elsewhere (`:326-358`). `_read` streams `head -c` for capped reads, else `download_file` to a temp dir; `write` = `upload_bytes` (gateway mkdirs parents) (`:360-415`).
- `prepare_execution`: `set_network(allow=[route hosts + config.allow])` or `deny=config.block` when `allow=["*"]`, then poll `get_network` until `applied` (60s) or **refuse to start the agent** (`:241-282`). Teardown `delete(id)` and release the shared client lease; atexit uses the sync client (`:417-446`).

**Creation limiters** (`runtimes/limiters.py`): a leaky bucket in `~/.cache/verifiers/limiter/<name>-<scope>.bucket`, advanced under `fcntl.flock`, so the rate holds across *all processes on the host* sharing `VF_RUN_ID` (broker workers included). Reservations are never released (a cancelled waiter still consumed its slot). Requires a wall clock comparable across processes (`:1-71`, `v1/utils/scope.py:12-27`).

### 3.7 Harness model

**Interface** (`v1/harness.py`):

| Member | Contract |
|---|---|
| `Harness[ConfigT](config)` | stateless value; one object per distinct harness config per env (`v1/env.py:116-124`) |
| `APPENDS_SYSTEM_PROMPT` | emit `system_prompt` separately; else it's folded into a string prompt with a warning (`:35-36, 64-88`) |
| `SUPPORTS_MCP` | may receive `mcp_urls`; else runs with tools are rejected (`compile.py:88-94`) |
| `SUPPORTS_TOOL_INTERCEPTION` | program asks the rollout's `/tool` gate before each tool call (`:38-42`) |
| `SUPPORTS_RESUME` | default `resume()` relaunches on the accreted conversation (`:43-45, 230-278`) |
| `EXECUTES_CODE` | False only for `null` (`:46-50`) |
| `SUPPORTS_SKILLS` | program discovers SKILL.md; `install_skills(runtime, dest)` copies host folders / `{runtime: path}` roots (`:51-54, 103-165`) |
| `NEEDS_CONTAINER` | True by default; only `bash`/`null` run on `subprocess` (`:55-59`, `compile.py:101-106`) |
| `setup(runtime)` | install before the execution timeout; network open |
| `launch(ctx, trace, runtime, endpoint, secret, mcp_urls, data[, tool_interception_url]) -> ProgramResult` | run to completion (abstract, `:284-306`) |
| `session(...) -> HarnessSession` | rollout-scoped handle; default adapts launch/resume; ACP overrides (`:185-212, 309-381`) |
| `score(trace, runtime)` | harness `@metric` methods → `trace.metrics` (`:214-228`) |
| `cleanup(trace, runtime)` | remove per-trace state, idempotent (`:280-282`) |

Model wiring convention: OpenAI-style programs get `base_url=endpoint` (ends in `/v1`), `api_key=secret`, `model=ctx.model`; Anthropic-style programs get `endpoint.removesuffix("/v1")` (claude-code, pi/kimi with `anthropic_messages`). Harness config fields common to all: `id` (default `bash`), `env`, `forward_env` (host vars forwarded by name), `mcp_header_env` (ACP MCP headers read from host env), `tool_timeout=600s` (MCP call timeout), `disabled_tools`, `skills` (`v1/configs/harness.py:26-67`).

**Harness catalog** (install path, pin, transport, MCP wiring, gate):

| id | Style | Install (in `setup`) | Model wiring | MCP | Tool gate |
|---|---|---|---|---|---|
| `bash` | bundled chat program | `prepare_uv_script(CHAT_PROGRAM_SOURCE)` (openai, mcp==2.0.0, httpx, httpx2, tenacity) | `--base-url/--api-key/--model` argv | `--mcp-config` JSON | `--tool-interception-url` (`bash/harness.py:54-127`) |
| `null` | bundled chat program, no local tools | same | same | same | same (`null/harness.py:17-54`) |
| `browser_use` | chat program + `browser-harness==0.1.13` | PEP 723 script | same | same | – (`browser_use/harness.py:54-122`) |
| `mini_swe_agent` | PEP 723 `mini-swe-agent==2.4.6` + litellm | uv script | `-c model.model_kwargs.api_base=endpoint`, `api_key=secret` | – | – (`mini_swe_agent/harness.py:18-73`) |
| `terminus_2` | PEP 723 `harbor==0.23.0` Terminus2 w/ a local `exec` env | uv script | `--base-url/--api-key` | – | – ; kills its tmux server in `finally` (`terminus_2/harness.py:20-78`) |
| `claude_code` | ACP | Node 22.19 → `/var/tmp/vf-node`; npm `@anthropic-ai/claude-code@2.1.278` + `claude-agent-acp@0.79.0` into `/var/tmp/vf-claude-agent-acp-*` | `ANTHROPIC_BASE_URL/API_KEY/MODEL`, `CLAUDE_CONFIG_DIR=.vf-claude/<trace>` | ACP session `mcp_servers` (`strictMcpConfig`) | settings.json `permissions.ask=["*"]` (`claude_code/harness.py:37-125`) |
| `codex` | ACP | npm `@openai/codex@0.155.1` + `codex-acp@1.12.0` → `/var/tmp/vf-codex-*` | `DEFAULT_AUTH_REQUEST` gateway `baseUrl=endpoint`, Bearer secret; `CODEX_HOME=/tmp/vf-codex-home-<trace>` | `config.toml` `mcp_servers` | PreToolUse hook `gate.mjs` with trusted hash (`codex/harness.py:57-253`) |
| `openclaw` | ACP (gateway + `openclaw acp`) | `install-cli.sh` v2026.9.5 → `/var/tmp/vf-openclaw-*` + patches its transcript sanitizer | provider `baseUrl=endpoint`, key via env | gateway config `mcp.servers` | – (`openclaw/harness.py:125-244`) |
| `pi` | ACP (`pi-acp`) | npm `pi-coding-agent@0.86.1`, `pi-mcp-adapter@2.34.0`, `pi-acp@0.0.33` → `/var/tmp/vf-pi/mcp` | `models.json` provider; transport chat/responses/anthropic | generated extension `mcp.js` | extension `gate.mjs` via `ctx.ui.confirm` (`pi/harness.py:60-221`) |
| `hermes_agent` | ACP (native) | GitHub tarball `v2026.9.14` + `uv sync --extra acp --extra mcp` → `/var/tmp/vf-hermes-agent-*` | `config.yaml` provider `api=endpoint`; `HERMES_HOME=/tmp/vf-hermes/<trace>` | ACP session | – (`hermes_agent/harness.py:35-123`) |
| `kimi_code` | ACP (native `kimi acp`) | `install.sh` v2.0.2 → `/tmp/vf-kimi-code` | `KIMI_MODEL_BASE_URL/API_KEY/NAME/PROVIDER_TYPE` | ACP session | config `permission.rules ask *` (`kimi_code/harness.py:49-144`) |
| `prime_agent` | ACP | sha256-verified release tarballs v0.9.5 (commit `a7d791b…`) via npm -g prefix `/var/tmp/vf-prime-agent/<commit>` | `models.json` provider `intercept` | ACP session | – ; lifecycle metadata → `trace.info["acp_lifecycle"]` (`prime_agent/harness.py:97-292`) |
| `rlm` | ACP (nano-rlm `rlm --acp`) | `git clone` nano-rlm @ `b425f2d…` + `install.sh` → `/tmp/vf-rlm-<sha>` | `session_meta["ai.prime.rlm/runtime-v1"].provider = {base_url, api_key}` + policy knobs | ACP session | – ; session snapshot → metrics, truncation stop conditions (`rlm/harness.py:144-305`) |

**Direct vs ACP, trainable vs eval-only.** 5 direct (`bash`, `null`, `browser_use`, `mini_swe_agent`, `terminus_2` subclass `Harness`); 8 ACP (the rest subclass `ACPHarness`). ACP is **orthogonal to trainability**. Every prime-rl *train* source uses the renderer client (`train_client_type="renderer"`, `orchestrator.py:202`, `algo/base.py:27`) → `TrainClient`, which accepts only the chat-completions dialect (`/v1/chat/completions`, `dialects/chat.py:425`) with unnamespaced function tools and raises otherwise (`v1/clients/train.py:40-44, 342-350`), so the model call fails at run time. Eval sources use the proxy `EvalClient` and accept every dialect.

| Harness | Default model API (how it is wired) | Trainable under prime-rl | Evidence |
|---|---|---|---|
| `bash`, `null`, `browser_use` | chat-completions (bundled OpenAI-SDK loop) | **yes** | `harnesses/utils/core.py:243-274` |
| `mini_swe_agent`, `terminus_2` | chat-completions (litellm `custom_llm_provider=openai`, `api_base=endpoint`) | **yes** | `mini_swe_agent/harness.py:59-63`, `terminus_2/program.py:69-70` |
| `pi` | chat-completions: `transport="chat_completions"` → `models.json` `api: "openai-completions"`, `baseUrl=endpoint` | **yes** at default; `responses` / `anthropic_messages` ⇒ eval-only | `pi/harness.py:52-54, 109-140` |
| `kimi_code` | chat-completions: `transport="chat_completions"` → `KIMI_MODEL_PROVIDER_TYPE=openai` (vs `openai_responses` / `anthropic`), `KIMI_MODEL_BASE_URL=endpoint` | **yes** at default (per the harness's own mapping; the kimi binary is outside the pin) | `kimi_code/harness.py:43-45, 82-91, 118-123` |
| `prime_agent` | chat-completions: `api: "openai-completions"` | **yes** | `prime_agent/harness.py:215-219` |
| `hermes_agent` | provider `openai`, `api=endpoint`; the vendor-native transport (`determine_api_mode`) is injected **only for eval clients**, so a train client gets Hermes's plain OpenAI provider | **yes (inferred)**: the train/eval branch exists for this reason; Hermes's default mode lives outside the pin | `hermes_agent/harness.py:73-89`, `hermes_agent/program.py:12-14` |
| `rlm` | nano-rlm gets only `provider={base_url, api_key}` (no API selector) | **yes (deployment evidence)**: 12 shipped configs train it through the renderer client, e.g. `examples/advanced/intellect-3.1/rl.toml:62-68`, which would fail on any other dialect; nano-rlm's source is outside the pin | `rlm/harness.py:223-229` |
| `claude_code` | Anthropic Messages (`ANTHROPIC_BASE_URL=endpoint−/v1`) | **no**: eval-only | `claude_code/harness.py:93-95` |
| `codex` | gateway auth to `endpoint`; speaks OpenAI Responses per verifiers' docs (`docs/v1/architecture.md:21`; the harness sets no API type, so it is Codex's own default), and its MCP tools arrive **namespaced** (the e2e asserts `tool.namespace`), which `tool_to_wire` refuses; either alone rules out training | **no**: eval-only | `codex/harness.py:238-249`, `tests/v1/test_e2e.py:415-423`, `clients/train.py:40-44` |
| `openclaw` | provider key = model-name prefix (`ctx.model.partition("/")`); the config sets only `baseUrl`/`apiKey`, so the API type comes from OpenClaw's built-in provider of that name | **undeterminable from the pin; depends on the model name**. Treat as eval-only until one request's route is observed. | `openclaw/harness.py:161, 192-199` |

Across shipped TOMLs under `examples/` and `configs/` the harness ids are only `null` (72×), `bash` (17×) and `rlm` (12×).

Every `ensure_installed` wraps the install in a filesystem lock (`flock`/`lockf`, else a symlink spinlock keyed `pid:starttime`) plus an optional "ready" test, so concurrent rollouts sharing a runtime install once (`harnesses/utils/install.py:8-56`).

**Bundled chat program** (`harnesses/utils/{core,compaction,mcp}.py` spliced into one PEP 723 script by `bundle_program`, `harnesses/utils/launch.py:17-32`):
- Loop: `chat()` streams chat-completions with `stream_options.include_usage`, retries transport errors with `x-stainless-retry-count` so the interception's body-digest replay guard dedups retries (`core.py:243-274`); accumulates reasoning fields the SDK doesn't (`core.py:192-240`); runs until a text-only reply (`core.py:338-420`).
- Local tools: `bash` (`subprocess.run(["bash","-c"])`, 3600s), `edit` (unique-string replace), `search` (Serper; key via argv, never env) (`core.py:37-189`); MCP tools are `<server>_<tool>` over streamable HTTP with 6-attempt reconnect-retry (`harnesses/utils/mcp.py:11-201`); tool results middle-truncated to 20 000 bytes (`compaction.py:18, 119-140`).
- Gate: when `--tool-interception-url` is set, each call first POSTs `{tool_call_id, name, arguments}` with `Bearer <secret>`; `deny` substitutes the returned message, `stop` raises (ending the program) (`core.py:313-335, 372-383`).
- Secrets: `--api-key` is passed over argv (not env) so tool subprocesses don't inherit it (`launch.py:24-26`); an initial Messages prompt goes through a file `.vf-initial-messages-<trace>.json` that the program deletes after reading (`launch.py:72-80`, `core.py:444-449`).
- **Compaction** (`compaction.py`): enabled by `BashHarnessConfig.compaction` (`bash/harness.py:32-51`); threshold = `summarize_at_tokens` or `context_window − 16 384` discovered from `/v1/models` fields (`max_model_len`, `context_length`, …) (`compaction.py:83-106`). Triggered by usage ≥ threshold or a recognized context-overflow 400/413 (`CONTEXT_OVERFLOW_MARKERS`, `compaction.py:40-80`); it asks the model for a checkpoint summary (`tool_choice="none"`, 3 attempts, falling back to the last usage-verified snapshot), and rebuilds `[system…, user: framing + summary]` (`compaction.py:155-256`). Without compaction, an overflow ends the run cleanly with the transcript so far (`core.py:348-358`). These summary calls go through the interception like any other model call (so they appear on the trace).

**ACP harnesses** (`v1/acp/__init__.py`, `v1/acp/runner.py`):
- `ACPHarness.session` requires `runtime.supports_live_processes` (subprocess, docker/podman/apptainer, modal, prime all qualify), calls `prepare_acp` → `ACPConfig(env, command, prompt, mcp_urls, system_prompt, session_meta, client_capabilities)` and `gate_tools` when a tool-interception URL is present (`acp/__init__.py:108-139`). `launch()` raises — ACP harnesses are session-only (`:141-153`).
- Transport: host ↔ `runner.py` (a PEP 723 script, `agent-client-protocol==0.12.1`, `UV_FROZEN=false`) via **8-byte big-endian length-prefixed JSON packets** on the live process's stdin/stdout (max 128 MiB) (`acp/__init__.py:28-29, 156-198`, `runner.py:324-346`). Operations: `{"operation":"prompt","config":{command, user_contents, mcp_urls, mcp_headers, system_prompt, session_meta, client_capabilities, tool_interception}}` → `{"ok", "result": ACPTurn{reply, stop_reason, response_metadata, update_metadata}, "error"?}`; `{"operation":"shutdown"}` → `{"ok","result":{"response_metadata"}}` (`runner.py:349-386`).
- The runner spawns the agent as its child (`spawn_agent_process(command…, env=os.environ)`), `initialize` + `new_session(cwd, mcp_servers, **session_meta)`, sends each segment as `session/prompt` (the system prompt is inlined as a `(system)…[user]` preface on the first turn), collects the visible reply from `AgentMessageChunk`s, and answers `request_permission` through `ToolGate` (`POST {tool_call_id, arguments}` to `/tool`, 120s; unreachable gate = deny; `stop` cancels the prompt) (`runner.py:49-151, 198-298`).
- Host side: the first turn lazily starts the process; each turn writes one packet and reads one reply under a lock; any exception kills the process ungracefully; `_require_model_turn` fails a segment that committed no model call (`acp/__init__.py:235-317, 168-176`). `close()` sends `shutdown` (10s), then waits 10s → `terminate` 5s → `kill` 5s (`:319-373`).
- Multi-turn: ACP sessions keep native state across `turn()`s, so ACP harnesses host users (user-sim, OpenEnv, TextArena) even with `SUPPORTS_RESUME=False` (`v1/agent.py:370-385`).

### 3.8 MCP tool servers (placement & reachability)

- A `Toolset` is an MCP server class (`@vf.tool` methods) with config `ToolsetConfig{colocated=False, runtime=SubprocessConfig(), url=None}` (task-scoped) or `SharedToolsetConfig{runtime, url}` (taskset-scoped) (`v1/mcp/toolset.py:16-34`).
- Launch (`mcp/launch.py:301-362`): in a non-subprocess runtime the framework **packages the server's source** (installed wheel re-packed, or `uv build --sdist` of the source project; cached per process/loop) and `uv pip install`s it into `<workdir>/.vf-venv` (`:75-276`); runs `python -m <module>` via `run_background` with `VF_CONFIG` (JSON config), `VF_STATE_URL`/`VF_STATE_SECRET`, and either `MCP_PORT=8000`+`MCP_HOST=0.0.0.0` (exposed out of a sandbox) or `MCP_PORT_FILE` (OS-chosen port, read back for ≤180s); probes `http://127.0.0.1:<port>/mcp` (180 × 1s).
- The server (`mcp/server.py:267-338`) sets `PR_SET_PDEATHSIG`, binds, writes its port, runs `setup()` then fetches its rollout's task from `<interception>/task` (bearer = state secret) for `setup_task`, and serves a **stateless** streamable-HTTP MCP app. Each tool call pulls/pushes typed per-rollout `State` over `<interception>/state` (last-write-wins) (`:169-247`). Shared servers get per-rollout state coordinates as HMAC-signed query parameters (`vf_state_url`, `vf_state_route`, `vf_state_signature`) (`:82-92, 150-167`, `launch.py:555-574`).
- Reachability (`reachable_url`, `launch.py:365-386`): colocated → `http://127.0.0.1:<port>`; otherwise `service.expose(port)`; a host-local service consumed by a remote harness is bridged with a `PrimeTunnel`; a non-colocated URL consumed by a local harness is passed through `harness_runtime.host_url` (Docker proxy) (`:445-460`). Restricted-network tool runtimes are only allowed when colocated (`:406-419`).
- Shared servers are started once per worker in `Env.serving()` (`v1/env.py:404-411`, `launch.py:501-552`).

### 3.9 Tasksets adapters

- **Harbor** (`tasksets/harbor/`): `HarborTaskset.load` runs `harbor download <dataset> --export -o <dir>` once into `~/.cache/harbor/<name>[_<sha12>]` (atomic publish), then every `task.toml` with an `instruction.md` becomes `HarborData`: prompt = instruction, `image = [environment].docker_image` (Dockerfile-only tasks rejected unless `ignore_dockerfile`), `workdir`, network allowlist from Harbor's network mode (`public→["*"]`, else `allowed_hosts`), resources (CPU, MB→GB, GPU `"type:count"`, scaled by `resource_multiplier`), timeouts dropped by default (`ignore_timeouts=True`), `task_dir` (a **host path**), `env`/`healthcheck`/`mcp_servers`, `artifacts`, `[[verifier.collect]]` hooks, and an optional separate-verifier `VerifierConfig` (`taskset.py:436-660, 805-819`). `HarborTask.setup` tars+uploads `environment/` and waits for the healthcheck; scoring stages `tests/` into `/tests`, clears reward files, runs `bash /tests/test.sh` with resolved `[verifier.env]`, reads `/logs/verifier/reward.json` (float or `{reward, …metrics}`, ≤1 MiB) else `reward.txt` else 0 (`taskset.py:193-375`). Task-declared MCP servers run as colocated `HarborMCPToolset` proxies (stdio/sse/streamable-http upstreams) (`toolset.py:21-86`). The package exports `HarborEnv` (an `IsolatedVerifierEnv`) as the taskset's default env: separate-verifier tasks defer scoring, collect artifacts, and grade in a freshly provisioned box with retries (`env.py:32-86`, `envs/isolated_verifier/env.py:60-182`).
- **NeMo-Gym** (`tasksets/nemo_gym/`): JSONL rows with `responses_create_params`; prompt parsed by the Responses dialect (`taskset.py:98-119`). A **shared** `NeMoGymToolset` (MCP, `TOOL_PREFIX=None`) bridges each rollout's Gym tools — either the Gym server's own MCP endpoint or direct `POST /<tool>` — using per-rollout cookies carried in `NeMoGymState` (`toolset.py:15-100`). `setup` posts the row to `/seed_session`; reward = `POST /verify` with the trace converted to a Responses object (`taskset.py:50-87`, `response.py:11-119`). `NeMoGymEnv.start` optionally launches the resource server and writes its URL into `taskset.config.task.resources_url` — the same object `_build_task` reads as `env.config.taskset.task`, since `Env.__init__` passes `config.taskset` by reference and `Taskset.__init__` stores it (`v1/env.py:97, 104`, `v1/taskset.py:43-44`); it launches the resource server (`nemo-gym==0.4.0` PEP 723 script) in a host subprocess runtime **per worker** on `127.0.0.1:<random>` (`taskset.py:122-165`, `server.py:1-72`).
- **OpenEnv** (`tasksets/openenv/taskset.py`): one task per `resets` entry, prompt `None`; `OpenEnvEnv.run` opens a `GenericEnvClient` (existing `base_url`, or `from_env(env, use_docker)` **on the env-server host**), fetches `/schema`, then drives `agents.player.interaction(task)`: each observation+action-schema JSON is a user turn, the reply is parsed as an action and stepped; per-step rewards summed into `openenv_reward` (`:73-134`).
- **TextArena** (`tasksets/textarena/taskset.py`): `INFINITE=True` generator of seeded games (seed = idx); the env replays `random.seed(seed)` + `ta.make` + `reset` synchronously (process-global RNG) and plays the engine as the user until `done`; reward `game_reward` (`:63-117`).
- **Reusable envs** (`v1/envs/`): `single_agent` (fallback env, one `agent` seat), `best_of_n` (n concurrent attempts in a `TaskGroup`, `best`/`pass_at_n` metrics), `user_sim` (modeled user on the `null` harness, `max_turns=8`, untrainable), `agentic_judge` (`shared-agentic-judge` borrows the solver's box via `agents.solver.provision(task)`; `agentic-judge` = isolated fresh box with restored artifacts; judge untrainable, verdict JSON at `/tmp/verdict.json`), `isolated_verifier` (deterministic re-scoring in a fresh box). The runtime-relevant bits: borrowed boxes (`Agent.run(runtime=box)`) are never started/stopped by the run, never retried, and reject tasks whose network policy differs from the box's (`v1/agent.py:86-140, 416-421`).

---

## 4. Interfaces & contracts

### 4.1 ZMQ wire (S3)

| Hop | Socket | Frames |
|---|---|---|
| orchestrator → server | client `DEALER` → server `ROUTER` (`ROUTER_MANDATORY=1`, `SNDHWM=RCVHWM=0` unlimited, `LINGER=0`) | send `[request_id, method, payload]`; ROUTER sees `[client_id, request_id, method, payload]` |
| server → orchestrator | ROUTER → DEALER | `[client_id, request_id, kind∈{delta, reply}, data]`; DEALER sees `[request_id, kind, data]` |
| broker → worker | broker `DEALER` (per worker, `LINGER=0`, default `SNDHWM=1000` — the one bounded hop, §7.2) → worker `ROUTER` on `ipc:///tmp/vf-pool-XXXX/<i>` | `[request_id, method, payload]` (worker's `client_id` = broker DEALER identity) |

Sources: `client.py:44-60, 109-112`, `server.py:55-64, 103-106, 177-179, 199-205`, `pool.py:96-108, 113-141`.

| Method | Request (msgpack of `model_dump(mode="json")`) | Response |
|---|---|---|
| `health` | `{}` | `HealthResponse{success=True, error=None}` (answered by broker inline) |
| `run` | `RunRequest{task_data: dict (dumped TaskData), client: ClientConfig, model: str, sampling: SamplingConfig}` | 0..n `delta` frames, then `RunResponse{success, error, head: dict (Episode minus traces), traces: [TraceSummary{id, nodes, calls}]}` |
| `cancel` | `CancelRequest{request_id}` | `CancelResponse{success, error, cancelled: bool}` |
| failure | – | `BaseResponse{success=False, error="<ExcType>: <msg>"}` → client raises `RuntimeError(error)`; unknown method → `error="unknown method '<m>'"` |

Types: `v1/serve/types.py:10-63`. `method` is a `ClassVar` — it travels only as the route frame, never in the payload. `client` is the `ClientConfig` union on `type` — `train` = `TrainClientConfig{base_url, api_key_var, headers, renderer, renderer_model_name, multiplex=256}`, `eval` = `{base_url, api_key_var, headers}` (`v1/configs/client.py:29-90`; built by `setup_client`, `orchestrator/clients.py:259-280`). **The API key never crosses the wire** — only the *name* of its env var; the worker resolves it from its own environment (`resolve_api_key`, `client.py:93-104`), so the env-server process must inherit that variable. `sampling` is `extra="allow"`, so provider keys (`top_k`, `extra_body.cache_salt`, …) ride as extras. Round-trip verified by execution (client subtype, sampling extras, cancel, failure-as-`RunResponse`, int map keys). Delta dict keys: `trace`, `discard`, `open`, `links`, `routing_repairs`, `nodes`, `calls`, `errors`, `extra_usage`, `request_rewrites`, `response_rewrites`, `set`, `pending` (`delta.py:192-246`).

### 4.2 Files & env vars

| Artifact | Writer | Reader | Notes |
|---|---|---|---|
| `<config_dir>/envs/<split>/<name>.json` (`EnvServerConfig`: `env`, `serve`, `address_file`, `log{level, json_logging}`) | launcher (`write_env_server_config`) | `env-server @ …` | `pathing.py:230-246` |
| `<config_dir>/envs/<split>/<name>.address` | env server (atomic tmp→replace) | orchestrator `wait_for_address` | `env_server.py:24-31`, `envs.py:50-62` |
| `<log_dir>/envs/<split>/<name>.log` | env server stdout/stderr | humans | `rl.py:267-280` |
| `/tmp/vf-pool-XXXX/<i>` | broker (ipc sockets) | workers | unlinked on shutdown (`pool.py:96, 266-285`) |
| `~/.cache/verifiers/limiter/*.bucket` | every process creating modal/prime sandboxes or prime tunnels | same | cross-process pacing (`limiters.py:20-51`) |
| `~/.cache/verifiers/runtimes/{subprocess/<name>, scripts, apptainer/images}` | runtimes | runtimes | subprocess workdirs, uv scripts, SIFs |
| `~/.cache/harbor/<dataset>` | orchestrator (taskset load) | env-server workers (`task_dir`) | **must be the same filesystem path** (§7) |

Env vars: `PRL_RUN_ID` → `VF_RUN_ID` (limiter scope; `env_server.py:57-60`, `scope.py:12-27`); `PRL_RUN_NAME` → sandbox labels; in-box: `VF_CONFIG`, `VF_STATE_URL`, `VF_STATE_SECRET`, `MCP_HOST`, `MCP_PORT`, `MCP_PORT_FILE` (tool servers, `launch.py:317-334`); Docker proxy `HTTP(S)_PROXY`, `NO_PROXY` (`docker/__init__.py:409-444`); harness-specific (see §3.7 table).

### 4.3 Config fields

**prime-rl `EnvServerConfig`** (`configs/env_server.py:10-45`): `env: vf.EnvConfig = SingleAgentEnvConfig()` (narrowed by env/taskset id), `serve: vf.ServeConfig`, `address_file: Path|None`, `log: LogConfig`, `output_dir`. Validation: env id required.

**Orchestrator `EnvConfig` (per source)** (`configs/orchestrator.py:156-194`): `env`, `serve` (only `pool`/`max_concurrent` are consumed by the launcher; `address` set = externally managed), `name` (≠ `"agg"`), `shuffle`.

**`vf.ServeConfig`** (`v1/configs/serve.py`):

| Field | Type / default | Effect |
|---|---|---|
| `pool.type` | `"elastic"` (default) \| `"static"` | discriminator |
| `pool.max_workers` (elastic) | `int\|None = None` | cap on workers; None = unbounded |
| `pool.multiplex` (elastic) | `int = 128` | spawn next worker when `in_flight ≥ 0.9·workers·multiplex` |
| `pool.num_workers` (static) | `int = 4` | pre-spawn; `1` = single in-process `EnvServer` (no broker; so is elastic `max_workers=1`) |
| `address` | `str\|None = None` | bind/connect address; launcher default binds `tcp://127.0.0.1:0` |
| `max_concurrent` | `int\|None = None` | per-worker episode semaphore; None = unbounded (orchestrator's `max_inflight` is then the only bound) |

**`EnvConfig` knobs relevant here** (`v1/configs/env.py:34-61`): `timeout.episode`, `timeout.finalize` (None), `retries` (whole-episode), `max_concurrent_agents=1`, `interception: elastic(multiplex=32) | server(tunnel: prime|custom{url, port}) | static(servers[])`.

**`AgentConfig`** (`v1/configs/agent.py:27-66`): `harness` (None → taskset default → `bash`), `runtime = PrimeConfig()`, `model/client/sampling` (None → run's), `max_turns/max_input_tokens/max_output_tokens/max_total_tokens`, `timeout{setup, rollout (None→task→4h; 0=unbounded), finalize, scoring}`, `retries`.

**Runtime configs** (discriminator `type`):

| type | Fields (default) |
|---|---|
| `subprocess` | – |
| `docker` / `podman` | `image` (`python:3.11-slim` / `docker.io/library/python:3.11-slim`), `workdir` (None→task→`/app`), `cpu`, `memory` (GB), `gpu` ("A100", "A100:2", "2"), `disk` (advisory), `allow=["*"]`, `block=[]` |
| `apptainer` | container fields; `workdir` must be absolute, non-root, no `..`/`:`/`,` |
| `modal` | `image`, `workdir`, `network_access=True`, `region`, `cpu=1.0`, `memory=2.0`, `gpu`, `disk=5.0` (ignored), `creates_per_sec=40.0`, `allow`, `block` (deny lists unsupported) |
| `prime` | `image`, `workdir`, `region`, `labels=[]`, `cpu=1.0`, `memory=2.0`, `gpu`, `disk=5.0`, `idle_timeout=3600s`, `creates_per_min=None`, `allow`, `block` |

Precedence (`compile.py:21-73`): CLI/TOML non-default > task (`TaskData.image/workdir/resources/network_allow/network_block`) > default; task network policy *intersects* the runtime's (`configs/runtime.py:88-126`); `allow=[]` or `"*"∈block` = framework-only; concrete allow + block lists are mutually exclusive (`configs/runtime.py:66-76`). A task needing a network policy on a runtime without one (subprocess, apptainer) is a hard error; a resource field the runtime lacks warns once.

---

## 5. Invariants & assumptions

1. **Served TaskData must round-trip through JSON whole.** The server refuses a TaskData class with any `exclude=True` field (`server.py:40-50`); the orchestrator ships `model_dump(mode="json")` and the worker validates into the declared type. Anything a task needs at run time must live in `TaskData` or the taskset/task config the *worker* was started with (the worker never calls `load()`).
2. **Worker config = launcher-time config.** The worker's env config comes from the JSON written at launch; orchestrator-side changes after launch don't reach it. The `ClientConfig`, model name and sampling are per-request, so weight updates / model renames need no worker restart (`server.py:89-95`).
3. **Orchestrator and env server share a host (and a filesystem)** for launcher-managed servers: loopback bind (`env_server.py:46`), the address file, and task payloads that carry host paths (`HarborData.task_dir`, `taskset.py:150-151, 196-201, 300-303`).
4. **All deltas precede the reply, and none are lost on a live connection** — `finish` hard-fails on a count mismatch (`delta.py:302-321`). The client stores deltas only in memory; there's no resumption across a broker/worker restart.
5. **Every model call goes through the interception endpoint**; a program that calls the provider directly (or uses a background "utility/title/recap" model) produces unrecorded tokens or fails under egress policy. Harnesses explicitly disable such calls (claude-code `extraArgs.name`, openclaw `utilityModel: ""`, hermes `title_generation.enabled: False`, `approvals.mode: off`) (`claude_code/harness.py:85-86`, `openclaw/harness.py:183`, `hermes_agent/harness.py:90-97`).
6. **Setup is online, execution is policed — when the policy is restricted** (default `allow=["*"]` polices nothing); framework routes (interception, MCP URLs) are always allowed and cannot be blocked on Docker (they can be on Prime deny rules — docs `harbor.md:100-106`, `egress.py:71-76`).
7. **One owned runtime per agent run**, named by the trace id; borrowed runtimes are never started/stopped/retried by the run (`rollout.py:185-219, 536-545`, `agent.py:416-421`).
8. **Teardown is cancellation-proof on the normal path** (shielded `stop`) and best-effort on interpreter exit (`atexit`); provider create calls are shielded until the id is captured (`modal.py:201-209`, `prime.py:197-215`).
9. **Run-scoped rate limits assume `VF_RUN_ID` is set and shared** by every process of a run on the host; without it each process gets its own bucket (a warning is logged) (`scope.py:12-27`).
10. **The `/tool` gate and model endpoint share one bearer (`secret`)**; the `/state`+`/task` channel uses a separate state secret (`interception/server.py:368-377`).

---

## 6. Extension points

### 6.1 Recipe: add a new sandbox provider

1. **Config + info.** In `v1/runtimes/<name>.py`: `class XConfig(NetworkPolicyConfig)` (or `BaseConfig` if you can't enforce egress) with `type: Literal["x"] = "x"`, `image`, `workdir: str|None = None`, and resource fields **named exactly** `cpu`, `memory` (GB), `gpu` (Modal-style string; use `parse_gpu`), `disk` — `resolve_runtime_config` maps `TaskData.resources` onto config fields *by name* and only when the config field is still at its default (`compile.py:57-72`). `class XRuntimeInfo(XConfig, BaseRuntimeInfo)`. Add a model validator rejecting policies you can't enforce (Modal rejects deny lists, `modal.py:100-114`).
2. **Runtime class** `class XRuntime(Runtime)`:
   - `__init__(self, config, name=None)`: `super().__init__(name)`; `self.config = config.model_copy(update={"workdir": config.workdir or "/app"})`; `self.info = XRuntimeInfo(**self.config.model_dump())`.
   - `is_local: ClassVar[bool]` — False if the box can't reach host loopback (forces a tunnel for the interception).
   - `async start()`: provision with `self.env` as the box's base env; wrap the create call in `run_shielded` **and set `info.id`/handle inside it**; wrap in `creation_limiter(rate, "x-sandbox", run_scope())` if the provider rate-limits; raise `SandboxError` for provisioning failures (one rollout's problem).
   - `async run(argv, env) -> ProgramResult`: non-zero exit is *data* (return it); raise `SandboxError` only for infra failure; merge `self.process_env(env)`; make cancellation kill the remote process if possible.
   - `async _read(path, max_bytes=None)`: must raise on a missing file; must enforce `max_bytes` at the source (or delegate to `super()._read` which uses `head|base64`).
   - `async write(path, data)`: create parents; resolve relative paths against `workdir`.
   - `cleanup()` (sync, idempotent, no event loop) and optionally `async teardown()`; don't clear the handle `cleanup` keys off before the first await (`base.py:185-191`).
   - Optional: `open_process` (required for every ACP harness), `run_background` (colocated tool servers, NeMo), `published_port`/`expose` (tool servers hosted in-box), `host_url` (if local but not host-network), `prepare_execution(routes)` (policy; `routes=None` must restore open egress), `scripts_dir`.
3. **Register** in `v1/runtimes/__init__.py`: import; add to the `RuntimeConfig` and `RuntimeInfo` unions (both discriminated on `type`); add `"x": XRuntime` to `_runtime_cls`; `__all__` (`runtimes/__init__.py:41-70, 104-134`); optionally re-export from `verifiers/v1/__init__.py:75-86` (`vf.XConfig`). Nothing else registers runtimes — no entry-point/plugin loader exists anywhere in verifiers (the only `entry_points` use is MCP source repackaging, `mcp/launch.py:113`). Both unions are **wire-load-bearing**: `rollout.open` stamps `trace.agent.runtime = runtime.info` (`rollout.py:197`), and the orchestrator's `WireEpisode` parse re-validates `AgentInfo.runtime: RuntimeInfo | None` (`trace.py:112`) and `agent.config.runtime: RuntimeConfig` (`WireAgentConfig`, `configs/agent.py:30, 69-74`) — verified by execution: an unknown `type` raises `ValidationError` on either field, so an orchestrator whose verifiers lacks the new type fails every episode. (Harness configs, by contrast, ride as extra-allow `WireHarnessConfig` and need no orchestrator-side registration.) Keep the provider SDK import lazy inside methods (as Modal does, `modal.py:175-180`): a module-level import (Prime's `prime_sandboxes`, `prime.py:16`) becomes a hard dependency of every process that imports `verifiers.v1`.
4. **Lockstep edits.** `utils/compile.py:115-134` (`cap_remote_agent_timeout` if the provider has a lifetime limit); `rollout.py:107-116` (special-case if your config has a "no network" flag like Modal's `network_access`); `mcp/launch.py` assumes `runtime.config.workdir` exists for non-subprocess runtimes and branches on `runtime.type == "subprocess"` (`launch.py:242-246, 322-345`); `agent._check_borrowed_placement` compares `config.image`/policy types (`agent.py:86-140`); docs.
5. **Tests to mirror.** `tests/v1/test_e2e.py` placement lists (`CHAT_PLACEMENTS`, `AGENTIC_PLACEMENTS`, `USER_RUNTIMES`, `ACP_RESUME_PLACEMENTS`, `TOOL_PLACEMENTS`, `TOOL_STATE_PLACEMENTS`, `SHARED_TOOL_PLACEMENTS`, `:133-268`), whose `pair()` helper applies `getattr(pytest.mark, <type>)` (`:17-19`) — so the mark **must be registered** in `deps/verifiers/pyproject.toml` `markers` (`:221-243`) or collection fails under `--strict-markers` (`:216`); the `tool_runtime` fixture (`conftest.py:65-72`, special-cases docker's `allow`), `run_v1_server` for the served path (`conftest.py:208-260`), and a provider fixture like `_configure_prime_runtimes` for labels/regions (`conftest.py:109-119`).
6. **Pitfalls.** `start()`/`prepare_execution()` run under no rollout deadline (§3.4) — bound your own provider waits (Prime: `max_attempts=180`, 60s policy poll). Cancellation between "provider scheduled the box" and "we hold its id" leaks paid sandboxes; exec APIs that report job output separately can drop output under load (Prime's retry loop, `prime.py:293-317`); image ENTRYPOINTs must be cleared or the keepalive never runs (`modal.py:219-222`, Docker `--entrypoint sleep`); `/tmp` may be a tiny tmpfs on VMs (install under `/var/tmp` or the workdir — `openclaw/harness.py:16`, `mcp/launch.py:238-246`); a live process you can't signal must fail the box closed (`modal.py:331-352`).

### 6.2 Recipe: add a new harness / agent scaffold

1. **Package & registration.** Create `v1/harnesses/<id_with_underscores>/{__init__.py, harness.py}`; `__init__.py` must export **exactly one** `Harness` subclass in `__all__` (config classes may also be exported). Resolution: id → `id.rsplit("/",1)[-1].split("@",1)[0].replace("-","_").lower()`; try `verifiers.v1.harnesses.<module>`, else a top-level importable module of that name (an installed third-party package) (`utils/loaders.py:70-126, 133-146`). A **taskset package** that exports a `Harness` makes it that taskset's default harness (`loaders.py:153-161`). Selection: `--env.agent.harness.id <id>` (role name for multi-agent envs). Don't put a second `Harness` subclass in `__all__` (e.g. re-exporting `ACPHarness` or a parent harness): ambiguity is a loud `ValueError` (`loaders.py:118-125`). Adding it to `v1/harnesses/__init__.py`'s re-exports is optional (the loader imports the subpackage directly). The module must be importable in every process that validates the config — `AgentConfig._resolve_harness` narrows `harness` via `harness_config_type` (imports it) wherever an env config is parsed, i.e. launcher and orchestrator too, not only the env-server worker (`configs/agent.py:53-66`).
2. **Config.** `class MyConfig(HarnessConfig)`; pin versions with `PinnedVersion`; `class MyHarness(Harness[MyConfig])` — the generic parameter is how `harness_config_type` narrows `--env.agent.harness.*` flags (`loaders.py:282-284`, `configs/agent.py:53-66`).
3. **Capability flags** (§3.7 table). Be conservative: `SUPPORTS_MCP=False` makes runs with tools fail fast; `NEEDS_CONTAINER=True` (default) keeps the program off the host.
4. **Pick a style.**
   - *Process style*: `setup(runtime)` installs (use `ensure_installed(runtime, directory, install, env, label, ready=…)` for concurrency-safe idempotent installs, or `runtime.prepare_uv_script(PEP723_source, env)` for Python programs); `launch(...)` builds env/args from `endpoint`, `secret`, `ctx.model`, `self.resolve_prompt(data)` (or `resolve_text_prompt`), `mcp_urls`, then `return await runtime.run_program(argv, env)`. Reuse `launch_chat_program` if you just need the bundled loop with extra args (`harnesses/utils/launch.py:35-86`). Implement `resume` or set `SUPPORTS_RESUME` if the program accepts a Messages prompt.
   - *ACP style*: subclass `ACPHarness`, call `super().setup(runtime)` at the end of `setup` (prepares the runner script), implement `prepare_acp(...) -> ACPConfig(env, command, prompt, mcp_urls?, system_prompt?, session_meta?, client_capabilities?)`; optionally `gate_tools` (+ `SUPPORTS_TOOL_INTERCEPTION=True`), `acp_turn_result`, `acp_close_result` to turn protocol metadata into trace metrics/stop conditions (`acp/__init__.py:64-153`, rlm/prime_agent as examples).
   - *In-process loop*: allowed as long as every model call goes to `endpoint`+`secret`; return a synthetic success `ProgramResult` (`harness.py:303-306`).
5. **Per-trace state & cleanup.** Key every config/home dir by `trace.id` and remove it in `cleanup(trace, runtime)` — required for borrowed/shared boxes (judge envs) (`claude_code/harness.py:120-125`).
6. **Tests.** Add rows to `CHAT_PLACEMENTS` / `AGENTIC_PLACEMENTS` / `ACP_RESUME_PLACEMENTS` with a mark named after the harness (`tests/v1/test_e2e.py:133-227`), registered in `deps/verifiers/pyproject.toml` `markers` (`:221-243`; `--strict-markers`); `harness` fixture resolves configs through `harness_config_type` (`conftest.py:79-82`).
7. **Trainability.** To be trainable under prime-rl the program must speak chat-completions with plain function tools (§3.7); Responses/Anthropic programs work for eval only. If the program can switch APIs, branch on `ctx.client.type == "train"` as hermes does (`hermes_agent/harness.py:88-89`).
8. **Pitfalls.** Install *everything* in `setup` — `launch` runs after the egress cut; `setup` shares the task's `timeout.setup` deadline. Disable telemetry, auto-update, background/utility model calls. Don't put secrets in env vars that model-driven tools inherit (pass via argv or config files, `bash/harness.py:95-112`). `run_program` must not be retried by your code (would fork the trace). Programs that treat an OpenAI base URL differently (Anthropic SDKs) need `/v1` stripped. Exit code ≠ 0 becomes `HarnessError` (or `SandboxError` if `alive()` fails) with the last 2 000 chars of stderr/stdout.

### 6.3 Other seams

- **New serving transport**: `EnvClient`/`EnvServer` speak plain ZMQ multipart; a replacement must preserve the "deltas then reply" ordering and `TraceSummary` check (`client.py`, `server.py`).
- **New tunnel**: subclass `Tunnel[Config]` (`bind_host`, `bind_port`, async-context `expose(port) -> url`) and add to `TunnelConfig` union + `make_tunnel` (`tunnel/base.py:24-47`, `tunnel/__init__.py:10-19`).
- **New taskset adapter**: a package exporting a `Taskset` subclass (+ optional `Env`, `Harness`, `Toolset`s) (`docs/v1/tasksets.md`, §3.9).

---

## 7. Gotchas & limitations

0. **A harness block without `id` silently becomes `bash`.** `AgentConfig._resolve_harness` narrows any non-`None` `harness` dict with default id `"bash"` (`configs/agent.py:59-65`, `loaders.py:47`). Setting only a knob, e.g. `env.agent.harness.tool_timeout = 900`, therefore **replaces the taskset's bundled default harness** with `bash`. The taskset default applies only while `harness` stays `None` (`configs/env.py` `agent_harnesses`). Verified by execution: `{"tool_timeout": 900}` → `BashHarnessConfig(id="bash")`. Always set `harness.id` together with any harness knob.
1. **Default runtime is remote Prime.** `AgentConfig.runtime = PrimeConfig()` (`configs/agent.py:30`): without Prime credentials `PrimeRuntime.__init__` fails (`prime.py:147`) and the interception mints prime tunnels. Set `--env.agent.runtime.type docker|subprocess` for local runs.
2. **The pool hides dead workers (confirmed).** `health` is answered by the broker from a pre-packed reply without consulting workers (`pool.py:43, 194-197`); the broker never polls `process.is_alive()` or its DEALER's peer, and there is no restart (`pool.py:18-20`; the process handle is used only in `_shutdown`, `:266-277`). Consequences, by case:
   - *Worker crashes at start* (bad config/import error in `load_environment`, `server.py:37`): it logs, re-raises and exits (`pool.py:367-371`); the broker lives on, health passes, and every `run` routed to it waits forever — no run timeout exists on the orchestrator side (`client.py:100-102, 208-217`; nothing in `envs.py`/`dispatcher.py` wraps `env.run`). All workers share the config, so all die; elastic spawns one more per ~115 stuck requests.
   - *Worker dies mid-run*: its in-flight runs never reply; its `active` count is frozen, so least-busy dispatch keeps handing it new requests whenever the frozen count is the minimum (`pool.py:225`); cancels for its runs are forwarded into the same dead queue (`:213-223`).
   - *Orchestrator recovery*: a hung **train** group is reclaimed only by the staleness sweep ($\text{step}-1-v_\text{start} >$ `max_off_policy_steps`, `dispatcher.py:399-437`) or an overload cut; **eval** and frozen-sourced train groups hang for good.
   - *Broker stall at 1000 (confirmed by execution)*: per-worker DEALERs keep libzmq's default `SNDHWM=1000` (`pool.py:130-131`); a pyzmq probe against an ipc peer that never bound, or bound then died, accepts exactly 1000 messages, then `await send_multipart` blocks. The broker awaits that send inline (`pool.py:221-223, 234-236`), so ≥1000 frames queued to one dead worker **freeze the whole broker** (no relays, no health). Unreachable under default elastic (≈115/worker); reachable with `static num_workers=N` / capped `max_workers` at in-flight ≳ 1000·N (e.g. `max_inflight=4096`, N≤4) — and transiently at startup while static workers still load the env.
   - In contrast `static num_workers=1` / `elastic max_workers=1` constructs the env in the main process, so a bad config kills the process and (locally) `monitor_process` fails the run; a crash mid-run still hangs its in-flight runs.
3. **Upscale-only, unbounded by default.** `max_workers=None` + `multiplex=128` spawns a new full worker process every ~115 in-flight episodes; workers are never reclaimed (`pool.py:145-160`). Each worker has its own interception pool, shared tool servers (e.g. a NeMo resource server per worker, `nemo_gym/taskset.py:122-158`), tunnels and memory. New workers get the backlog routed to them *before* they finish loading the env (`pool.py:146-150`).
4. **No per-episode timeout on the wire, and few inside.** The orchestrator's only escape is cancellation (stale/overload/superseded/shutdown); eval episodes are never staleness-dropped (`dispatcher.py:399-437`). Inside the worker, by default only the agent segment is bounded (4h); setup, finalize, scoring and `env.timeout.episode` are `None`, and provisioning/tunnel/tool-server/cut steps sit outside every deadline (§3.4) — a wedged `docker` CLI call or provider wait hangs the episode indefinitely. Set `env.timeout.episode` (wraps setup+run of the whole episode, `v1/env.py:268-289`) as the practical watchdog.
5. **Serial episode validation per source.** `EnvClient._decode_slots = BoundedSemaphore(1)` validates one `WireEpisode` at a time per env client (in a thread) (`client.py:60, 137-152`); long multi-trace episodes from a single high-throughput source can queue here.
6. **Unlimited ZMQ queues** (`SNDHWM=RCVHWM=0` on client and frontends) mean backpressure turns into memory growth rather than blocking — except the broker→worker DEALER hop (HWM 1000), where it turns into a broker-wide stall (§7.2).
7. **Per-rollout install cost.** Each agent run gets a *fresh* box, so container runtimes re-run `harness.setup` every rollout: Node download, `npm install`, `uv sync` of PEP 723 scripts (their cache lives inside the box, `base.py:142-144`), MCP server package install (`mcp/launch.py:237-276`). Only `subprocess` shares script envs across rollouts (`subprocess.py:81-89`). Bake the pinned paths (`/var/tmp/vf-node`, `/var/tmp/vf-claude-agent-acp-*`, …) into the image to amortize. Prime's first use of a new image triggers a ~10-minute VM image build (`prime.py:79-82, 216-228`).
8. **`CompactionFailed` is not caught** by the chat loop (`grep` finds no handler outside `compaction.py`), so when every checkpoint attempt fails the bundled program exits non-zero → `HarnessError`, despite the class docstring saying "the caller ends the run cleanly" (`compaction.py:71-72`, `core.py:338-358`).
9. **Stale docstrings about ports.** `OrchestratorConfig.env_sources` says the order "fixes each source's deterministic env-server port" and `envs.py:3-6` says addresses derive from the source's position — the code binds OS-assigned ports and publishes them via address files (`configs/orchestrator.py:789-803`, `env_server.py:46`).
10. **Harbor tasks carry host paths.** `HarborData.task_dir` points into the orchestrator host's `~/.cache/harbor`; the worker tars `environment/` and `tests/` from it (`taskset.py:196-201, 300-303`). An externally managed env server on another host needs the same cache at the same path.
11. **Network-policy coverage is uneven.** Docker/Podman: full host+SNI filtering through the in-process proxy; Modal: SNI domain allowlist on 443 only, no deny lists, domain-fronting caveat (`modal.py:238-243`); Prime: host allow/deny, deny rules may block framework routes (`docs/v1/harbor.md:104-106`); Apptainer/subprocess: none (tasks requiring a policy are rejected). Restricted tool runtimes must be colocated (`mcp/launch.py:406-419`).
12. **Prime can't expose ports** — any tool server in its own Prime sandbox fails; colocate it (`prime.py:340-344`).
13. **Docker needs host group access and, when restricted, a locally built `localhost/verifiers-network:1` image** (built on first use from `alpine:3.22` — requires registry access once) (`docker/__init__.py:104-121`). The proxy listener trick needs Python in the task image or pulls `python:3.11-alpine` (`:274-289`).
14. **Harness-program timeouts**: bundled bash tool 3600s/command, chat client 600s read timeout when bash is on (None otherwise), MCP discovery 60s (non-bash), MCP call `tool_timeout` 600s, ACP gate 120s, ACP shutdown 10s+5s+5s (`core.py:148-159, 450-474`, `configs/harness.py:52`, `runner.py:52-57`, `acp/__init__.py:336-348`). The agent timeout (default 4h, 24h cap on Modal) bounds all of it.
15. **Subprocess runtime strips only `*API_KEY*` env vars** from the inherited host environment; everything else (tokens named differently, `HF_TOKEN`, cloud creds) is visible to model-driven tools (`subprocess.py:26-29, 101`).
16. **Codex doesn't append system prompts natively** (`APPENDS_SYSTEM_PROMPT = False  # TODO`, `codex/harness.py:58`); `terminus_2` and `mini_swe_agent` support neither MCP nor disabled tools.
17. **OpenEnv `use_docker` and `from_env` run on the env-server host**, not in a sandbox (`openenv/taskset.py:32-34, 79-85`); TextArena mutates the process-global `random` (safe only because reset is await-free, `textarena/taskset.py:65-69`).

---

## 8. For a custom framework

**Keep (essential design).**
- *Orchestrator owns data, env workers are stateless executors.* Shipping the dumped task on every request removes per-worker dataset loads and makes workers horizontally interchangeable. Keep it, but make the task payload self-contained (no host paths — Harbor's `task_dir` is the counterexample).
- *Interception-as-contract.* Recording at a proxy you control, rather than parsing agent transcripts, is what lets 13 heterogeneous agent CLIs produce identical, token-exact traces. Any scaffold that can be pointed at an OpenAI/Anthropic base URL becomes *recordable*; in prime-rl it becomes *trainable* only if it speaks chat-completions, because token-exact recording is done by a renderer that re-tokenizes chat messages (§3.7). A custom framework that wants Responses/Anthropic scaffolds trainable must add per-dialect rendering. This is the single most valuable idea in this area.
- *Two-phase networking* (trusted online setup → policed execution with framework routes always allowed) and *cancellation-proof teardown* (shielded create-with-id-capture, shielded stop, sync atexit backstop). These are the hard-won correctness properties for paid sandboxes at scale.
- *Delta streaming with count-checked assembly*: cheap live visibility and a correctness check for free.

**Simplify.**
- The pool: replace upscale-only elastic spawning with a fixed worker count sized from the orchestrator's `max_inflight`, and add worker liveness (heartbeats or broker-side process polling) plus fail-fast on worker death — today a crashed worker is a silent hang (§7.2). Health should mean "a worker can run an episode". Never `await` a bounded send inside a single-threaded broker loop; use non-blocking sends and treat `EAGAIN` as worker death.
- Put a timeout (or at least a watchdog) on the wire for `run` requests, keyed to the agent timeout + slack.
- One transport for host↔sandbox commands. prime-rl has four exec mechanisms (CLI exec, Modal SDK, Prime background jobs + live process, subprocess). A custom framework on a single cluster (e.g. Apptainer/Docker on SLURM) can pick one and drop tunnels entirely by colocating sandboxes with env workers on the host network.
- Pre-baked images per harness instead of per-rollout installs; the `ensure_installed` lock machinery then becomes a sanity check.

**Replace / reconsider.**
- The Docker egress proxy (fd passing into the container netns + in-process h11 proxy + iptables helper container) is clever but heavy; if you control the cluster, use network namespaces/CNI policies or a sidecar proxy.
- The ACP double hop (host → PEP 723 `runner.py` → agent process, length-prefixed JSON inside, JSON-RPC ACP inside that) exists to normalize multi-turn session state across agent CLIs. If you only need single-segment rollouts, process-style harnesses are far simpler; add ACP only for user-sim/multi-turn environments.
- Serial per-client episode validation and unbounded ZMQ HWMs are throughput/memory hazards at high concurrency; validate in a pool and set HWMs.

**Coupling points to watch.** `Trace` field set ↔ `delta.py` field lists (unit-tested); `TaskData` serializability ↔ server refusal; `TaskResources` field names ↔ runtime config field names; `is_local` ↔ tunnel requirement ↔ interception bind; `SERVICE_PORT=8000` ↔ Docker/Modal publish; `VF_RUN_ID`/`PRL_RUN_NAME` ↔ limiter scope/labels.

---

## 9. Open questions

1. Runtime confirmation of §7.2 (static analysis + a pyzmq HWM probe agree; no live env-server test was run): kill one worker mid-run and observe (a) its runs hang, (b) the broker keeps routing to it, (c) with a static pool and >1000 frames queued to it, the broker freezes.
2. Config-dir agreement (A to confirm): the launcher writes absolute `address_file` paths into each `EnvServerConfig` (`pathing.py:230-246`); the orchestrator recomputes them from `get_config_dir(output_dir)` = `$PRL_ATTEMPT_CONFIG_DIR` (exported by every sbatch template) else `<output_dir>/configs/latest/resolved` (the symlink `create_attempt_dirs` just repointed) (`pathing.py:38-72, 158-168`). Local agreement therefore rests on `orchestrator.output_dir == run_dir` and no concurrent launch in the same run dir.
3. Apptainer `--cpus/--memory` generally need cgroups v2 delegation for unprivileged users; whether the runtime works with those flags on Mila/DRAC nodes is not verifiable statically.
4. `codex` / `openclaw` model API (chat-completions vs Responses) is decided inside their binaries; resolve by capturing one request's route on the interception (`/v1/chat/completions` vs `/v1/responses`).
