# Cross-process contract catalog — prime-rl @ b944873

> Scope: everything that crosses a process or repo boundary in prime-rl — addresses, sockets, HTTP routes, wire schemas, files, env vars, counters — organised by plane, with a desync detector at the end.
> Pins: prime-rl `b944873`, verifiers `69cc0f9`, renderers `6b8da3f`, prime-envs `b677502`, vLLM wheels `0.29.0`.
> Sources: section docs `sections/01…11`, `_brief/LEDGER.md`, `00-mental-model.md`, plus direct reads of the checkout for every row marked with a code cite.
> **vLLM caveat:** every claim about vLLM *internals* (pause `keep` = freeze, prefix cache survives updates, DP utility fan-out, `device.index` under internal DP, `prompt_logprobs` skipping prefix-cache reads, `StatelessProcessGroup` binding) was read in vLLM **0.24** source; the pin is **0.29.0**. Those rows carry `[vLLM 0.24]`.

## How to use this doc

- **Find the plane** (§1 processes/discovery, §2 env control, §3 rollout data, §4 admin, §5 batches, §6 weights, §7 run dir, §8 step/version accounting, §9 cross-repo). Each row has a **stable ID** (`E3`, `W7`, …); §10 maps every ID to "what breaks and how it shows up".
- **Implementing a compatible component?** Read the plane's schema table top to bottom; every field is listed in wire order with the producer and consumer code.
- **Debugging a hang or silent corruption?** Go to §10 first, then jump to the ID.
- `→ 06 §3.8` = section doc and heading. Code cites are `path:line` with these prefixes:
  `src/` = `src/prime_rl/`, `cfg/` = `packages/prime-rl-configs/src/prime_rl/configs/`, `vf/` = `deps/verifiers/verifiers/v1/`, `rnd/` = `deps/renderers/renderers/`, `tpl/` = `src/prime_rl/templates/`.
- "Nothing checks" means no runtime or config-time validator exists at the pin; the contract holds only by construction.

---

## 1. Process inventory and discovery

### 1.1 Processes

| ID | Process | Spawned by | Count / placement | Binds | Finds peers via | Cite |
|---|---|---|---|---|---|---|
| P1 | `rl` launcher (`PRL::Launcher`) | user | 1, submit host (SLURM single-node: re-run inside the allocation) | nothing (picks a free torchrun rdzv port, `get_free_port`, TOCTOU) | writes `configs/attempt_N/resolved/*.json`; supervises children (monitor thread each, 1 s poll; teardown = SIGTERM process tree, 60 s, SIGKILL) | `src/entrypoints/rl.py:116-423, 647-705`; `src/utils/process.py:64-105` → 01 §2.1-2.2 |
| P2 | Router (`vllm-router` fork; or llm-d EPP+Envoy on SLURM) | local: the `inference` process (`start_router`); SLURM: inference node 0 | 1 per run | `server.host or 0.0.0.0` : `server.port` (8000); Prometheus `server.port+21000` | `--worker-urls http://<host>:<backend_port>` (local) or per-rank URLs (SLURM); session affinity `--request-id-headers x-session-id`, policy `sticky_least_loaded` | `src/entrypoints/inference.py:165-191, 216-228`; `tpl/_launch_router.sh.j2:71-85` → 06 §3.12 |
| P3 | vLLM engine (API server(s) + EngineCore(s) + GPU workers with `worker_extension_cls`) | `inference` entrypoint (in-process `server(config)`) | local: 1 server, `dp` engines behind `backend_port` (8100); SLURM multi-node: one `uv run inference` per DP rank on `BACKEND_PORT+k`, `--server.host 0.0.0.0` | `backend_port(+k)` | nothing; it is found (router worker URLs, orchestrator `admin_base_url`) | `src/inference/vllm/server.py:59-63, 212-241`; `tpl/_launch_rank.sh.j2:16-39` → 06 §2 |
| P4 | Env server = broker `EnvServerPool` (ROUTER) + spawned worker `EnvServer`s (ROUTER on `ipc:///tmp/vf-pool-XXXX/<i>`) | launcher, one per `(split, source)` with `serve.address = None`, started before the orchestrator (no readiness wait) | orchestrator's host (single-node box; SLURM: `ORCH_PROCID` node) | `tcp://127.0.0.1:0` (OS port) unless `serve.address` | publishes its address to `envs/<split>/<name>.address`; reaches inference with the `ClientConfig` shipped in each `RunRequest` | `src/entrypoints/env_server.py:24-51`; `vf/serve/pool.py:334-366` → 08 §2 |
| P4a | Interception server (aiohttp) + renderer pool, inside each env-server **worker** | `Env.serving()` in the worker | 1 elastic pool per worker (32 rollouts/server), renderers process-global (256 rollouts/renderer) | `127.0.0.1:0`, or tunnel `bind_host/bind_port` | harness gets `runtime.host_url(base_url+"/v1")` + per-rollout secret | `vf/interception/server.py:423-459`; `vf/rollout.py:247-261` → 07 §2 |
| P5 | Runtime + harness program (+ MCP tool servers) | env-server worker, per rollout, named by trace id | subprocess / docker / apptainer on the orchestrator host; Prime/Modal remote (tunnelled) | tool servers: `MCP_PORT=8000` or `MCP_PORT_FILE` | interception endpoint + secret; `/state`, `/task` with the state secret | `vf/rollout.py:177-356` → 08 §3.4 |
| P6 | Orchestrator (1 asyncio process) | launcher | 1; CPU; single-node box or trainer node 0 (`orchestrator_on_inference` → last inference node) | ZMQ PUB `rollout_transport.host:port` (5555) + PULL `port+1` (5556) | router `model.client.base_url`; engines `admin_base_url`; env servers via address files; trainer via `broadcasts/step_N/` markers | `src/orchestrator/orchestrator.py:184-401`; `src/transports/batch/zmq.py:27-34` → 03 §2 |
| P7 | Trainer ranks (torchrun, FSDP2 SPMD) | launcher `torchrun --role=trainer … -m prime_rl.trainer.rl.train @ trainer.json` | `num_train_gpus` (single node) / all GPUs of trainer nodes | rank 0: NCCL TCPStore `weight_broadcast.host:port` (29501); torchrun rdzv; optional metrics server `0.0.0.0:8000` | ZMQ SUB+PUSH to `rollout_transport.host`; markers; NCCL/NIXL peers | `src/entrypoints/rl.py:330-376`; `src/transports/weights/nccl.py:131-137` → 05 §2 |
| P8 | ModelExpress server + redis (NIXL only) | SLURM templates (trainer node 0, or single-node SLURM); **not launched locally** | 1 | gRPC 8001; redis 6379 (6380 if MX port is 6379) | every NIXL participant publishes/discovers there | `tpl/multi_node_rl.sbatch.j2:268-313` → 06 §3.9, 01 §3.4.7 |
| P9 | SFT online-eval (`python -m prime_rl.eval.online`) | `sft` launcher | 1 | — | alternative **weight consumer**: acks broadcasts (§6), uses the same `WeightReceiver` | `src/eval/online.py:44-197` → 11 §3.15 |
| P10 | Dashboard daemon | launchers (`ensure_dashboard`, TTY only) | 1 per user/host | `127.0.0.1:7788` (+up to 99) | reads run dirs registered in `~/.cache/prime-rl/dashboard/dirs.json` | `src/entrypoints/dashboard.py:31-114` → 11 §3.17 |
| P11 | Optional infra: llm-d EPP (gRPC 9002 / health 9013 / metrics 9090) + Envoy, llm-d `pd-sidecar` (8300+k), mooncake master (50051, HTTP 8080) + client (50052), Dynamo frontend | SLURM templates / user | — | as listed | file-discovered endpoints (llm-d), `MOONCAKE_CONFIG_PATH`, Dynamo discovery (`GET <discovery_url>/v1/rl/workers`) | `tpl/_launch_router.sh.j2:11-70`; `tpl/_mooncake_store.sh.j2:11-42`; `src/inference/dynamo.py:26-151` → 06 §3.10, §3.12 |

Start order (local): inference → env servers → orchestrator → torchrun (`src/entrypoints/rl.py:202-376`). Nobody waits in the launcher: the orchestrator waits for inference (`/health`, 3600 s), env servers (address file 600 s + health 600 s) and the trainer's startup broadcast (1200 s); trainer and orchestrator rendezvous over READY + the marker handshake (→ 01 §2.1).

### 1.2 Discovery mechanisms

| ID | Mechanism | Who writes → who reads | What it carries | Cite |
|---|---|---|---|---|
| D1 | **Resolved JSON per process** — `<cmd> @ configs/attempt_N/resolved/<x>.json`; each child re-runs `cli()` on it | launcher → every child | every static host/port/URL, `inference_world_size`, `num_train_workers`, `pad_to_multiple_of`, transports | `src/entrypoints/rl.py:89-113`; `src/utils/config.py:15-22` → 02 §3.9 |
| D2 | **Env-server address file** `…/resolved/envs/<split>/<name>.address` (atomic `tmp`+`replace`) — the only dynamic discovery | env server → orchestrator (`wait_for_address`, 0.5 s poll, 600 s) | `tcp://127.0.0.1:<port>` | `src/entrypoints/env_server.py:24-31`; `src/orchestrator/envs.py:50-62`; `src/utils/pathing.py:222-227` → 01 §3.7 |
| D3 | **Config dir resolution** `get_config_dir(output_dir)` = `$PRL_ATTEMPT_CONFIG_DIR`, else `<output_dir>/configs/latest/resolved` (legacy `configs/resolved` if `latest` absent) | orchestrator (to locate D2 files) | attempt pinning | `src/utils/pathing.py:158-169` |
| D4 | **Multi-node CLI overrides** computed in bash from `scontrol show hostnames` | sbatch → orchestrator / trainer | `--model.client.base-url $INFER_URLS`, `--model.client.admin-base-url $ADMIN_URLS`, `--weight_broadcast.host $MASTER_ADDR` (NCCL) / `$WEIGHT_BROADCAST_HOST` (NIXL), `--rollout_transport.host $ORCH_ADDR` (both sides) | `tpl/multi_node_rl.sbatch.j2:85-161, 518-519, 534-535, 563-565` → 01 §3.4 |
| D5 | **Broadcast markers** `broadcasts/step_N/.sender_ready` etc. | trainer ↔ weight consumer | version offers and acks (§6.1) | `src/transports/weights/base.py:23-26` |
| D6 | **ModelExpress** (NIXL) | all NIXL peers | `WorkerMetadata{worker_rank, nixl_metadata}` per role | `src/transports/weights/nixl/model_express.py:25-91` → 06 §3.9 |
| D7 | **Single-node auto URLs**: `base_url` default `http://localhost:8000/v1`; `admin_base_url = [http://<host>:<backend_port>/v1]` | `RLConfig` validators → orchestrator JSON | router vs. engine split | `cfg/shared.py:177`; `cfg/rl.py:846-854` → 02 §3.6 #21 |
| D8 | **GPU placement** by `CUDA_VISIBLE_DEVICES` (ints only): inference `[0,n_infer)`, trainer next `n_train`; orchestrator/env servers get **no** override | launcher | device partition | `src/entrypoints/rl.py:144-162`; `src/utils/process.py:36-44` → 01 §3.2 |

### 1.3 Ports

| ID | Port (default) | Owner (binds) | Connects | Config / derivation | Cite |
|---|---|---|---|---|---|
| N1 | 8000 | router | orchestrator data/eval clients, env-server TrainClients, `finish_session` | `inference.server.port` | `cfg/inference.py:20` |
| N2 | 8100 (+k per DP rank on SLURM) | vLLM API server | router, AdminPlane, metrics collector | `backend_port = server.port+100` when a router is set | `cfg/inference.py:534-538` |
| N3 | 29000 | router Prometheus | Prometheus (not scraped by prime-rl) | `server.port + 21000` | `src/entrypoints/inference.py:188-189` |
| N4 | 5555 / 5556 | orchestrator ZMQ PUB / READY PULL | every trainer rank (SUB / PUSH) | `rollout_transport.{host,port}`; READY = `port+1` | `src/transports/batch/zmq.py:27-34`; `cfg/shared.py:242-252` |
| N5 | 29501 | trainer rank 0 (`StatelessProcessGroup` TCPStore) | every inference GPU worker | `weight_broadcast.port`; trainer host `localhost` (single) / `0.0.0.0` (multi) | `src/transports/weights/nccl.py:131-137`; `cfg/rl.py:788-789` |
| N6 | 8001 (+ redis 6379/6380) | ModelExpress | NIXL trainer ranks, workers, orchestrator | `weight_broadcast.port` (NIXL) | → 06 §4.5 |
| N7 | 29500 / free port | torchrun rendezvous | trainer ranks | local `localhost:<get_free_port()>`; multi-node `$MASTER_ADDR:29500` | `src/entrypoints/rl.py:330-345`; `tpl/multi_node_rl.sbatch.j2:137-138` |
| N8 | OS-assigned | env-server broker | orchestrator `EnvClient` | D2 | `src/entrypoints/env_server.py:46` |
| N9 | OS-assigned (127.0.0.1) or tunnel bind | interception server | harness programs, MCP servers (`/state`, `/task`) | `runtime.host_url` / tunnel URL | `vf/interception/server.py:443-459` |
| N10 | 8000 (opt-in) | trainer metrics/health server | Prometheus / k8s probes | `trainer.metrics_server.port` — **clashes with N1 on one host** (`OSError` at trainer start) | `cfg/shared.py:226-231` → 11 §7.12 |
| N11 | 13345 | vLLM external-LB DP RPC | DP group peers | `vllm.data_parallel_rpc_port` | → 06 §4.5 |
| N12 | 8100+k / 8200+k, 5600+k, 8300+k | P/D prefill / decode engines, NIXL KV side channel, llm-d sidecar | router / EPP / peer ranks | template loops | `tpl/multi_node_rl.sbatch.j2:356-457` |
| N13 | 50051 / 8080 / 50052 | mooncake master / metadata / client | vLLM mooncake connector | template | `tpl/_mooncake_store.sh.j2:11-42` |
| N14 | 7788 | dashboard | browser | `--port` | → 11 §3.17 |

Static ports are per-host singletons: two local runs on one host collide on N4 and N5 (plain `bind` → `EADDRINUSE`), not only N1/N2 (→ 01 §7.4).

### 1.4 Environment variables with cross-process meaning

| ID | Var | Set by | Read by | Effect | Cite |
|---|---|---|---|---|---|
| V1 | `PRL_RUN_ID` | `rl`/`sft`/`eval` launcher (`setdefault` uuid); sbatch fallback | orchestrator (`run.id`, sandbox labels), env server (→ `VF_RUN_ID`), W&B (`WANDB_RUN_ID`), prime monitor | run identity | `src/entrypoints/rl.py:652-654`; `src/orchestrator/orchestrator.py:219-224` |
| V2 | `PRL_RUN_NAME` | launcher | orchestrator, env server (sandbox base labels) | display name / labels | `src/entrypoints/env_server.py:36-37` |
| V3 | `PRL_ATTEMPT_CONFIG_DIR`, `PRL_ATTEMPT_LOG_DIR` | SLURM templates, `eval`/`sft` | `prepare_attempt_dirs`, `get_config_dir` | pin a child to its attempt (address files, logs); must be set together | `src/utils/pathing.py:60-72, 158-169` |
| V4 | `VF_RUN_ID` | env server (`setdefault` from `PRL_RUN_ID`) | verifiers limiters (`~/.cache/verifiers/limiter/<name>-<scope>.bucket`) | run-scoped sandbox/tunnel creation rate | `src/entrypoints/env_server.py:60` → 08 §3.6 |
| V5 | `CUDA_VISIBLE_DEVICES` | user (input) → launcher (output, protected) | `get_physical_gpu_ids`; vLLM; torch | GPU partition; ints only | `src/utils/process.py:36-44` |
| V6 | `RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `LOCAL_WORLD_SIZE` | torchrun | trainer `World`; monitors (`RANK`/`DP_RANK` gate rank-0-only registration) | SPMD topology; a leaked `RANK≠0` silently disables orchestrator monitors | `src/trainer/world.py:7-13`; `src/monitors/__init__.py:67-69` |
| V7 | `MASTER_ADDR`, `MASTER_PORT` | multi-node sbatch | torchrun; orchestrator `--weight_broadcast.host` (NCCL) | trainer node 0 | `tpl/multi_node_rl.sbatch.j2:137-138, 563` |
| V8 | `INFER_URLS`, `ADMIN_URLS`, `ROUTER_ARGS` | multi-node sbatch | orchestrator CLI; router CLI | data-plane URL; per-engine admin URLs **in rank order** | `tpl/multi_node_rl.sbatch.j2:94-127, 534-535` |
| V9 | `ORCH_ADDR`, `ORCH_PROCID` | multi-node sbatch | orchestrator + trainer `--rollout_transport.host` | ZMQ host (a hostname; libzmq binds the one interface it resolves to) | `tpl/multi_node_rl.sbatch.j2:158-161, 519, 565` |
| V10 | `WEIGHT_BROADCAST_HOST`, `MODEL_EXPRESS_PORT`, `MODEL_EXPRESS_REDIS_PORT` | multi-node sbatch | trainer + orchestrator (NIXL) | ModelExpress endpoint | `tpl/multi_node_rl.sbatch.j2:141-151, 518, 564` |
| V11 | `VLLM_API_KEY` (= `client.api_key_var`) | user | AdminPlane (`Authorization: Bearer`), env-server worker `TrainClient` | engine auth; the **name** crosses the wire, the value is read from each process's own env | `cfg/shared.py:180`; `src/orchestrator/clients.py:294-296`; `vf/configs/client.py:93-104` |
| V12 | `PRIME_API_KEY`, `PRIME_TEAM_ID`, `PRIME_INFERENCE_URL` | user / `prime login` | verifiers default clients (judges, Prime runtime, tunnels), prime monitor | default runtime is a Prime sandbox; missing key → `"EMPTY"` | `vf/configs/client.py:23-104` → 10 §4.5 |
| V13 | `WANDB_SHARED_MODE=1`, `WANDB_RUN_ID=$PRL_RUN_ID`, `WANDB_SHARED_LABEL=orchestrator\|trainer`, `WANDB_SHARED_PRIMARY/FINISHER`, `WANDB_ARGS`, `WANDB_PROGRAM` | launcher (first four protected) | W&B monitor | trainer + orchestrator write one shared W&B run (online only) | `src/entrypoints/rl.py:171-174, 302-312, 351-363` → 11 §3.7 |
| V14 | `DEFAULT_COMMON_ENV_VARS` (`CUDA_DEVICE_ORDER=PCI_BUS_ID`, `PYTHONUNBUFFERED=1`, `OMP_NUM_THREADS=1`, `GIT_LFS_SKIP_SMUDGE=1`), trainer `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, inference `VLLM_WORKER_MULTIPROC_METHOD=spawn`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`, `VLLM_ENGINE_READY_TIMEOUT_S=4200`, `UCX_TLS=all` | launcher per child | children | inference **re-applies its defaults in-process**, so RL-level `[env_vars]` for these keys are reset there — use `[inference.env_vars]` | `src/utils/process.py:17-33`; `src/entrypoints/inference.py:210` → 01 §3.2 |
| V15 | `PROTECTED_ENV_VARS` | — | `reject_protected_env_vars` | `CUDA_VISIBLE_DEVICES`, `PRL_RUN_ID`, `PRL_RUN_NAME`, `WANDB_RUN_ID`, `WANDB_SHARED_{MODE,LABEL,PRIMARY,FINISHER}` cannot be set via `env_vars` | `cfg/shared.py:13-36` |
| V16 | `PRIME_NO_MOE_LORA`, `PRIME_DP_COORDINATOR_STARTUP_TIMEOUT` (300), `VLLM_USE_V2_MODEL_RUNNER`, `VLLM_ENFORCE_STRICT_TOOL_CALLING=0`, `VLLM_PLUGINS` | inference process | vLLM API server → workers (cross the spawn boundary) | patch selection, sampling-mask capture needs V2 runner | `src/inference/server.py:7-49`; `src/inference/vllm/server.py:218-221` → 06 §3.1 |
| V17 | `NCCL_P2P_DISABLE`, `NCCL_SHM_DISABLE` | `disable_nccl_p2p_if_unavailable` (trainer master + inference workers, if unset and no NVLink) | NCCL | broadcast group transport | `src/utils/nccl.py:8-42` |
| V18 | `UCX_*` (`UCX_NET_DEVICES`, `UCX_TLS=rc_x,…` setdefault) | NIXL agents | UCX | the inference default `UCX_TLS=all` (V14) pre-empts the NIXL setdefault on vLLM workers | `src/transports/weights/nixl/agent.py:151-210` → 06 §7 |
| V19 | `VLLM_NIXL_SIDE_CHANNEL_HOST/PORT`, `GLOO_SOCKET_IFNAME`, `MOONCAKE_CONFIG_PATH`, `PYTHONHASHSEED=0`, `VLLM_RPC_BASE_PATH`, `NCCL_IB_HCA` | SLURM templates | vLLM ranks | P/D KV transfer, mooncake, RPC socket isolation | `tpl/multi_node_rl.sbatch.j2:248-250, 356-457`; `tpl/_launch_rank.sh.j2:18-23` |
| V20 | `PRL_OUTPUT_DIR`, `PRIME_LOG_LEVEL`, `PRIME_VF_LOG_LEVEL`, `NEVER_CLEAN` | user | config `default_factory` (evaluated **in the launcher**, frozen into JSON) | output root, log levels, disable `--clean` | `cfg/shared.py:199-204`; `src/utils/config.py:10-12` → 02 §2 |
| V21 | In-box: `VF_CONFIG`, `VF_STATE_URL`, `VF_STATE_SECRET`, `MCP_HOST`, `MCP_PORT`, `MCP_PORT_FILE`, `HTTP(S)_PROXY`, `NO_PROXY` | env-server worker → runtime | MCP tool servers, harness | tool-server wiring; docker egress proxy (restricted policy only) | `vf/mcp/launch.py:317-334`; `vf/runtimes/docker/__init__.py:409-444` → 08 §4.2 |

---

## 2. Env control plane (orchestrator ↔ env server)

### 2.1 Sockets

| ID | Hop | Socket (side) | Options | Frames | Cite |
|---|---|---|---|---|---|
| E1 | orchestrator → env server | `EnvClient`: `DEALER` connect, **one per env/source**, all episodes multiplexed | `LINGER=0`, `SNDHWM=0`, `RCVHWM=0` (unbounded) | send `[request_id, method, payload]` | `vf/serve/client.py:44-60, 109-112` |
| E2 | env server front | broker `ROUTER` (pool) or worker `ROUTER` (single) bound at the address | `ROUTER_MANDATORY=1`, `SNDHWM=RCVHWM=0`, `LINGER=0` | receives `[client_id, request_id, method, payload]`; sends `[client_id, request_id, kind, data]` | `vf/serve/server.py:55-64`; `vf/serve/pool.py:185-238` |
| E3 | broker → worker | one `DEALER` per worker → worker `ROUTER` on `ipc:///tmp/vf-pool-XXXX/<i>` | `LINGER=0`, **default `SNDHWM=1000`** (the one bounded hop) | `[request_id, method, payload]` | `vf/serve/pool.py:113-141` → 08 §4.1 |
| E4 | env server → orchestrator | DEALER view | — | `[request_id, kind ∈ {b"delta", b"reply"}, data]`; anything but 3 frames is dropped with a warning | `vf/serve/client.py:66-91` |

### 2.2 Methods

`method` is a `ClassVar`; it travels only as the route frame, never in the payload. Payloads are `msgpack.packb(request.model_dump(mode="json"), use_bin_type=True)` (`vf/serve/client.py:109`).

| ID | Method | Request | Response | Who answers | Cite |
|---|---|---|---|---|---|
| E5 | `health` | `{}` | `HealthResponse{success=True, error=None}` as a `reply` frame | **broker inline** from a pre-packed reply — never touches a worker, so it passes even when every worker is dead | `vf/serve/pool.py:194-197`; `vf/serve/types.py:21-26` |
| E6 | `run` | `RunRequest` (§2.3) | 0..n `delta` frames, then one `reply` = `RunResponse{success, error, head, traces}` | least-busy worker (`min(workers, key=active)`) | `vf/serve/types.py:42-63`; `vf/serve/server.py:97-123` |
| E7 | `cancel` | `CancelRequest{request_id: <run's wire id>}` sent fire-and-forget with a **fresh** request id when the orchestrator cancels an episode task | `CancelResponse{success, error, cancelled: bool}`; the cancelled run still replies `success=False, error="Cancelled: rollout aborted"` | broker routes to the worker holding the target, else answers `cancelled=False` inline | `vf/serve/client.py:118-129, 154-163`; `vf/serve/types.py:29-39` → 08 §3.2 |
| E8 | failure | — | `BaseResponse{success=False, error="<ExcType>: <msg>"}` → client raises `RuntimeError(error)`; unknown method → `"unknown method '<m>'"` | any | `vf/serve/client.py:132-135` → 08 §4.1 |

### 2.3 `RunRequest` (field-level, as prime-rl fills it)

| ID | Field | Type | prime-rl value (train) | Consumer | Cite |
|---|---|---|---|---|---|
| E9 | `task_data` | `dict` | `group.task.data.model_dump(mode="json")` — **full dump, no `exclude_none`**; same dict for all `group_size` requests of a group | worker `_build_task`: `data_cls.model_validate(task_data)` → `task_cls(data, env.config.taskset.task)` (the **server's own** task config) | `src/orchestrator/dispatcher.py:599`; `vf/serve/server.py:80-87` |
| E10 | `client` | `ClientConfig` union, discriminator `type` | `TrainClientConfig{type:"train", base_url, api_key_var, headers, renderer: RendererConfig\|null, renderer_model_name: <orchestrator model.name>, multiplex: 256}`; eval: `EvalClientConfig{type:"eval", base_url, api_key_var, headers}` | the worker builds its own HTTP client per distinct config; **the API key never crosses the wire** — only its env-var name | `src/orchestrator/clients.py:259-280`; `vf/configs/client.py:29-90` |
| E11 | `model` | `str` | live policy name (`config.model.name`; LoRA adapters are registered under this same name) or the frozen source's name | imposed on every harness request by `dialect.apply_overrides` | `src/orchestrator/dispatcher.py:547-553`; `vf/interception/server.py:596` |
| E12 | `sampling` | `SamplingConfig` (`extra="allow"`) | `{temperature, top_p, logprobs: True, [max_completion_tokens → alias max_tokens], extra_body: {top_k (−1 default, or configured/512), min_p: 0.0, return_token_ids: True, …user keys, cache_salt: str(policy_version_at_start)}}`; frozen sources drop `logprobs` and get no salt | worker `ModelContext.sampling`; the TrainClient flattens it (§3.4) | `cfg/orchestrator.py:91-108, 758-769`; `src/orchestrator/envs.py:121-125`; `vf/types.py:263-284` |

### 2.4 Streaming and reply

| ID | Item | Contract | Cite |
|---|---|---|---|
| E13 | Delta field classes | `HEADER_FIELDS = (version, id, verifiers, task)` sent once on `open`; `LIST_FIELDS = (nodes, calls, errors, extra_usage, request_rewrites, response_rewrites)` append-only; `SCALAR_FIELDS = (agent, tools, mm_token_type_id_map, rewards, metrics, info, root_reply, is_completed, ok, stop_condition, timing, num_input_tokens, num_output_tokens, num_total_tokens)` re-sent whole on change. A unit test asserts they cover every serialized `Trace` field | `vf/serve/delta.py:44-73`; `tests/v1/test_serve_delta.py:74-82` (verifiers) |
| E14 | Delta dict keys | `trace`, `discard`, `open`, `links`, `routing_repairs`, `nodes`, `calls`, `errors`, `extra_usage`, `request_rewrites`, `response_rewrites`, `set`, `pending`. In-place mutations of sent nodes allowed only for `semantic_parents` links and a **repair of the last routed-experts row** (may widen dtype) | `vf/serve/delta.py:167-246` → 08 §3.3 |
| E15 | Encoding | msgpack with `default=msgpack_encoder` (ndarray/tensor → `{"__torch_tensor__", dtype, shape, data}`); deltas/replies dump `mode="python", serialize_as_any=True`; unpack `strict_map_key=False` (int node-index keys) | `vf/serve/delta.py:76-88`; `vf/serve/encoding.py` |
| E16 | Ordering | `DeltaStreamer.__aexit__` flushes before the reply is built ⇒ **all deltas precede the reply**; flushes serialized by a lock; cursor advances only after a successful send | `vf/serve/delta.py:124-165` |
| E17 | Reply | `RunResponse{head: dump(episode, exclude={"traces"}), traces: [TraceSummary{id, nodes, calls}]}` | `vf/serve/server.py:115-123`; `vf/serve/delta.py:324-329` |
| E18 | **Count assertion** | `EpisodeAssembly.finish(head, summaries)`: every summary's trace must exist and `len(nodes)`/`len(calls)` must equal the server's counts, else `RuntimeError` ("a delta was lost") | `vf/serve/delta.py:302-321` |
| E19 | Validation | assembled record → `WireEpisode.model_validate` in a thread, **one at a time per `EnvClient`** (`BoundedSemaphore(1)`) | `vf/serve/client.py:137-152` |

### 2.5 `WireEpisode` and what prime-rl does with it

`WireEpisode = Episode[WireTaskData, State, WireAgentConfig]` (`vf/episode.py:166`): `id, env{id,name}, task{type, data, key, hash}, group, run, ok, errors, traces[Trace]` (`vf/episode.py:87-100`). `WireTaskData` is `extra="allow"`; agent configs parse loose, **but the runtime union is strict**: an unknown `runtime.type` fails validation of every episode (→ 08 §6.1 step 3).

| ID | Step (orchestrator) | Contract | Cite |
|---|---|---|---|
| E20 | Episode-level failure | if `not episode.ok`, every still-ok trace gets `episode.last_error` (or `EpisodeFailed`) and `ok=False` — partial episodes never train | `src/orchestrator/envs.py:146-152` |
| E21 | **Task provenance** | `(episode.task.key, episode.task.hash) == (task.key, task.hash)` of the dispatched task; `hash = task_key(data.model_dump(mode="json", exclude_none=True))`, `key` defaults to `hash`. Mismatch → `ValueError` escapes the dispatcher task → orchestrator exits 1 | `src/orchestrator/dispatcher.py:116-120, 712`; `vf/task.py:150-164` → 10 §3.5, 03 §7.24 |
| E22 | Empty checks | no traces + ok → `EmptyEpisode`; ok trace with `num_turns == 0` → `EmptyTrajectory` | `src/orchestrator/dispatcher.py:668-679` |
| E23 | Stamping | `env.name`, `group = GroupInfo(id)`, `run = TrainRunInfo(id=$PRL_RUN_ID, name=$PRL_RUN_NAME, work=Train\|EvalWorkInfo(step=group.step, policy=PolicySpan(start=group.policy_version_at_start, end=policy.version) \| None))` (`None` for frozen-sourced train) | `src/orchestrator/dispatcher.py:705-729`; `vf/episode.py:31-61` |
| E24 | Fields read downstream | `Trace.{branches, nodes, reward, num_*_tokens, num_turns, has_error, agent.trainable, agent.name, id, info}`, `Branch.{nodes, trainable, token_ids, logprobs, reference_logprobs, advantages, multi_modal_data, mm_token_type_ids, routed_experts, sampling_mask, index}`; algorithms **write** `MessageNode.{advantages, reference_logprobs, loss_weights}` | `src/orchestrator/trajectories.py:91-180` → 07 §4.1 |

### 2.6 Timeouts and HWMs

| ID | Item | Value | Cite |
|---|---|---|---|
| E25 | `run` request | **no timeout** ("rollouts run untimed"); only cancellation ends it | `vf/serve/client.py:100-102` |
| E26 | address file wait / health wait | 600 s each (`ENV_SERVER_STARTUP_TIMEOUT`); health poll 1 s interval, ≤2 s per probe | `src/orchestrator/envs.py:42, 99-104`; `vf/serve/client.py:165-184` |
| E27 | Worker-side defaults | only the agent segment is bounded (4 h); setup, finalize, scoring, `env.timeout.episode` = none; provisioning/tunnel/cut = no deadline | → 08 §3.4, §7.4 |
| E28 | HWMs | client/front `0` (unbounded ⇒ memory growth), broker→worker `1000` (≥1000 frames to a dead worker freeze the whole broker) | → 08 §7.2 |
| E29 | Supervision | broker answers health itself; no worker restart or death detection; a dead worker's runs hang forever (train groups reclaimed only by the staleness sweep; eval/frozen never) | `vf/serve/pool.py:18-20, 194-197` → 08 §7.2 |

---

## 3. Rollout data plane

```
harness ──POST {endpoint}/chat/completions (Bearer <model_secret>)──► interception (env-server worker)
   ──TrainClient: render/bridge (renderer)──► POST {base_url−/v1}/inference/v1/generate  (X-Session-ID: trace.id) ──► router ──► engine
eval:  interception ──EvalClient relay──► POST {base_url}/chat/completions ──► router ──► engine
orchestrator ──prefill score──► /inference/v1/generate (prompt_logprobs)       orchestrator ──► router /finish_session
```

### 3.1 Harness → interception

| ID | Item | Contract | Cite |
|---|---|---|---|
| R1 | Endpoint handed to the harness | `endpoint = runtime.host_url(base_url + "/v1")` (subprocess/apptainer: loopback; docker: egress-proxy capability URL `…/.vf-host/<token>/v1`; Prime/Modal: tunnel URL), `secret = model_secret` | `vf/rollout.py:247-261` → 08 §3.5 |
| R2 | Routes | POST `/v1/chat/completions` (**only trainable dialect**), `/v1/responses`, `/v1/messages`, `/v1/messages/count_tokens` (eval only); GET `/v1/models` (relayed, 30 s); POST `/tool`; GET/PUT `/state`; GET `/task`. Dialect = route posted to | `vf/interception/server.py:423-437` → 07 §4.2 |
| R3 | Auth | model secret via `Authorization: Bearer` (Anthropic: `x-api-key`) looked up in `sessions`; unknown → 401. `/state` + `/task` use a separate **state secret** (or shared-server secret + `X-Verifiers-State-Route: <trace.id>`) | `vf/interception/server.py:586-589`; `vf/dialects/base.py:299-302` |
| R4 | Headers consumed | `x-stainless-retry-count` + body hash (retry coalescing), `Idempotency-Key` (stripped upstream), `X-ACP-Model-Request-ID` (stripped) | `vf/interception/server.py:606-691` → 07 §3.5.2 |
| R5 | Request rules for training | chat dialect, `n == 1`, un-namespaced **function** tools only; the program's own sampling fields (temperature, `max_tokens`, `stop`, …) are **ignored** — the run's `ctx.sampling` is authoritative | `vf/clients/train.py:40-52, 342-350, 367-370`; `vf/dialects/chat.py:512-514` |
| R6 | Status codes to the program | 400 stop/refusal/overlong prompt (`OverlongPromptError` → `ProviderError(400)`), 401 unknown secret, 409 rollout concluded, provider status or 502/503/504. `NotImplementedError` (wrong dialect), `MalformedGenerateResponseError` and "renderer has no tool support" are plain exceptions → **502, not stashed on `session.error`** (SDK retries resample the turn) | `vf/interception/server.py:847-870`; `vf/clients/train.py:444-449` → 09 §2 |
| R7 | Streaming | training never relays; the whole response is generated, then framed as one SSE chunk + `[DONE]`; keepalives after 60 s grace | `vf/interception/server.py:238-310, 597-600` |

### 3.2 `TrainClient` → `/inference/v1/generate` request

`POST {base_url.rstrip("/").removesuffix("/v1")}/inference/v1/generate` — mounted at the **server root**, not under `/v1` (`rnd/client.py:338-339`). Built by `renderers.client.generate` (`rnd/client.py:191-352`) from `TrainClient.get_response` (`vf/clients/train.py:329-452`).

| ID | Field / header | Value | Cite |
|---|---|---|---|
| R8 | `Authorization` | `Bearer $<api_key_var>` via the worker's `AsyncOpenAI` client (`max_retries=0`, connect 5 s, no read timeout) | `vf/clients/base.py:12-31` |
| R9 | `X-Session-ID` | `trace.id`, same on every turn of a rollout (router session affinity) | `vf/clients/train.py:440-442`; `vf/clients/client.py:16-18` |
| R10 | `model` | `body["model"]` after `apply_overrides` = `RunRequest.model` | `vf/clients/train.py:367` |
| R11 | `token_ids` | bridged ids (`bridge_to_next_turn(prev_prompt, prev_completion, tail)`) when the tail is `[tool*, user?]`, no images, and the prefix ends at a contiguous sampled assistant node; else a full `render(…, add_generation_prompt=True)` | `vf/clients/train.py:386-426`; `vf/graph.py:504-531` |
| R12 | `sampling_params` | `ctx.sampling.wire_args()` (`extra_body` flattened, typed fields win) **minus** `chat_template_kwargs` and `cache_salt`; then **forced** `stop_token_ids = renderer.get_stop_token_ids()`, `logprobs = 1`, `skip_special_tokens` default `False`; plus `routed_experts_prompt_start = max(len(prev_prompt)+len(prev_completion)−1, 0)` only on a bridged turn | `vf/clients/train.py:368-370, 409-412`; `rnd/client.py:310-313` |
| R13 | `cache_salt` | top-level body field (popped out of sampling), `str(group.policy_version_at_start)` (§3.5) | `rnd/client.py:330-331` |
| R14 | `features` | multimodal only: `{mm_hashes, mm_placeholders, kwargs_data}` built with vLLM internals **in the client process** (Qwen3-VL/Qwen3.5 family, Gemma 4 only) | `rnd/client.py:321-329, 488-637` → 09 §3.9 |
| R15 | `content_parts`, `priority` | `process_multimodal=False` path (unused by verifiers) / passthrough | `rnd/client.py:320-333` |
| R16 | Pre-flight | `GET {base}/v1/models` once per `(base_url, model)` → `max_model_len` of the card with `id == model`; `len(prompt) > cap` → `OverlongPromptError` without contacting the engine; lookup failure silently disables the check (cached forever) | `rnd/client.py:63-109, 302-308` |

**Engine-side constraints on the request** (`[vLLM ≥0.28 native, not readable]` for the rejection itself): with `enable_return_sampling_mask` on, the engine rejects any request with `temperature <= 0` or without `top_k > 0` — **engine-wide, including eval**; `top_k ≤ 512` (`TRAIN_TOP_K_BOUND`) because the trainer pads masks per micro batch (`cfg/inference.py:472-473`; `cfg/orchestrator.py:509-515, 658-681`; `cfg/rl.py:613-643`) → 06 §3.3.

### 3.3 `/inference/v1/generate` response (fields consumed)

| ID | Field | Contract | Cite |
|---|---|---|---|
| R17 | `choices[0].token_ids` | completion ids; engine stops **on** a stop id and returns it as the last token, without trailing template scaffold (bridges re-add the `\n`) | `rnd/client.py:355-356` → 09 §5.5 |
| R18 | `choices[0].logprobs.content[i]` | exactly one entry per completion token, `{token: "token_id:<id>", logprob: finite float ≠ −9999.0}`; `logprobs_mode = processed_logprobs` ⇒ logprob of the post-temperature/top-p/top-k distribution. Any violation → `MalformedGenerateResponseError` (→ 502) | `rnd/client.py:137-188`; `cfg/inference.py:635-636` |
| R19 | `choices[0].routed_experts` | `{data: base64(raw uint8\|uint16), shape: [tokens, layers, topk], start: int, dtype: "uint8"\|"uint16"}` (uint16 iff any expert id > 255); only this object form is allowed (router merges P/D objects). The base64 is spliced out of the raw JSON before `json.loads` | `src/inference/vllm/routed_experts.py:11-33`; `src/inference/vllm/serving_tokens.py:58-87`; `rnd/client.py:112-134` |
| R20 | `choices[0].sampling_mask` | vLLM native: one list of surviving vocab ids per completion token → verifiers `SamplingMask{ids: int32 flat, counts: int32}` | `rnd/client.py:376-379`; `vf/types.py:194-217` |
| R21 | `choices[0].finish_reason` | `stop`/`length`/…; client rewrites `stop → tool_calls` when ≥1 tool call parsed `OK` | `rnd/client.py:381-394` |
| R22 | `prompt_token_ids`, `mm_placeholders` | required only on the `content_parts` path | `rnd/client.py:357-366` |
| R23 | `usage` | includes `prompt_tokens_details` (`enable_prompt_tokens_details=True`); verifiers overwrites `Usage` with `len(prompt_ids)`/`len(completion_ids)` | `cfg/inference.py:640-641`; `vf/clients/train.py:106-173` |

### 3.4 How each `sampling_params` key gets there

| ID | Key | Origin | Path |
|---|---|---|---|
| R24 | `temperature`, `top_p` | `TrainSamplingConfig` (per source, merged over `orchestrator.train.sampling`) | `to_sampling_args` → `RunRequest.sampling` → `wire_args()` (`cfg/orchestrator.py:91-108`) |
| R25 | `max_tokens` | `max_completion_tokens` (alias) | `vf/types.py:270-284` |
| R26 | `top_k`, `min_p`, `return_token_ids` | `extra_body` defaults `top_k=-1`, `min_p=0.0`, `return_token_ids=True` for policy-sourced envs; `top_k=512` injected when only `top_p` truncates | `cfg/orchestrator.py:673-681, 758-769` |
| R27 | `logprobs` | `True` from `to_sampling_args`, overwritten to `1` by `generate()` | `rnd/client.py:312` |
| R28 | `stop_token_ids`, `skip_special_tokens` | renderer / client | `rnd/client.py:311-313` |
| R29 | `routed_experts_prompt_start` | verifiers, bridged turns only | `vf/clients/train.py:409-412` |
| R30 | trainer temperature | **not** from the engine: the sink stamps `temperatures = [env sampling.temperature] * len` | `src/orchestrator/train_sink.py:279-283` |

### 3.5 `cache_salt` path

`dispatcher` (`cache_salt = str(group.policy_version_at_start)` for live-policy work, `None` for frozen; fixed at **group open**, `src/orchestrator/dispatcher.py:559-565`) → `Env._sampling` puts it in `extra_body.cache_salt` (`src/orchestrator/envs.py:121-125`) → msgpack `RunRequest.sampling` → worker `ctx.sampling` → `wire_args()` flattens → `sampling_params.pop("cache_salt")` (`vf/clients/train.py:370`) → top-level `/generate` field (`rnd/client.py:330-331`). Eval: stays top-level in the relayed chat body (`vf/dialects/chat.py:589-597`). The same salt is reused on **every turn** of the episode (→ 07 §3.5.3, 03 §7.1).

Contract (R31): the prefix cache is **never reset** on a weight update (`/pause` hard-wired `clear_cache=False`, no `reset_prefix_cache` anywhere) `[vLLM 0.24]`, so the salt is the only guard — and it is per group, not per dispatch (§8.6).

### 3.6 Session affinity and release

| ID | Item | Contract | Cite |
|---|---|---|---|
| R32 | Router affinity | `vllm-router --request-id-headers x-session-id`, policy `sticky_least_loaded` (first request → least-loaded worker, later turns stick) `[router internals UNVERIFIED]` | `src/entrypoints/inference.py:182-183` → 06 §3.12 |
| R33 | `/finish_session` | `POST {router root}/finish_session?session_id=<trace id>` for every trace id seen in deltas (`delta["trace"]`) or the final episode, in the episode task's `finally` (shielded). `_admin_post(timeout_s=5)`: retries 5xx/timeouts up to 10 s / 10 attempts, then logged at debug. **No-op unless `admin_base_url` is set** | `src/orchestrator/clients.py:94-129`; `src/orchestrator/dispatcher.py:583-610` |

### 3.7 Prefill scoring (OPD/OPSD reference logprobs)

| ID | Item | Contract | Cite |
|---|---|---|---|
| R34 | Request | `POST {base−/v1}/inference/v1/generate {"model", "token_ids", "sampling_params": {"max_tokens": 1, "temperature": 1.0, "top_p": 1.0, "prompt_logprobs": 1}}` — **no `cache_salt`, no `top_k`** | `src/orchestrator/clients.py:523-544` |
| R35 | Response | `prompt_logprobs[i]` dict → **first entry**'s logprob (vLLM inserts the actual token first `[vLLM 0.24]`); position 0 (`None`) → 0.0 | `src/orchestrator/clients.py:545-556` |
| R36 | Cache | any `prompt_logprobs` request skips prefix-cache reads ⇒ always recomputed under current weights `[vLLM 0.24]`; `opd`/`opsd` + truncated train sampling is rejected at config time (would hit R-constraint above) | `cfg/orchestrator.py:684-690` → 06 §3.3 |
| R37 | Process requirement | the orchestrator must import `vllm` (`GenerateResponse`) | `src/orchestrator/clients.py:528` |

### 3.8 Eval path

Eval groups always use `clients.eval_client` = `EvalClientConfig` → interception relays the native chat body to `{base_url}/chat/completions` through the router, **no tokens** on the trace; routed experts are stripped from chat responses by a server patch (`src/orchestrator/dispatcher.py:547-553`; `src/inference/patches.py:388-423`) → 07 §3.5.4.

---

## 4. Admin plane (orchestrator → every engine, bypassing the router)

Clients: `setup_admin_clients` — one `httpx.AsyncClient` per URL in `admin_base_url` (else `[base_url]`), `/v1` stripped, `Authorization: Bearer $<api_key_var>` if set, `max_connections=4, max_keepalive_connections=1`, `timeout=None` (`src/orchestrator/clients.py:283-308`). `_admin_post`: tenacity retry on ≥500 / `TimeoutException` / `TransportError` (never 4xx), `stop_after_delay(2·timeout_s) | stop_after_attempt(10)`, exp. wait 1–10 s, per-attempt `Timeout(connect=10, read=timeout_s, write=60, pool=10)` (`src/orchestrator/clients.py:370-412`).

### 4.1 Routes

| ID | Route | Payload | Server action | Client timeout / retry | Caller | Cite |
|---|---|---|---|---|---|---|
| A1 | `GET /health` | — | stock vLLM | 1 s poll until 2xx, total `wait_for_ready_timeout` 3600 s; **404 ⇒ treated as ready**; router health too when `admin_base_url` set | `AdminPlane.wait_for_ready` | `src/orchestrator/clients.py:154-163, 332-367` |
| A2 | `GET /v1/models` | — | stock vLLM | none; must list `model_name` unless `skip_model_check` | same | `src/orchestrator/clients.py:311-329` |
| A3 | `POST /init_broadcaster` (NCCL) | `{host, port, rank_offset = i·(W // n_clients), inference_world_size: W, timeout}` | `collective_rpc("init_broadcaster", (host, port, rank_offset, W, timeout, session_id="default"))` → each worker joins the NCCL group (§6.3) | plain `post`, **no timeout, no retry**; `HTTPStatusError` 404 → warning, **any other status silently swallowed**; transport errors propagate | `NCCLWeightReceiver.initialize` | `src/orchestrator/clients.py:165-203`; `src/inference/vllm/server.py:133-146` |
| A4 | `POST /init_broadcaster` (NIXL) | same + `session_id` | worker creates NIXL agent, publishes to ModelExpress (`inference_world_size` ignored) | `_admin_post(timeout_s=max(300, timeout))` | `init_nixl_broadcast` | `src/orchestrator/clients.py:491-520` |
| A5 | `POST /pause` | query `mode=keep&clear_cache=false` (**ignored**) | always `pause_generation(mode="keep", clear_cache=False)` = freeze in place, requests and KV parked, not aborted, not drained `[vLLM 0.24]` | 300 s/attempt, ≤600 s total | `_pause_engines`, gathered | `src/orchestrator/clients.py:415-422`; `src/inference/vllm/server.py:66-70` |
| A6 | `POST /update_weights` | `{"weight_dir": <posix str> \| null}` | `collective_rpc("update_weights_from_path", (weight_dir,))` on every worker; returns after every worker (and every DP engine behind the URL `[vLLM 0.24]`) finished | 720 s/attempt, 1440 s total; **retry unsafe under NCCL/NIXL** (a retry queues behind the running collective and waits for a broadcast that never comes); safe for filesystem | `AdminPlane.update_weights`, gathered | `src/orchestrator/clients.py:205-232, 392`; `src/inference/vllm/server.py:79-83` → 06 §3.6.1d |
| A7 | `POST /resume` | — | `resume_generation()` | 300 s/attempt; always runs in `finally`; idempotent | `_resume_engines`, gathered | `src/orchestrator/clients.py:425-433`; `src/inference/vllm/server.py:73-76` |
| A8 | `POST /load_lora_adapter` | `{"lora_name": <base model name>, "lora_path": <broadcasts/step_N>}` | forces `load_inplace=True`, loads, resets stored flag to `False`; returns `{"status":"ok"}` or vLLM `ErrorResponse` code | 30 s/attempt, 120 s total, retries 404/500/transport (NFS lag); **no pause** | `FileSystemWeightReceiver` (LoRA) | `src/orchestrator/clients.py:436-488`; `src/inference/vllm/server.py:86-117` |
| A9 | `GET /metrics` (+ one-time `GET /v1/models` for `max_model_len`) | — | vLLM Prometheus | every 5 s, 5 s timeout, errors swallowed; feeds adaptive concurrency (no metrics ⇒ cap stuck at `min_inflight`) | `InferenceMetricsCollector` | `src/orchestrator/inference_metrics.py:296-374` → 03 §3.7 |
| A10 | `GET /liveness` | — | `collective_rpc("liveness_probe")` bounded by `liveness_timeout_seconds` (30) → 200 / 503 | — | k8s probes only (off by default) | `src/inference/vllm/server.py:120-130` |
| A11 | Dynamo | `GET <discovery_url>/v1/rl/workers` (default `base_url` port+1); NCCL update = `/pause` → `POST /collective_rpc {"method","timeout","args","kwargs":{}}` expecting `{"results":[None]}` (single POST, **no retry**, failure ⇒ plane `terminal`) → `/resume` | Dynamo frontend | — | `DynamoAdminPlane` | `src/inference/dynamo.py:26-151, 309-425` → 06 §3.10 |

### 4.2 Ordering rules

| ID | Rule | Consequence of violation | Cite |
|---|---|---|---|
| A12 | **`admin_base_url` order == GPU rank order.** `rank_offset = i · (W // len(clients))`; every URL must own exactly `W/n` GPUs; `inference_metrics_roles` uses the same order; P/D `ADMIN_URLS` = all prefill ranks then all decode ranks, per replica | NCCL ranks collide/skip (§6.3); NIXL maps wrong ranks | `src/orchestrator/clients.py:137-140, 173, 200`; `tpl/multi_node_rl.sbatch.j2:97-124`; `cfg/rl.py:813-821` |
| A13 | Update sequence (watcher, under `update_lock`): `wait_published(N)` → `ckpt_step = N` → `on_version_pending(N)` (dispatcher barrier + stale-group drop, **before** any pause) → `receiver.receive(N)` → `policy.version = N` → `on_new_version` → hooks (`trigger_eval`, `on_policy_update` → dispatch gate + `version_advanced`) | drop-after-resume races NIXL KV-transfer flushes and crashes the decode engine | `src/orchestrator/watcher.py:79-131` |
| A14 | Inside `receive`: FS = ack → wait `.finished` → pause → update → resume; NCCL = pause → **ack in `on_paused`** → update (engines enter the collective) → resume; NIXL = ack → wait `policy:<N>:ready` → pause → update(null) → resume → send `policy:<N>:complete` | NCCL ack before pause ⇒ trainer collectives start while engines serve | `src/transports/weights/filesystem.py:68-75`; `src/transports/weights/nccl.py:207-213`; `src/transports/weights/nixl/nixl.py:477-502` |
| A15 | One version at a time, strictly in order; `policy.version` advances only after **every** engine applied it; any failure raises out of `receive` and kills the watcher (→ orchestrator exit) | skipping a version strands the trainer on its `.receiver_ready` | `src/orchestrator/watcher.py:113-116` → 06 §3.6 |

---

## 5. Batch plane (orchestrator → trainer)

Serialization: `msgspec.msgpack.Encoder()` / `Decoder(type=list[MicroBatch])`, one `list[MicroBatch]` per DP rank per step (`src/transports/batch/base.py:15, 34`). All structs are `array_like=True` ⇒ **positional msgpack arrays**; `omit_defaults` does not drop positions (all 19 `MicroBatch` positions are always encoded, unset optionals as `nil`). Appending a defaulted field is compatible both ways; inserting or reordering is not (→ 06 §3.11).

### 5.1 `TrainingSample` (orchestrator-internal; produced by `trace_to_samples`, consumed in-process by `prepare_batch` — never on the trainer wire)

`src/transports/batch/types.py:30-87`; → 04 §4.1. All per-token lists have length `L = len(token_ids)`.

| # | Field | Type | Meaning |
|---|---|---|---|
| 0 | `token_ids` | `list[int]` | branch tokens |
| 1 | `mask` | `list[bool]` | trainable sampled tokens (shared node trained in first branch only) |
| 2 | `logprobs` | `list[float]` | sampling logprobs $\log\mu$, 0.0 off-sample |
| 3 | `temperatures` | `list[float]` | `[env temperature]*L` (sink) |
| 4 | `env_name` | `str` | resolved env name (`"all"`/`"agg"` reserved) |
| 5 | `ref_logprobs` | `list[float] \| None` | prefill reference logprobs |
| 6 | `mm_kwargs` | `dict[str, EncodedTensor] \| None` | image tensors, cat dim 0 |
| 7 | `routed_experts` | `RoutedExperts \| None` | `[L, layers, topk]`, dtype from payload |
| 8 | `mm_token_type_ids` | `list[int] \| None` | 0 text / 1 image / 2 video |
| 9–11 | `rl_weights`, `ce_weights`, `ref_kl_weights` | `list[float] \| None` | component streams; `None` = absent |
| 12 | `advantages` | `list[float] \| None` | per-token credit |
| 13 | `sampling_mask` | `SamplingMask \| None` | kept vocab ids per token |
| 14–15 | `trace_id`, `branch_index` | `str \| None`, `int \| None` | identity for trainer annotations |

### 5.2 `MicroBatch` (the wire payload)

`src/transports/batch/types.py:91-125`; trainer tensors from `src/trainer/rl/data.py:202-267`; → 04 §4.2, 06 §4.3.

| # | Field | Wire type | Trainer tensor | Notes |
|---|---|---|---|---|
| 0 | `input_ids` | `list[int]` | `[1,L]` int64 | $L$ ≤ `seq_len` + < `pad_to_multiple_of` padding (variable per micro batch) |
| 1 | `loss_mask` | `list[bool]` | `[1,L]` bool | first token of every packed sample must be False |
| 2 | `advantages` | `list[float]` | `[1,L]` f32 | always present (zeros if none) |
| 3 | `inference_logprobs` | `list[float]` | `[1,L]` f32 | $\log\mu$ |
| 4 | `position_ids` | `list[int]` | `[1,L]` int64 | restart at 0 per sample and for padding |
| 5 | `sequence_lengths` | `list[int]` | python list | per packed sample (loss split, annotations); last includes padding |
| 6 | `temperatures` | `list[float]` | `[1,L]` f32 | padding 1.0 |
| 7 | `env_names` | `list[str]` | python list | `""` = padding |
| 8 | `seq_lens` | `list[int]` | `[segments]` int64 | attention segments → `cu_seqlens` (== `sequence_lengths` at the pin) |
| 9 | `ref_logprobs` | `list[float] \| None` | `[1,L]` f32 \| None | 0.0 back-fill |
| 10 | `routed_experts` | `RoutedExperts \| None` | `[1,L,layers,topk]` int32 | `frombuffer(dtype).reshape(shape).to(int32)`; required iff `enable_router_replay` |
| 11 | `mm_kwargs` | `dict[str, EncodedTensor] \| None` | dict, no batch dim | `**`-unpacked into forward |
| 12 | `mm_token_type_ids` | `list[int] \| None` | `[1,L]` int64 | |
| 13–15 | `rl_weights`, `ce_weights`, `ref_kl_weights` | `list[float] \| None` | `[1,L]` f32 \| None | back-fill 1.0 / 0.0 / 0.0; padding 0.0 |
| 16 | `sampling_mask` | `SamplingMask \| None` | `[1,L,K]` int32, −1 padded, `K = max(max(counts),1)` | row = mask of the token at that position; trainer shifts left |
| 17–18 | `trace_ids`, `branch_indices` | `list[str] \| None`, `list[int] \| None` | python lists | per sequence; `""`/−1 unknown; consumed by `AnnotationWriter` |

Sub-structs (`src/transports/batch/types.py:7-25`): `EncodedTensor = [dtype: str, shape: list[int], data: bytes]`; `RoutedExperts = [data: bytes, shape: [seq_len, layers, topk], dtype: str]`; `SamplingMask = [ids: bytes (int32), counts: bytes (int32 per position; 0 = no mask)]`, `len(ids) == 4·Σcounts`. `lora_num_tokens` is synthesized trainer-side (`[L]` int32). **No step, no policy version** on any struct.

### 5.3 Grid shape contract

| ID | Rule | Checked? | Cite |
|---|---|---|---|
| B1 | Grid = `list[list[MicroBatch]]`, outer index = DP rank, `len == num_train_workers`, all inner lists equal length | sender asserts rectangularity + width | `src/transports/batch/zmq.py:68-70`; `src/transports/batch/filesystem.py:20-22` |
| B2 | `num_train_workers` == trainer DP degree (`world // cp`); `pad_to_multiple_of` == trainer `cp` | **nothing checks at runtime**; auto-filled only by the `rl` launcher | `cfg/rl.py:687-714` |
| B3 | Trainer `dp_rank = rank // (world_size // dp_world_size)`; every rank (incl. CP peers) builds a receiver on its `dp_rank` — correct only because `cp` is the innermost mesh dim | by construction | `src/trainer/rl/data.py:190-193` → 05 §5.2 |
| B4 | Every DP rank gets the same number of micro batches, and the same modality at every index (dummy micro batches pad; they cost a full fwd+bwd) | packer | `src/trainer/batch.py:847-911` → 04 §3.7 |
| B5 | Samples > `seq_len` are silently truncated by the packer at ship time | none | `src/trainer/batch.py:415-441` |

### 5.4 ZMQ transport (default)

| ID | Socket | Side | Endpoint | Behaviour | Cite |
|---|---|---|---|---|---|
| B6 | data `PUB` (asyncio) `bind` | orchestrator | `tcp://{host}:{port}` (5555) | `SNDHWM=hwm` (10); per rank `send_multipart([b"data_rank\|<r>\|", msgpack], copy=False)`; **never blocks — drops past a subscriber's HWM** | `src/transports/batch/zmq.py:27-29, 66-80` |
| B7 | ready `PULL` `bind` | orchestrator | `tcp://{host}:{port+1}` (5556) | first `send` blocks until READY from `data_world_size` **distinct rank ids** (a `set` — CP peers dedup) | `src/transports/batch/zmq.py:32-34, 47-64` |
| B8 | data `SUB` `connect` | each trainer rank | same | `SUBSCRIBE b"data_rank\|<r>\|"` (trailing `\|` makes the prefix match exact); `RCVHWM=hwm`; `poll()` then `recv_multipart` | `src/transports/batch/zmq.py:96-101, 121-134` |
| B9 | ready `PUSH` `connect` | each trainer rank | `port+1` | sends `str(data_rank)` **once**, in the constructor | `src/transports/batch/zmq.py:107-112` |

Safety of B6 rests on the ship gate (§8.3): ≤ `TARGET_LAG + 1` = 2 unconsumed steps ≪ HWM 10. The READY/CP-dedup gap is masked because every rank builds its receiver before the collective startup broadcast, which the orchestrator must consume before shipping (`src/trainer/rl/train.py:231-236, 267-273`).

### 5.5 Filesystem transport

`output_dir/batches/step_<S>/rank_<r>.bin`, written as `rank_<r>.bin.tmp` then `rename`, in a worker thread; receiver `sync_wait_for_path` (1 s poll, no timeout) then decode (`src/transports/batch/filesystem.py:17-62`). Nothing garbage-collects `batches/` during a run.

### 5.6 Counter coupling (no step id)

| ID | Rule | Cite |
|---|---|---|
| B10 | Sender and receiver each hold a local counter initialised to their own `progress.step` and advanced once per batch. ZMQ: counter is **log-only**, coupling is by message order; FS: counter is the directory name | `src/transports/batch/zmq.py:45, 80, 119, 133`; `src/transports/batch/filesystem.py:24-27, 47` |
| B11 | Orchestrator `sender` created with `progress.step = resume_step + 1`; trainer `DataLoader(output_dir, progress.step, …)` after `progress.step = checkpoint_step + 1` | `src/orchestrator/orchestrator.py:255-266`; `src/trainer/rl/train.py:217, 231-236` |
| B12 | **Orchestrator-only restart is unsupported**: new ZMQ sender's READY barrier never completes (ranks sent READY once); FS rewrites `batches/step_{resume+1…}` while the trainer waits on its own later step; `sync_startup` expects a fresh trainer startup broadcast | → 03 §3.12, 06 §7 |

---

## 6. Weight plane (trainer → inference, driven by the consumer)

### 6.1 The four-marker handshake (every transport)

Directory `output_dir/broadcasts/step_<N>/` (`src/utils/pathing.py:287-292`). Markers `src/transports/weights/base.py:23-26`.

| ID | Marker | Writer / when | Reader / how | Cite |
|---|---|---|---|---|
| W1 | `.sender_ready` | trainer master, after `rm -rf step_N; mkdir` (reset per attempt) | consumer: `next_version(current)` = max published step (watcher polls 1 s); `wait_published` polls 0.2 s | `src/transports/weights/base.py:53-64, 139-156` |
| W2 | `.receiver_ready` | consumer `_ack`: FS/NIXL at the start of `receive`; NCCL inside `on_paused` after all engines paused | trainer master polls 0.1 s; `TimeoutError("No receiver joined the broadcast within …")` after `weight_broadcast.timeout` (1200 s) — bounds **only** the ack wait | `src/transports/weights/base.py:75-93, 158-160`; `cfg/shared.py:40-43` |
| W3 | `.started` | trainer master after the ack | nobody | `src/transports/weights/base.py:65` |
| W4 | `.finished` | trainer master after `_broadcast` returns on all ranks | FS receiver (`wait_for_path`, 1 s poll, **no timeout**) | `src/transports/weights/base.py:66-68`; `src/utils/pathing.py:414-425` |
| W5 | Retention | master `_clean(N)` keeps `N` and `N−1`; at startup prunes `> start−1`; launcher `clean_future_steps` deletes `≥ resume_step` | — | `src/transports/weights/base.py:29-36, 101-108`; `src/utils/pathing.py:379-397` |

`WeightSender.broadcast` is `@final` and runs on all ranks: master does the handshake; non-masters enter `_broadcast` immediately and each transport must hold them (NCCL/FS `dist.barrier()`, NIXL object-broadcast + notifications) so their gathers do not start before inference is paused (`src/transports/weights/base.py:53-70`; `src/transports/weights/nccl.py:182-190`).

### 6.2 Transport selection

`rl` default **NCCL** unless LoRA or no `[inference]` → filesystem (`cfg/rl.py:447-451`); standalone trainer/orchestrator default filesystem; LoRA ⇒ filesystem; trainer, orchestrator and `inference.weight_broadcast.type` must agree (`cfg/rl.py:452-493`; `packages/prime-rl-configs/src/prime_rl/utils/validation.py:306-318`). Server worker class from `WORKER_EXTENSION_CLS[type]` (`src/inference/vllm/server.py:59-63`).

### 6.3 NCCL

| ID | Item | Contract | Cite |
|---|---|---|---|
| W6 | Group | trainer master only: `StatelessProcessGroup.create(host, port, rank=0, world_size=W+1, store_timeout=timeout)` + `PyNcclCommunicator`; rank 0 binds a TCPStore at exactly `(host, port)` `[vLLM 0.24]` | `src/transports/weights/nccl.py:117-140, 162-179` |
| W7 | **Rank formula** | worker of admin client $i$ with local device $d$: $\text{rank} = 1 + i\cdot\frac{W}{n_{\text{clients}}} + d$, world $W+1$; `d = self.device.index` (index within the server's `CUDA_VISIBLE_DEVICES`; with internal DP = `dp_local_rank·tp + tp_rank` `[vLLM 0.24]`); the inference side derives nothing — $W$ and `rank_offset` come from the orchestrator | `src/inference/vllm/worker/nccl.py:91-126`; `src/orchestrator/clients.py:173, 200` |
| W8 | $W$ (`inference_world_size`) | single-node: `dp × tp` **computed before DP auto-fill** (undercount bug for local launches with DP unset; single-node SLURM heals by re-parse); multi-node: `total_infer_nodes × gpus_per_node`; P/D: total infer GPUs | `cfg/rl.py:458-469, 687-709, 778-792` → 06 §3.8 |
| W9 | Message sequence per version | `long[1] = L+1` (L = `get_max_layer_num`); then for each group — **non-layer keys first**, then `layer_prefix{i}.*` for i = 0..L−1: `long[1]` = len(pickle); `uint8[size]` = `pickle.dumps({torch.dtype: [(name, shape, numel), …]})` (first-seen dtype order); per dtype one flat `cat([v.flatten() …])` | `src/transports/weights/nccl.py:24-81, 143-159`; `src/inference/vllm/worker/nccl.py:24-86` |
| W10 | Names / dtypes | HF names (`convert_layer_to_hf` if `is_prime_state_dict(group)`, else `revert_weight_conversion`); DTensors cast to **bf16** (hard-coded) or fp32 when `keep_in_fp32_for_weight_transfer(name)`, **before** `full_tensor()`; non-DTensor buffers keep native dtype; pickle of torch dtypes ⇒ compatible torch both sides | `src/transports/weights/nccl.py:84-114, 129` |
| W11 | Receiver | zero-copy views into the concatenated buffer → `load_weights_checkpoint_layerwise` | `src/inference/vllm/worker/nccl.py:31-57, 132-147` |
| W12 | Lifetime | group formed once; not rebuildable without restarting both sides | → 06 §7 |

### 6.4 Filesystem (full model)

| ID | Item | Contract | Cite |
|---|---|---|---|
| W13 | Writer | every rank all-gathers every DTensor (bf16 / declared fp32), keeps a byte-balanced **layer-atomic** partition, converts to HF, writes `tmp-rank{r}-*.safetensors` (≤5 GB), exchanges shard maps, renames to `model-XXXXX-of-YYYYY.safetensors` (or `model.safetensors` if one shard); master writes `model.safetensors.index.json` **only if > 1 shard** | `src/transports/weights/filesystem.py:34-58`; `src/utils/weights.py:131-252` (index `:248`) |
| W14 | Not written | no `config.json`, tokenizer or generation config — the dir is weights-only; completion signal is `.finished`, not the index | `src/utils/weights.py:202-252` → 06 §3.7 |
| W15 | Reader | `/update_weights {"weight_dir": step_dir}` → worker `DefaultModelLoader.Source(weight_path)` + layerwise reload (re-runs `process_weights_after_loading`, so online FP8 is redone inference-side) `[vLLM-internal UNVERIFIED]`; idempotent (safe to retry) | `src/inference/vllm/worker/filesystem.py:28-55` |

### 6.5 LoRA (filesystem only)

| ID | Item | Contract | Cite |
|---|---|---|---|
| W16 | Files | master writes `adapter_model.safetensors` (unsharded) + `adapter_config.json` = `{peft_type: LORA, r, lora_alpha, lora_dropout, target_modules: sorted suffixes, modules_to_save, …}` | `src/transports/weights/filesystem.py:35-52`; `src/trainer/lora.py:314-360` |
| W17 | Keys | `"{module_fqn}.lora_{A,B}.weight"` (HF names via `convert_adapter_to_hf`), **no** PEFT `base_model.model.` prefix (vLLM strips it only if present); MoE per-expert `"<experts>.<e>.<proj>.lora_{A,B}.weight"`; patched loader accepts bare and qualified expert names | `src/trainer/lora.py:58-73`; `src/inference/patches.py:449-606` |
| W18 | Load | receiver sees `adapter_config.json` → `/load_lora_adapter {lora_name: <base model name>, lora_path}` on every engine, no pause; requires `max_loras=1`, `api_server_count=1`; `modules_to_save` rejected (only adapters ship) | `src/transports/weights/filesystem.py:72-73`; `cfg/inference.py:116-118, 214-215`; `cfg/trainer.py:801-805` |

### 6.6 NIXL (receiver-driven RDMA)

| ID | Item | Contract | Cite |
|---|---|---|---|
| W19 | ModelExpress identity | `SourceIdentity{mx_version:"0.3.0", mx_source_type: WEIGHTS, model_name:"prime-rl-weights", extra_parameters:{role, session_id}}`; `WorkerMetadata{worker_rank, nixl_metadata}`; `wait_for(role, count)` polls 50 ms; RPC retries 120 s | `src/transports/weights/nixl/model_express.py:25-91` |
| W20 | Roles | `trainer` rank 0 (`trainer-table`: encoded `TrainerTensorTable`), `inference` rank `rank_offset + device.index`, `orchestrator` rank 0 | `src/transports/weights/nixl/nixl.py:344-353, 458-475`; `src/inference/vllm/worker/nixl.py:104-115` |
| W21 | `TrainerTensorTable` (msgpack) | `{agents: [TrainerAgent{name, metadata: bytes, device_id}], staging_buffer_count, groups: [TrainerGroup{name ∈ non_layer \| layer.<i>, tensors: [TrainerTensor{name (trainer/prime naming), wire_dtype: bfloat16\|float32, shape, shards: [TrainerShard{agent, offset, numel, addr}]}]}]}` | `src/transports/weights/nixl/trainer_tensor_table.py:8-60` |
| W22 | Notifications | group credit `f"{group:08x}:{generation:016x}"`; policy `f"policy:{step:016x}:{ready\|complete}"`; agent names `f"{role}-{hostname}-r{rank}"` | `src/transports/weights/nixl/agent.py:26-31, 147-148` |
| W23 | Sequence | trainer `policy:S:ready` → orchestrator pauses + `/update_weights {weight_dir: null}` → per group: serving ranks copy to staging, notify each worker; worker READs with the group notification (= credit back) → orchestrator resumes, sends `policy:S:complete` → trainer barrier | `src/transports/weights/nixl/nixl.py:372-447` → 06 §3.9 |
| W24 | Shape/dtype rules | FSDP **dim-0** contiguous shards only (`Shard(1)` ⇒ plan-build failure); bf16/fp32 only; custom (`PreTrainedModelPrimeRL`) models only; receiver replays the prime→HF `conversion_chain` symbolically with `LazyWeight` (view ops only) inside the vLLM worker (imports `prime_rl.trainer.models`); trace + plan static per worker lifetime; trainer reads `state_dict()` **once** ⇒ default `qkv` fusion serves stale q/k/v on ≥2-rank trainers | `src/transports/weights/nixl/graph.py:35-55, 312-346`; `src/transports/weights/nixl/nixl.py:121-188, 326-338` → 06 §3.9, 05 §7 |

---

## 7. On-disk layout of the run dir

`run_dir = output_dir / run.dir` (default `run.name`); `trainer.output_dir = orchestrator.output_dir = run_dir` (`cfg/rl.py:379-396`). Resume is keyed by `run.name`; auto names carry a random suffix.

| ID | Path (under `run_dir` unless absolute) | Writer | Reader | When | Lifetime / cleanup | Cite |
|---|---|---|---|---|---|---|
| F1 | `configs/attempt_N/command.txt`, `rl.toml` | launcher | humans, dashboard | per launch (`command.txt` first writer wins) | kept forever | `src/utils/pathing.py:172-219` |
| F2 | `configs/attempt_N/resolved/{trainer,orchestrator,inference}.json` (+ `rl.json` single-node SLURM; `sft.json`, `eval.json`) | launcher | each child's `cli()` (the inter-process contract) | before spawn | new attempt per launch; old kept; `--dry-run` writes them too | `src/entrypoints/rl.py:89-113, 596` |
| F3 | `configs/attempt_N/resolved/envs/<split>/<name>.json` | launcher `write_env_server_config` (`{env, serve, address_file, log}`) | env server | before spawn | per attempt | `src/utils/pathing.py:230-246` |
| F4 | `…/resolved/envs/<split>/<name>.address` | env server (atomic) | orchestrator (0.5 s poll, 600 s) | after bind | **never deleted**; a requeue reusing the pinned attempt can read a stale address | `src/entrypoints/env_server.py:24-31` → 01 §7.18 |
| F5 | `configs/latest`, `logs/latest` → `attempt_N` | `create_attempt_dirs` (atomic symlink) | `get_config_dir` fallback, tools | per launch (also on `--dry-run`) | repointed each launch | `src/utils/pathing.py:29-57` |
| F6 | `logs/attempt_N/{orchestrator,trainer,inference}.log`, `trainer/torchrun/…/<rank>/std*.log`, `inference/node_<n>[_rank<k>].log`, `inference/router.log`, `envs/<split>/<name>.log` | launcher-redirected stdout/stderr | humans, dashboard, CI regexes | continuous | kept | → 01 §3.8 |
| F7 | `launcher/rl.sbatch`, `launcher/logs/job_<id>.log`, `launcher/.trainer.done`, `.orchestrator.done` | `rl_slurm`; trainer node-rank 0 / orchestrator subshell on exit 0 | `sbatch`; every node's watcher (10 s) | submit / end of run | done files `rm -f`'d at job start | `tpl/multi_node_rl.sbatch.j2:72-77, 580-596` |
| F8 | `batches/step_<S>/rank_<r>.bin` (`.tmp` → rename) | orchestrator (FS transport) | trainer DP rank r (all CP peers) | per ship | **never GC'd in-run**; launcher deletes `> resume_step` (all on fresh run) | `src/transports/batch/filesystem.py:17-35`; `src/utils/pathing.py:379-397` |
| F9 | `broadcasts/step_<N>/{.sender_ready,.receiver_ready,.started,.finished}` + FS weights / LoRA adapter | trainer master (+ all ranks' shards) / consumer (`.receiver_ready`) | consumer, trainer master, vLLM workers | every version + startup | trainer keeps `N`, `N−1`; startup prunes `> start−1`; launcher deletes `≥ resume_step` | `src/transports/weights/base.py:29-36, 101-108` |
| F10 | `checkpoints/step_<N>/trainer/{.metadata, __<rank>_0.distcp}` (RL: no dataloader) | all trainer ranks, synchronous DCP | trainer resume; `tools/convert_dcp_to_*` | every `ckpt.interval` (not last) + final | trainer `maybe_clean` `rmtree`s the **whole** `step_N` (incl. `orchestrator/`, `weights/`); **no completeness marker** | `src/trainer/ckpt.py:197-211, 290-322` → 05 §3.16 |
| F11 | `checkpoints/step_<N>/orchestrator/progress.pt` (`{progress, train_source{rng, envs{sampler, gates}}}`, atomic `mkstemp`+`os.replace`) | orchestrator | orchestrator resume (missing → `FileNotFoundError`, even with `skip_progress`) | after shipping N if `N % interval == 0` (not `max_steps`); teardown save of last shipped step on exception/SIGINT (not SIGTERM) | cleaned only via F10 | `src/orchestrator/ckpt.py:22-60`; `src/orchestrator/orchestrator.py:433-441, 958-974` |
| F12 | `<ckpt.output_dir>/checkpoints/step_<N>/trainer/` | trainer (when rl-level `[ckpt] output_dir` set; propagates to trainer only) | trainer, launcher | as F10 | orchestrator keeps `<run_dir>/checkpoints` ⇒ two step sets; orchestrator ckpts **never cleaned** | `packages/prime-rl-configs/src/prime_rl/utils/validation.py:90-91` → 03 §3.15 |
| F13 | `monitors/file/metrics.jsonl` (`{step\|null, time, …, producer}`) | **trainer and orchestrator** (same file, rows tagged `producer`), line-buffered append | dashboard, benchmarks | each `monitors.log(dict)` | append-only | `src/monitors/file/monitor.py:40-60` → 11 §3.2 |
| F14 | `monitors/file/traces/stream/NNNNN.jsonl[.zst]` + `stream.index.jsonl` | orchestrator only (single writer) | dashboard, `eval --resume` | each arriving episode (all subset) | chunks sealed at 5 GiB and on clean `finalize` | `src/monitors/file/monitor.py:128-152` → 11 §3.3 |
| F15 | `monitors/file/traces/annotations/{orch,trainer,…}/` + `<producer>.index.jsonl` | orchestrator at ship (`effective`, `ship.step`, advantages); trainer per step (`trainer_logprobs`, `entropies`, gathered to rank 0) | dashboard fold | ship / step | append-only | `src/orchestrator/annotations.py:33-49`; `src/trainer/rl/annotations.py:32-81` |
| F16 | `monitors/file/traces/live/<trace>.jsonl`, `live/pending/<id>.json`, `monitors/file/plan.json`, `monitors/prime/run.json` | orchestrator FileMonitor (live wiped on first call); prime monitor | dashboard | ≤0.5 s / eval epoch start / init | live files removed on `done` | → 11 §3.5-3.8 |
| F17 | `modelexpress/`, `mooncake/node_<n>/config.json`, `wandb/`, `step_N/weights[-FP8]/` | SLURM NIXL/mooncake blocks, W&B SDK, offline tools | services, humans | — | — | → 01 §3.8 |
| F18 | Host-level: `~/.cache/prime-rl/dashboard/{daemon.json,dirs.json,.lock}`, `~/.cache/verifiers/limiter/*.bucket` (flock, keyed by `VF_RUN_ID`), `/tmp/vf-pool-XXXX/<i>` (ipc), `~/.cache/harbor/<dataset>` (**orchestrator and env server must see the same path**: `HarborData.task_dir`) | launchers / verifiers / broker / orchestrator taskset load | same + env-server workers | — | ipc unlinked on shutdown; others persist | → 08 §4.2, 10 §4.4 |

---

## 8. Step and version accounting

### 8.1 Conventions

| ID | Quantity | Convention | Cite |
|---|---|---|---|
| S1 | Orchestrator `progress.step` = $S$ | 1-indexed; **the batch being collected**; `+= 1` right after `sender.send` | `src/orchestrator/types.py:26-33`; `src/orchestrator/orchestrator.py:636-637` |
| S2 | Trainer `progress.step` = $s$ | 1-indexed; step $s$ consumes batch $s$ from weights $v_{s-1}$, takes one optimizer step, broadcasts $v_s$; `+= 1` at loop end | `src/trainer/rl/train.py:253-296, 596, 719` |
| S3 | Policy version $V$ | 0-indexed, $v_0$ = base; `policy.version` = latest version **applied on inference** | `src/orchestrator/watcher.py:113-116` |
| S4 | `ckpt_step` | latest version **published** (≥ $V$) | `src/orchestrator/watcher.py:86-91` |
| S5 | Group stamp | `policy_version_at_start = policy.version` at group **open**; all members, dispatched one at a time, share it and its salt | `src/orchestrator/dispatcher.py:525-534, 559-567` |
| S6 | Episode span | `PolicySpan(start = group version, end = policy.version at emission)`; `None` for frozen-sourced | `src/orchestrator/dispatcher.py:717-721` |
| S7 | Checkpoint `step_k` | "step $k$ finished (trainer) / shipped (orchestrator)"; resume at $k+1$ | `src/trainer/rl/train.py:216-217`; `src/orchestrator/orchestrator.py:251-257` |

### 8.2 Timeline

```
batch s (collected while progress.step == s) ──ship──▶ trainer step s: θ = v_{s-1} ──1 optim step──▶ v_s ──broadcast──▶ inference applies v_s ──▶ policy.version = s
```

### 8.3 Gates and bounds

| ID | Mechanism | Condition | Blocks | Where | Cite |
|---|---|---|---|---|---|
| S8 | `TARGET_LAG` | `= 1`, hard-coded constant | — | orchestrator | `src/orchestrator/orchestrator.py:95` |
| S9 | **Dispatch gate** `dispatch_allowed` | open iff `lead = (S−1) − V ≤ TARGET_LAG` ⇔ $V \ge S-2$; re-evaluated after each ship and each applied version | all new train scheduling, **including remaining members of open groups**; eval ignores it | dispatcher `fill_inflight` | `src/orchestrator/orchestrator.py:976-1000`; `src/orchestrator/dispatcher.py:443-476` |
| S10 | **Ship gate** | batch $S$ ships only when $V \ge S-1-\text{TARGET\_LAG} = S-2$; bare `version_advanced.wait()` — **no timeout, no health check** | the main loop (so `out_q` stops draining) | `finalize_train_batch` | `src/orchestrator/orchestrator.py:607-625` |
| S11 | Update barrier | `policy_update_pending` + `scheduling_lock` from `on_version_pending` to `on_new_version` | every new dispatch (train and eval) for the whole `receive` (FS: the whole trainer export) | dispatcher | `src/orchestrator/dispatcher.py:399-441` |
| S12 | **Staleness sweep** (hard guarantee) | void queued traces with `policy.start < min_fresh = (S−1) − max_off_policy_steps` (default 8); full sweep once per $S$ + per inserted group; no-op while `min_fresh ≤ 0` | shipping of stale traces | `TrainSink._drop_stale` | `src/orchestrator/train_sink.py:180-219`; `src/orchestrator/utils.py:42-45` |
| S13 | Stale-group drop (compute saver) | same bound, computed from current $S$, on `on_version_pending`, before pausing engines | in-flight + unscheduled members; emits `GroupCancellation(reason="stale")` via bounded `out_q.put` | dispatcher | `src/orchestrator/dispatcher.py:399-437` |
| S14 | Trainer lockstep | trainer blocks on `.receiver_ready` for every version ⇒ at most one un-acked offer | trainer | `WeightSender.broadcast` | `src/transports/weights/base.py:53-93` |

Consequences: when batch $S$ ships, only $S-1$ and $S$ are beyond the applied version ⇒ ≤ 2 unconsumed batches (keeps ZMQ under HWM 10). Staleness of a trace trained in batch $s$: $\text{total} = \max(0,(s-1)-\text{start})$, $\text{in\_flight} = \min(\text{total}, \text{end}-\text{start})$, $\text{in\_queue} = \text{total}-\text{in\_flight}$ (`src/orchestrator/utils.py:48-61`); bounded $0 \le (s-1)-\text{start} \le M$. A group opens at $\Delta \in \{0,1\}$ but later members can start at $\Delta \ge 2$ (start is pinned at open). `max_off_policy_steps = 0` voids everything dispatched at lead 1 (→ 03 §3.9).

### 8.4 What advances when

| ID | Counter | Owner | Advanced by | Cite |
|---|---|---|---|---|
| S15 | orchestrator `progress.step` | orchestrator | after `sender.send` | `src/orchestrator/orchestrator.py:636-637` |
| S16 | ZMQ/FS sender counter | orchestrator | inside `send` (log-only for ZMQ) | `src/transports/batch/zmq.py:80`; `filesystem.py:27` |
| S17 | receiver counter | each trainer rank | inside `receive` | `src/transports/batch/zmq.py:133`; `filesystem.py:61` |
| S18 | trainer `progress.step` | trainer | end of loop iteration, after broadcast + ckpt + metrics | `src/trainer/rl/train.py:719` |
| S19 | broadcast step dir | trainer master | `broadcast(model, step)`: startup `v_{start−1}`, then `v_s` after every optimizer step | `src/trainer/rl/train.py:263-276, 581-597` |
| S20 | `ckpt_step` / `policy.version` | watcher | after `wait_published` / after `receive` returns from every engine | `src/orchestrator/watcher.py:79-121` |
| S21 | `policy_version_at_start` | dispatcher | at group open only | `src/orchestrator/dispatcher.py:532` |

### 8.5 Startup and resume pairing

| ID | Step | Contract | Cite |
|---|---|---|---|
| S22 | Step resolution | launcher, trainer and orchestrator **each** resolve: `resume.dir` → its `step_<N>`; else `resume.step`; else max `step_*` **name** under their ckpt dir — no completeness check. Launcher uses it only to clean `batches/ > N`, `broadcasts/ ≥ N` | `src/entrypoints/rl.py:671-686`; `src/trainer/rl/train.py:137-143`; `src/orchestrator/orchestrator.py:240-246`; `src/utils/pathing.py:301-311` |
| S23 | Trainer start | load `step_N/trainer`, `progress.step = N+1`, prune broadcasts `> N`, broadcast **startup `v_N`** (v0 from scratch) before waiting for batch $N+1$ | `src/trainer/rl/train.py:205-221, 263-276` |
| S24 | Orchestrator start | `progress.step = N+1` (forced, even with `skip_progress`), sender at $N+1$, `watcher.sync_startup(N)` with timeout `ckpt.wait_for_weights_timeout or 1200` ⇒ `policy.version = N`, hooks fire (eval, gate) | `src/orchestrator/orchestrator.py:255-266, 317-320, 400`; `src/orchestrator/watcher.py:48-54` |
| S25 | Must agree | both resume from the same $N$; the latest `step_N` must contain both `trainer/` and `orchestrator/progress.pt`. Orchestrator writes `step_N/orchestrator` up to 2 steps before the trainer writes `step_N/trainer`, and crash-teardown saves a step the trainer never will | → 03 §3.15, 05 §3.16 |
| S26 | Lost on resume | in-flight episodes, dispatcher groups, sink buffers, eval queue, concurrency cap, RAE baselines; `StandardSampler` cursor advanced at group open ⇒ those tasks are **skipped** | → 03 §7.15 |

### 8.6 Version purity is not guaranteed

A group, a multi-turn episode, and a single mid-decode request can straddle a swap: members after the swap keep the old salt, later turns reuse old-weight prefix KV, and a request frozen by `pause(keep)` resumes on new weights over old KV `[vLLM 0.24]`. $\mu$ is then a hybrid; the IS ratio still uses the logprobs actually sampled, staleness is over-counted (conservative), and nothing tags affected tokens (→ 03 §7.1, 04 §3.11, 06 §3.6.1).

---

## 9. Cross-repo contracts

### 9.1 prime-rl ↔ verifiers

| ID | Contract | prime-rl side | verifiers side |
|---|---|---|---|
| X1 | Client configs on the wire (`ClientConfig` union: `TrainClientConfig{renderer, renderer_model_name, multiplex}` / `EvalClientConfig`) | `setup_client` (`src/orchestrator/clients.py:259-280`); `train_client_type="renderer"` (`src/orchestrator/orchestrator.py:199-205`) | `vf/configs/client.py:29-90` |
| X2 | Serve protocol (`EnvClient`, `RunRequest`, `WireEpisode`, delta assembly) | `src/orchestrator/envs.py:95-153` | `vf/serve/{client,server,types,delta,pool}.py` |
| X3 | Env-server hosting | `serve_env(**pool_serve_kwargs(serve.pool), address, address_queue, log_setup, config_data=env_config_data(env), max_concurrent)`; `set_base_sandbox_labels` (`src/entrypoints/env_server.py:44-51`) | `vf/serve/pool.py:289-371` |
| X4 | Taskset loading **in both processes** (`vf.load_taskset` client-side; `load_environment` server-side, never `load()`); config narrowing imports the env package in launcher, orchestrator, env server, eval | `src/orchestrator/envs.py:105`; `cfg/orchestrator.py:160-176` | `vf/utils/loaders.py:70-126, 302-321`; `vf/serve/server.py:36-39` |
| X5 | Task identity (`key`, `hash`) round-trip; `TaskData` JSON-safe, no `exclude=True` fields | `src/orchestrator/dispatcher.py:116-120` | `vf/task.py:150-164`; `vf/serve/server.py:43-50` |
| X6 | Trace/Branch read + node annotation write (advantages/reference_logprobs compact over `mask=True`; `loss_weights` full-length) | `src/orchestrator/trajectories.py:91-180`; `src/orchestrator/algo/routing.py:11-51` | `vf/trace.py:194-310, 532-570`; `vf/graph.py:83-166` |
| X7 | Provenance types `PolicySpan`, `TrainWorkInfo`, `EvalWorkInfo`, `TrainRunInfo`, `GroupInfo` | `src/orchestrator/dispatcher.py:715-728` | `vf/episode.py:31-61` |
| X8 | Sampling schema `SamplingConfig` (`extra="allow"`, `wire_args` flattens `extra_body`) | `src/orchestrator/envs.py:121-125` | `vf/types.py:263-284` |
| X9 | Runtime union is wire-load-bearing: an orchestrator whose verifiers lacks a runtime `type` fails every episode | — | `vf/runtimes/__init__.py:41-70, 104-134`; `vf/trace.py:112`; `vf/configs/agent.py:30, 69-74` → 08 §6.1 |

### 9.2 verifiers ↔ renderers

| ID | Contract | verifiers | renderers |
|---|---|---|---|
| X10 | Renderer construction from `renderer_model_name or model` + `RendererConfig` + `chat_template_kwargs` (pool keyed on all three) | `vf/clients/train.py:222-307, 369-376` | `create_renderer`, `load_tokenizer` (`rnd/base.py:1171-1313, 1386`) |
| X11 | `render()` / `bridge_to_next_turn()` → `RenderedTokens{token_ids, message_indices, sampled_mask, is_content, message_roles, message_tool_names, multi_modal_data}`; bridge either proves exact prefix or returns `None` | `vf/clients/train.py:386-426`; `vf/graph.py:774-935` | `rnd/base.py:223-315, 716-845` |
| X12 | `generate()` signature and return dict (`completion_ids`, `completion_logprobs`, `routed_experts`, `sampling_mask`, `prompt_attribution`, …); error types (`OverlongPromptError` mapped to 400; `MalformedGenerateResponseError` is a plain `ValueError` → 502) | `vf/clients/train.py:428-452` | `rnd/client.py:191-423` |
| X13 | Parsed tool calls (`ParsedToolCall.status`; verifiers drops `UNKNOWN_TOOL`/nameless) | `vf/clients/train.py:119-131` | `rnd/base.py:559-617` |

### 9.3 renderers / verifiers ↔ vLLM (with prime-rl's server extension)

| ID | Contract | Client side | Server side |
|---|---|---|---|
| X14 | Token-in route `/inference/v1/generate` at server root; request/response schema §3.2-3.3 | `rnd/client.py:310-423` | vLLM `ServingTokens` + `PrimeRlServingTokens` (`src/inference/vllm/serving_tokens.py:58-87`) |
| X15 | Logprob format `token: "token_id:<id>"`, one per completion token, no `-9999.0` | `rnd/client.py:137-188` | vLLM, `logprobs_mode=processed_logprobs` (`cfg/inference.py:635-636`) |
| X16 | `routed_experts` compact object `{data, shape, start, dtype}` | `rnd/client.py:112-134`; `vf/graph.py:689-757` | `src/inference/vllm/routed_experts.py:11-33` |
| X17 | Stop semantics: engine stops on a renderer stop id and returns it; no trailing scaffold | `rnd/client.py:311`; bridges | vLLM |
| X18 | `/v1/models` exposes `max_model_len` (vLLM extension); router passthrough `[UNVERIFIED]` | `rnd/client.py:63-109` | vLLM / vllm-router |
| X19 | Multimodal `features` encoded with vLLM internals in the client process | `rnd/client.py:572-637` | vLLM `mm_serde` |

### 9.4 prime-rl ↔ vLLM (server, admin, weights)

| ID | Contract | Cite |
|---|---|---|
| X20 | App-factory monkeypatches (`build_app`, `init_app_state`, `run_api_server_worker_proc`) mount the admin routes; `vllm.general_plugins` entry point `apply_shared_vllm_patches` runs in every vLLM process — a broken patch import **aborts startup** (only the module import itself is guarded) | `src/inference/vllm/server.py:149-206`; `src/inference/patches.py:4-25`; `pyproject.toml:50-51` |
| X21 | Worker extension API: `init_broadcaster(host, port, rank_offset, inference_world_size, timeout, session_id)`, `update_weights_from_path(arg)`, `liveness_probe()` invoked via `collective_rpc` | `src/inference/vllm/worker/nccl.py:88-147` |
| X22 | Weight names/layout: NCCL/FS ship HF names (conversion chain `convert_to_hf`/`convert_layer_to_hf`); NIXL ships trainer names and imports trainer model code inside the vLLM worker; declared fp32 keys; FP8 ignore patterns must match between trainer and inference | `src/transports/weights/nccl.py:103-114`; `src/utils/weights.py:94-112`; `src/transports/weights/nixl/graph.py:312-346` → 05 §6.1 step 4 |
| X23 | `pause_generation(mode="keep", clear_cache=False)`, `resume_generation()`, layerwise reload helpers, private symbols patched (`Scheduler._update_from_kv_xfer_finished`, `DPCoordinator._wait_for_zmq_addrs`, `LoRAModel.from_local_checkpoint` copy) | `src/inference/vllm/server.py:66-76`; `src/inference/patches.py:127-185, 449-606, 767-797` |
| X24 | Version pin: `vllm>=0.29.0` declared, `[tool.uv.sources]` pins exact v0.29.0 wheels; router wheel `vllm_router-0.2.1` (fork); `nixl-cu13==0.10.1`; ModelExpress 0.3.0 | `pyproject.toml:59, 70, 272-278` → 06 §7 |

### 9.5 Tokenizer-identity requirement

| ID | Tokenizer | Loaded from | Where | Cite |
|---|---|---|---|---|
| X25 | Renderer (produces every trained token id) | `load_tokenizer(renderer_model_name)` = orchestrator `model.name`, **not** `[tokenizer]` (its `name`/`chat_template` never reach the renderer) | env-server worker | `src/orchestrator/clients.py:85-91`; `vf/clients/train.py:278-288` → 09 §2 |
| X26 | Engine | `inference.vllm.model` (must equal `orchestrator.model.name`) — the token-in path applies no chat template; the engine's vocab defines logprobs/sampling masks | vLLM | `packages/prime-rl-configs/src/prime_rl/utils/validation.py:216` (`validate_shared_model_name`) → 02 §3.7 |
| X27 | Trainer | `trainer.model.name` (may differ, e.g. FP8 inference variant); ids go straight into its embedding | trainer | → 02 §3.7 |
| X28 | Rule | renderer tokenizer == engine tokenizer == trainer vocab, incl. special/stop ids. **Nothing checks this.** The orchestrator validator keys `auto` resolution on `tokenizer.name or model.name` (`cfg/orchestrator.py:715`) while the runtime uses `model.name` ⇒ a local checkpoint path silently gets `DefaultRenderer`. OPD (student ids → teacher) and SFT-distill (teacher ids → student) also assume a shared vocabulary, unchecked | → 09 §5.6, §7.1; 04 §7.10 |

---

## 10. Desync detector

| ID(s) | Contract | What breaks when the sides disagree | How it manifests |
|---|---|---|---|
| D2, F4 | Env-server address file | orchestrator reads a stale address from a previous job in a reused attempt dir | `TimeoutError` "did not become healthy" after 600 s |
| E5, E29 | Broker health / worker liveness | a worker crashed at `load_environment` or mid-run | health passes; routed `run`s hang forever; train groups reclaimed only by the staleness sweep; eval hangs |
| E28 | Broker→worker HWM 1000 | ≥1000 frames queued to a dead worker (static pool / capped workers) | whole broker freezes: no relays, no health |
| E9, E21, X5 | `TaskData` JSON round-trip / `(key, hash)` | lossy field, rewriting validator, or `key` depending on non-wire state | `ValueError "Episode task provenance … does not match"` → orchestrator exits 1 on first episode |
| E13, E18 | Delta field sets / count check | a new `Trace` field not routed in `delta.py`, or a lost delta | `RuntimeError "served trace … assembled N nodes / M calls, server reports …"` |
| X9 | Runtime union versions | orchestrator's verifiers lacks the worker's runtime type | every episode fails `WireEpisode` validation |
| R5, R6 | Chat-only training | harness speaks Responses/Anthropic or namespaced tools (`codex`, `claude_code`) | per-turn 502 (retried by SDK), trace ends as `HarnessError`; real cause only in `trace.calls[*].error` |
| R12, R26 | Sampling-mask capture ↔ request knobs | `temperature 0` or no `top_k > 0` while capture is on (incl. eval on the same server) | vLLM rejects the request `[vLLM ≥0.28, unread]`; eval fails at request time |
| R18, X15 | Logprob format | engine without prime-rl's settings / a frozen endpoint not serving `/inference/v1/generate` with `token_id:N` | `MalformedGenerateResponseError` → 502, turn resampled or rollout dies |
| R19, X16 | Routed-experts object | P/D or llm-d path returning another form; dtype mismatch across packed samples | parse failure, or packer refuses to co-pack; replay on + missing → trainer `ValueError` |
| R16, X18 | `max_model_len` preflight | router doesn't proxy `/v1/models`, or engine restarted with a new cap | pre-flight silently off (overlong prompts reach the engine) or stale cap |
| R13, R31, S5 | `cache_salt` per group, cache never reset | members/turns after a swap | no error; larger $\lvert\log r\rvert$, more IPO/IcePop masking, higher `mismatch_kl`; $\mu$ ≠ any $\pi_v$ |
| R33 | `/finish_session` needs `admin_base_url` | bare router `base_url` | sticky sessions never released (router-side growth) `[router internals UNVERIFIED]` |
| R30 | Trainer temperature from config | a trainable seat with its own `sampling.temperature` | trainer divides by the wrong $T$; silent ratio bias; `temperature=0` (untruncated) → inf/NaN logprobs |
| A3, W8 | `/init_broadcaster` + $W$ | local single-node DP auto-fill undercounts $W$ | worker rank ≥ world → assert → 500 **swallowed**; first `/update_weights` fails → retries → orchestrator dies at startup sync |
| A12, W7 | Admin URL order = rank order; uniform GPUs per URL | reordered/non-uniform `ADMIN_URLS` | NCCL ranks collide or leave holes → group never forms / hangs at `init_broadcaster` (no client timeout) |
| A6 | `/update_weights` retry | a transfer > 720 s under NCCL/NIXL | retry queues behind the collective; engine frozen until 1440 s; orchestrator dies; trainer then times out |
| A5 | `/pause` semantics | client expects drain (per stale comments) | requests continue after resume on new weights over old KV; no abort `[vLLM 0.24]` |
| A13, S13 | Drop-before-pause via bounded `out_q` | `out_q` full during a ship-gate hold with a stale group pending | watcher blocks in `drop_group`, version never advances, dispatch frozen; trainer fails after 1200 s with "No receiver joined the broadcast" |
| S10 | Ship-gate hold has no health check | watcher dies during a hold | orchestrator hangs until the trainer's 1200 s handshake timeout |
| W2, S14 | Lockstep marker handshake | consumer dead/slow (e.g. online eval mid-epoch), or a skipped version | trainer `TimeoutError "No receiver joined the broadcast within 1200s"` |
| W4 | FS `.finished` wait has no timeout | trainer dies mid-export | watcher stranded; dispatch blocked (update barrier) |
| W9, W10 | NCCL metadata pickle / torch dtypes | incompatible torch between trainer and vLLM | unpickle failure or tensor bytes misread as metadata |
| W12 | One-shot NCCL group | restart of only trainer or only inference | cannot rejoin; hang at startup broadcast |
| W13-W15 | FS layout | reader expects `config.json` or an index for one shard | only relevant to external readers; vLLM workers load fine |
| W16-W18 | LoRA PEFT layout | NCCL/NIXL with LoRA, `modules_to_save`, `max_loras>1`, `api_server_count>1` | config-time rejection; otherwise adapter not found (404 retried 120 s) |
| W24 | NIXL static plan + default `qkv` fusion | ≥2-rank trainer with fusions on | **silent**: inference keeps startup q/k/v forever; `shard_fused_on_dim1` → plan-build error "no trainer shard owns element" |
| W24 | NIXL op whitelist / dtypes | loader or conversion uses `cat`/`stack`/`new_empty`, or FP8 weights | `UnsupportedOpError` at trace time (first `/update_weights`) |
| B2 | `num_train_workers` vs trainer DP | too many workers | ZMQ: sender blocks forever at READY; FS: extra `rank_r.bin` silently dropped |
| B2 | same | too few workers | some trainer ranks wait forever for their topic/file |
| B2, B3 | `pad_to_multiple_of` vs CP; cp innermost | mismatch / mesh order change | CP divisibility assertion; wrong `dp_rank` ⇒ ranks read other replicas' batches |
| B4 | Equal micro-batch count & modality per rank | packer change or foreign producer | FSDP collectives hang |
| B6, S10 | PUB HWM vs ship gate | `TARGET_LAG` raised or gate bypassed | silent batch drop past HWM; the trainer then consumes the next message as the current step (ZMQ counters are log-only) |
| B9, B12 | READY sent once | orchestrator-only restart | first `send` blocks forever |
| B10, B11, S22 | No step id on the batch wire | trainer and orchestrator resume from different $N$ (separate `ckpt.output_dir`, orchestrator-only `step_N`) | trainer `FileNotFoundError` on load, or orchestrator `sync_startup` times out after 1200 s waiting for the wrong $v_N$; with ZMQ, batches silently pair with wrong steps |
| §5.1-5.2 | Positional msgspec | orchestrator and trainer on commits with inserted/reordered fields | fields silently misassigned or decode `ValidationError` |
| B5 | Packer truncation | sample longer than `seq_len` | sampled tail and its credit silently dropped |
| F10, F11, S25 | No completeness marker; whole-dir cleanup | crash mid-DCP-save; `--resume.step K` rewind; orchestrator-only `step_N` | `FileNotFoundError` / DCP metadata error on resume; bare `--resume` later jumps to the abandoned timeline |
| F12 | `ckpt.output_dir` split | set at rl level | divergent step sets → startup timeout; orchestrator ckpts never cleaned |
| F13 | Shared `metrics.jsonl` / W&B run | same keys (`time/step`, `time/save_ckpt`) from both producers | interleaved series; disambiguate by `producer` `[W&B merge UNVERIFIED]` |
| V14 | Inference re-applies default env | RL-level `[env_vars]` for a default key | value silently reset inside inference |
| V6 | Monitor rank gate | `RANK` leaks into orchestrator env | orchestrator registers no monitors, silently |
| N4, N5, N10 | Static per-host ports | two runs on one host; metrics server on 8000 | `EADDRINUSE` / `OSError` at startup |
| X25-X28 | Tokenizer identity | local-path `model.name`, `[tokenizer]` override, mismatched teacher/student vocab | `DefaultRenderer` (no tools → 502; no bridge → branch explosion); or silently wrong ids/logprobs |
| X20, X24 | vLLM private-API coupling | vLLM upgrade renames a patched symbol | every vLLM process fails at startup |
| X4, X9 | Env package importable everywhere | package missing on launcher/orchestrator host | `ModuleNotFoundError` at config parse (before anything spawns) |
| F18 | Harbor host paths | external env server on another host | task setup fails reading `task_dir` |
