# Deployment topology & launch — prime-rl @ b944873

> Scope: seam **S1** (launcher → process spawn, GPU/node placement, discovery/rendezvous for single-node, SLURM, k8s) and the launcher side of **S8** (run dir, attempts, resume/clean).
> Files read in full: `src/prime_rl/entrypoints/rl.py` (705), `inference.py` (255), `orchestrator.py` (28), `trainer.py` (26), `env_server.py` (65), `sft.py` (565), `eval.py` (203), `dashboard.py` (127); `src/prime_rl/templates/single_node_rl.sbatch.j2` (126), `multi_node_rl.sbatch.j2` (617), `inference.sbatch.j2` (364), `single_node_sft.sbatch.j2` (42), `multi_node_sft.sbatch.j2` (280), `_launch_rank.sh.j2` (39), `_launch_router.sh.j2` (86), `_mooncake_store.sh.j2` (42), `llmd/{envoy.yaml.j2 (97), endpoints.yaml.j2 (41), epp_pd.yaml.j2 (56), epp_estimate.yaml.j2 (34)}`; `k8s/README.md` (72), `k8s/prime-rl/{Chart.yaml, values.yaml (156), templates/deployment.yaml (378), service.yaml (141), pvc.yaml (16), _helpers.tpl (59), examples/reverse-text.yaml (44), examples/reverse-text/{train,orch,infer}.toml}`; `src/prime_rl/utils/{pathing.py (425), process.py (105), heartbeat.py (53), utils.py (179), nccl.py (42)}`, `src/prime_rl/_compat.py` (54); `Dockerfile.cuda` (297), `scripts/docker-entrypoint.sh` (76), `scripts/install.sh` (149), `install_llmd.sh` (86), `install_modelexpress.sh` (60), `install_nixl_from_source.sh` (96); `pyproject.toml` (356). Plus the callers needed to confirm discovery: `src/prime_rl/orchestrator/envs.py` (50–274), `src/prime_rl/orchestrator/orchestrator.py` (200–259), `src/prime_rl/transports/batch/zmq.py` (1–125), `src/prime_rl/transports/weights/nccl.py` (110–214), `src/prime_rl/inference/vllm/worker/nccl.py` (full), `src/prime_rl/orchestrator/clients.py` (165–203), `src/prime_rl/trainer/ckpt.py:169,332`, `src/prime_rl/orchestrator/ckpt.py:25`, `tests/unit/utils/test_pathing.py`, `tests/unit/test_configs.py`. Docs cross-checked: `docs/scaling.md`, `docs/inference.md`, `docs/configuration.md`, `skills/{configs,install,training/start-run}/SKILL.md`, `examples/extra/dynamo/*`.
> Related docs: `02-config-system.md` (how `RLConfig` resolves and splits), `03-orchestrator.md` (client/admin side, env clients), `05-trainer.md` (checkpoint/resume internals), `06-inference-and-transports.md` (vLLM server, router internals, weight/rollout transports), `08-harnesses-runtimes-serve.md` (env server internals), `11-observability-eval-ops.md` (dashboard, monitors, W&B shared mode).

---

## 1. Mental model

prime-rl is **three independent programs glued by a launcher**: a vLLM-based **inference** server (fronted by a router), an asyncio **orchestrator** (drives envs, builds batches, controls weight updates on inference), and a torchrun-launched FSDP **trainer**. A fourth kind of process — one **env server** per environment source — hosts rollouts (agent/harness execution) and is spawned next to the orchestrator. Every one of these is a standalone console script (`pyproject.toml:40-48`: `rl`, `sft`, `inference`, `trainer`, `orchestrator`, `eval`, `env-server`, `dashboard`) that parses its own config with `pydantic-config` (`@ file` + CLI flags). The `rl` launcher's whole job is: parse one `RLConfig`, split it into per-component **resolved JSON configs** on disk, then start each component with `<cmd> @ <component>.json` plus the right env vars / `CUDA_VISIBLE_DEVICES`.

There are three placement backends. **Local single-node** (`rl_local`, `src/prime_rl/entrypoints/rl.py:116-423`) is a Python supervisor that `Popen`s every component on one host and partitions GPUs by `CUDA_VISIBLE_DEVICES`. **SLURM** (`rl_slurm`, `rl.py:585-644`) renders a Jinja2 sbatch template and `sbatch`es it; the single-node template simply re-runs `uv run rl` on the compute node (which then takes the local path), while the multi-node template is a hand-written bash orchestration of per-node roles. **Kubernetes** is a thin Helm chart (`k8s/prime-rl/`) that creates three StatefulSets (orchestrator / inference / trainer) sharing a ReadWriteMany PVC — it does **not** use the `rl` launcher at all; you run each component yourself.

Discovery is deliberately primitive and mostly **static**: well-known ports on known hostnames, injected either as config fields in the resolved JSON (single-node: `localhost`) or as CLI overrides computed in bash from `scontrol show hostnames` (multi-node). The only dynamic discovery is (a) env servers bind an **OS-assigned port** and publish it to an **address file** the orchestrator polls, and (b) Dynamo worker discovery (inference side, E owns). Everything else — router URL, per-rank admin URLs, NCCL/ModelExpress rendezvous host, ZMQ rollout host, torchrun master — is computed by the launcher/template and passed as flags. Think of the resolved config directory `configs/attempt_N/resolved/` as the **contract between processes**, and the run directory as the **shared bus** (filesystem transports, address files, done-markers, checkpoints).

## 2. Where it runs

### 2.1 Process inventory — local single-node `uv run rl @ rl.toml`

```
uv run rl  ─────────────────────────────── PRL::Launcher  (rl.py:699-701; supervisor, no GPU)
 ├─ inference @ <cfg>/inference.json ───── PRL::Inference  CUDA_VISIBLE_DEVICES=<first num_infer_gpus>   (rl.py:204-234)
 │    ├─ vllm-router --port 8000 --worker-urls http://localhost:8100 ...                      (inference.py:165-191)
 │    └─ vLLM OpenAI server on :8100 (api servers ×api_server_count, engine cores ×dp, TP workers; E owns)
 ├─ env-server @ <cfg>/envs/<split>/<name>.json ── PRL::EnvServer, one per launcher-managed source  (rl.py:262-292)
 │    └─ verifiers env worker pool (G owns)            binds tcp://127.0.0.1:<OS port> → writes <name>.address
 ├─ orchestrator @ <cfg>/orchestrator.json ── PRL::Orchestrator  (no CUDA_VISIBLE_DEVICES override)  (rl.py:294-325)
 └─ torchrun --role=trainer --nproc-per-node=<num_train_gpus> -m prime_rl.trainer.rl.train @ <cfg>/trainer.json
      └─ N trainer ranks (PRL::Trainer)     CUDA_VISIBLE_DEVICES=<next num_train_gpus>        (rl.py:327-376)
 (+ detached `dashboard` daemon, one per user/host, start_new_session=True)                (dashboard.py:78-114)
```

Start order in `rl_local`: inference → env servers → orchestrator → trainer (`rl.py:202-376`). Nothing waits for readiness in the launcher; readiness is pushed down: the orchestrator waits for inference (`ClientConfig.wait_for_ready_timeout = 3600`, `packages/prime-rl-configs/src/prime_rl/configs/shared.py:174`) and for each env server's address file + health (`ENV_SERVER_STARTUP_TIMEOUT = 600.0`, `src/prime_rl/orchestrator/envs.py:42,95-104`); the trainer and orchestrator rendezvous over the rollout transport's READY barrier (ZMQ, §3.7) and the weight broadcast handshake.

Frozen-model endpoints (`FrozenModelConfig`, e.g. an OPD teacher or SFT-distill sampling source) are **never started** by the launcher — it only logs them (`rl.py:244-257`). If `[inference]` is absent, no policy server is started either and a warning says the orchestrator will hang waiting for `orchestrator.model.client.base_url` (`rl.py:235-242`) — this is how the Dynamo example runs (`examples/extra/dynamo/README.md:65-71`).

### 2.2 Lifecycle and supervision (local)

- **Start**: each child gets `stdout`/`stderr` redirected to its own log file (`rl.py:208-222`; comment "If we don't log stdout, the server hangs"). A daemon `Thread(target=monitor_process)` per child does `process.wait()`, appends `RuntimeError("<Name> failed with exit code N")` to a shared `error_queue` if non-zero, then sets that child's `stop_event` (`src/prime_rl/utils/process.py:96-105`). Because stderr is a file, `process.stderr` is `None` and no stderr text is appended.
- **Steady state**: main thread polls every 1 s (`rl.py:383-393`): any entry in `error_queue` → log, `cleanup_threads`, `cleanup_processes`, `sys.exit(1)`.
- **Success condition**: *both* trainer and orchestrator stop events set (`rl.py:381-383`), then both return codes must be 0 (`rl.py:395-406`); then inference and env servers are torn down (`rl.py:410-412`).
- **Failure of a non-terminal child with exit 0** (e.g. inference or an env server exiting cleanly) is *not* an error — the run keeps going and the orchestrator will fail/hang on its own.
- **Signals**: `SIGTERM` handler and `KeyboardInterrupt` both call cleanup and `exit(1)` (`rl.py:194-200,414-418`); any other exception cleans up and re-raises (`rl.py:419-423`).
- **Teardown**: `cleanup_processes` walks each live child's **whole process tree** with `psutil` (children first, then parent), sends `SIGTERM`, waits up to 60 s, then `SIGKILL`s the tree (`process.py:64-93`). This reaches grandchildren behind `uv`/`torchrun` regardless of process groups. Monitor threads are joined with a 5 s timeout (`process.py:58-61`).

### 2.3 Process inventory — multi-node SLURM (`deployment.type = "multi_node"`)

One sbatch job with `num_train_nodes + total_infer_nodes` nodes (the template's `num_infer_nodes` is `deployment.total_infer_nodes` = per-replica nodes × `num_infer_replicas`, `rl.py:549`), `--ntasks-per-node=1 --gres=gpu:<gpus_per_node> --exclusive` (`multi_node_rl.sbatch.j2:3-22`). Node order is `scontrol show hostnames $SLURM_JOB_NODELIST`; **inference nodes come first**, trainer nodes after (`multi_node_rl.sbatch.j2:85-90`). One `srun --kill-on-bad-exit=1 bash -s` task per node runs the role logic (`:233`).

```
HOSTNAMES = [I_0 ... I_{NI-1} | T_0 ... T_{NT-1}]           SLURM_PROCID == index (assumed; see §9)

I_0   : router (vllm-router, or llm-d EPP+Envoy) on ROUTER_PORT          (_launch_router.sh.j2; :459-462)
        + dp_per_node vLLM API servers on BACKEND_PORT+k (TP slice each)  (:464-488)
        + [mooncake_master + metadata server if kv_offload=mooncake]      (_mooncake_store.sh.j2:25-33)
I_j   : dp_per_node vLLM API servers on BACKEND_PORT+k                   (+ mooncake_client)
I_last: [orchestrator + env servers, iff deployment.orchestrator_on_inference]   (:158)
T_0   : torchrun node_rank 0 (MASTER_ADDR, rdzv :29500; NCCL weight-broadcast TCPStore :29501 on trainer rank 0)
        + orchestrator + env servers (default ORCH_PROCID = NUM_INFER_NODES)     (:153-161, :529-574)
        + [ModelExpress server + redis, iff weight_broadcast=nixl and slurm.launch_modelexpress]  (:268-313)
T_i   : torchrun node_rank i
```

P/D disaggregated inference replaces the inference-node block with prefill/decode role groups (§3.4.4). With `num_infer_nodes = 0` (fake-data trainer benchmark) there is no router/orchestrator at all (`rl.py` config `validate_deployment`, `packages/prime-rl-configs/src/prime_rl/configs/rl.py:338-356`).

### 2.4 Lifecycle — multi-node SLURM

Batch-script phase (runs once on the batch host): `set -e`, export topology vars, compute URLs, `uv sync` (once, or per node via `srun` when `slurm.shared_fs = false`), optional `pre_run_command`, then a **per-node cleanup** `srun` that `pkill -9`s stale `python.*prime_rl`, `torchrun`, `vllm*`, llm-d, mooncake, (modelexpress/redis) processes and wipes `/dev/shm/vllm-*`, `/tmp/vllm-*`, `/tmp/torch-*`, `/tmp/torchelastic_*` (`multi_node_rl.sbatch.j2:163-221`).

Launch phase (per node, inside `srun --kill-on-bad-exit=1`): every long-lived component is backgrounded with `... | tee` under `set -o pipefail` (`:237`). **Completion** is signalled through the shared FS: trainer node-rank 0 touches `$LAUNCHER_DIR/.trainer.done` on exit 0; the orchestrator touches `.orchestrator.done` on exit 0 (`:72-77,521-525,568-572`). Every node runs a watcher `( until [ -f TRAINER_DONE ] && [ -f ORCHESTRATOR_DONE ]; do sleep 10; done ) &` (`:580-585`) and loops on `wait -n -p FINISHED_PID` until either some background job exits non-zero or the watcher finishes (`:590-596`).

- Clean completion → each node `kill -TERM`s its remaining jobs and exits 0 (`:612-616`).
- **Any non-zero exit** of any backgrounded job on any node (trainer, orchestrator, router, a vLLM rank, an env server) → that node **sleeps `slurm.cleanup_grace_period` (default 3600 s)** without signalling anything, so in-flight checkpoints on it and on peers can flush, *then* kills its jobs and exits non-zero; `srun --kill-on-bad-exit=1` then reaps every other node (`:598-611`; `shared.py:112-113`). Consequence: a crashed router or env server keeps the whole allocation idle for up to an hour by default.

Standalone `inference.sbatch.j2` uses the simpler `wait -n` → kill-all → exit (`inference.sbatch.j2:353-363`) — no done-files, no grace period.

### 2.5 Kubernetes

Three StatefulSets `<release>-{orchestrator,inference,trainer}` (`k8s/prime-rl/templates/deployment.yaml:1-378`), each with a headless service `<release>-<role>-headless` (`clusterIP: None`) plus an optional ClusterIP service (`service.yaml:1-141`), all mounting one PVC `<release>-shared-data` (RWX, `storageClassName: nfs`, `1Ti`, mounted at `/data`; `pvc.yaml:1-16`, `values.yaml:14-22`). Containers run `sleep infinity` unless `<role>.autoStart: true`, in which case `/bin/bash -c <role>.command` (`deployment.yaml:33-39,139-145,266-272`). The image's entrypoint (`scripts/docker-entrypoint.sh`) raises `ulimit -n 32000`, optionally clones/syncs an override source (`PRIME_RL_REF`, `PRIME_RL_REPO`) or reinstalls verifiers (`VERIFIERS_VERSION`), then `exec "$@"`. Details and limitations in §3.6 and §7.

## 3. Mechanics

### 3.1 `rl()` — the top-level launch path (`rl.py:647-705`)

1. `main()` sets the process title `PRL::Launcher` and calls `rl(cli(RLConfig))` (`rl.py:699-701`). All config resolution (shared-field propagation, GPU/DP auto-fill, run-name generation) happens inside `cli()` → `RLConfig.model_validate` (see `02-config-system.md`).
2. **Run identity** (runtime-only, never in config): `os.environ.setdefault("PRL_RUN_ID", uuid4().hex)`; `os.environ["PRL_RUN_NAME"] = config.run.name` (`rl.py:652-654`). Every spawned child inherits both (they are `PROTECTED_ENV_VARS`, `shared.py:13-24`).
3. **Run-dir guard** `validate_run_dir(run_dir, output_dir, resuming, clean, ckpt_output_dir)` (`rl.py:656-661`; `src/prime_rl/utils/pathing.py:339-376`):
   - resuming → no checks at all;
   - `clean` (and not `$NEVER_CLEAN`) → `rmtree(run_dir)` and `rmtree(ckpt_output_dir)` if distinct; refuses if `run_dir` is not under `output_dir`;
   - otherwise raise `FileExistsError` if the run dir has artifacts beyond `configs/`, `launcher/`, or an empty `logs/` tree (`has_run_artifacts`, `pathing.py:320-336`), or if a separate `ckpt.output_dir` already has checkpoints.
4. `mkdir` run dir and `ckpt.output_dir` (`rl.py:662-664`).
5. **Resume-step resolution & stale-step cleanup** (`rl.py:671-686`): if `resume.dir` → step parsed from `step_<N>`; else `resume.step`; else latest `step_*` under `get_ckpt_dir(ckpt.output_dir or run_dir)` = `<base>/checkpoints` (`pathing.py:257-311`). Then `clean_future_steps(run_dir, resume_step)` deletes `batches/step_k` for `k > resume_step` and `broadcasts/step_k` for `k >= resume_step`; from scratch it passes `-1` and wipes all of both (`pathing.py:379-397`). **The resolved step is used only for cleaning** — it is *not* written into the sub-configs; trainer and orchestrator each re-resolve "latest" themselves (§5).
6. Unless `dry_run`: `pre_download_model(trainer.model.name, skip_weights=trainer.model.debug.random_init)` → `huggingface_hub.snapshot_download` (skipped for local paths) (`rl.py:688-691`; `src/prime_rl/trainer/model.py:69-88`).
7. Dispatch: `rl_slurm` if `config.slurm` else `rl_local` (`rl.py:693-696`).

### 3.2 Local launch `rl_local` (`rl.py:116-423`)

**Attempt dirs & configs.** `prepare_attempt_dirs(run_dir)` either reuses the pinned attempt from `$PRL_ATTEMPT_CONFIG_DIR`/`$PRL_ATTEMPT_LOG_DIR` (both or neither) or creates `configs/attempt_{n+1}/resolved/` and `logs/attempt_{n+1}/` (n = max over both trees) and atomically repoints `configs/latest` and `logs/latest` symlinks (`pathing.py:21-72`). `write_launch_artifacts` writes `configs/attempt_N/command.txt` (`shlex.join(["uv","run","rl",*argv])`, only if absent) and copies root `@ *.toml` files to `configs/attempt_N/rl.toml` (multiple files concatenated with `# @ <path>` headers — a record, not necessarily re-loadable TOML) (`pathing.py:172-219`). `write_subconfigs` dumps (`rl.py:89-113`):

| File | Content |
|---|---|
| `trainer.json` | `dump_resolved_config(config.trainer)` |
| `orchestrator.json` | `dump_resolved_config(config.orchestrator)` |
| `inference.json` | `config.inference` minus `{deployment, slurm, output_dir, dry_run}`; `router = None` iff `deployment.type == "multi_node"` |
| `envs/<split>/<name>.json` | per launcher-managed source: `{env, serve, address_file, log:{level: orchestrator.log.vf_level, json_logging}}` (`pathing.py:230-246`) |

Launcher-managed sources are `orchestrator.env_sources` (train then eval) whose `serve.address is None` (`rl.py:55-61`; `configs/orchestrator.py:788-803`). `dry_run` returns right after writing configs (`rl.py:129-131`).

**GPU assignment** (`rl.py:144-162`): `num_infer_gpus = deployment.num_infer_gpus if inference else 0`. Local ids `[0, num_infer_gpus)` → inference, `[num_infer_gpus, num_infer_gpus + num_train_gpus)` → trainer. Physical ids come from `get_physical_gpu_ids()`: parse `$CUDA_VISIBLE_DEVICES` as a comma-separated list of **ints**, else `pynvml.nvmlDeviceGetCount()` (`process.py:36-44`). Requesting more than available raises. Leftover GPUs are unused. Orchestrator and env servers get **no** `CUDA_VISIBLE_DEVICES` override (they inherit the launcher's).

**Port sanity check**: if `[inference]` is set, the port in `orchestrator.model.client.base_url` must equal `inference.server.port` (`rl.py:176-186`). The host is not checked.

**Env layering** for every child: `{**os.environ, **DEFAULT_COMMON_ENV_VARS, [**DEFAULT_<ROLE>_ENV_VARS], **config.env_vars, **config.<component>.env_vars, <launcher-forced vars>}` — later wins (`rl.py:212-219,272-277,302-312,351-363`). **Exception — inference re-applies defaults in-process:** `inference_local` does `os.environ.update({**DEFAULT_COMMON_ENV_VARS, **DEFAULT_INFERENCE_ENV_VARS, **config.env_vars})` where `config.env_vars` is only `inference.env_vars` (`entrypoints/inference.py:210`). So an RL-level `[env_vars]` entry that collides with a default key (`CUDA_DEVICE_ORDER`, `PYTHONUNBUFFERED`, `OMP_NUM_THREADS`, `GIT_LFS_SKIP_SMUDGE`, `VLLM_WORKER_MULTIPROC_METHOD`, `PYTORCH_CUDA_ALLOC_CONF`, `VLLM_ENGINE_READY_TIMEOUT_S`, `UCX_TLS`) is silently reset inside the inference process (local and SLURM alike); put such overrides in `[inference.env_vars]`.

| Default set (`process.py:17-33`) | Vars |
|---|---|
| `DEFAULT_COMMON_ENV_VARS` (all) | `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `PYTHONUNBUFFERED=1`, `OMP_NUM_THREADS=1`, `GIT_LFS_SKIP_SMUDGE=1` |
| `DEFAULT_TRAINER_ENV_VARS` | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| `DEFAULT_INFERENCE_ENV_VARS` | `VLLM_WORKER_MULTIPROC_METHOD=spawn`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`, `VLLM_ENGINE_READY_TIMEOUT_S=4200`, `UCX_TLS=all` |

Launcher-forced (not overridable, enforced by `reject_protected_env_vars`, `shared.py:27-36`): `CUDA_VISIBLE_DEVICES` (inference, trainer), `WANDB_SHARED_MODE=1`, `WANDB_RUN_ID=$PRL_RUN_ID`, `WANDB_SHARED_LABEL=orchestrator|trainer` (orchestrator, trainer); plus `LOGURU_FORCE_COLORS=1`, `WANDB_PROGRAM="uv run rl"`, `WANDB_ARGS=json(sys.argv)` (these three can be overridden by env_vars since they come earlier) (`rl.py:171-174,302-312,351-363`). Env servers get only common defaults + `env_vars` + `orchestrator.env_vars` (`rl.py:272-277`).

**Trainer command** (`rl.py:330-345`):
```
torchrun --role=trainer --rdzv-endpoint=localhost:<get_free_port()> --rdzv-id=<uuid4 hex>
         --log-dir=<logs>/trainer/torchrun --local-ranks-filter=<trainer.log.ranks_filter, default 0>
         --redirect=3 --tee=3 --nproc-per-node=<num_train_gpus>
         -m prime_rl.trainer.rl.train @ <cfg>/trainer.json
```
`get_free_port()` binds port 0, reads it, closes the socket (`src/prime_rl/utils/utils.py:159-167`) — a TOCTOU window. No `--rdzv-backend` is passed, so torchrun's default `static` backend applies (torch 2.11 `torch/distributed/run.py:446-450,778-781,867-868`): `--rdzv-endpoint` is simply the rank-0 TCPStore `MASTER_ADDR:PORT`, `--node-rank` is honoured and `--rdzv-id` is unused (same for multi-node). `--redirect=3` works only as an argparse prefix abbreviation of torchrun's `--redirects`. All rank output goes to `logs/attempt_N/trainer/torchrun/<...>/<rank>/std*.log`; the console tee of ranks in `ranks_filter` lands in `logs/.../trainer.log`.

**Inference command**: `inference @ <cfg>/inference.json` (console script). Inside (`src/prime_rl/entrypoints/inference.py:194-239`): apply `{DEFAULT_COMMON, DEFAULT_INFERENCE, config.env_vars}` to `os.environ`; `setup_vllm_env(config)`; if `config.router` is set, `start_router` spawns
```
vllm-router --policy <router.policy, default sticky_least_loaded> --host <server.host or 0.0.0.0> --port <server.port>
  --worker-urls http://<localhost|server.host>:<backend_port> --intra-node-data-parallel-size <dp_local or dp>
  --request-id-headers x-session-id --request-timeout-secs <router.request_timeout_secs=14400>
  --worker-startup-timeout-secs 4200 --prometheus-port <server.port + 21000>
```
(`inference.py:165-191`), then rewrites `config.server.port = config.backend_port` so the vLLM server binds behind the router (`:220`), and starts a watcher thread that `SIGTERM`s its own process if the router dies (`:222-228`). The llm-d router is rejected for single-node (`configs/inference.py:526-532`). The engine is started in-process by `prime_rl.inference.vllm.server.server(config)` (E owns).

**Env server command**: `env-server @ <cfg>/envs/<split>/<name>.json` (`rl.py:262-292`). Inside (`src/prime_rl/entrypoints/env_server.py:34-61`): `os.environ.setdefault("VF_RUN_ID", $PRL_RUN_ID or uuid)` (an inherited `VF_RUN_ID` wins) so verifiers keys run-scoped limiters by run; sandbox base labels = `[$PRL_RUN_NAME]`; `serve_env(**pool_serve_kwargs(serve.pool), address=serve.address or "tcp://127.0.0.1:0", address_queue, log_setup, config_data, max_concurrent)`; a daemon thread takes the bound address from the queue and writes it atomically (`tmp.write_text` + `replace`) to `address_file` (`env_server.py:24-31`).

**Orchestrator command**: `orchestrator @ <cfg>/orchestrator.json` → `asyncio.run(run_orchestrator(config))` (`src/prime_rl/entrypoints/orchestrator.py:19-24`). Heavy imports are deferred so `--help` is fast (same pattern in `entrypoints/trainer.py`).

### 3.3 SLURM single-node (`rl_slurm` + `single_node_rl.sbatch.j2`)

`rl_slurm` (`rl.py:585-644`): create attempt dirs; write **only** `rl.json` (whole `RLConfig` minus `{slurm, dry_run, clean}`), render `single_node_rl.sbatch.j2` to `<run_dir>/launcher/rl.sbatch`, and unless `dry_run`, start the dashboard and `subprocess.run(["sbatch", script])`. The template (`single_node_rl.sbatch.j2:1-126`): `#SBATCH --nodes=1 --gres=gpu:<gpus_per_node> --exclusive`, output to `<run_dir>/launcher/logs/job_%j.log`; exports `PRL_ATTEMPT_CONFIG_DIR/LOG_DIR` (pinning the attempt), `cd project_dir`, sources `.env`, activates `.venv`, `uv sync --all-extras --all-packages`, runs `pre_run_command`, sets `PRL_RUN_ID` if unset, optionally boots ModelExpress+redis locally for NIXL (`:43-125`), and finally **`uv run rl @ <cfg>/rl.json`** (`:126`). On the compute node `slurm` is absent from `rl.json`, so the re-invoked `rl` takes the **local** path above, reusing the pinned attempt (`prepare_attempt_dirs`) — i.e. single-node SLURM == local launch inside an allocation — with one difference: the compute node re-parses the *already-resolved* `RLConfig` from `rl.json`, so every phase-2 validator runs a second time on its own output (this is why the single-node NCCL world size is correct under SLURM but not locally, §7.5), and `rl()`'s `validate_run_dir`/`clean_future_steps`/`pre_download_model` run again on the compute node. Note the nested `inference.slurm` survives in `rl.json` (propagated from the top-level `[slurm]`, see `02`), but is stripped again from `inference.json`.

### 3.4 SLURM multi-node (`multi_node_rl.sbatch.j2`)

`rl_slurm` writes the full per-component subconfigs (`write_subconfigs`, with `inference.json.router = None` so per-rank processes run bare engines) and renders the template with the variables of `rl.py:538-578` (or `:491-537` for disaggregated): node counts, `gpus_per_node`, `router`, `router_port = inference.server.port`, `backend_port`, `inference_tp`, `inference_enable_expert_parallel`, `inference_data_parallel_rpc_port`, `dp_per_node = gpus_per_node // tp`, mooncake vars, `use_nccl_broadcast`, `use_zmq_transport`, `use_nixl_broadcast`, `launch_modelexpress`, `modelexpress_{host,port,redis_port}` (redis 6379, or 6380 if the MX port is 6379, `rl.py:476`), `ranks_filter`, `orchestrator_on_inference`, per-component env-var dicts (defaults ⊕ `env_vars` ⊕ component `env_vars`, `rl.py:448-459`), and `train_env_names`/`eval_env_names`.

#### 3.4.1 Topology variables & URLs (batch host)

`HOSTNAMES=(scontrol show hostnames)`; `INFER_HOSTS=HOSTNAMES[0:NI]`; `TRAIN_HOSTS=HOSTNAMES[NI:]` (`:85-90`).
- `INFER_URLS = http://${INFER_HOSTS[0]}:${ROUTER_PORT}/v1` — the **single global router** on inference node 0 (`:94`).
- Non-disaggregated: for every inference node `n` and local DP rank `k < INFERENCE_DP_LOCAL = GPUS_PER_NODE / INFERENCE_TP`: `ADMIN_URLS += http://host_n:(BACKEND_PORT+k)/v1`, `ROUTER_ARGS += http://host_n:(BACKEND_PORT+k)` (`:117-124`).
- Disaggregated: per replica, all prefill nodes' `PREFILL_PORT+k` then all decode nodes' `DECODE_PORT+k`, as `--prefill`/`--decode` router args (`:98-115`).
- `MASTER_ADDR = HOSTNAMES[NUM_INFER_NODES]` (= trainer node 0), `MASTER_PORT = 29500` (`:137-138`).
- `ORCH_PROCID = NUM_INFER_NODES` (trainer node 0) or `NUM_INFER_NODES - 1` (last inference node) with `orchestrator_on_inference`; `ORCH_ADDR = HOSTNAMES[ORCH_PROCID]` (`:158-161`).
- NIXL: `WEIGHT_BROADCAST_HOST = MASTER_ADDR` if the job launches ModelExpress, else the configured host; `MODEL_EXPRESS_PORT`, `MODEL_EXPRESS_REDIS_PORT` (`:141-151`).
- Also exported: `PRL_RUN_ID` (`${PRL_RUN_ID:-uuid}` — the launcher's value normally arrives through sbatch's inherited environment), `PRL_RUN_NAME`, `WANDB_RUN_ID=$PRL_RUN_ID`, `WANDB_SHARED_MODE=1` (`:79-82`); `PRL_ATTEMPT_CONFIG_DIR/LOG_DIR` (`:63-64`). Log symlinks `logs/.../trainer.log → trainer/node_0.log`, `inference.log → inference/node_0.log` (`:65-67`).

#### 3.4.2 Inference node role (`SLURM_PROCID < NUM_INFER_NODES`, `:331-490`)

Exports inference env vars (defaults ⊕ config), `LOCAL_IP=$(hostname -I | awk '{print $1}')`, `ulimit -l unlimited` if KV offload, includes the mooncake store block (§3.5), computes `REPLICA_IDX = rank / NODES_PER_INFER_REPLICA`, `RANK_IN_REPLICA`. `NCCL_IB_HCA` is set from `ibv_devinfo` InfiniBand HCAs on every node (`:248-250`). Then (non-disaggregated, `:459-488`):
- node 0 calls `launch_router regular "$ROUTER_ARGS" "$ROUTER_PORT" <policy> <log>/inference/router.log`;
- for each local DP rank `d`: `RANK_GPUS = d*TP .. d*TP+TP-1`; if EP enabled and the replica has >1 DP rank, launch an **external-LB DP rank** of a replica-wide DP group (`data_parallel_rank = RANK_IN_REPLICA*DP_LOCAL + d`, `data_parallel_address` = own `LOCAL_IP` on the replica head node else the replica head hostname, group size `NODES_PER_INFER_REPLICA*DP_LOCAL`, rpc port `INFERENCE_DATA_PARALLEL_RPC_PORT`); otherwise an independent single-engine server (`dp=1`).

`launch_inference_rank <port> <gpus> <dp> <rpc> <extra_json> <nixl_port> <log>` (`_launch_rank.sh.j2:15-39`) runs
```
CUDA_VISIBLE_DEVICES=<gpus> [VLLM_NIXL_SIDE_CHANNEL_PORT=<nixl>] VLLM_RPC_BASE_PATH=/tmp/vllm-rpc-$USER-$SLURM_JOB_ID-<port> \
  uv run inference @ $CONFIG_DIR/inference.json --server.host 0.0.0.0 --server.port <port> \
     --vllm.data-parallel-size <dp> --vllm.data-parallel-size-local 1 --vllm.api-server-count 1 \
     [--vllm.data-parallel-rpc-port <rpc>] [--vllm @ <rpcbase>/vllm-overrides.json] 2>&1 | tee -a <log> &
```
Per-rank vLLM JSON overrides (e.g. `data_parallel_rank`, `data_parallel_address`, role `all2all_backend`, `compilation_config`) go through a nested `--vllm @ file` because dict values cannot be inlined as flags. Rank 0 logs to `inference/node_<n>.log`, rank k>0 to `inference/node_<n>_rank<k>.log` (`_launch_rank.sh.j2:7-13`). Each rank process is the `inference` entrypoint with `router = None` → a bare vLLM server on `--server.port`.

#### 3.4.3 Trainer node role (`:491-527`)

`TRAIN_NODE_RANK = SLURM_PROCID - NUM_INFER_NODES`; exports trainer env vars; then
```
WANDB_SHARED_LABEL=trainer uv run torchrun --role=trainer --nnodes=$NUM_TRAIN_NODES --nproc-per-node=$GPUS_PER_NODE \
  --node-rank=$TRAIN_NODE_RANK --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT --rdzv-id=job_$SLURM_JOB_ID \
  --log-dir=$LOG_DIR/trainer/torchrun --tee=3 --redirects=3 --local-ranks-filter=<ranks_filter> \
  -m prime_rl.trainer.rl.train @ $CONFIG_DIR/trainer.json \
  [--weight-broadcast.host $WEIGHT_BROADCAST_HOST]   (nixl) \
  [--rollout_transport.host $ORCH_ADDR]              (zmq) \
  2>&1 | sed -u 's/^\[[a-zA-Z]*[0-9]*\]://' | tee -a $LOG_DIR/trainer/node_$TRAIN_NODE_RANK.log
```
All GPUs of every trainer node are trainer ranks. For NCCL broadcast, the resolved `trainer.json` already carries `weight_broadcast.host = "0.0.0.0"` (set by `RLConfig.auto_setup_deployment`, `configs/rl.py:778-792`) so trainer rank 0 binds the TCPStore on all interfaces.

#### 3.4.4 Disaggregated P/D inference (`:356-457`)

Per replica (`NODES_PER_INFER_REPLICA = prefill_nodes + decode_nodes` of one island): nodes with `RANK_IN_REPLICA < NUM_PREFILL_NODES` are **prefill**, others **decode**; each role group is subdivided into sub-replicas of `nodes_per_{prefill,decode}_replica`. Each role sub-replica is one external-LB DP group rooted at its head host (`DP = nodes_per_role_replica * DP_LOCAL`, `data_parallel_rank = ROLE_RANK*DP_LOCAL + k`), ports `PREFILL_PORT+k` (8100) / `DECODE_PORT+k` (8200), NIXL side-channel `VLLM_NIXL_SIDE_CHANNEL_HOST=$LOCAL_IP`, port `5600+k`. Roles get `all2all_backend = deepep_high_throughput` (prefill) / `deepep_low_latency` + `compilation_config={"cudagraph_mode":"FULL_DECODE_ONLY"}` (decode), merged with `prefill_vllm_overrides`/`decode_vllm_overrides`; decode nodes force `UCX_NET_DEVICES=mlx5_0:1` (hard-coded). `GLOO_SOCKET_IFNAME` is derived from the interface owning `LOCAL_IP`; `UCX_NET_DEVICES` from active InfiniBand ports (fallback RoCE, fallback `all`). Router node 0 runs `launch_router pd`. With llm-d, **every decode node** also runs `pd-sidecar --port DECODE_SIDECAR_PORT(8300)+rank --vllm-port DECODE_PORT --kv-connector nixlv2 ...` (`:426-445`). The KV transfer connector itself is built by `InferenceConfig.to_namespace()` from `use_pd_kv_transfer` (E owns). The orchestrator's `inference_metrics_roles` is auto-filled to match `ADMIN_URLS` order (`configs/rl.py:813-821`).

#### 3.4.5 Router launch (`_launch_router.sh.j2`)

- **vllm-router**: `vllm-router --policy <policy> --host 0.0.0.0 --port <port> --request-id-headers x-session-id --request-timeout-secs <..> --prometheus-port <port+21000> --worker-startup-timeout-secs 4200 --log-level debug` plus `--worker-urls <per-rank urls>` or `--vllm-pd-disaggregation --prefill ... --decode ...` (`:71-85`). Note: unlike the local launcher, no `--intra-node-data-parallel-size` — every DP rank is its own endpoint ("external LB").
- **llm-d**: renders `epp.yaml` (`llmd/epp_estimate.yaml.j2` or `epp_pd.yaml.j2`), `endpoints.yaml` (one entry per DP rank; decode entries point at the **sidecar** port) and `envoy.yaml` into `$LOG_DIR/inference/llmd/`; replaces `__LLMD_ADDR_<i>__` placeholders with each host's IPv4 from `getent hosts` (file-discovery rejects hostnames); runs `epp --pool-name prime-rl --pool-namespace slurm --config-file ... --grpc-port 9002 --grpc-health-port 9013 --metrics-port 9090` and `envoy -c envoy.yaml` (`:11-70`). Envoy listens on `router_port`, routes `/v1/` and `/inference/v1/` to an `ORIGINAL_DST` cluster keyed by the `x-gateway-destination-endpoint` header that EPP sets via `ext_proc` (gRPC to `127.0.0.1:9002`), admin on `127.0.0.1:9901`, circuit breakers raised to 100000 (`llmd/envoy.yaml.j2:5-97`). Binaries come from `scripts/install_llmd.sh` into `third_party/llmd/bin/` (fork `S1ro1/llm-d-router@1ca4243` for token-in P/D, Envoy 1.36.0 extracted with `crane`).

#### 3.4.6 Orchestrator + env servers (`:528-575`)

On `SLURM_PROCID == ORCH_PROCID` (top-level, so the `wait -n` covers it): exports orchestrator env vars; starts **one `uv run env-server @ $CONFIG_DIR/envs/<split>/<name>.json` per launcher-managed source** in the background (logs to `$LOG_DIR/envs/<split>/<name>.log`); then
```
WANDB_SHARED_LABEL=orchestrator uv run orchestrator @ $CONFIG_DIR/orchestrator.json \
   --model.client.base-url $INFER_URLS --model.client.admin-base-url $ADMIN_URLS \
   [--weight_broadcast.host $MASTER_ADDR (nccl) | $WEIGHT_BROADCAST_HOST (nixl)] [--rollout_transport.host $ORCH_ADDR (zmq)] \
   2>&1 | tee $LOG_DIR/orchestrator.log
```
Env servers bind `127.0.0.1` (§3.2), so they **must** be co-located with the orchestrator — which is why they launch on `ORCH_PROCID`. Role env vars are plain `export`s in the node's shell (`:334-336,498-500,538-540`), so on the default `ORCH_PROCID` (= trainer node 0) the orchestrator and env servers inherit the trainer's exported env (`DEFAULT_TRAINER_ENV_VARS` + `trainer.env_vars`) under their own; with `orchestrator_on_inference` they inherit the inference node's env (incl. P/D role vars). Locally the three sets stay separate.

#### 3.4.7 NIXL / ModelExpress

With `weight_broadcast.type = "nixl"` the template verifies the UCX build under `$PROJECT_DIR/third_party/ucx` has `cuda_copy` and `rc_verbs` transports, and (if `slurm.launch_modelexpress`, default `True`) trainer node 0 starts `redis-server --port <redis_port> --bind 127.0.0.1` and `MX_METADATA_BACKEND=redis ... modelexpress-server --port <mx_port> --database-path $LOG_DIR/modelexpress/models.db`; every node then waits up to 120 s for `WEIGHT_BROADCAST_HOST:MODEL_EXPRESS_PORT` (`multi_node_rl.sbatch.j2:252-327`). Binaries come from `scripts/install_modelexpress.sh` (ModelExpress v0.3.0 via cargo, redis 7.4.2) and UCX/NIXL from `scripts/install_nixl_from_source.sh`.

### 3.5 Mooncake KV-offload store (`_mooncake_store.sh.j2`)

Only with `inference.kv_cache_offload.type = "mooncake"` (SLURM-only; `configs/rl.py:673-685`). Every inference node: `PYTHONHASHSEED=0`, `MOONCAKE_CONFIG_PATH=$OUTPUT_DIR/mooncake/node_<rank>/config.json`. The **first host of the whole job nodelist** (`scontrol ... | head -1`, i.e. inference node 0) runs `mooncake_master -rpc_port=50051 -enable_http_metadata_server ... -http_metadata_server_port=8080 -default_kv_lease_ttl=3600000 [-enable_offload -root_fs_dir=<disk>]`; every node waits for `:50051`/`:8080`, runs `mooncake_client -host=$LOCAL_IP -port=50052 -master_server_address=<head>:50051 -metadata_server=http://<head>:8080/metadata -protocol=rdma -global_segment_size=<cpu.num_bytes>` and writes the standalone-store JSON config (`:10-42`).

### 3.6 Kubernetes mechanics

- **Pod DNS**: `<release>-<role>-<i>.<release>-<role>-headless.<namespace>.svc.cluster.local`. The chart pre-computes, for the orchestrator and trainer pods, `INFERENCE_URL = http://<release>-inference-0.<release>-inference-headless.<ns>.svc.cluster.local:<inference.service.port>/v1` and `ADMIN_INFERENCE_URLS` = the comma-joined list over all inference replicas (`deployment.yaml:62-74,299-311`). Also injected: `POD_NAME`, `POD_IP`, `POD_NAMESPACE`, `ROLE`, `STATEFUL_REPLICAS`, optional `WANDB_API_KEY`/`HF_TOKEN` from a secret. **No prime-rl code reads any of these** (grep over `src/`, `packages/`) — they exist only for shell expansion in user commands.
- **Commands** (example, `k8s/prime-rl/examples/reverse-text.yaml:15-44`): orchestrator `uv run orchestrator @ .../orch.toml --output-dir /data/outputs --client.base-url $INFERENCE_URL`, inference `uv run inference @ .../infer.toml`, trainer `uv run trainer @ .../train.toml --output-dir /data/outputs`, each followed by `; sleep infinity`.
- **Shared PVC role**: `/data` is the data plane for filesystem transports (trainer and orchestrator both use `output_dir=/data/outputs`, the default filesystem weight broadcast writes `broadcasts/step_N` there, and the inference pod loads from the same path) and for checkpoints.
- **Probes** (off by default): inference `startupProbe`/`readinessProbe` `GET /health`, `livenessProbe` `GET /liveness` on `inference.service.port` (`deployment.yaml:198-220`); trainer probes `GET /health` on `trainer.service.port` — requires `trainer.metrics_server` to be configured (`values.yaml:129-144`). `ServerConfig.liveness_timeout_seconds` (30 s) advises probe `timeoutSeconds ≥` it (`configs/inference.py:23-24`).
- **Manual vs automated**: with `autoStart: false` (the default, `values.yaml:29-30,62,104`) you `kubectl exec` into pods and run commands by hand (`k8s/README.md:14-17`). No launcher runs, so nothing does GPU partitioning, run-dir management, subconfig splitting, env-server spawning, or supervision. See §7 for what breaks.

### 3.7 Discovery & rendezvous — every address

| Link | Who listens | Who connects | How the address is learned | Default port(s) |
|---|---|---|---|---|
| Client-facing inference (OpenAI + `/inference/v1/generate`) | router (vllm-router / Envoy) | orchestrator (`model.client.base_url`), env servers via orchestrator-supplied client | single-node: `ClientConfig.base_url` default `http://localhost:8000/v1` (`shared.py:177`), or auto-set when no source is policy-sourced (`configs/rl.py:839-845`); multi-node: `--model.client.base-url $INFER_URLS` | `inference.server.port = 8000` |
| Engine behind router | vLLM server(s) | router, admin clients | `backend_port` auto = `server.port + 100` when a router is set and `backend_port` unset (`configs/inference.py:534-538`); multi-node per-rank `BACKEND_PORT + k` | 8100 (+k) |
| Admin plane (pause / update_weights / resume / init_broadcaster / metrics) | each engine directly | orchestrator | single-node: `admin_base_url = [http://<host>:<backend_port>/v1]` auto (`configs/rl.py:846-854`); multi-node: `--model.client.admin-base-url $ADMIN_URLS` (one per DP rank) | 8100+k; P/D 8100+k / 8200+k |
| Router metrics | router | Prometheus | `server.port + 21000` | 29000 |
| P/D prefill/decode | vLLM role servers | router / llm-d EPP | template loops | 8100+k / 8200+k; llm-d decode sidecar 8300+k |
| vLLM DP coordination (external-LB groups) | replica/role head | other ranks of the DP group | `data_parallel_address` (head `LOCAL_IP` or head hostname) + `--vllm.data-parallel-rpc-port` | `vllm.data_parallel_rpc_port = 13345` |
| NIXL KV side channel (P/D) | each vLLM rank | peer ranks | `VLLM_NIXL_SIDE_CHANNEL_HOST=$LOCAL_IP`, `_PORT=5600+k` | 5600+k |
| Env server (verifiers serve protocol) | env server | orchestrator (`EnvClient`) | address file `configs/attempt_N/resolved/envs/<split>/<name>.address` (written by the server, polled by `wait_for_address`, `orchestrator/envs.py:50-62`), located via `get_config_dir(output_dir)` = `$PRL_ATTEMPT_CONFIG_DIR` or `<output_dir>/configs/latest/resolved` (`pathing.py:158-169`); or explicit `serve.address` | OS-assigned on 127.0.0.1 |
| Rollout batches (ZMQ) | orchestrator: `PUB` bind `tcp://host:port`, `PULL` bind `tcp://host:port+1` (`transports/batch/zmq.py:27-34`) | each trainer data rank: `SUB` + `PUSH` READY (`zmq.py:96-112`) | `rollout_transport.host` (default `localhost`; multi-node `--rollout_transport.host $ORCH_ADDR` on **both** sides). libzmq resolves a hostname in `bind()` to **one** address (verified with libzmq 4.3.5: `bind("tcp://<hostname>:0")` → the node's primary IP), so `ORCH_ADDR` works iff the hostname resolves to a routable IP on the orchestrator node (not `127.0.1.1`) | 5555 / 5556 |
| Rollout batches (filesystem) | — | — | `<run_dir>/batches/step_N` on shared FS | — |
| Weight broadcast (NCCL) | trainer world-master rank 0 hosts the `StatelessProcessGroup` TCPStore at `host:port` as group rank 0 (`transports/weights/nccl.py:131-137,172-179`) | every inference worker joins as rank `1 + rank_offset + device.index` of world `inference_world_size + 1` (`inference/vllm/worker/nccl.py:91-126`), triggered by orchestrator `POST /init_broadcaster {host, port, rank_offset, inference_world_size, timeout}` per admin client, `rank_offset = i * (inference_world_size // #clients)` (`orchestrator/clients.py:165-203`; plain `post`, `httpx.Timeout(None)`, non-404 HTTP errors swallowed) | trainer `host`: `localhost` (single) / `0.0.0.0` (multi); orchestrator `host`: `localhost` / `--weight_broadcast.host $MASTER_ADDR`. `inference_world_size` is a config value, never discovered — wrong for local single-node DP (§7.5) | 29501 |
| Weight broadcast (NIXL) | ModelExpress gRPC server (+ redis) | trainer ranks, inference workers, orchestrator | `weight_broadcast.host/port`; multi-node `$WEIGHT_BROADCAST_HOST` injected on trainer + orchestrator CLIs | 8001 (redis 6379/6380) |
| Weight broadcast (filesystem) | — | — | `<run_dir>/broadcasts/step_N` on shared FS | — |
| Trainer torch.distributed | torchrun rendezvous | other nodes | local: `localhost:<free port>`; multi-node: `$MASTER_ADDR:29500`, `rdzv-id=job_$SLURM_JOB_ID` | 29500 |
| Mooncake store | head inference node master / metadata, per-node client | vLLM mooncake connector | `MOONCAKE_CONFIG_PATH` JSON | 50051 / 8080 / 50052 |
| llm-d | EPP gRPC / health / metrics, Envoy admin | Envoy (ext_proc) | fixed | 9002 / 9013 / 9090 / 9901 |
| Dynamo (optional) | Dynamo frontend + RL discovery | orchestrator admin plane | `model.client.dynamo.discovery_url`, default = `base_url` with port + 1, path stripped (`inference/dynamo.py:103-116`; E owns) | — |
| Dashboard | `dashboard` daemon | browser | `~/.cache/prime-rl/dashboard/daemon.json` `{pid, url}` | (J owns) |
| Trainer metrics/health (optional) | trainer | Prometheus / k8s probes | `trainer.metrics_server.{host=0.0.0.0, port=8000}` (`shared.py:226-231`) | 8000 |

### 3.8 Output directory layout created by launchers

```
<output_dir>/                               # RLConfig.output_dir, default $PRL_OUTPUT_DIR or "outputs"  (utils/config.py:10-12)
└── <run.dir>/                              # = run.name unless run.dir set; run_dir = output_dir / run.dir (configs/rl.py:252-255)
    ├── configs/
    │   ├── attempt_<N>/
    │   │   ├── command.txt                 # uv run rl <argv...>   (first writer wins)
    │   │   ├── rl.toml                     # copy of root @ TOML(s)
    │   │   └── resolved/                   # PRL_ATTEMPT_CONFIG_DIR
    │   │       ├── rl.json                 # SLURM single-node only
    │   │       ├── trainer.json  orchestrator.json  inference.json
    │   │       └── envs/<train|eval>/<name>.json, <name>.address
    │   └── latest -> attempt_<N>
    ├── logs/
    │   ├── attempt_<N>/                    # PRL_ATTEMPT_LOG_DIR
    │   │   ├── trainer.log  orchestrator.log  inference.log   (multi-node: symlinks to node_0 logs)
    │   │   ├── trainer/torchrun/<rdzv-run-id>/attempt_0/<rank>/{stdout,stderr}.log   (torchrun --log-dir layout)
    │   │   ├── trainer/node_<k>.log  inference/node_<n>[_rank<k>].log  inference/router.log  inference/llmd/  inference/sidecar_<n>.log
    │   │   ├── envs/<split>/<name>.log
    │   │   └── modelexpress/ (multi-node NIXL)
    │   └── latest -> attempt_<N>
    ├── launcher/                            # SLURM only: rl.sbatch, logs/job_<jobid>.log, .trainer.done, .orchestrator.done
    ├── checkpoints/step_<N>/               # trainer (unless ckpt.output_dir) + orchestrator (D/B)
    ├── broadcasts/step_<N>/                 # weight broadcast markers/weights (D/E)
    ├── batches/step_<N>/                    # filesystem rollout transport (C/E)
    ├── monitors/file/{metrics.jsonl, plan.json, ...}, monitors/prime/run.json   (J)
    ├── modelexpress/                        # single-node SLURM NIXL state
    └── mooncake/node_<n>/                   # mooncake store
<ckpt.output_dir>/checkpoints/step_<N>/      # trainer checkpoints when [ckpt] output_dir is set (trainer/ckpt.py:332)
```
The standalone `inference` SLURM path has no run name: attempts are created directly under `InferenceConfig.output_dir` (`entrypoints/inference.py:138`).

### 3.9 Other launchers (topology summary)

- **`sft`** (`entrypoints/sft.py`): `SFTConfig` is *not* split — the trainer re-parses the same `sft.json` (`-m prime_rl.trainer.sft.train @ sft.json`). Local: optional `inference` (online evals) on the first `num_infer_gpus`, one `env-server` per eval source, a `python -m prime_rl.eval.online @ eval.json` process with `PRL_ATTEMPT_CONFIG_DIR/LOG_DIR` + `PRL_LOG_DIR` and shared-W&B primary/finisher roles, then `torchrun --nproc-per-node=num_train_gpus` (`sft.py:290-498`). Without `[inference]`, `CUDA_VISIBLE_DEVICES` is not set for the trainer at all (`sft.py:320-334,455-456`). Success waits for trainer **and** online-eval (`sft.py:464-477`). Multi-node (`multi_node_sft.sbatch.j2`): inference nodes first; inference node 0 runs router, env servers and the online-eval process (`--client.base-url $INFER_URLS --client.admin-base-url $ADMIN_URLS [--weight-broadcast.host $MASTER_ADDR]`), trainer nodes run torchrun with `--weight-broadcast.host 0.0.0.0` for NCCL; done files are `.sft_trainer_done_$SLURM_JOB_ID` / `.sft_eval_done_$SLURM_JOB_ID`; **no grace period** (`:54-61,145-280`). `NEVER_CLEAN` also disables stale-eval-artifact cleanup (`sft.py:501-512`).
- **`eval`** (`entrypoints/eval.py`): single process; expands shorthands (`<taskset-id>`, `--env.*`, `-c N`) into a JSON `--source` flag, spawns one env server per source without `serve.address`, sets `PRL_ATTEMPT_*` in its own env, runs `run_eval` in-process, cleans env servers on exit (`eval.py:48-199`).
- **`inference`** standalone: local (router + engine, §3.2) or SLURM via `inference.sbatch.j2` (single-node: one `uv run inference @ config` per node with the router; multi-node: router on node 0 + per-rank engines, **each node an independent replica**, EP groups node-local; disaggregated: as §3.4.4 but single replica set) (`inference.sbatch.j2:199-351`).
- **`dashboard`**: `ensure_dashboard(output_dir)` registers `output_dir.resolve()` in `~/.cache/prime-rl/dashboard/dirs.json` under an `fcntl` lock, reuses a live daemon from `daemon.json` (pid alive + `GET <url>/api/runs` OK), and only spawns a new one when stdout is a TTY and the `dashboard` extra (fastapi, uvicorn) is importable; waits up to 10 s (`dashboard.py:31-114`). Called only on non-dry-run launches (`rl.py:142,635`).

### 3.10 Resume & clean from the launcher's side (S8)

- Resume is enabled by `[resume]` / bare `--resume` (latest), `--resume.step N`, or `--resume.dir <other_run>/checkpoints/step_N` (step and dir mutually exclusive; dir must be named `step_<N>`) (`shared.py:54-78`). `RLConfig.auto_setup_resume` copies the same `ResumeConfig` into `trainer.resume` and `orchestrator.resume` (`configs/rl.py:398-405`); `validate_shared_ckpt_config` requires them equal and requires `ckpt` on both sides or neither, with equal `interval` (`utils/validation.py:194-213`).
- Launcher steps: skip run-dir guard; resolve the step (§3.1 step 5) only to delete stale `batches/` (> step) and `broadcasts/` (≥ step); a new `attempt_<N+1>` is created, old attempts are kept.
- The **run directory is keyed by `run.name`**: auto-generated names contain a random suffix (`<envs>--<model>--<8 hex>`, `configs/rl.py:257-265`), so resuming requires re-passing the same `--run.name`.
- Trainer resolves its checkpoint root as `ckpt.output_dir or output_dir` → `<root>/checkpoints` (`trainer/ckpt.py:169,332`); orchestrator always `<run_dir>/checkpoints` (`orchestrator/ckpt.py:25`) — shared `ckpt.output_dir` propagates to the trainer only (`utils/validation.py:90-91`); the launcher uses the trainer's root.
- **Who picks the step.** Launcher (`rl.py:671-686`), trainer (`trainer/rl/train.py:137-143`) and orchestrator (`orchestrator/orchestrator.py:240-246`) each run the same rule: `resume.dir` → its `step_<N>`; else `resume.step`; else `resolve_latest_ckpt_step` = max over `glob("step_*")` directory **names** (`pathing.py:295-311`) — no completeness check, no look inside. Both then resume at `N+1`. With the default layout all three glob the same dir and agree on `N`; with a separate `ckpt.output_dir` the trainer and orchestrator glob different trees and can pick different `N` (trainer broadcasts `v{N_t}` at startup while the orchestrator's `sync_startup` waits for exactly `v{N_o}` → startup timeout, `STARTUP_WEIGHT_WAIT_TIMEOUT_S = 1200`, `orchestrator.py:100,316-317`; `watcher.py:48-54`).
- **"Latest" can be an orchestrator-only directory.** The orchestrator checkpoints step `S` right after *shipping* batch `S` (`orchestrator.py:640,958-973`) — up to `TARGET_LAG+1` steps before the trainer finishes and saves `S` (`trainer/rl/train.py:597-611`). Its teardown `finally` also saves `step_{last shipped}` **regardless of `ckpt.interval`** whenever `main_loop` exits by exception or SIGINT/cancellation (`orchestrator.py:433-441`); SIGTERM (what the local launcher and SLURM send) kills it without that save (no handler). So a trainer crash, or a Ctrl-C of a local run (SIGINT reaches the whole foreground process group), easily leaves `checkpoints/step_K/orchestrator/progress.pt` with no `step_K/trainer/`. A bare `--resume` then resolves `K` everywhere and the trainer raises `FileNotFoundError("Checkpoint not found …")` (`trainer/ckpt.py:253-267`). Recover with `--resume.step <last step that has trainer/>`.
- **Partial writes.** Trainer DCP saves are synchronous and unmarked: a crash mid-save leaves a `step_N/trainer/` that `dcp_load` fails on, and "latest" still picks it. Orchestrator writes are atomic (`mkstemp` + `os.replace`, `orchestrator/ckpt.py:31-45`); a missing `progress.pt` raises `FileNotFoundError` even with `ckpt.skip_progress` (`orchestrator/ckpt.py:47-56`). Neither side falls back to an older step.
- **Rewinds leave the abandoned timeline.** `--resume.step K` filters only the trainer's in-memory list (`trainer/ckpt.py:173-177`); `clean_future_steps` removes only `batches/`/`broadcasts/`, so `checkpoints/step_{>K}` stay on disk and a later bare `--resume` jumps back onto them. Resuming a run that already reached `max_steps` makes the trainer start step `max_steps+1` (its loop breaks only after a step with `progress.step >= max_steps`, `trainer/rl/train.py:258,717-719`) and wait for a batch the orchestrator refuses to ship (`orchestrator.py:588-592`) [static].
- **Component-only restarts.** Restarting just the orchestrator is unsupported with ZMQ: trainer ranks push their READY once at init (`transports/batch/zmq.py:107-112`), so a new orchestrator's READY barrier never completes. Restarting just the trainer re-joins NCCL only if the inference group can be rebuilt (it cannot without an engine restart, see `06`).

## 4. Interfaces & contracts

### 4.1 Launcher-owned CLI/argv contracts

| Child | argv | cwd | Env (beyond `os.environ`) |
|---|---|---|---|
| inference (local) | `inference @ <cfg>/inference.json` | launcher cwd | common + inference defaults + `env_vars` + `inference.env_vars` + `CUDA_VISIBLE_DEVICES` |
| inference rank (SLURM multi) | `uv run inference @ inference.json --server.host 0.0.0.0 --server.port P --vllm.data-parallel-size DP --vllm.data-parallel-size-local 1 --vllm.api-server-count 1 [...]` | `$PROJECT_DIR` | exported inference vars + `CUDA_VISIBLE_DEVICES` slice, `VLLM_RPC_BASE_PATH`, [`VLLM_NIXL_SIDE_CHANNEL_*`], [`MOONCAKE_CONFIG_PATH`, `PYTHONHASHSEED=0`], `NCCL_IB_HCA` |
| env server | `env-server @ <cfg>/envs/<split>/<name>.json` | same | common + `env_vars` + `orchestrator.env_vars`; reads `PRL_RUN_ID`, `PRL_RUN_NAME`; sets `VF_RUN_ID` |
| orchestrator | `orchestrator @ <cfg>/orchestrator.json` [+ multi-node overrides] | same | common + `env_vars` + `orchestrator.env_vars` + `WANDB_SHARED_*`, `WANDB_RUN_ID` |
| trainer | `torchrun ... -m prime_rl.trainer.rl.train @ <cfg>/trainer.json` [+ overrides] | same | common + trainer defaults + `env_vars` + `trainer.env_vars` + `WANDB_SHARED_*` + `CUDA_VISIBLE_DEVICES` (local) |

### 4.2 Environment variables with cross-process meaning

| Var | Set by | Read by | Meaning |
|---|---|---|---|
| `PRL_RUN_ID` | `rl`/`sft`/`eval` launcher (`setdefault`), sbatch fallback | orchestrator (`orchestrator.py:219`), eval runner, prime monitor, env server (→ `VF_RUN_ID`) | run identity; also `WANDB_RUN_ID` |
| `PRL_RUN_NAME` | launcher, multi-node sbatch | orchestrator & env server (sandbox labels) | display name |
| `PRL_ATTEMPT_CONFIG_DIR`, `PRL_ATTEMPT_LOG_DIR` | SLURM templates, `eval`/`sft` for online-eval | `prepare_attempt_dirs`, `get_config_dir` | pin a child to its launch attempt (address files, logs) |
| `PRL_OUTPUT_DIR` | user | `default_output_dir()` | default `output_dir` |
| `PRL_LOG_DIR` | sft launchers | online-eval | log dir |
| `NEVER_CLEAN` | user | `rl`/`sft`/`eval` | disables `clean` (and sft stale-eval cleanup); does **not** disable `clean_future_steps` in `rl` |
| `WANDB_SHARED_MODE/LABEL/PRIMARY/FINISHER`, `WANDB_RUN_ID`, `WANDB_PROGRAM`, `WANDB_ARGS` | launchers | W&B monitor (J) | single shared W&B run |
| `PRIME_LOG_LEVEL`, `PRIME_VF_LOG_LEVEL` | user | `LogConfig` defaults (`shared.py:200-204`) | log levels |
| `CUDA_VISIBLE_DEVICES` | user (input) / launcher (output) | `get_physical_gpu_ids` | GPU partition; must be integer indices |
| `PYDANTIC_CONFIG_PLAIN/WIDE`, `NO_COLOR`, `FORCE_COLOR` | user | pydantic-config | error/help rendering |
| `NCCL_P2P_DISABLE`, `NCCL_SHM_DISABLE` | `disable_nccl_p2p_if_unavailable` (if unset and no NVLink found) | NCCL | set in trainer master & inference workers before the broadcast group (`utils/nccl.py:8-42`) |

### 4.3 Files crossing process boundaries (launcher-relevant)

| Path | Writer | Reader | When |
|---|---|---|---|
| `configs/attempt_N/resolved/*.json` | launcher | each component's `cli()` | before spawn |
| `.../envs/<split>/<name>.address` | env server (atomic rename) | orchestrator `wait_for_address` (0.5 s poll, 600 s timeout) | after bind |
| `launcher/.trainer.done`, `.orchestrator.done` | multi-node trainer rank 0 / orchestrator subshell on exit 0 | every node's watcher (10 s poll) | end of run |
| `logs/latest`, `configs/latest` | `create_attempt_dirs` (atomic symlink replace) | tools, `get_config_dir` fallback | per launch |
| `~/.cache/prime-rl/dashboard/{daemon.json,dirs.json,.lock,daemon.log}` | launcher / dashboard | launcher / dashboard | per launch |

### 4.4 Launcher-relevant config fields

| Field | Type / default | Effect |
|---|---|---|
| `deployment.type` | `single_node` (default) \| `multi_node` | placement backend; multi-node requires `[slurm]` |
| `deployment.gpus_per_node` | `8` | `--gres`, dp_per_node, trainer nproc (multi) |
| `deployment.num_train_gpus` / `num_infer_gpus` | `1` / `1` (single) | GPU split; sum ≤ `gpus_per_node` |
| `deployment.num_train_nodes` | required (multi) | trainer nodes |
| `deployment.num_infer_nodes` | `None` → inferred from `inference.deployment` (multi) | inference nodes **per replica**; `0` = no inference/orchestrator (fake data) |
| `deployment.num_infer_replicas` | `1` | replicates the whole inference island |
| `deployment.nodes_per_fsdp_group` | `None` | sets `trainer.model.dp_replicate = num_train_nodes / nodes_per_fsdp_group` |
| `deployment.orchestrator_on_inference` | `False` | orchestrator + env servers on last inference node |
| `slurm.{job_name="prime-rl", project_dir=".", template_path, partition="cluster", nodelist, exclude, account, time, pre_run_command, launch_modelexpress=True, cleanup_grace_period=3600, shared_fs=True}` | `SlurmConfig` (`shared.py:81-137`) | sbatch header + template behaviour; `project_dir` is `.resolve()`d on the submit host |
| `inference.server.{host=None, port=8000}`, `inference.backend_port` (8100 / auto), `inference.router` (`vllm-router` default \| `llm-d` \| `None`) | `configs/inference.py:16-24,443-453` | ports and router |
| `inference.vllm.{tensor_parallel_size, data_parallel_size, data_parallel_size_local, api_server_count, data_parallel_rpc_port=13345}` | auto-filled by RLConfig | per-node engine layout |
| `weight_broadcast` | `None` → NCCL (or filesystem with LoRA / no inference) | transport; host/port rendezvous |
| `rollout_transport` | `None` → resolved from sub-configs (default ZMQ `localhost:5555`) | orchestrator→trainer transport |
| `env_vars`, `<component>.env_vars` | `{}` | env layering; cannot contain `PROTECTED_ENV_VARS` |
| `output_dir`, `run.name`, `run.dir`, `clean`, `resume`, `dry_run`, `dashboard` | see §3 | run dir management |
| `orchestrator.train/eval.source[*].serve.address` | `None` | `None` → launcher spawns the env server; set → externally managed |

## 5. Invariants & assumptions

1. **Sub-configs are re-validated in the child.** Each process runs `cli()` on its JSON, so every validator must be idempotent on its own output (e.g. `propagate_shared_fields` accepts equal shared/sub values for this reason, `utils/validation.py:18-25`). The launcher mutates sub-configs after construction without `validate_assignment`, so the child's resolved view can differ from the launcher's in-memory one (see `02`).
2. **The run directory (and `launcher/`) must be on a filesystem shared by all nodes** for multi-node: done-markers, address files for co-located env servers aside, filesystem transports, checkpoints, and `$CONFIG_DIR` JSONs are all read cross-node. The project checkout + `.venv` must be shared unless `slurm.shared_fs = false` (`scaling.md:185`; `multi_node_rl.sbatch.j2:167-179`).
3. **SLURM_PROCID ↔ HOSTNAMES index**: the template assumes task `i` of `srun --ntasks-per-node=1` runs on `HOSTNAMES[i]` (block distribution in nodelist order). Mooncake alone uses `$SLURMD_NODENAME` to pick its head.
4. **Env servers are loopback-bound** unless `serve.address` is set → they must share a host with their orchestrator (and with their eval process).
5. **Only one inference router URL, many admin URLs.** Admin traffic must bypass routers; clients learn per-engine URLs only from `admin_base_url` (single-node auto, multi-node `$ADMIN_URLS`).
6. **NCCL broadcast world** = trainer rank 0 + `inference_world_size` workers; `inference_world_size` must equal the number of inference GPUs that will join (`configs/rl.py:778-792` multi-node = `total_infer_nodes*gpus_per_node`; single-node = the **pre-auto-fill** `dp*tp` from `auto_setup_weight_broadcast`, `configs/rl.py:458-469`, correct only if the user's `dp*tp == num_infer_gpus`, §7.5), and `rank_offset = client_index * (inference_world_size // #admin_clients)` assumes every admin client fronts the same number of GPUs.
7. **Static ports are per-host singletons**: 8000/8100 (router/engine), 29501 (NCCL TCPStore), 5555/5556 (ZMQ), 29500 (torchrun, multi-node) — two runs on one host must move all of them.
8. **Trainer/orchestrator agree on resume independently**: the launcher does not pin a step; both resolve "latest" = highest `step_*` directory name in their own checkpoint dirs. Agreement holds only if they share the checkpoint tree **and** the latest dir contains both `trainer/` and `orchestrator/progress.pt` — neither is guaranteed (§3.10).
9. `get_physical_gpu_ids` assumes `CUDA_VISIBLE_DEVICES` holds integer indices and that NVML order == `PCI_BUS_ID` order (children get `CUDA_DEVICE_ORDER=PCI_BUS_ID`).

## 6. Extension points

- **Custom SLURM template**: `--slurm.template-path path/to/x.sbatch.j2` (`shared.py:88-89`; auto-default in `configs/rl.py:857-867`). The template receives the variables in `rl.py:479-578` (single-node: `job_name…shared_fs` from `SlurmConfig.template_vars` + `config_path, config_dir, log_dir, output_dir, launcher_dir, launcher_log_dir, gpus_per_node` + modelexpress vars). Includes (`_launch_rank.sh.j2`, `_launch_router.sh.j2`, `_mooncake_store.sh.j2`, `llmd/*`) resolve relative to the custom template's directory for `rl` (loader = template's parent only, `rl.py:433`) — copy them alongside; `sft` also adds the bundled templates dir to the loader search path (`sft.py:148-152`).
- **Adding a new component process** (e.g. a reward-model server) to local runs: add its config to `RLConfig`, a writer in `write_subconfigs` (`rl.py:89-113`), a `Popen` + `monitor_process` thread in `rl_local` (pattern `rl.py:259-292`), add it to `rl_config_components` and `format_log_message`; for multi-node, add a role block to `multi_node_rl.sbatch.j2` and pass its address to consumers as CLI overrides; decide whether it is terminal (done-file) or supporting. Keep it out of the completion condition unless it must finish.
- **Externally managed env servers**: set `serve.address` on a source; the launcher stops spawning it and the orchestrator connects directly (`configs/orchestrator.py:163-164`; `rl.py:55-61`). This is the hook for k8s env-server pods.
- **External inference**: omit `[inference]` and point `orchestrator.model.client.base_url` / `admin_base_url` (or `dynamo`) at it; set LoRA/router-replay/sampling-mask flags on that server yourself (warnings in `configs/rl.py:563-567,582-585,627-634`).
- **Per-role env/vLLM knobs (P/D)**: `inference.deployment.{prefill,decode}_{env_vars,vllm_overrides}`.
- **Run-level env**: `[env_vars]` / `[<component>.env_vars]` (cannot override protected vars).
- **Image / source override on k8s**: `PRIME_RL_REF`, `PRIME_RL_REPO`, `VERIFIERS_VERSION` env on the pod (`scripts/docker-entrypoint.sh:16-73`).

## 7. Gotchas & limitations

1. **`--dry-run` is not side-effect free** (verified by running `rl()` with `--dry-run` on a scratch run dir). The path is `rl()` → `validate_run_dir` → `mkdir` → `clean_future_steps` → `rl_local`/`rl_slurm` (attempt dirs, `command.txt`, JSONs, `rl.sbatch`) → `return` (`rl.py:656-686,124-131,592-633`); only model download, dashboard, GPU check, process launch and `sbatch` are gated. `--clean --dry-run` **rmtree's the run dir** (and a separate `ckpt.output_dir`); `--resume --dry-run` deletes `broadcasts/step_≥N` and `batches/step_>N` for the resolved `N`; a from-scratch dry run on a dir with only `configs/`/`launcher/` wipes every `batches/`/`broadcasts/` step (`rl.py:684-686`). Each dry run also repoints `configs/latest`/`logs/latest`, which a live run's orchestrator uses to find env-server address files if it (re)reads them (§3.7).
2. **`--resume` without `--run.name` silently trains from scratch** in a fresh auto-named dir: the guard is skipped, `resolve_latest_ckpt_step` warns "No checkpoints found … Starting from scratch" and returns `None` (`pathing.py:301-311`).
3. **Grace period on crash**: multi-node RL holds the allocation for `cleanup_grace_period` (3600 s) after *any* component exits non-zero — including routers, env servers, vLLM ranks (`multi_node_rl.sbatch.j2:598-611`). Set `slurm.cleanup_grace_period = 0` for fast-fail debugging.
4. **Two local runs on one host collide on more than the inference port.** `docs/scaling.md:46-59` shows changing only `inference.server.port`/`base_url`, but with defaults (`reverse-text/rl.toml` → NCCL broadcast + ZMQ transport) while `backend_port` (+100) and router metrics (+21000) follow `server.port` automatically, both runs still bind NCCL TCPStore `localhost:29501` (trainer rank 0 `listen_socket.bind` in vLLM `StatelessProcessGroup.create`) and ZMQ `5555/5556` (`zmq.py:29,34`) — plain `bind`s, so the second run fails with `EADDRINUSE` rather than cross-talking. Move `weight_broadcast.port` and `rollout_transport.port` too.
5. **Single-node NCCL/NIXL world size is computed before DP auto-fill (bug, local launches).** `auto_setup_weight_broadcast` (`configs/rl.py:458-469`) sets `inference_world_size = vllm.data_parallel_size * tp` on trainer + orchestrator *before* `auto_setup_deployment` (`:696-703`) rewrites `data_parallel_size = num_infer_gpus // tp` (whenever `dp*tp != num_infer_gpus`, overriding even an explicit DP); the single-node branch never recomputes it (multi-node does, `:778-792`). Verified by running the validators on `configs/basic/reverse-text/rl.toml`: `num_infer_gpus=2` → `dp=2, inference_world_size=1`; `=4, tp=2` → `dp=2, ws=2`; explicit `dp=4` with 2 GPUs → `dp=2, ws=4` (over-count, api_server_count left at 4). The trainer's `trainer.json` carries the stale value verbatim; **SLURM single-node self-heals** because the compute node re-validates `rl.json` through `RLConfig` (re-parse gives `ws=2`). Receiver side: the orchestrator posts one `/init_broadcaster` per admin URL with `rank_offset = i * (ws // #clients)` (`orchestrator/clients.py:166-203`); vLLM fans `collective_rpc` to every DP engine (`DPLBAsyncMPClient.call_utility_async`), and each worker joins as rank `1 + rank_offset + device.index` of world `ws + 1` (`inference/vllm/worker/nccl.py:111-126`) with `device.index = dp_local_rank*tp + tp_rank`. Under-count ⇒ DP engine ≥1 computes rank ≥ world, `StatelessProcessGroup` asserts `rank < world_size` (vLLM 0.24 source, `vllm/distributed/utils.py:204`; pin requires ≥0.29), `/init_broadcaster` returns 500, and `initialize_nccl` **swallows it** (non-404 branch, `clients.py:192-196`); engine 0 + trainer form a 2-rank group, and **the run fails at the first weight sync**: the other engines have no `nccl_broadcast_receiver`, so `/update_weights` errors there (and its 5xx retries re-enter the broadcast on engine 0). NIXL under-count makes the trainer wait for only `ws` inference registrations (`transports/weights/nixl/nixl.py:384-387`). **Affected: every local single-node `rl` launch with `num_infer_gpus > tp` and DP not set to `num_infer_gpus / tp`** — including the shipped `examples/basic/hendrycks-sanity/rl.toml` (dp 4, ws 1), `configs/ci/nightly/multimodal_color_codeword.toml` (4 GPUs, no DP) and wordle with 3 inference GPUs (dp 3, ws 1). **Workaround:** set `inference.vllm.data_parallel_size = num_infer_gpus / tp` explicitly (all `configs/ci/nightly-fft/*` and `docs/scaling.md:38-42` do), or launch through single-node SLURM.
6. **`CUDA_VISIBLE_DEVICES` must be integer indices**; UUID/MIG forms make `int(token)` raise (`process.py:44`).
7. **Orchestrator and env servers inherit all GPUs** of the launcher (no `CUDA_VISIBLE_DEVICES` override). Anything in them that initializes CUDA lands on GPU 0 (shared with inference).
8. **Trainer metrics server default port 8000 collides with the inference router** on a single node if `trainer.metrics_server` is enabled without changing its port (`shared.py:227`, `configs/inference.py:20`).
9. **A component exiting with code 0 is not an error** locally; e.g. a clean inference exit leaves the orchestrator to hang on its own timeouts.
10. **`get_free_port` race** for the local torchrun rendezvous (bind-close-reuse).
11. **Custom `rl` templates don't see bundled includes** (loader search path is only the custom template's dir, `rl.py:433`), unlike `sft`.
12. **`uv run` inside the sbatch** (every rank, router-less inference, orchestrator, env servers) triggers uv's default project sync check per process on the shared FS [UNVERIFIED whether `.env` sets `UV_NO_SYNC`]; the template comment "Do not sync" only refers to the explicit `uv sync`.
13. **llm-d**: rejected for single-node and for routed-expert return (router replay) (`configs/inference.py:500-532`, `configs/rl.py:588-605`); `grpc-health-port` is 9013 because mooncake's metrics use 9003 on node 0 (`_launch_router.sh.j2:63-64`); decode nodes hard-code `UCX_NET_DEVICES=mlx5_0:1`.
14. **Mooncake offload is SLURM-only** (`configs/rl.py:673-685`); its head is "first host of the job", which is inference node 0 only because inference nodes are listed first.
15. **Multi-node inference within RL vs standalone differ**: in RL, an EP DP group spans all nodes of a replica (`multi_node_rl.sbatch.j2:470-480`); standalone `inference.sbatch.j2` makes each node an independent replica with node-local EP (`:328-344`).
16. **Kubernetes path is a scaffold, and the shipped example is broken** (first three items and the admin-URL item verified by parsing the example configs with `cli()`; pod behaviour not run):
    - `examples/reverse-text/infer.toml` contains only `[model] name = ...`, but `InferenceConfig` has no `model` field (it is `vllm.model`) and forbids extras → `uv run inference @ infer.toml` should fail validation. `tests/unit/test_configs.py:52-70` only requires *some* config class to parse each k8s TOML, so this passes CI (Orchestrator/Trainer parse it).
    - The orchestrator command passes `--client.base-url`, but `OrchestratorConfig` has `model.client`, not `client` → unknown key rejected by `extra="forbid"`.
    - `uv run trainer` runs a single process (no torchrun) — `trainer.gpu.count > 1` gives more GPUs but one rank; multi-GPU needs a hand-written `torchrun ... -m prime_rl.trainer.rl.train`.
    - Standalone orchestrator defaults `rollout_transport` to ZMQ `localhost` (`configs/orchestrator.py:567`, `configs/trainer.py:694`) → trainer and orchestrator in different pods cannot connect unless you set `rollout_transport.host` (orchestrator bind `0.0.0.0`, trainer → orchestrator pod DNS) or switch both to filesystem.
    - `num_train_workers`/`pad_to_multiple_of` are not auto-filled outside the `rl` launcher (`configs/orchestrator.py:594-598`).
    - **No env servers are spawned** by standalone `orchestrator`; it waits up to 600 s for `<output_dir>/configs/latest/resolved/envs/train/<name>.address` and then fails, unless each source sets `serve.address` to an env server you run yourself (`orchestrator/envs.py:95-104`; `pathing.py:158-169`). The chart has no env-server workload.
    - `ADMIN_INFERENCE_URLS` is comma-joined (`deployment.yaml:62-74`), but `admin_base_url` is `list[str]` parsed from space-separated CLI tokens → `['http://a…/v1,http://b…/v1']`, one bogus URL (verified); and the listed URLs are the router port (8000), whereas admin ops must bypass the router (default `InferenceConfig.router = vllm-router`, engine on 8100).
    - The orchestrator Service exposes ports 8000/29501 that the orchestrator does not listen on; nothing is exposed for ZMQ 5555/5556 (not required for pod-to-pod traffic).
    - No supervision/restart semantics beyond k8s container restarts; `; sleep infinity` in the example commands keeps a failed container "Running".
17. **`docs/configuration.md` drift**: the example `--orchestrator.train.source.0.args ...` cannot work — pydantic-config has no list-index paths (acknowledged in `entrypoints/eval.py:52-53` and `skills/configs/SKILL.md:66`). `docs/scaling.md:152-156` says `fused_lm_head_token_chunk_size` defaults to 1024; the config default is 8192 (`configs/trainer.py:347`). `docs/scaling.md:247` mentions `nixl_cu12-*.whl` while the pin is `nixl-cu13==0.10.1` (`pyproject.toml:70`).
18. **Stale env-server address files on re-submission.** Address files live in the attempt's config dir and are never deleted; only a *new* attempt gets a clean dir. Re-running the same `launcher/rl.sbatch` (as the dry-run message suggests, or a SLURM requeue) reuses the pinned `PRL_ATTEMPT_CONFIG_DIR`, so the orchestrator can read the previous job's `<name>.address` before the new server overwrites it, then time out on health after 600 s (`orchestrator/envs.py:51-62,95-104`) [static]. The multi-node template `rm -f`s only the done-files (`:77`).
19. **A crashed orchestrator can exit 0.** `Orchestrator.stop()` (run in the `finally` of `start()` on every exit path) calls `os._exit(0)` if teardown exceeds `SHUTDOWN_TIMEOUT_S = 300` (`orchestrator.py:90,1043-1052`), bypassing the crash exit code: locally the launcher then counts the orchestrator as finished; multi-node the subshell touches `.orchestrator.done` (`:569-571`).
20. **Monitors are rank-gated on `RANK` / `DP_RANK`** (`monitors/__init__.py:67`). Neither the local launcher nor any template sets these for the orchestrator or env servers (`RANK` is only set by torchrun for trainer ranks), so they appear only if inherited from the user's shell/job environment (e.g. `rl` started inside another torchrun/accelerate job) or set in `orchestrator.env_vars` (not protected) — in which case the orchestrator silently registers no monitors.
21. **The launcher host needs every env package installed**: `cli(RLConfig)` imports each source's taskset while validating (observed: validation fails with the taskset's own `ImportError` when it is missing), before anything is spawned.

## 8. For a custom framework

- **Keep**: (a) *resolved-config-per-process* as the inter-process contract — every process is independently runnable and debuggable with `<cmd> @ resolved.json`; (b) attempt directories with `latest` symlinks and `command.txt`; (c) killing whole process trees (psutil) rather than process groups; (d) OS-assigned ports + atomically-written address files for anything spawned dynamically (env servers) — this is the one discovery mechanism that is race-free and multi-run safe; (e) a single client URL + explicit per-engine admin URLs (routers must never carry control-plane traffic); (f) done-files + `--kill-on-bad-exit` as a cheap cross-node completion protocol.
- **Simplify / replace**: static well-known ports are the main multi-tenancy liability (§7.4); generalize the address-file pattern to *every* listener (router, ZMQ PUB, NCCL TCPStore, ModelExpress) and have consumers wait on a small registry (file dir or KV) keyed by run id. Replace the ~600-line bash multi-node template with a Python per-node agent (same code path as local), which would also remove the validator-order and "launcher view ≠ child view" divergences. Pin the resume step in the launcher and pass it explicitly to trainer and orchestrator instead of letting each resolve "latest", and define "latest" as the newest step with a completeness marker written after *both* sides saved it. Derive broadcast world sizes from the placement plan (or have engines report their worker count) rather than from config arithmetic.
- **Coupling points to respect**: vLLM external-LB DP semantics (`data_parallel_rank/address/rpc_port`, one API server per DP rank) drive both the router worker list and the weight-broadcast rank arithmetic (`rank_offset = i * gpus_per_server`, `device.index` as local rank). If you change engine layout, change broadcast rank assignment in lockstep.
- **k8s**: treat the Helm chart as a pod scaffold only. A real k8s backend needs: an env-server workload with `serve.address`, a router/engine port split in URLs, torchrun (or a JobSet/PyTorchJob) for the trainer, rollout/broadcast host injection, and a supervisor equivalent to `rl_local`'s error queue.

## 9. Open questions

1. Nightly CI history for the two configs without explicit DP (§7.5): they should fail at the first weight sync; if they pass, some pinned-vLLM detail differs from the static reading.
2. Does `vllm-router` proxy `/health` and `/liveness` for the k8s probes when the router is enabled on the probed port? (Router internals are external — partial per E.)
3. Is `SLURM_PROCID ↔ HOSTNAMES[i]` guaranteed on the target clusters (custom `--distribution` or heterogeneous allocations would break role assignment)?
4. Does `sbatch` in the target environments export the launcher's `PRL_RUN_ID` (`--export=ALL` default), or does the template's fallback mint a different id than the launcher printed?
