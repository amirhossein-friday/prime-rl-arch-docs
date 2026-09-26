# Observability, Eval & Ops — prime-rl @ b944873

> Scope: metric/episode monitors (file, W&B, Prime platform), logging, heartbeats, Prometheus server, what the orchestrator/trainer measure, MFU/throughput accounting, standalone + online eval and eval resume, the local dashboard, trace tooling, and the test/CI/benchmark surface.
>
> Files read in full (LOC): `src/prime_rl/monitors/__init__.py` (175), `monitors/base.py` (75), `monitors/prime.py` (276), `monitors/wandb/{__init__,monitor,overview}.py` (3/164/261), `monitors/file/{__init__,monitor}.py` (3/176), `monitors/file/traces/{__init__,__main__,chunks,index,update,live}.py` (33/6/115/146/100/289), `utils/logger.py` (290), `utils/metrics_server.py` (176), `utils/heartbeat.py` (53), `utils/pathing.py` (425), `packages/prime-rl-configs/.../configs/monitors.py` (71), `configs/eval.py` (167), `eval/{eval,online,resume,runner}.py` (68/223/169/345), `entrypoints/eval.py` (203), `entrypoints/dashboard.py` (127), `dashboard/server.py` (2257), `dashboard/README.md` (52), `orchestrator/{metrics,inference_metrics,periodic_logger,live,annotations}.py` (546/458/56/68/49), `trainer/perf.py` (245), `trainer/utils.py` (344), `trainer/rl/annotations.py` (90), `tools/convert_traces_to_hf_dataset.py` (195), `tests/conftest.py` (131), `tests/utils.py` (245), `tests/integration/{conftest,dashboard_smoke,test_reverse_text,test_benchmark_regression,test_gsm8k_eval}.py`, `tests/nightly/test_hendrycks_sanity.py`, `tests/unit/{orchestrator/test_metrics,eval/test_resume,eval/test_cli,train/test_perf}.py`, `.github/workflows/{cpu_tests,gpu_tests,nightly_tests,benchmarks,style}.yaml`, `benchmarks/scripts/run_single_benchmark.py` (321), `docs/{development,eval,training}.md`, `skills/{training/monitor-run,eval,dashboard}/SKILL.md`, `deps/verifiers/verifiers/v1/episode.py` (169). Read in part (call sites): `orchestrator/orchestrator.py` 130–1077, `trainer/rl/train.py` 70–140 & 228–756, `trainer/sft/train.py` monitor call sites, `entrypoints/rl.py` 110–390, `orchestrator/dispatcher.py` (live relay, gauges), `orchestrator/utils.py` 30–125, `utils/utils.py` 40–165, `utils/async_utils.py` 20–70, `configs/{shared,rl,orchestrator,trainer}.py` observability fields. Static JS (`dashboard/static/app.js` 7200, `index.html`) skimmed only for tabs and poll cadence.
>
> Related docs: 01-deployment-topology-and-launch.md (log dirs / attempt dirs / env propagation are A's), 03-orchestrator.md (dispatcher, sinks, eval source/sink), 04-algorithms-loss-data-path.md (loss tensors that become trainer metrics; staleness S9), 05-trainer.md, 06-inference-and-transports.md (weight receivers used by online eval), 07-verifiers-core.md (Episode/Trace schema, delta stream).

---

## 1. Mental model

prime-rl's observability is **one fan-out abstraction plus one on-disk format**. Every process that has something to report (orchestrator, RL trainer, SFT trainer, `eval`, `online-eval`) calls module-level functions in `prime_rl.monitors` (`log`, `log_annotations`, `log_live`, `log_eval_plan`, `log_eval_epoch`, `finalize`), which fan out to every *registered* monitor on that process's rank 0 (`src/prime_rl/monitors/__init__.py:48-175`). There are exactly three monitor backends: **File** (on by default), **W&B** (off by default), **Prime platform** (off by default; train flavour on the orchestrator, eval flavour on `uv run eval`). A monitoring failure never raises out of the fan-out (`monitors/__init__.py:126-130`), but a *configured* monitor whose `init` fails crashes the process (`monitors/__init__.py:60-64`).

The **file monitor is the system of record**. It writes, under `<run_dir>/monitors/file/`: a JSONL of metric rows (`metrics.jsonl`), an append-only **trace stream** of every finished `vf.Episode` in arrival order (chunked, zstd-sealed, with a byte-offset index), per-producer **annotation streams** (post-hoc facts: which traces were shipped, their advantages, the trainer's recomputed per-token logprobs/entropies), a **live** directory of in-flight rollout deltas, and an eval `plan.json`. Episodes are serialized **exactly once**; everything learned later is an append-only *update* keyed by `trace_id` that readers fold back on (`monitors/file/traces/update.py:1-11`). The dashboard, `eval --resume`, the benchmark harness and CI all read these files rather than W&B.

Metrics are organised on two axes. **Step-keyed rows** (training step, or eval step = policy version) come from the orchestrator (episode statistics per shipped batch), the trainer (loss/entropy/KL/perf/time) and eval epochs. **Time-keyed rows** (`step=None`) come from background samplers: the vLLM `/metrics` scraper (`inference/*`) and the orchestrator's periodic pipeline view (`dispatcher/*`, `concurrency/*`, `watcher/*`, `event_loop_lag/*`). Episode statistics are computed on a matrix of **scope** (`train/agg`, `train/<env>`, `eval/<env>`) × **subset** (`all` = everything that arrived, `effective` = the exact cohort shipped/scored) × **level** (episode-level sums vs per-agent trace-level metrics) (`orchestrator/metrics.py:277-401`). Most misreadings of prime-rl dashboards come from mixing these axes.

Evaluation is **not a separate engine**: `uv run eval` and SFT's `online-eval` drive the orchestrator's own `Dispatcher` + `EvalSource` + `EvalSink` + `ConcurrencyController` + `InferenceMetricsCollector` through an `EvalRunner` (`src/prime_rl/eval/runner.py:1-12`), and log through the same monitors. RL online eval runs inside the orchestrator itself (`orchestrator.py:775-799, 911-956`).

---

## 2. Where it runs

| Process | Monitors registered (rank 0 only) | `producer` tag | Heartbeat beats on | Other ops surfaces |
|---|---|---|---|---|
| Orchestrator (`uv run orchestrator`, spawned by `rl`) | file, wandb, prime(train) — `orchestrator.py:208-217` | `orch` | every shipped batch (`orchestrator.py:745-746`) | InferenceMetricsCollector (5 s poll), PeriodicLogger (`log.interval`, default 10 s), live-delta relay (0.5 s) |
| RL trainer (torchrun, `prime_rl.trainer.rl.train`) | file, wandb — `trainer/rl/train.py:85-93` | `trainer` | every step on `world.is_master` (`train.py:97-99, 714-715`) | Prometheus `MetricsServer` on master, `HealthServer` on other nodes' local rank 0 (`train.py:102-112`); AnnotationWriter (per-step gather) |
| SFT trainer | file, wandb (`overview_flavor="sft"`) — `trainer/sft/train.py:80-90` | `trainer` | every step on `world.rank == 0` (`sft/train.py:94-96`) | — |
| `online-eval` (SFT launcher sidecar) | file, wandb (no prime) — `eval/online.py:47-55` | `online-eval` | every landed episode (`runner.py:236-238`) | InferenceMetricsCollector, PeriodicLogger |
| `uv run eval` | file, wandb, prime(eval) — `eval/eval.py:33-42` | `eval` | every landed episode | env-server children, dashboard daemon auto-start |
| Dashboard daemon (`uv run dashboard`) | none — pure reader | — | — | FastAPI/uvicorn on `127.0.0.1:7788+` |
| Inference (vLLM) | none | — | — | vLLM Prometheus `/metrics` scraped by the orchestrator/eval |

Rank gating: `monitors.setup` returns early unless `int(os.environ.get("RANK", os.environ.get("DP_RANK", "0"))) == 0` (`monitors/__init__.py:67-69`), so non-zero ranks' fan-out calls are no-ops. Only torchrun sets `RANK` (trainers); no launcher, sbatch template or k8s chart exports `RANK`/`DP_RANK` into the orchestrator/eval env (the templates' `*_RANK` names are unexported shell locals), and nothing else in the repo reads `DP_RANK` — a `RANK` leak can only come from the user's shell. The trainer's per-token annotations are explicitly gathered to rank 0 before logging (`trainer/rl/annotations.py:69-81`).

Lifecycle (every monitor-carrying process):

```
start ──► setup_logger ──► monitors.setup(...)          (init: wandb.init / pr.init / open files)
      ──► [steady] monitors.log(dict, step) | log(episodes, step, kind, subset)
                   log_annotations(updates) | log_live(events) | log_eval_plan | log_eval_epoch
      ──► clean exit: stop background loggers ─► monitors.finalize()  (seal chunks, wandb.finish, pr finish)
      ──► crash: finalize is NOT called (orchestrator.py:442-455, eval.py:65-68, online.py:205-208);
                 @clean_exit calls wandb.finish(exit_code=1) (utils/utils.py:60-66); prime SDK's
                 atexit hook marks the platform run crashed (monitors/prime.py:54-58)
```

---

## 3. Mechanics

### 3.1 Monitor registry and fan-out

- `MONITORS: list[Monitor]` is a process-global (`monitors/__init__.py:45`); `setup` asserts it is called once (`:66`).
- Construction order is prime → wandb → file (`:71-91`); each `await monitor.init(**kwargs)` runs sequentially, then the monitor is appended (`:97-100`). A raise in `init` propagates (a configured monitor must work).
- `log(data, step, kind="train", subset="effective")` dispatches by type: a `dict` goes to `Monitor.log_metrics(metrics, step)`; an `Episode | list[Episode]` goes to `Monitor.log_episodes(episodes, step, kind, subset)` (`monitors/base.py:32-45`). Each monitor call is awaited in order and wrapped in `try/except Exception → warning` (`monitors/__init__.py:126-130`) — slow monitors delay later ones and the caller.
- Optional hooks with no-op defaults: `log_annotations`, `log_live`, `log_eval_plan`, `log_eval_epoch`, `finalize` (`monitors/base.py:56-75`).
- `get(monitor_cls)` returns the registered instance of a type or `None` (`monitors/__init__.py:104-107`).

Which monitor consumes what:

| Call | File | W&B | PrimeTrain | PrimeEval |
|---|---|---|---|---|
| `log(dict, step)` | row in `metrics.jsonl` | `wandb.log` | `run.log_metrics` (queued) | ignored (`prime.py:177-178`) |
| `log(eps, kind, "all")` | append to trace stream | ignored (`wandb/monitor.py:155-156`) | ignored | eval only: stream to platform eval (`prime.py:244-255`) |
| `log(eps, kind, "effective")` | ignored (`file/monitor.py:134-135`) | ignored | train only: `run.log_episodes` (`prime.py:124-132`) | ignored |
| `log_annotations` | producer's annotation stream | — | — | — |
| `log_live` | `traces/live/` files | — | — | — |
| `log_eval_plan` | `plan.json` merge | — | — | opens platform evaluation |
| `log_eval_epoch` | — | — | — | `run.finish(metrics)` |

### 3.2 File monitor: `metrics.jsonl`

- Path: `get_file_monitor_dir(output_dir) / config.path` = `<run_dir>/monitors/file/metrics.jsonl` (`file/monitor.py:40`, `utils/pathing.py:277-280`; `FileMonitorConfig.path` default `metrics.jsonl`, absolute paths win — `configs/monitors.py:27-29`).
- Opened in append mode, **line-buffered** (`open(path, "a", buffering=1)`) "so a concurrently-running dashboard can tail the file" (`file/monitor.py:43-44`).
- Each `log_metrics` writes one row `{"step": step|null, "time": time.time(), **sanitized, "producer": <producer>}` (`file/monitor.py:57-60`). `sanitize` recursively drops NaN/±inf and warns with the dotted paths (`utils/utils.py:134-156`; `file/monitor.py:51-55`).
- **One row per `monitors.log` call**, not per step: the RL trainer alone emits 5 rows per step (perf, optim, tensor stats, time, disk — `trainer/rl/train.py:662-697`); readers must merge rows by `(step, producer)` (the benchmark harness merges by step: `benchmarks/scripts/run_single_benchmark.py:215-221`).
- In an `rl` run, the orchestrator and the trainer both open the *same* `metrics.jsonl` (both `output_dir`s are forced to the run dir — `configs/rl.py:380-396`); rows are distinguished only by `producer`.

### 3.3 File monitor: the trace stream (`traces/stream/`)

Directory layout (`monitors/file/traces/__init__.py:13-30`):

```
<run_dir>/monitors/file/
├── metrics.jsonl
├── plan.json                               # eval epochs' expected episode counts
├── eval.json                               # (uv run eval only) config stamped beside the episodes
└── traces/
    ├── stream/00000.jsonl.zst              # sealed chunk (seekable zstd)
    ├── stream/00001.jsonl                  # live chunk (plain, tail-able)
    ├── stream.index.jsonl                  # one row per episode: summary + (chunk, offset)
    ├── annotations/<producer>/00000.jsonl  # trace updates, one writer per producer
    ├── annotations/<producer>.index.jsonl  # {trace_id, chunk, offset, info}
    └── live/<trace_id>.jsonl, live/pending/<dispatch_id>.json
```

Write path (`file/monitor.py:128-152`):
1. Only `subset == "all"` is written; the `effective` cohort "writes no second copy" (`:134-135`).
2. In a worker thread (`asyncio.to_thread`, awaited so appends never interleave — `:150-152`), for each episode: `record = episode.to_record(float_decimals=config.float_decimals)`; `chunk, offset = stream.append(orjson.dumps(record, OPT_APPEND_NEWLINE|OPT_SERIALIZE_NUMPY))`; `self._logged += 1`; write `index_row(self._logged, record, chunk, offset)` to the index (`:142-146`).
3. Flush the stream **before** the index so "a row a reader sees points at a record it can read" (`:120-126, 147-148`).
4. On relaunch, `_logged` is initialised from the existing index line count so numbering continues (`:37-39`).

`Episode.to_record` = `model_dump(mode="json", exclude={"traces": {"__all__": EXCLUDE_FIELDS}}, exclude_none=True, context={"float_decimals": ...})` (`deps/verifiers/verifiers/v1/episode.py:153-163`), with `EXCLUDE_FIELDS = {"nodes": {"__all__": {"multi_modal_data", "routed_experts", "sampling_mask"}}}` (`deps/verifiers/verifiers/v1/trace.py:43-51`). Per-token floats are rounded to `FileMonitorConfig.float_decimals` (default 4, `configs/monitors.py:40-43`).

`ChunkedJsonl` (`file/traces/chunks.py:61-115`):
- `append(line)` rolls to a new numbered chunk when `size + len(line) > max_bytes` (and the current chunk is non-empty) — a line never spans chunks (`:81-88`). `max_bytes = FileMonitorConfig.chunk_bytes` default `5 * 1024**3` (`configs/monitors.py:31`).
- Rolling seals the finished chunk on a **non-daemon thread** (`:111-115`): write `NNNNN.jsonl.zst.tmp` with `pyzstd.SeekableZstdFile(level=3, max_frame_content_size=4 MiB)`, atomically rename to `.zst`, then unlink the plain file (`:48-58`). `compress=False` disables sealing.
- On open, a relaunch appends to a still-plain last chunk or starts a new one after a sealed one, and re-seals any other plain chunk (left mid-seal by a crash) (`:70-79`).
- `close()` seals the live chunk too ("a finished run keeps nothing plain") and joins seal threads (`:93-101`). This only happens in `finalize` (`file/monitor.py:172-176`) — i.e. only on a clean exit. The dashboard uses "no plain `*.jsonl` left in `stream/`" as the "eval finished" signal (`dashboard/server.py:340`).
- Readers use `open_chunk(dir, n)`: plain file if present else the seekable `.zst` (`:39-45`).

Index row (`file/traces/index.py:47-146`): `summarize_episode` computes per-episode `rewards` (per-reward-name mean over traces), `metrics`, `timing` (flattened phase→seconds), `cost` (sum of `calls[].usage.cost`), `line`, `offset`, `id`, `kind` (from `traces[].info.kind`, else `run.work.type`, else `run.type`, else `"eval"` — `:22-29`), `trace_ids`, `env` (`env.name` else `env.id`), `group`, `ok`, `num_errors`, `reward` (mean over reward-bearing traces of $\sum_r \text{score}_r \cdot \text{weight}_r$ — `:71-78`), `advantage` (mean of `trace.info.advantage`), token counts, `turns` (count of assistant nodes), `branches` (#leaves), `stop_condition` (last trace's), `truncated` (stop in `TRUNCATING_STOPS` or last successful call `finish_reason=="length"` — `:32-44`), `dispatch`/`arrival` times and `duration` = arrival − dispatch (`:128-136`). `index_row` drops `rewards/metrics/timing` and adds `chunk` (`:139-146`).

### 3.4 Annotations: post-hoc facts about traces

Record (`file/traces/update.py:28-40`): `{"version": 1, "trace_id", "info": {...}, "branches": [{"index": i, "advantages"|"trainer_logprobs"|"entropies": [float|null, ...]}]}`. Streams are **full-length over the branch's token prefix**, `null` at unknown positions, so a reader never needs the producer's loss mask (`:1-11`).

Producers:
1. **Orchestrator at ship time** — `stamp_batch(effective.vf_episodes, step)` (`orchestrator/annotations.py:33-49`) emits `info = {"effective": True, "ship": {"step", "time"}, "advantage"?}` plus per-trainable-branch `advantages`. Called for train at `orchestrator.py:630-631` and for each finished eval epoch at `orchestrator.py:920-922` / `runner.py:268-270` (so for eval, `ship.step` = the eval step). The scalar advantage reaches `trace.info.advantage` only on the model copies built by `EpisodeCollection.vf_episodes` (`orchestrator/metrics.py:441-456`).
2. **Trainer per step** — `AnnotationWriter.export(micro_batch, model_output)` builds per-sample `trainer_logprobs` and `entropies` streams over each packed sample's span, null outside the loss mask, first position nulled (right-shift crosses the packing boundary), trailing pad trimmed (`trainer/rl/annotations.py:32-67`); CP ranks > 0 skip (`:29, 33-34`). `flush()` `dist.gather_object`s every rank's records to rank 0 and calls `monitors.log_annotations` (`:69-81`). Created unconditionally (`trainer/rl/train.py:239-241`), exported every micro-batch (`:543`), flushed every step (`:559`).

Arrival stamping (before the `all` log): `stamp_arrival` writes `trace.info.kind`, `trace.info.dispatch = {step: work.step, time: trace.timing.start}`, `trace.info.arrival = {step, time}` (`orchestrator/annotations.py:20-30`; called at `orchestrator.py:545`, `runner.py:236`).

File write: `log_annotations` appends to `annotations/<producer or "unknown">/` and writes `update_index_row = {trace_id, chunk, offset, info}` to `annotations/<producer>.index.jsonl` (`file/monitor.py:154-170`, `update.py:21-25`).

Fold (reader side, `update.py:67-100`): apply updates in write order, deep-merge `info` (newest wins), and project each branch stream onto the branch's root-to-leaf node path (`branch_node_paths`, `:43-58`), keeping only `mask`-sampled positions; a node takes a stream only if fully covered with non-null values. Returns #nodes carrying `trainer_logprobs`.

### 3.5 Live traces (in-flight rollouts)

Source → sink:
1. Env servers stream deltas per rollout (`verifiers.v1.serve.delta`; schema owned by G/F). The dispatcher's `on_delta` callback folds each delta into its `InflightEpisode.live` bookkeeping (`orchestrator/live.py:44-56`) and queues `{"delta": delta, "dispatch": dispatch_info(meta)}` (`orchestrator/dispatcher.py:583-591`).
2. Bookkeeping events: `{"pending": dispatch_id, "dispatch": {...}}` at dispatch (`dispatcher.py:614`, `live.py:34-36`), `{"dispatched": dispatch_id}` once the first trace streams or the episode leaves (`live.py:39-41`), `{"done": trace_id}` when an episode finishes (`monitors/base.py:60-63`).
3. `publish_live` flushes the buffer to `monitors.log_live` every `LIVE_INTERVAL_S = 0.5` s (`dispatcher.py:62, 382-397`). Buffer cap `LIVE_EVENT_CAP = 20_000`; past it the deltas are trimmed to the newest `LIVE_EVENT_CAP // 2` = 10,000 (bookkeeping events always kept), warned once (`dispatcher.py:63-65, 368-380`). `retire(meta)` emits `dispatched` (if no trace ever streamed) and one `{"done": trace_id}` per live trace when an episode leaves the in-flight set (`dispatcher.py:361-366`).
4. `FileMonitor.log_live` (`file/monitor.py:71-118`): first call wipes `traces/live/` (stale from a previous attempt; "only the process that streams clears them"); `pending` → atomic write of `live/pending/<id>.json`; `dispatched` → unlink placeholder; `done` or `delta.discard` → unlink `live/<trace>.jsonl`; otherwise append the delta (first line — the one with `"open"` — carries `"dispatch"`), batching all lines per file per flush.

`dispatch_info` fields: `id, kind, env, group, task (name or "idx=N"), policy_version, step, started` (wall time reconstructed from monotonic, `orchestrator/live.py:17-31`).

Readers: `read_live`/`LiveFolds` fold lines with `verifiers.v1.serve.EpisodeAssembly` (torn trailing line waits; a file shorter than consumed means a new attempt → refold) (`file/traces/live.py:63-136`); `stage()` derives `pending/boot/setup/running/finalize/scoring/done/error` from `timing` spans (`:151-167`); `live_rows` merges pending placeholders with streaming traces (`:233-241`); `live_etag` = md5 of `name:size` of all live/pending files (`:244-259`). CLI: `uv run python -m prime_rl.monitors.file.traces <run_dir> [<trace_id>]` (`__main__.py`, `live.py:262-285`).

### 3.6 Eval plan

`log_eval_plan(env, step, expected)` is called when an eval epoch starts — orchestrator: `expected = group_size × len(examples)` (`orchestrator.py:793-798`); runner: `eval_sink.batch_size_for(env)` (`runner.py:185-186`). `FileMonitor` merges `{env: {str(step): expected}}` into `plan.json` via tmp+replace (`file/monitor.py:62-69`). The dashboard's eval progress bar reads it (`dashboard/server.py:341-342`). `PrimeEvalMonitor` uses it to open the platform evaluation (`prime.py:241-242`).

### 3.7 W&B monitor

`WandbMonitor.init` (`wandb/monitor.py:30-140`):
- `$WANDB_ARGS` (JSON argv set by the launcher) replaces `sys.argv` so W&B records the user's command (`:40-43`; set at `entrypoints/rl.py:307, 357`). `WANDB_DISABLE_WEAVE=1` default (`:46`).
- **Shared mode** iff `$WANDB_SHARED_MODE == "1"` and `$WANDB_MODE` not in `{disabled, offline}` (`:50-53`). Then `run_id = $WANDB_RUN_ID`, `label = $WANDB_SHARED_LABEL`, primary iff `label == $WANDB_SHARED_PRIMARY` (default `"orchestrator"`), finisher iff `label == $WANDB_SHARED_FINISHER` (default primary) → `wandb.Settings(mode="shared", x_label, x_primary, x_update_finish_state)` (`:54-71`).
  - `rl` launcher: always shared; `WANDB_RUN_ID = $PRL_RUN_ID`; labels `orchestrator` / `trainer` (`entrypoints/rl.py:167-174, 310-311, 360-361`). `SharedWandbConfig.offline=True` is rejected (`configs/rl.py:81-93`).
  - `sft` launcher: shared only when `[eval]` is configured; primary `trainer`, finisher `online-eval` (`entrypoints/sft.py:336-348, 421, 454`).
- Else `mode = os.environ.get("WANDB_MODE", "offline" if config.offline else "online")` and `run_id=None` (`:72-77`). So with `WANDB_MODE=offline|disabled` (CI sets `offline`, `gpu_tests.yaml`) an `rl` run's trainer and orchestrator each create their **own** W&B run — the shared-run contract only holds online.
- `wandb.init(id, resume="allow" if id, project, entity, name, group, tags, dir=output_dir, config=config.model_dump(), settings)` with retries: 30 × 10 s for non-primary shared processes waiting for the primary to create the run, else 5 × 10 s for transient `CommError` (+`ServerResponseError` in shared mode); `wandb.teardown()` between attempts to clear the stream mux (`:79-116`).
- Axes: `wandb.define_metric("*", step_metric="step")` (`:118`). Every `log_metrics` stamps `_timestamp = time.time()`; step-keyed rows add `"step": step` (`:153`). For `step=None` rows the **first path component** of each key gets `define_metric(f"{prefix}/*", step_metric="_timestamp")` on first sight (`:144-151`).
- Overview view: on the primary (or sole) online process, `ensure_overview_view` builds a curated saved view (`:126-138`). `build_sections` flavours `rl` / `sft` / `eval` (`wandb/overview.py:151-188`): train sections (`train/agg` + per-env when >1 env) with `effective/num_total_tokens|num_turns|num_branches/mean` and regex panels for per-agent `reward/mean` (all+effective), `all/*/has_error/mean`, `effective/*/is_truncated/mean`; eval sections with `avg@.*` regexes + `all/cancelled/mean`; `stability` (`optim/grad_norm`, `entropy/all/mean`, `mismatch_kl/all/mean`, `kl_ent_ratio/mean`); `inference` (fleet aggregate + min/max tail panels, x = `RelativeTime(Wall)`); `performance` (`perf/mfu`, `time/step`, `time/wait_for_batch`, `time/wait_for_policy`) (`overview.py:35-91`). A view is reused if its `view_signature` (env sets + panel set) matches, else a versioned `overview-vN` is created (`overview.py:207-261`). The dashboard mirrors these constants in `static/app.js` (`overview.py:3-5`; app.js 425-435).
- `log_episodes` is a no-op: W&B receives **scalars only** (`:155-156`).
- `finalize` calls `wandb.finish(exit_code=0)` in a thread; explicit because in shared mode the atexit finish "does not land the run state" (`:158-164`).

### 3.8 Prime platform monitors

`PrimeTrainMonitor` (orchestrator only; `configs/rl.py:105-106`) — thin wrapper over the `prime_runs` SDK (`monitors/prime.py:51-138`):
- `init`: if `$RUN_ID` set, attach (`pr.init(id=...)`); else register with `name`, `model`, `environments=[env.env_id ...]`, `TrainingSpec(max_steps, batch_size, rollouts_per_example=group_size, seq_len, wandb_project)`, and the full config dump (`:68-92`). Mode is `$PRIME_RUNS_MODE` (`pr.MODE_ENV`) or `"online"`; base URL from `$PRIME_API_BASE` with trailing `/rft` stripped (`:44-48, 98-105`). `finish_timeout = 60 s` (`:26`). Writes `monitors/prime/run.json = {"kind":"train","id","url"}` atomically for the dashboard's "view on platform" link (`:29-36, 106-110`; path `utils/pathing.py:270-274`).
- `log_metrics`: sanitize (warn on dropped), then `run.log_metrics(metrics, step=step)` in a thread (a queue put onto the SDK's uploader thread) (`:114-122`).
- `log_episodes`: only `kind=="train"` and `subset=="effective"` → `run.log_episodes` (`:124-132`). Per the docstring the SDK uploads every 10th step's samples as Parquet (`:54-58`) — `[UNVERIFIED: SDK-internal]`.
- `finalize`: `run.finish` (drains queue). A process exiting without finish is reported crashed by the SDK's atexit hook (`:54-58`).

`PrimeEvalMonitor` (`uv run eval` only) (`prime.py:141-276`): one platform evaluation per `(env, step)`, opened lazily under an `asyncio.Lock` on first `log_eval_plan`/episode (`run_for`, `:215-239`); a failed open is cached as `None` so the epoch does not retry (`:221-226`). `$PRIME_RUNS_EVAL_ID` attaches to a pre-created evaluation and requires exactly one source (`:164-169, 182-194`). Episodes with `kind=="eval", subset=="all"` stream as they land (`:244-255`); `log_eval_epoch` finishes with `pr.metrics.from_episodes(episodes)` (`:257-266`); `finalize` closes unfinished evaluations as `CANCELLED` (`:268-276`). Online mode requires `$PRIME_API_KEY` or `prime login` at init (`:151-155`). `run.json` holds `{"kind":"eval","run_id":$PRL_RUN_ID,"evaluations":{env:{step,id,url}}}`.

### 3.9 What gets measured

#### 3.9.1 Episode-statistics machinery (`orchestrator/metrics.py`)

- `TraceRecord(episode, trace, sampled, admitted, cancelled)` — every trace joined with its selection state (`:31-60`).
- `Stat(values)` → `{prefix}/{mean,max,min,p10,p90}` (linear-interpolated percentiles; empty → `{}` for `to_dict`, `0.0` for accessors) (`:63-102`).
- `TraceMetrics.to_dict(prefix, subset)` per agent (`:191-274`): distributions `reward, num_total_tokens, num_input_tokens, num_output_tokens, num_turns, num_branches` (5 stats each); rates `is_truncated, is_completed` (`/mean` only); `timing/{setup,agent,finalize,scoring,agent/model,agent/harness,total}` (5 stats; `total` = sum of the four phases, `:126-163`); `metrics/<name>` and `rewards/<name>` (5 stats; `None` values count as `0.0` on both subsets, `:166-188`); `stop_condition/generation_truncated` (share truncated and not `prompt_too_long`) and `stop_condition/<cond>` (share among traces with a recorded condition) (`:229-239`). `all`-only: `has_error/mean`, `cancelled/mean`, `error/<type>` (**counts**), and group solve rates `solved_none|solved_all|solved_some` (`:245-256, 268-272`).
- `EpisodeMetrics.to_wandb(prefix, subset)` (`:337-349`): **episode-level** `{prefix}/{subset}/{num_total_tokens,num_input_tokens,num_output_tokens,num_turns,num_branches}/<stat>` (summed over an episode's traces, `:294-318`), `all`-only `has_error/mean` (`not ok or any trace error`) and `cancelled/mean`, then **agent-level** `{prefix}/{subset}/{agent}/...` from `TraceMetrics`. Trace-level metrics never pool across agents (`tests/unit/orchestrator/test_metrics.py:141, 173`).
- `TrainMetrics` adds `{agent}/is_trainable/mean` and `{agent}/is_admitted/mean` (`:352-367`). `EvalMetrics` adds `{agent}/avg@{group_size}` (mean reward) and, on `effective` only and only if all rewards ∈ {0,1}, unbiased `pass@k` / `pass^k` for $k \in \{1,2,4,\dots\} \le n$ averaged over groups (`:370-401`; `orchestrator/utils.py:97-117`):
  $$\text{pass@}k = 1-\frac{\binom{n-c}{k}}{\binom{n}{k}},\qquad \text{pass}^k=\frac{\binom{c}{k}}{\binom{n}{k}}.$$
- Subsets: `TrainEpisodes.effective` keeps traces that are `admitted ∧ sampled ∧ ¬cancelled ∧ ¬has_error ∧ agent.trainable` (`:495-510`). `EvalEpisodes.effective` keeps `¬has_error ∧ agent.trainable` (`:536-542`).

#### 3.9.2 Orchestrator per-step rows (`finalize_train_batch`, `orchestrator.py:573-762`)

After the batch is shipped (`sender.send`, `:636`) and checkpointed:
- Episode matrix: `train/agg` and `train/<env>` × `all` (full arrival window `batch.episodes`) and `effective` (`batch.cohort.effective`) (`:643-649`).
- `dispatch_failure_metrics`: `{prefix}/all/dispatch_failure/mean` = failures/attempts and `/<error_type>` counts (`orchestrator/metrics.py:18-28`; `:650-663`).
- `progress/{tokens (full window), input_tokens, output_tokens (effective only), rollouts, tasks, total_tokens, total_rollouts, total_tasks}`; `time/{step, pack, save_ckpt, wait_for_policy}`; `step` (`:669-690`).
- Staleness of the shipped cohort `off_policy/{mean,max}`, `off_policy/{in_flight,in_queue}/{mean,max}` (from `episode_staleness`: $\text{total}=\max(0,(s-1)-\text{policy.start})$, $\text{in\_flight}=\min(\text{total},\text{policy.end}-\text{policy.start})$, $\text{in\_queue}=\text{total}-\text{in\_flight}$ — `orchestrator/utils.py:48-60`), `off_policy/dropped` (`:694-706`). (S9 owned by C.)
- `batch/<env>` (share of traces), `curriculum/<env>/{admission_rate, sampler/<name>, <custom>}` (`:707-709`; `train_source.py:56-65`).
- Console: `Step N | <time> | Reward x.xxxx | Trainable a/b (..%) | Turns | Branches | Max Off-Policy | Error | Cancelled | Truncation` with per-env `╰─` lines when >1 env (`:864-909`). Quality numbers are over the shipped cohort; `Error`/`Cancelled` over the full window.
- Warnings: ≤10% trainable effective traces (`:600-605`), `wait_for_policy ≥ active step time` (`:712-718`), >50% of the window discarded with stale/errored/no-signal breakdown (`:720-742`).

#### 3.9.3 Eval epoch rows

Orchestrator `finalize_eval_batch` (`orchestrator.py:911-956`) and `EvalRunner.finalize_eval_batch` (`runner.py:261-307`) both emit `eval/<env>/{all,effective}/...` via `EvalEpisodes.metrics.to_wandb`, `eval/<env>/all/dispatch_failure/*`, `eval/<env>/policy_version` and `step`. Differences:
- **`policy_version`**. Orchestrator = min over the epoch's episode policy-span starts *and* failures' `policy_version` (`:923-928`). RL in-orchestrator eval is **not version-pinned**: groups open at whatever version is current (`dispatcher.py:532`), weight updates keep landing while eval is in flight (eval dispatch only pauses during an update, `:450-457`), and in-flight episodes continue across the swap. So the logged value is a *lower bound* (≥ the trigger step), and the epoch can mix versions. Runner = `batch.step` (`runner.py:287`), which is exact: standalone eval runs one epoch at step 0, and SFT online eval reloads weights only between epochs (`online.py:164-197`).
- **`cancelled`**. `EvalEpisodes` never sets cancelled ids, so the generic `eval/<env>/all/cancelled/mean` (and the per-agent one) is always `0.0`. The runner overwrites the episode-level key with `cancelled/total_attempts` and adds `.../all/cancelled/count` only when the epoch was cut (`runner.py:284-286`). Its `dispatch_failure/mean` denominator also includes cancellations (`:278`).
- **Console**. Orchestrator: `Evaluated <env> | Policy v<min> | t | Reward | Turns | Branches | Error | Truncation` (`orchestrator.py:951-956`). Runner: `Evaluated <env> (Step N) | t | Reward | Turns | Branches | Error | Truncation`, or a WARNING-level `Partially evaluated ...` line when cancelled (`runner.py:294-307`).

#### 3.9.4 Time-keyed pipeline view (`PeriodicLogger`)

`PeriodicLogger(name, collect, interval=log.interval)` ticks every `interval` seconds (default `LogConfig.interval = 10.0`, `configs/shared.py:212-213`), logs the console body at INFO and `monitors.log(payload, step=None)` (`orchestrator/periodic_logger.py:33-50`). Orchestrator payload (`orchestrator.py:801-862`): `dispatcher.gauges()` (`dispatcher/{inflight/train, inflight/eval, queued/eval, mode (1=PREFER_EVAL), off_policy/max, off_policy/mean}` — `dispatcher.py:857-867`), `dispatcher.metrics.drained()` (`dispatcher/{cancelled,errored}/{train,eval,<env>}` counts since the last tick, **cleared on read** — `dispatcher.py:93-113`), `watcher.gauges()` (`watcher/{policy_version, update_count, last_update_weights_time, last_wait_for_ckpt_time}`), `concurrency.gauges()` (`concurrency/{max_inflight, turnover, capacity, signal, growth_multiplier}` — `concurrency.py:331-338`), and `event_loop_lag/{min,mean,median,p90,p99,max,n}` from a 100 ms sleep-drift sampler (`utils/async_utils.py:25-67`). The eval runner's payload omits watcher and lag (`runner.py:309-327`).

#### 3.9.5 Inference metrics (`InferenceMetricsCollector`)

- Polls `GET /metrics` on every admin client every `POLL_INTERVAL = 5.0` s with `FETCH_TIMEOUT = 5.0` (`inference_metrics.py:17-18, 296-305, 323-341`), plus a one-time `GET /v1/models` per endpoint for `max_model_len` (`:360-374`).
- Parses only `vllm:*` counter/gauge/histogram families; strips `vllm:`; sums samples across extra labels per engine label except `num_requests_waiting_by_reason` → `num_requests_waiting_reason_<reason>`; `vllm:cache_config_info` labels kept separately (`:88-119`).
- Per engine (`engine_id = "<server|prefill|decode><i>.<engine label>"`, `:122-147, 80-81`): gauges and counters verbatim, histograms as `_sum/_count`, and interval-derived `<counter>:rate` (per s), `<hist>:rate`, `<hist>:mean`, and per-engine `prefix_cache_hit_rate` (`:168-211`) → keys `inference/<engine_id>/<name>`.
- Scopes `agg` (all engines) and, when both P/D roles exist, `prefill`/`decode`: `inference/<scope>/<name>/{min,max,sum,mean,median}`, pooled histogram quantiles `/{p50,p90,p99}`, pooled `prefix_cache_hit_rate/pooled` (`:227-260, 415-449`).
- Always feeds `EngineLoadSample`s to the concurrency controller (`on_load`, `:347-352, 376-413`); logging to monitors only if `log_metrics` (`OrchestratorConfig.collect_inference_metrics`, default `True`, `configs/orchestrator.py:552-553`) — rows are `step=None` (`:354-358`).
- `probe()` scrapes up to 3× outside the loop; `EvalRunner` fails fast if no engine metrics and the concurrency band is not pinned (`runner.py:145-159`).

#### 3.9.6 RL trainer rows (`trainer/rl/train.py:619-697`)

- `perf/{throughput, throughput_per_gpu, mfu, peak_memory}` (`:655-662`), `optim/{lr, grad_norm}` (`:665-671`).
- Tensor stats: `Tensors.compute_stats()` all-gathers every key across ranks and emits `{key}/{mean,median,std,min,max}` (NaN when empty) (`trainer/utils.py:240-283`). Keys include `loss`, `entropy/all`, `entropy/<env>`, `mismatch_kl/all`, `mismatch_kl/<env>` (only over sampled-token positions, `train.py:517-541`), MoE stats, and every `loss_tensors` key from `compute_loss` (C owns). Derived `kl_ent_ratio/mean = mismatch_kl/all/mean / entropy/all/mean` (`:673-677`). `filter_rl_trainer_tensor_stats_for_wandb` drops `trainer_probs/`, `inference_probs/`, keeps only `{mean,std,max}` of 3-part `entropy/*` keys, and only `mean`/`max` (or 3-part `mean,std,max`) for `is_masked/`, `mismatch_kl/`, `masked_mismatch_kl/`, `unmasked_mismatch_kl/`, `ref_kl/...` (`trainer/utils.py:286-319`). The filter applies to **all** monitors, not only W&B (it is applied before `monitors.log`, `train.py:680`).
- `time/{step, wait_for_batch, load_data, broadcast_weights, save_ckpt, forward_backward}` (`:683-692`).
- `system/ckpt_disk_{free_gib, used_gib, total_gib, free_ratio}` via `shutil.disk_usage(<run_dir>/checkpoints)` (`trainer/utils.py:100-116`).
- Console `Step N | t | Loss | Entropy | Mismatch KL | Grad. Norm | LR | Throughput .. tokens/s | MFU ..% | Peak Mem. .. GiB [| Max Vio | Routing Conf.]` (`:642-652`); warning when `wait_for_batch ≥ active step time` (`:634-641`).

#### 3.9.7 SFT trainer rows

`progress/{epoch, num_samples, num_tokens, <subset>/ratio_samples, <subset>/ratio_tokens}`, `perf/*`, `optim/*`, `loss/{mean, perplexity = exp(min(loss,20)), nan_count}`, `val/{loss, perplexity}` every `val.interval`, `time/{step, save_ckpt, broadcast_weights, forward_backward}`, `system/*`, MoE stats (`trainer/sft/train.py:370-382, 596-662`).

### 3.10 Performance accounting (`trainer/perf.py`)

Peak FLOPs by `torch.cuda.get_device_name()` substring: A100 312 TF; H100/H200 989 TF (NVL 835, PCIe 756); GB200/GB300 2.5 PF; B200/B300 2.25 PF; MI300X/MI325X 1307.4 TF; **anything else → 312 TF with a warning** (`perf.py:71-109`).

Active matmul params per token $N_\text{act}$ (`perf.py:111-175`): attention Q/K/V/O (GQA, or MLA with `q_lora_rank`/`kv_lora_rank`), dense MLP $3 \cdot I \cdot h$ per dense layer, sparse MLP $3 \cdot I_\text{moe} \cdot h$ per (shared + top-k routed) expert per sparse layer, router $E \cdot h$ per sparse layer, plus LM head $V \cdot h$ (embeddings excluded).

FLOPs per token (`perf.py:177-213`), with $L$ layers, $H$ heads, $d_h = h/H$, $T$ = `seq_len`:
$$F_\text{tok} = 6N_\text{act} + 12\,L\,H\,d_h\,T \quad(\text{full FT}),\qquad F_\text{tok} = 4N_\text{act} + 2N_\text{trainable}^{\neg\text{lora}} + 6N_\text{lora} + 12LHd_hT \quad(\text{LoRA}).$$
Here $d_h$ is `hidden_size // num_attention_heads` (`perf.py:182-186`), **not** `config.head_dim`, which $N_\text{act}$ does use (`:121`). Models with `head_dim` ≠ $h/H$ therefore get a wrong attention term. The test value encodes this: Qwen3-0.6B at $T=1024$ gives $N_\text{act}=595{,}984{,}384$ and $F_\text{tok}=3{,}928{,}227{,}840$ (`tests/unit/train/test_perf.py:8-35`). Since $F_\text{tok}-6N_\text{act} = 12\cdot28\cdot16\cdot64\cdot1024$, the term uses $d_h=64$ where Qwen3's `head_dim` is 128, so attention FLOPs are halved.

$$\text{MFU}\,[\%] = 100 \cdot \frac{F_\text{tok}\cdot \text{TPS}}{P_\text{peak}\cdot W},\quad W = \text{world\_size (all trainer GPUs)} \qquad (\texttt{perf.py:68-69}).$$

- **RL trainer**: $\text{TPS} = N_\text{tok} / t_\text{fwd+bwd}$ with $N_\text{tok} = |\text{dp}| \cdot \text{seq\_len} \cdot n_\text{micro}$, where $n_\text{micro}$ = number of micro-batches on this rank (`train.py:292, 298, 623-629`). `seq_len = micro_batches[0]["input_ids"].shape[1]` is the length of **this rank's first micro-batch only**. Micro-batches are *not* padded to `model.seq_len`: FFD bins are variable-length (≤ seq_len), padded only to `pad_to_multiple_of`, and then balanced across ranks (`trainer/batch.py:690-727, 747-774, 876-893`; the "[1, seq_len]" docstring at `:883` is wrong). So the token count is an extrapolation from one bin. Benchmarks don't see this, because `FakeDataLoader` rows are exactly `seq_len` long (`trainer/rl/data.py:103-167`). $t_\text{fwd+bwd}$ spans from after `get_batch` to after `scheduler.step()` — includes grad clipping and the optimizer step, excludes batch wait/load, weight broadcast and checkpoint (`train.py:297, 578-579`). Single-step, unsmoothed.
- **SFT trainer**: sliding window over the last 10 full step wall-times via `count_tokens`/`get_tokens_per_second` = $\sum_{i\ge 1} n_i / (t_\text{last} - t_\text{first})$ (`perf.py:42-59`; `sft/train.py:566-576`), with $N_\text{tok} = |\text{dp}| \cdot \text{config.data.seq\_len} \cdot (\text{batch\_size}/|\text{dp}|)$. The first step logs throughput/MFU `0` (`get_* or 0`, window < 2).
- `perf/throughput_per_gpu = throughput / world_size`; `perf/peak_memory = torch.cuda.max_memory_reserved()/2^{30}` after `reset_peak_memory_stats()` at step start (`train.py:255, 630`).
- `get_perf_counter` is a process singleton: the model and `seq_len` of the **first** call are frozen (`perf.py:237-245`). In RL that is step 1's first-micro-batch length, which then fixes the attention term $T$ for the whole run. SFT always passes `config.data.seq_len`, so it is unaffected.

### 3.11 Logging

- `setup_logger(log_level, tag, json_logging, log_file, console_level)` builds a private loguru `Logger(core=Core())` so third-party code cannot hijack it (`utils/logger.py:101-182`). Console format `HH:mm:ss  LEVEL [tag] message` (+ ` [file::line]` at DEBUG) with colour; `logger.critical` is replaced with a no-op (`:176-177`).
- **JSON mode** (`LogConfig.json_logging`, default False, `configs/shared.py:206-207`): console sink is `json_sink` (flat JSON to stdout) with `enqueue=True` and a `traceback_patcher` that pre-formats exceptions (`:60-70, 143, 163-164`). Entry: `{timestamp, level, message, module, function, line, exception?, tag?, extra?}`; progress events `{timestamp, level, type:"progress", desc, current, total, percent, step?, extra?}` (`:16-57`). `ProgressTracker` uses tqdm in text mode, emits progress JSON every 10% in JSON mode (`:201-265`).
- `log_file` adds a plain-text file sink at `log_level` (used only by `uv run eval`, whose stdout no launcher redirects — `entrypoints/eval.py:139-140`). `console_level="ERROR"` quiets the eval console while running (`entrypoints/eval.py:192-194`).
- Stdlib → loguru: `InterceptHandler(prefix)` re-emits stdlib records at the matching level, forcing `logger.complete()` in JSON mode on exceptions (`:73-98`); `intercept_vf_logging` installs it on `verifiers.v1` (WARN in orchestrator and runner, `orchestrator.py:145`, `runner.py:63`; `orchestrator/utils.py:64-70`). Env servers get `setup_env_server_logging(vf_level, json_logging)` (`orchestrator/utils.py:73-80`) with `vf_level` from `LogConfig.vf_level` (`utils/pathing.py:242`).
- Levels: `LogConfig.level` default `$PRIME_LOG_LEVEL` or `info`; `vf_level` default `$PRIME_VF_LOG_LEVEL` or `info`; `TrainerLogConfig.ranks_filter=[0]` → `torchrun --local-ranks-filter` (`configs/shared.py:199-218`; `entrypoints/rl.py:337`).
- Files (A owns the layout): `<run_dir>/logs/attempt_<n>/` with `logs/latest` symlink (`utils/pathing.py:38-57`): `orchestrator.log`, `trainer.log` (launcher-captured torchrun stdout; `--tee=3` from ranks in `ranks_filter`), `trainer/torchrun/<rdzv>/attempt_0/<rank>/{stdout,stderr}.log` (`--log-dir`, `--redirect=3`), `inference.log`, `envs/{train,eval}/<name>.log`, `eval.log` (`entrypoints/rl.py:209, 267, 297, 336-339, 348`; `pathing.py:95-143`). SLURM job logs under `<run_dir>/launcher/logs/*job_*.log` (`pathing.py:115, 253-254`).

### 3.12 Heartbeat (Better Stack)

`Heartbeat(url).beat()` spawns a daemon thread doing `requests.get(url, timeout=1)` unless one is already pending — beats coalesce, never block (`utils/heartbeat.py:8-53`). Configured per process via `heartbeat: HeartbeatConfig(url)` (`configs/shared.py:221-223`). Beat cadence differs: trainer per step, orchestrator per shipped batch, eval runner per landed episode with deliberately **no startup ping** so boot latency is not a silent gap (`runner.py:81-87`; `configs/eval.py:29-34`).

### 3.13 Prometheus metrics server (trainer only)

`MetricsServerConfig(port=8000 [1..65535], host="0.0.0.0")` (`configs/shared.py:226-231`), `TrainerConfig.metrics_server = None` default (`configs/trainer.py:729`). `HealthServer` = stdlib `HTTPServer` on a daemon thread: `GET /health → 200 "ok\n"`, else 404 (`utils/metrics_server.py:20-74`). `MetricsServer(HealthServer)` adds `GET /metrics` in Prometheus text format from an isolated `CollectorRegistry` (`:77-136`). Gauges: `trainer_step`, `trainer_loss`, `trainer_throughput_tokens_per_sec`, `trainer_last_step_timestamp_seconds`, `trainer_grad_norm`, `trainer_peak_memory_gib`, `trainer_learning_rate`, `trainer_mfu_percent`, `trainer_entropy`, `trainer_mismatch_kl`, `trainer_kl_ent_ratio` (set only if entropy > 0) (`:88-105, 151-176`). Updated once per step on the master (`train.py:700-711`). `HTTPServer(...)` binds in the calling (main) thread (`metrics_server.py:58, 144`), so a taken port raises `OSError` at trainer startup rather than failing silently in the daemon thread. The SFT trainer, orchestrator and eval expose no Prometheus endpoint. Inference exposes vLLM's own `/metrics` on each engine (E owns), and the local `vllm-router` gets `--prometheus-port = server.port + 21000` (29000 by default) (`entrypoints/inference.py:188-189`). The only consumer of the trainer's `/health` and inference's `/liveness` is the k8s chart's probes, which are disabled by default (`k8s/prime-rl/templates/deployment.yaml:199-216, 345-362`; `values.yaml:83`).

### 3.14 Standalone eval (`uv run eval`)

Launcher `entrypoints/eval.py:103-199`:
1. `expand_shorthands`: leading `<taskset-id>` and `--env.<path> <value>` fold into one JSON `--source '[{...}]'`; `-c N` → `--concurrency.min_inflight N --concurrency.max_inflight N`; refuses shorthands next to a TOML defining `[[source]]` (`:48-84`; `tests/unit/eval/test_cli.py`).
2. Parse `EvalConfig` (`configs/eval.py:48-118`); `$PRL_RUN_ID` defaulted to a uuid, `$PRL_RUN_NAME = run.name` (`:129-131`). Run name auto `<envs>--<model>--<8hex>` (`configs/eval.py:104-118`).
3. `validate_run_dir` (refuses reuse unless `--resume` or `--clean`; `NEVER_CLEAN` env disables clean) → `prepare_attempt_dirs` → log file `logs/attempt_n/eval.log` → write launch artifacts, `resolved/eval.json`, and one `resolved/envs/eval/<name>.json` per source without `serve.address` (`:133-156`).
4. `--dry-run` exits here. Else `ensure_dashboard(output_dir)` (`:162`), spawn one `env-server @ <json>` per such source with stdout/stderr → `logs/attempt_n/envs/eval/<name>.log` (`:165-179`), SIGTERM handler cleans children, console → ERROR, `asyncio.run(run_eval(config))` (`:184-199`).

`Eval.run` (`eval/eval.py:25-57`): (resume prelude, §3.16) → `monitors.setup(producer="eval", overview_flavor="eval")` → `resume.stamp_config` (writes `monitors/file/eval.json` **after** monitors start, i.e. only for an attempt that actually runs) → `runner.setup()` → `runner.start()` → `fired = eval_source.trigger(0)` → `run_epoch(fired, 0, restored=...)` → `drain()`. `run_eval` finalizes monitors only on clean exit (`:60-68`).

`EvalRunner.setup` (`runner.py:77-166`): `InferenceClient` + `AdminPlane` (policy client; `model` from config), `EvalEnvs(config.source, env_addresses, config_dir).start()` (addresses via published address files, S3), `admin_plane.wait_for_ready`, `EvalSource(intervals=None for eval)`, `EvalSink`, `Policy(version=0)`, `ConcurrencyController` (fallback per-episode cost = max `max_completion_tokens` or 8192), an eval-only `Dispatcher(train_envs=None, ..., max_off_policy_steps=0)`, controller bound **without** `on_overload` (eval episodes are never cancelled on load), `InferenceMetricsCollector` probe + start, `PeriodicLogger(name="Eval")`.

`run_epoch(fired, step, restored, superseding_step)` (`runner.py:172-249`): `log_eval_plan` per env; switch dispatcher to `PREFER_EVAL`; land restored episodes first; then loop on `dispatcher.out_q`: `GroupCancellation → eval_sink.cancel`, `DispatchFailure → eval_sink.fail`, episode → `stamp_arrival`, `heart.beat()`, `land()` (= `monitors.log([ep], step, "eval", "all")` + `eval_sink.add`); a completed `EvalBatch` → `finalize_eval_batch` (effective log + annotations + `log_eval_epoch` + metric row + console line). With `superseding_step` (online eval), poll every `POLL_INTERVAL_S = 2.0` and, when a newer published checkpoint appears, `dispatcher.cancel_eval_step(step)` once.

Default client is Prime Inference (`PRIME_INFERENCE_URL = "https://api.pinference.ai/api/v1"`, key `PRIME_API_KEY`, model `deepseek/deepseek-v4.1-flash`, concurrency pinned 128) (`configs/eval.py:45-63`).

### 3.15 Online eval

**RL** (inside the orchestrator; B owns): the weight watcher's `on_update` → `trigger_eval(step)` fires due envs via `EvalSource.trigger(step, force=is_final)`, flips `PREFER_EVAL`, logs the plan; eval episodes flow through the same `main_loop` → `eval_sink` → `finalize_eval_batch` (`orchestrator.py:386-388, 548-553, 775-799`). Skipped on the resume step unless `eval.retrigger_on_resume` (`:780-781`).

**SFT** (`python -m prime_rl.eval.online @ eval.json`, spawned by the `sft` launcher; `eval/online.py`): `SFTOnlineEvalConfig` = `ScheduledEvalConfig` + `ServedEvalConfig` (+ `cancel_on_new_checkpoint=True`, `model`, `weight_broadcast`, `broadcasts_dir`, `max_steps`, `resume_step`, `output_dir`, `log`, `monitors: MonitorsConfig`) (`configs/eval.py:121-167`). `main` writes `resolved/eval.json` into the (launcher-pinned) attempt dir, sets up logging (no file sink; stdout goes to `eval.log` via the launcher) and runs (`online.py:211-219`). `OnlineEval.run`: monitors (`producer="online-eval"`, flavour `sft`), `runner.setup(skip_first_step, is_resumed)`, `setup_weight_receiver(broadcasts_dir, weight_broadcast or FileSystemWeightBroadcastConfig(), admin_plane, model)` (`online.py:44-74`). `watch()`: rendezvous with the trainer's startup broadcast (`sync_startup(startup_step, timeout=1200)`), eval step 0 unless resumed (or re-fire on resume with `retrigger_on_resume`), then poll `broadcasts/step_*` every 2 s: skip steps ≤ `last_step`; warn+skip due steps deleted by broadcast cleaning (`deleted_due_steps`); stop at the first unpublished step unless a newer one is published (then treat it as abandoned); for each published step `maybe_run_eval(step, reload_weights=True, force=is_final)`; exit once `last_step ≥ max_steps` (`online.py:76-135, 150-162`). `maybe_run_eval` **always receives the broadcast** even when no env is due, because a live-transport trainer blocks inside the handshake (`online.py:164-197`); weight reload is bracketed by `dispatcher.on_version_pending(step)` / `on_new_version(step)`. E owns the receiver protocol.

### 3.16 Eval resume (`eval/resume.py`)

`uv run eval ... --run.name X --resume`:
1. `previous_config(run_dir)`: the `eval.json` stamped beside landed episodes (current `monitors/file/` first, else newest archive) (`resume.py:63-69`).
2. `check_config(previous, current)`: dotted-path diff (`config_diff`, lists by index; different lengths = the whole list differs) must lie within `RESUMABLE = (resume, clean, dry_run, dashboard, num_examples, group_size, concurrency, client, log, monitors, source.*.num_examples, source.*.shuffle, source.*.group_size, source.*.serve)` (`:31-99`). Model, sampling and env changes raise.
3. `take_landed(run_dir)`: read every record of every attempt's stream (`monitors/file.attempt_N/traces/stream` oldest first, then current), keep `ok` records deduplicated by `id`, then **rename** `monitors/file` → `monitors/file.attempt_{k+1}`; nothing is deleted (`:114-135`). A torn last line ends a stream (`:102-111`).
4. After monitors start (fresh `monitors/file`), `plan(landed, eval_envs)`: target per `(env, task.key)` = `group_size`× occurrences in `env.examples`; keep landed episodes up to target as `vf.WireEpisode`; `owed` = target − kept; `groups` maps a task key to the landed group id so owed rollouts complete that group (`:138-169`). `eval_source.restore(owed, groups)` queues only owed rollouts (`tests/unit/eval/test_resume.py:56-64`).
5. Restored episodes are re-`land`ed first in `run_epoch` → re-streamed into the fresh stream, counted in the epoch metrics and platform upload (`runner.py:203-204`).

### 3.17 Dashboard

**Launch & discovery** (`entrypoints/dashboard.py`): state dir `~/.cache/prime-rl/dashboard/` with `daemon.json` (`{pid, url, started}`), `dirs.json` (registered output dirs), `daemon.log`, `.lock` (flock) (`:31-45`). Every launcher (`rl`, `sft`, `eval`; `dashboard: bool = True`) calls `ensure_dashboard(output_dir)`: mkdir + register dir; if `daemon.json`'s pid is alive and `GET <url>/api/runs` answers within 1 s, reuse; else, only in a TTY and only if `fastapi` and `uvicorn` import and a `dashboard` binary exists, spawn it detached (`start_new_session=True`, output to `daemon.log`) and wait ≤10 s for discovery (`:48-114`). The URL is printed as the launcher's last banner (`:117-121`).

**Server** (`dashboard/server.py:2192-2253`): args `output_dirs...` (default `default_output_dir()`), `--port 7788`, `--host 127.0.0.1`, `--isolated`. Non-isolated instances register their dirs and serve the union of CLI dirs + `dirs.json` (re-read on mtime change) (`:100-145`). `free_port` scans 100 ports up from `--port` with `SO_REUSEADDR` (`:2150-2165`). The first non-isolated instance `claim_daemon`s `daemon.json` (refuses if another live pid holds it) and releases on exit (`:2168-2189`). `uvicorn.run(..., timeout_graceful_shutdown=2)`, GZip ≥1000 B, `Cache-Control: no-cache` on `/` and `/static/*` (`:50-51, 2116-2124, 2245`).

**Data sources** — filesystem only, never W&B/network except HF tokenizer downloads (`Tokenizer.from_pretrained(model)` for token decoding, failure → `None`; `:1675-1687`). A run dir is any child of a tracked dir having `configs|logs|monitors|traces.jsonl` (`:91-94`); duplicate names across dirs are qualified `<parent>:<name>` (`:148-165`). Run type from resolved config files: `sft.json` → sft; `orchestrator.json|trainer.json` → rl; `eval.json` → eval (`:241-249`).

**HTTP API** (all GET unless noted):

| Route | Reads | Returns |
|---|---|---|
| `/api/runs` | scan tracked dirs, `run_meta` each | runs sorted by mtime (`:369-377`) |
| `/api/runs/{run}` | configs, `metrics.jsonl` first row, `logs/latest/*.log` mtimes, stream, `plan.json`, `monitors/prime/run.json` | `{name,type,finished,eval_plan,platform,model,dataset,has_validation,env,total_episodes,eval_totals,max_steps,train_envs,eval_envs,has_metrics,last_step,started,updated,created,mtime}` (`:300-366`) |
| `/api/runs/{run}/logfiles?attempt=` | `logs/attempt_n/**.log` (+ legacy `logs/envs`) | `{attempt, attempts, files:[{id,component,label,size,master}]}` (`:409-444`) |
| `/api/runs/{run}/log?file&start&end&tail` | byte range (≤2 MB), tail snaps to line, EOF drops torn line | `{text,start,end}` (`:447-469`) |
| `/api/runs/{run}/configs?attempt=`, `/config?file&attempt` | `configs/attempt_n/*.toml`, `command.txt`, `resolved/**.json` concatenated | (`:482-506`) |
| `/api/runs/{run}/reports`, `/report?file` | `<run>/reports/*.md` (frontmatter `title`) | (`:530-549`) |
| `/api/runs/{run}/metrics?offset=` | `metrics.jsonl` from byte offset; first chunk 4 MB then 16 MB | `{rows, offset, size}`; a truncated/replaced file resets offset (`:560-582`) |
| `/api/runs/{run}/episodes?step&kind&env&episode&errors_only&sort&order&offset&limit≤512&start&end&etag&upto` | index rows + annotation-derived `step` | paged rows; `upto` pins the list length for stable scrolling; `{unchanged}` on etag hit (`:1560-1620`) |
| `/api/runs/{run}/episodes/histogram` | index rows | arrival counts in "nice" bins (`:1623-1656`) |
| `/api/runs/{run}/rollouts` | annotation-derived `(kind, step)` pairs | `{steps:[{step,kinds}]}` (`:1659-1661`) |
| `/api/runs/{run}/live?etag`, `/live/{trace_id}?etag` | `traces/live/` via `LiveFolds` | rows / one assembled trace (`:1758-1796`) |
| `/api/runs/{run}/episodes/series?kind&etag&after` | full summaries (parsed from records, sidecar-cached) | per-episode series incl. `rewards/*`, `metrics/*`, `timing/*` (`:1799-1848`) |
| `/api/runs/{run}/episodes/{line}?tokens&rendered` | one record by `(chunk, offset)`, annotations folded, `train_annotations {step,nodes,eps}`; optional `token_strs` and full-branch decode | (`:1884-1916`) |
| `/api/runs/{run}/episodes/{line}/timeline` | one record | lanes/spans/semantic edges (`:2105-2108`) |
| `POST /api/view`, `GET /api/view/events` (SSE) | validates address against the FS | fan-out navigation command to connected tabs; `409` when none connected (`:1919-2102`) |
| `/`, `/static/*` | `static/index.html`, `app.js`, uPlot | UI |

Key mechanics: a row's `step` is the **min `ship.step` among `effective` annotations for any of its traces**, else `None` — "only a cohort ties to a step" (`:1470-1513, 620-629`). All caches are LRU (64 files) keyed by size/version since streams are append-only; `file_checkpoint` (64 bytes before the old EOF) detects rewrites (`:59-88, 632-672, 758-767`). Full episode summaries are persisted as sidecars in `~/.cache/prime-rl/dashboard/<sha256(path)[:24]>.json` (format 4, at most every 20 s) — the dashboard never writes into run dirs (`:743-803`). `ipo_eps` for the stable-mask overlay reads `trainer.json` `loss.eps` if `loss.type == "ipo"`, else 0.1 (`:903-908`).

**UI** (static JS, skimmed): tabs `overview | metrics | config | traces | logs | report` (`static/index.html`), polls `/api` every `POLL_MS = 5000` and live rollouts every `LIVE_POLL_MS = 1000` (`static/app.js:15, 7157-7167`). Overview mirrors the W&B sections; metrics tab charts every key; eval overview shows block progress bar (from `plan.json`), avg@k/pass@k tiles, beeswarms, usage/timing panes (docs/eval.md:92; skills/dashboard/SKILL.md:12-18). Trace viewer: Transcript/Messages/Rendered, Timeline, Replay, Semantic (README.md:3-48).

### 3.18 Trace tooling

- `tools/convert_traces_to_hf_dataset.py <traces.jsonl> --name <repo|dir> [--subset] [--split] [--public] [--local]`: one row per **branch** of every trace, unfiltered: `messages` (`message_to_wire`), JSON-encoded `tools`, `calls`, scalar outcome columns (`reward, stop_condition, has_error, is_truncated, ...`) and JSON-encoded variable-schema metadata (`:46-114`); push to Hub (private by default) or write `<name>/<subset>/<split>.parquet` + dataset-card `configs` (`:117-191`). It reads a **single plain JSONL file** (`:160-168`).
- `python -m prime_rl.monitors.file.traces <run_dir> [<trace_id>]` (§3.5).

### 3.19 Tests, CI, benchmarks — how a framework builder validates changes

Layout: `tests/unit/` (hermetic; subdirs `eval, inference, orchestrator, train{,/models,/rl,/sft}, transports, utils`, plus `test_configs.py`, `test_dashboard_timeline.py`, `test_parsers.py`), `tests/integration/` (full `uv run rl|sft|eval` on tiny models), `tests/nightly/` (example configs to convergence thresholds) (`docs/development.md:21-27`). Markers `gpu`, `slow`, `--strict-markers` (`pyproject.toml:341-346`). Unit files marked GPU: most of `train/models/*`, `train/rl/test_{fused_lm_head,loss}.py`, `train/test_{model,state_offload}.py`, `transports/test_nccl_broadcast.py` (grep of `mark.gpu`).

Fixtures (`tests/conftest.py`): autouse `setup_logger("debug")`/`reset_logger`, env snapshot/restore, `reset_world`; module-scoped `cleanup_zombies` `pkill`s `torchrun`/`VLLM` only when `USERNAME_CI` set; session `output_dir` = `$PYTEST_OUTPUT_DIR` or a tmp dir (never deleted); `run_process(cmd, env, timeout)` → SIGTERM then SIGKILL on timeout; W&B project gets `-local` suffix off-CI.

**Integration assertions parse console SUCCESS lines** (`tests/utils.py`): orchestrator `Reward:?\s+(\d+\.\d{4})` must go up / be ≥ threshold; trainer `Mismatch KL:?\s+(\d+\.\d{4})` avg over last 3 steps ≤ 0.01 (reverse-text) or within `[0, 5e-4]` (nightly hendrycks); eval `Evaluated gsm8k .*Reward ... | Turns ... | Branches ...` with reward ≥ 0.5 and branches == 1.0 (`tests/integration/test_reverse_text.py:102-120`, `test_gsm8k_eval.py:40-60`, `tests/nightly/test_hendrycks_sanity.py:58-87`). `test_reverse_text` also resumes the run and converts the final DCP checkpoint to bf16/fp8 and back, asserting byte-equality between direct and chained fp8 (`:123-153`). Six integration modules (`reverse_text`, `_rl_opd`, `_rl_sft`, `_sft`, `_sft_lora`, `alphabet_sort`; not lora/moe/gsm8k_eval/benchmark) append `make_dashboard_test(...)`. That is a Playwright/Chromium smoke of all dashboard tabs against the artifacts just produced (`tests/integration/dashboard_smoke.py:33-184`); it launches a non-`--isolated` `uv run dashboard`, so it joins the user's dashboard registry. `test_gsm8k_eval` needs no GPU: it runs `uv run eval` against the default Prime Inference API (`PRIME_API_KEY`, network) and parses `logs/latest/eval.log` (`test_gsm8k_eval.py:21-60`).

CI workflows:

| Workflow | Trigger | Runs | Where |
|---|---|---|---|
| `cpu_tests.yaml` | PR + push main | `pytest tests/unit -m "not gpu"`; slim `prime-rl-configs` wheel install that must import all configs without `torch, transformers, vllm, wandb, ring_flash_attn, prime, liger_kernel, loguru` in `sys.modules` | `ubuntu-latest` |
| `gpu_tests.yaml` | non-draft PR + push main (cancel-in-progress on PRs) | `pytest tests/unit -m gpu` (45 min); integration matrix `reverse_text_sft, reverse_text_sft_lora, reverse_text, reverse_text_rl_opd, reverse_text_rl_sft, alphabet_sort, reverse_text_lora, reverse_text_moe, gsm8k_eval` on `vm`, `benchmark_regression` on `4xa6000` (60 min); `WANDB_MODE=offline`; uploads `/tmp/outputs/**/*.log` on failure | self-hosted, container `pytorch/pytorch:2.10.0-cuda13.0-cudnn9-devel` |
| `nightly_tests.yaml` | daily 11:00 UTC + dispatch (optional single file) | every `tests/nightly/test_*.py`, one job each, 24 h timeout | `research-cluster` |
| `nightly-fft.yaml` | daily 11:00 UTC + dispatch (optional `config_file`, `image_tag`) | per `configs/ci/nightly-fft/*.toml`: resolve the newest pullable `v*.dev*` GHCR image, prepend `name = "nightly-<cfg>-<date>-<run_id>"`, then fire-and-forget `prime train <cfg> --image-tag <tag> -e WANDB_API_KEY -e HF_TOKEN` onto Prime's hosted platform (30 min dispatch job; results are not checked here) | `ubuntu-latest` → hosted |
| `benchmarks.yaml` | manual dispatch (image, `set_baselines`) | ~60-cell trainer matrix (model × GPU × rl/sft × LoRA × seq_len × AC × attn × cp/ep) → `aggregate_results.py --regression-threshold 0.05` → PR updating `benchmarks/results/BENCHMARKS.md` (+ baselines) | 1×/8× A6000/H100/H200/B200 |
| `style.yaml` | PR + push | `ruff check`, `ruff format --check` (v0.15.14) | ubuntu |
| `build_image`, `devx_tag`, `tag-and-release`, `sync-docs` | release plumbing | image build, tags, docs sync | — |

Benchmarks: `run_single_benchmark.py` runs the trainer **directly under torchrun with fake data** (`--data.fake.batch-size = micro_batches × num_gpus`, `max_steps=4`, `--model.compile`) and aggregates `perf/mfu`, `perf/throughput`, `time/step`, `perf/peak_memory` from `metrics.jsonl`, merging rows per step and dropping the first (warmup) step (`benchmarks/scripts/run_single_benchmark.py:119-249`). `test_benchmark_regression` compares 1- and 4-GPU Qwen3-0.6B RL runs (seq 65536, FA2, Recompute) against committed baselines: peak memory ≤ +1%, MFU/throughput/step time within ±15% (comments say 5%) (`tests/integration/test_benchmark_regression.py:21-283`). Baselines in `benchmarks/baselines/*.json` are `{config, metrics:{mfu,throughput,step_time:{mean,std,min,max}, peak_memory:{gib,pct}}}`.

---

## 4. Interfaces & contracts

### 4.1 Files written/read

| Path (under `<run_dir>`) | Writer | Reader(s) | When |
|---|---|---|---|
| `monitors/file/metrics.jsonl` | every process's FileMonitor (rows tagged `producer`) | dashboard, benchmark harness | each `monitors.log(dict)` |
| `monitors/file/traces/stream/NNNNN.jsonl[.zst]` + `stream.index.jsonl` | orchestrator / eval / online-eval (the process that runs episodes) | dashboard, `eval --resume`, `zstd -dcf \| jq` | each arriving episode |
| `monitors/file/traces/annotations/{orch,trainer,eval,online-eval}/` + `.index.jsonl` | the named producer | dashboard (fold) | ship time / trainer step / eval epoch end |
| `monitors/file/traces/live/<trace>.jsonl`, `live/pending/<id>.json` | FileMonitor of the dispatching process | dashboard (`LiveFolds`), traces CLI | every ≤0.5 s |
| `monitors/file/plan.json` | FileMonitor | dashboard | eval epoch start |
| `monitors/file/eval.json` | `resume.stamp_config` (`uv run eval`) | `resume.previous_config` | after monitors start |
| `monitors/file.attempt_N/` | `resume.take_landed` (rename) | resume | on `--resume` |
| `monitors/prime/run.json` | Prime monitors | dashboard top bar | init / epoch open |
| `wandb/` | wandb SDK (`dir=output_dir`) | — | — |
| `logs/attempt_N/...`, `logs/latest` | launcher / eval (A) | dashboard logs tab, tests | — |
| `configs/attempt_N/{command.txt, <entry>.toml, resolved/**.json}` | launcher (A) | dashboard config tab, `run_meta` | launch |
| `reports/*.md` | agents/humans | dashboard report tab | — |
| `~/.cache/prime-rl/dashboard/{daemon.json, dirs.json, daemon.log, .lock, <sha>.json}` | launchers / dashboard | launchers / dashboard | — |

### 4.2 Record schemas

- **metrics row**: `{"step": int|null, "time": float, "<key>": number, ..., "producer": str}` (NaN/inf dropped).
- **stream record**: `vf.Episode.to_record()` — `id, env{id,name}, task{type,data,key,hash}, group{id}?, run{type,id,name,work{type,step,policy{start,end}}}?, ok, errors[], traces[], num_input_tokens, num_output_tokens, num_total_tokens` (`episode.py:87-138`); each trace minus `nodes[].{multi_modal_data,routed_experts,sampling_mask}`, with `info.{kind,dispatch{step,time},arrival{step,time}}` stamped at arrival. Trace field semantics: F owns (07-verifiers-core.md). Fields the readers here rely on: `nodes[].{message,parent,token_ids,mask,logprobs,timestamp,semantic_parents}`, `calls[].{node,time{start,end},usage{...,cost},finish_reason,error}`, `rewards{name:{score,weight}}`, `metrics`, `timing` tree, `stop_condition`, `errors`, `is_completed`, `ok`, `agent{name,trainable,config{client{renderer_model_name},model}}`.
- **stream index row**: `{cost, line, offset, chunk, id, kind, trace_ids, env, group, ok, num_errors, reward, advantage, input_tokens, output_tokens, turns, branches, stop_condition, truncated, dispatch, arrival, duration}`.
- **annotation update**: `{version:1, trace_id, info, branches:[{index, advantages?|trainer_logprobs?|entropies?}]}`; index row `{trace_id, chunk, offset, info}`.
- **live file line**: delta from `verifiers.v1.serve` (`trace`, `open`, `calls`, `set`, `errors`, `discard`, ... — G/F own) with `dispatch{id,kind,env,group,task,policy_version,step,started}` on the `open` line.
- **plan.json**: `{env: {"<step>": expected_episodes}}`.

### 4.3 Environment variables

| Var | Effect | Where |
|---|---|---|
| `RANK`, `DP_RANK` | monitors register only on rank 0 | `monitors/__init__.py:67` |
| `WANDB_SHARED_MODE`, `WANDB_RUN_ID`, `WANDB_SHARED_LABEL`, `WANDB_SHARED_PRIMARY` (default `orchestrator`), `WANDB_SHARED_FINISHER`, `WANDB_MODE`, `WANDB_ARGS`, `WANDB_DISABLE_WEAVE` | W&B shared mode / command capture | `wandb/monitor.py:40-77` |
| `PRL_RUN_ID`, `PRL_RUN_NAME` | run identity stamped on episodes; W&B shared run id; platform eval config | `entrypoints/eval.py:129-131`, `runner.py:90-91`, `prime.py:158` |
| `RUN_ID`, `PRIME_RUNS_MODE` (`disabled` opt-out), `PRIME_API_BASE`, `PRIME_API_KEY`, `PRIME_RUNS_EVAL_ID`, `PRIME_TEAM_ID` | Prime platform | `prime.py:21-26, 44-48, 70, 101, 151-164`; docs/training.md:394 |
| `PRIME_LOG_LEVEL`, `PRIME_VF_LOG_LEVEL` | default log levels | `configs/shared.py:200-204` |
| `PRL_ATTEMPT_CONFIG_DIR`, `PRL_ATTEMPT_LOG_DIR` | pin child processes to a launch attempt | `utils/pathing.py:60-72` |
| `PRL_OUTPUT_DIR` | default output dir (also dashboard default) | `configs/eval.py:74-76` |
| `NEVER_CLEAN` | disables `--clean` for eval | `entrypoints/eval.py:133` |
| `USERNAME_CI`, `PYTEST_OUTPUT_DIR`, `GITHUB_REF_NAME` | test fixtures | `tests/conftest.py` |

### 4.4 Ports & endpoints

| Surface | Default | Routes |
|---|---|---|
| Trainer MetricsServer | `0.0.0.0:8000` (off unless `trainer.metrics_server` set) | `/metrics`, `/health` (master); `/health` (other nodes' local rank 0) |
| vllm-router Prometheus | `server.port + 21000` (29000) | router's own metrics (not scraped by prime-rl) |
| Dashboard | `127.0.0.1:7788`, bumps up to +99 | §3.17 table |
| vLLM (scraped) | per `client.base_url` / admin clients | `/metrics`, `/v1/models` (read) |
| Better Stack | `heartbeat.url` | GET |

### 4.5 Config fields

| Field | Type / default | Effect |
|---|---|---|
| `monitors.file` | `FileMonitorConfig \| None = FileMonitorConfig()` per process (on; `--no-monitors.file` disables). On `rl`, `SharedMonitorsConfig.file` defaults `None`, so trainer and orchestrator keep their own default (on). `rl --no-monitors.file` does reach both: pydantic-config encodes it as the string `"None"` (`deps/pydantic-config/.../cli.py:1182-1188`), and the presence pass fills `"None"` into absent sub-blocks (`utils/validation.py:166-179`). **But only the `path` leaf is propagated** (`:107-108`): shared `monitors.file.{chunk_bytes, compress, float_decimals}` validate and are then silently ignored (the `rl` launcher never reads `config.monitors`). Set them per sub-config. | local metrics + traces |
| `monitors.file.path` | `Path("metrics.jsonl")` | metrics file name |
| `monitors.file.chunk_bytes` | `5 GiB` | stream roll size |
| `monitors.file.compress` | `True` | zstd sealing |
| `monitors.file.float_decimals` | `4` (`None` = full) | rounding of per-token floats in records/annotations |
| `monitors.wandb.{project="prime-rl", entity, name (←run.name), group, tags, offline=False}` | off by default | W&B |
| `orchestrator.monitors.prime.name` / `eval monitors.prime.name` | off by default; name ← run.name | Prime platform |
| `log.{level, vf_level, json_logging=False, log_data=False, interval=10.0}`; trainer `log.ranks_filter=[0]` | | logging + periodic cadence |
| `heartbeat.url` | `HeartbeatConfig \| None = None` on trainer, SFT, orchestrator, eval | Better Stack |
| `trainer.metrics_server.{port=8000, host="0.0.0.0"}` | `None` | Prometheus |
| `orchestrator.collect_inference_metrics` | `True` | log `inference/*` rows (poll always runs) |
| `orchestrator.inference_metrics_roles` | `list["prefill"\|"decode"] \| None` | P/D scopes |
| `trainer.trace_path` / `memory_profiler_path` | `None`; trace requires `max_steps < 10` | torch profiler chrome trace per rank / CUDA memory snapshots (`train.py:247-250, 721-727`; `configs/trainer.py:772-780`) |
| eval: `model`, `client`, `concurrency` (pinned 128), `num_examples=-1` (`-n`), `group_size=1` (`-r`), `run`, `output_dir`, `clean`, `dry_run`, `dashboard=True`, `resume=False`, `tasks_per_minute`, `heartbeat`, `log`, `monitors` | `configs/eval.py:13-118` | standalone eval |
| online eval: `interval=100`, `skip_first_step=False`, `retrigger_on_resume=False`, `cancel_on_new_checkpoint=True` | `configs/orchestrator.py:391-418`, `configs/eval.py:130-133` | scheduling |

---

## 5. Invariants & assumptions

1. **Single writer per stream.** The trace stream has exactly one writer per run (the process that runs episodes: orchestrator in `rl`, eval runner in `eval`/`online-eval`); each annotation stream has one writer (its `producer`). Appends within a process are serialized by awaiting `to_thread` (`file/monitor.py:150-152`). Nothing but naming enforces this.
2. **Stream-before-index flush** (`file/monitor.py:120-126`) — any index row a reader sees points to a complete record; readers treat a torn trailing line as "not yet" (`server.py:1450-1459`, `live.py:66-67`, `resume.py:102-111`).
3. **Append-only everything** — dashboard caches key on file size (`server.py:607-610, 1664-1667`); any rewrite that keeps the size breaks caching except where `file_checkpoint` is checked (bare `traces.jsonl` and sidecars).
4. **Every episode is serialized once**, in arrival order, whatever its kind/fate (errored, rejected, cancelled-late, never batched) — the stream is crash-durable evidence (`orchestrator.py:535-546`).
5. **Only `effective` cohorts tie to a step.** Step membership comes solely from `effective` annotations' `ship.step` (`server.py:1503-1509`); `all` is the stream.
6. **Annotation streams are full-length over the branch's token prefix** with nulls; a node gets a stream only if fully non-null over its sampled positions (`update.py:84-99`). The trainer's writer relies on `micro_batch["trace_ids"]`, `branch_indices`, `sequence_lengths`, `loss_mask`, `env_names` (padding marked `env_name == ""`) (`trainer/rl/annotations.py:35-52`) — C/E must keep these fields on the micro-batch wire.
7. **A W&B top-level key prefix is either step-keyed or time-keyed, never both** — the first `step=None` row re-axes the whole `prefix/*` to `_timestamp` (`wandb/monitor.py:147-150`).
8. **`step` key present in step-keyed dicts** (every producer adds `"step": step` to its dict; the file row's `step` field comes from the argument, the dict's `step` overwrites it with the same value).
9. **Monitors finalize only on a clean exit** — finalize is what seals the live chunk and marks W&B/platform runs finished; the dashboard's "eval finished" depends on it.
10. **SFT online eval must receive every offered broadcast** (trainer blocks in the handshake) and evaluates each epoch at exactly one policy version (`online.py:1-11, 164-197`). RL in-orchestrator eval has **no** such pin (§3.9.3).
11. **Resume validity**: landed eval episodes may only be reused if model/sampling/env config is unchanged (`resume.py:31-99`), and are matched by `task.key`.

---

## 6. Extension points

**A new monitor backend** (e.g. MLflow, TensorBoard, ClickHouse):
1. Subclass `prime_rl.monitors.base.Monitor`; implement `log_metrics(metrics, step)` (handle `step=None` as wall-time rows) and `log_episodes(episodes, step, kind, subset)`; optionally `log_annotations`, `log_live`, `log_eval_plan`, `log_eval_epoch`, `finalize`, and an `init(**kwargs)` that names the kwargs it needs.
2. Add a config class in `packages/prime-rl-configs/src/prime_rl/configs/monitors.py` and a field on `MonitorsConfig` (+ `SharedMonitorsConfig` in `configs/rl.py:96-106` if `rl` should propagate it).
3. Extend `monitors.setup` signature and the construction block (`monitors/__init__.py:48-91`) — there is **no registry**; setup is a hand-written if-chain — and update **every call site** of `monitors.setup` (orchestrator `:208`, RL trainer `:86`, SFT trainer `:81`, eval `:33`, online eval `:47`).
4. Heavy blocking I/O must go to a thread (`asyncio.to_thread`) as the file/prime monitors do; the fan-out awaits sequentially.

**A new metric**: add keys to an existing `monitors.log(dict, step)` call (orchestrator `finalize_train_batch` dict, trainer dicts) or, for episode statistics, add a property to `TraceMetrics.stats()` / `EpisodeMetrics.to_wandb` (`orchestrator/metrics.py`). Time-keyed gauges: add to a `gauges()` method consumed by `collect_pipeline_view`. If it should appear on the curated views, update both `monitors/wandb/overview.py` constants and `dashboard/static/app.js` (`overview.py:3-5`). Env-specific scalars flow automatically via `trace.metrics` / `trace.rewards` → `.../<agent>/metrics/<name>` / `rewards/<name>`.

**A new annotation producer**: build records with `make_update(trace_id, info=..., branches={idx: {"advantages"|"trainer_logprobs"|"entropies": [...]}})` and call `monitors.log_annotations`; to add a new per-token stream name, extend `STREAM_FIELDS` (`update.py:17`) — the dashboard folds only those.

**A new eval driver**: reuse `EvalRunner(config, run_dir)` → `setup()` → `start()` → `eval_source.trigger(step)` → `run_epoch(fired, step, superseding_step=...)` → `drain()`/`stop()` (`runner.py`); config must satisfy `ServedEvalConfig` (sources, client, concurrency, tasks_per_minute, heartbeat, `env_addresses`) plus `model`, `log.interval`.

**Dashboard**: agent-generated code ("not meant to be read or edited by humans", `server.py:3`); extend by adding FastAPI routes reading run-dir files; verify via `tests/integration/dashboard_smoke.py`. Agent control: `POST /api/view` with `{run, tab ∈ {metrics,config,traces,logs,report}, step, kind, subset, episode|line, trace, branch, report, highlight:[{node, quote, prefix?, suffix?, reason?, field∈{content,reasoning}}]}` (`server.py:1928-2048`).

**Heartbeat / Prometheus**: `HeartbeatConfig` is a URL; `MetricsServer.update` has a fixed gauge list — add a `Gauge` in `__init__` and a parameter in `update` and the call in `train.py:700-711`.

---

## 7. Gotchas & limitations

1. **`time/step` and `time/save_ckpt` mean different things per process but share a key.** Trainer `time/step` = full trainer step (`train.py:634, 684`); orchestrator `time/step` = sink-to-sink batch cycle (`orchestrator.py:584-586, 685`). In an `rl` run both go to the **same shared W&B run and the same `metrics.jsonl`** (rows distinguished only by `producer`). The curated "performance" panels plot bare `time/step` (`overview.py:47-52`). Confirmed writers: `rl.py:171-174` sets one `WANDB_RUN_ID` for both processes, with labels `orchestrator` (`:311`) and `trainer` (`:361`); primary = finisher = orchestrator (the `wandb/monitor.py:58-63` defaults). `[UNVERIFIED]` how W&B shared mode resolves two writers of one key at one step — expect interleaved/overwritten series; disambiguate via `producer` in `metrics.jsonl`. Also `[UNVERIFIED]`: the orchestrator (finisher) can `wandb.finish` right after `wait_for_final_broadcast`, while the trainer still logs its last step's rows after that broadcast (`train.py:586-697`).
2. **`all` vs `effective`.** Quality metrics (reward, truncation, turns) should be read on `effective`; `has_error`, `cancelled`, `error/<type>`, `solved_*` exist **only on `all`**; `pass@k` exists **only on `effective`** and only for binary rewards (silently absent otherwise) (`orchestrator/metrics.py:268-272, 399-400, 370-373`). Eval `effective` = non-errored and trainable-agent traces — not "trained on".
3. **Reward keys are per agent**: `train/<scope>/<subset>/<agent>/reward/mean` — there is no pooled `.../reward/mean` (`test_metrics.py:141, 173`). Single-agent envs usually use agent `agent`. Episode-level token/turn counts **sum over all agents** (including judges).
4. **Stale docs**: `docs/training.md:117-130` lists `reward/{all,env}/mean`, `seq_len/...`, `empty_rollouts`, `errored_rollouts` — none of these keys exist; `docs/eval.md:108-113` lists `eval/<env>/all/seq_len/mean` (does not exist; use `num_total_tokens`). `skills/training/monitor-run/SKILL.md` has the correct scheme. `collect_inference_metrics`' docstring says "Mirror ... to W&B (requires wandb)" but rows go to every monitor including file (`inference_metrics.py:354-358`).
5. **Empty sets read as 0.** `Stat.mean()` of no values is `0.0` (`metrics.py:69-70`) — the console `Reward 0.0000` can mean "no effective traces". Unscored reward/metric entries (`None`) count as `0.0` on both subsets (`metrics.py:182`). Trainer tensor stats for empty keys are NaN (all five stats), and `std` of a single-element key is NaN too (unbiased `torch.std`, `trainer/utils.py:263-276`). These are **dropped with a warning** by the file and prime monitors but sent to W&B as NaN (`wandb/monitor.py:142-153` does not sanitize).
6. **`error/<type>` and `dispatch_failure/<type>` are counts**, not rates; `dispatch_failure/mean` is a rate (`metrics.py:22-27, 271`).
7. **`progress/tokens` (full window) vs `progress/{input,output}_tokens` (effective only)** don't add up; `progress/total_*` are logged **before** the step's increment (`orchestrator.py:669-690, 748-750`).
8. **RL `perf/*` are approximations; RL vs SFT MFU are not comparable.**
   - Timing windows differ: RL uses the single-step fwd-bwd(+optimizer) time, SFT a 10-step sliding window of full wall time (§3.10).
   - RL throughput extrapolates rank 0's **first micro-batch length** to every micro-batch ($|\text{dp}|\cdot\text{len(mb}_0)\cdot n_\text{micro}$). That is padded tokens, not loss tokens, and it can over- or under-state, because bins are variable-length.
   - The attention term uses step 1's first-bin length $T$, frozen by the singleton (`perf.py:237-245`), for the whole run. It treats packed short documents as one long sequence, and it uses $d_h = h/H$ instead of `head_dim` (halved for Qwen3).
   - Unknown GPUs (e.g. RTX A6000 in CI) are costed as A100 (312 TF), with a warning (`perf.py:107-109`). CI's MFU baselines are therefore relative, not true MFU.
9. **Time-keyed rows** (`inference/*`, `dispatcher/*`, `concurrency/*`, `watcher/*`, `event_loop_lag/*`) carry `step: null`; they cannot be joined to training steps. `dispatcher/{cancelled,errored}/*` are per-tick drained counts (per `log.interval`), not cumulative.
10. **`inference/*` raw counters are cumulative** (`engine_values` passes counters through verbatim) — use the `:rate` variants. Fleet means hide a sick engine; the overview pairs mean with min/max for that reason (`overview.py:54-79`). Metric names depend on the vLLM version (`RATIO_METRICS` lists both legacy and OpenMetrics names).
11. **Trainer annotation cost is paid even when the file monitor is off.** `AnnotationWriter` is built unconditionally and does a per-step `dist.gather_object` of Python float lists of every trained token's logprob+entropy to rank 0 (`train.py:239-241, 543, 559`; `annotations.py:69-81`); with `--no-monitors.file` rank 0 then drops them, and `float_decimals` becomes `None` (full precision). A potential per-step stall at large batch × long sequence. `[UNVERIFIED: magnitude]`
12. **Metrics server default port 8000 = inference `server.port` default 8000** (`configs/shared.py:227`, `configs/inference.py:20`). The clash is real in a default single-node `rl` run with `--trainer.metrics-server`: the local `vllm-router` (router on by default, `configs/inference.py:446`) binds `0.0.0.0:8000`, with the engine behind it on 8100 (`entrypoints/inference.py:165-177, 216-220`). Whichever process binds second fails. Inference is spawned before torchrun (`rl.py:204-362`), so that is normally the trainer: its main-thread `HTTPServer` bind raises `OSError` and the run dies at startup. Fix: `--trainer.metrics-server.port`. Multi-node and k8s are unaffected (trainer hosts ≠ inference hosts). Only the RL trainer has the server; non-master nodes serve `/health` only.
13. **Crashes leave artifacts in "running" shape**: no finalize → live chunk stays plain, W&B run state relies on `@clean_exit`'s `wandb.finish(exit_code=1)`, platform run marked crashed by SDK atexit. The dashboard reads an unfinalized eval as `stopped` once files go quiet (skills/dashboard/SKILL.md:18).
14. **`logger.critical` is a no-op** (`logger.py:176-177`) — anything logged at CRITICAL disappears.
15. **Console lines are a test contract.** Integration/nightly tests regex the `SUCCESS` lines (`Reward x.xxxx`, `Mismatch KL x.xxxx`, `Evaluated <env> ... | Turns | Branches`) from `logs/latest/*.log` (`tests/utils.py:24-225`). Reformatting them breaks CI.
16. **`convert_traces_to_hf_dataset.py` expects a single plain `traces.jsonl`**, but the file monitor writes a chunked, zstd-sealed stream; decompress first (`zstd -dcf monitors/file/traces/stream/* > traces.jsonl`). Its rows lack annotation-derived facts (advantages, cohort membership, trainer logprobs) — only arrival-time data (`convert_traces_to_hf_dataset.py:160-168`).
17. **Eval resume archives, never deletes**: every resume renames `monitors/file` to `file.attempt_N`; the dashboard only shows the current attempt's stream. `check_config` treats a changed-length `source` list as a change of `source` (not resumable) (`test_resume.py:85-86`). `RESUMABLE` is matched with `fnmatch`, whose `*` also crosses dots. So `source.*.group_size` also admits a change to any deeper leaf that ends in `group_size` / `num_examples` / `shuffle` / `serve` (e.g. inside `source.0.env...`) (`resume.py:89-94`).
18. **Eval adaptivity requires vLLM `/metrics`**; against external APIs (Prime Inference) the band must be pinned or setup raises (`runner.py:145-159`).
19. **Online eval cancellation**: with `cancel_on_new_checkpoint=True` a slow eval epoch is partially evaluated and logged with `cancelled/{count,mean}` — `avg@k` is then over fewer episodes (`runner.py:284-301`). With `False`, the trainer can idle behind slow evals (`configs/eval.py:130-133`).
20. **Dashboard auto-start only in a TTY** and only if the `dashboard` extra is installed (`entrypoints/dashboard.py:92-102`); the daemon serves the code of the checkout that started it (kill it after switching branches — skills/dashboard/SKILL.md:31). It downloads HF tokenizers for token views (network).
21. **W&B-only views lose episodes**: the W&B monitor logs no samples/tables; episodes live only in the file stream (and, for train, every-10th-step samples on the Prime platform).
22. **Benchmark tolerance docstring is stale**: the module docstring says "within 5%" and "exactly the same" memory, but the check is two-sided `METRIC_TOLERANCE = 0.15` on MFU/throughput/step time and one-sided `≤ +1%` on memory (the inline comment explains the A6000 fleet drift) (`test_benchmark_regression.py:1-24, 176-222`). The manual `benchmarks.yaml` aggregate uses a separate 5% threshold. `docs/development.md` omits `gsm8k_eval` from the GPU matrix list.
23. **Multi-node shared-FS appends** — orchestrator and trainer (possibly on different nodes) append to one `metrics.jsonl`; POSIX `O_APPEND` atomicity across NFS/Lustre clients is not guaranteed. `[UNVERIFIED]`

---

## 8. For a custom framework

**Essential design to keep**
- **Episode-once + append-only annotations.** Serializing each rollout once at arrival and recording later facts (cohort membership, ship step, advantages, trainer logprobs/entropies) as keyed updates is the right shape for async off-policy RL: it is crash-durable, cheap to append, and lets a viewer overlay "what the loss saw" on "what the policy generated" without rewriting gigabytes. Keep the full-length-with-nulls convention for per-token streams — it decouples the writer's loss mask from readers.
- **`all` vs `effective` as a first-class axis.** Selection bias is the #1 misreading in async RL (errors, stale drops, curriculum rejections, zero-signal groups). Computing every statistic on both the arrival window and the shipped cohort is cheap and diagnostic. Keep per-agent trace-level metrics separate from episode sums for multi-agent envs.
- **Step-keyed vs wall-time rows.** Inference/pipeline health must be sampled on wall time; conflating them with training steps hides stalls.
- **Offset index beside the stream + seekable compression.** Makes 100 GB runs browsable; the 4 MiB-frame seekable zstd trick is worth copying.
- **Eval reuses the rollout engine.** One dispatcher/env/concurrency stack for train and eval, and resume from the episode stream keyed by `task.key`. Take the SFT online-eval discipline (weights reload only between epochs → one version per epoch), not RL's in-orchestrator eval, whose epochs straddle updates and report only a min-version lower bound.

**Simplify / replace**
- Replace the if-chain `monitors.setup` with a small registry (`name → (ConfigCls, MonitorCls, init-kwargs builder)`); make fan-out concurrent or per-monitor queued so one slow backend never stalls the orchestrator loop.
- Namespace metrics by producer (`trainer/time/step`, `orch/time/step`) to kill the shared-key ambiguity (Gotcha 1), and emit one merged row per step per producer rather than N partial rows.
- Make trainer annotations opt-in and sampled (e.g. every k steps / a fraction of samples), and gather tensors not Python lists.
- Define MFU once and use it identically for RL and SFT: fwd+bwd model FLOPs over the actual per-micro-batch token counts, per-document attention lengths and the real `head_dim`, over a stated time window. Keep a hardware table with an explicit "unknown" rather than defaulting to A100.
- Treat console lines as UX only; have tests assert on `metrics.jsonl` rows instead of regexing logs.
- Keep a tiny read-only dashboard server over files (it is a big productivity win) but own a stable, documented JSON schema for records/index/annotations; prime-rl's is implicit in reader code.

**Coupling points to watch**: the trace record schema (verifiers `Episode`/`Trace`), the live delta schema (`verifiers.v1.serve`), micro-batch fields consumed by `AnnotationWriter` (trace_ids, branch_indices, sequence_lengths, env_names, loss_mask), vLLM metric names, and the W&B shared-mode env contract set by launchers.

---

## 9. Open questions

1. How W&B shared mode merges the orchestrator's and trainer's same-named keys (`time/step`, `time/save_ckpt`, `step`) at the same step — needs a W&B SDK read or a live run. (Gotcha 1.)
2. `prime_runs` SDK internals: the every-10th-step sample cadence and Parquet encoding (`monitors/prime.py:54-58`) are only asserted in docstrings; lives in the external `prime-runs` package.
3. Whether `vllm-router` (external binary) forwards `/liveness` to the engine. The k8s liveness probe targets the service port, i.e. the router when the router is on (`deployment.yaml:206-212`). This only matters if probes are enabled.
4. Exact delta schema (`open`, `calls`, `set`, `errors`, `discard`, preview fields) and `EpisodeAssembly.apply` semantics — `deps/verifiers/verifiers/v1/serve/` (G/F).
5. The full list of `loss_tensors` keys that become trainer metrics (e.g. `is_masked/*`, `ref_kl/*`) — `trainer/rl/loss.py` (C).
6. Magnitude of the per-step `AnnotationWriter` gather at production scale (Gotcha 11) — needs profiling.
7. Cross-node `O_APPEND` safety of the shared `metrics.jsonl` on the shared filesystems prime-rl targets (Gotcha 23).
8. How the W&B shared-mode finisher (orchestrator) closing the run interacts with the trainer's trailing last-step rows (Gotcha 1).
