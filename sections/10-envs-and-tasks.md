# Environments & tasks — prime-rl @ b944873

> Scope: how an env named in a prime-rl TOML becomes running code (S3: resolution, installation, import, construction), the anatomy of a v1 env package, the prime-envs catalog, mixing envs, and the add-a-new-env recipe. Protocol and pool internals of the env server belong to G (`08-harnesses-runtimes-serve.md`). Task, trace, and scoring semantics belong to F (`07-verifiers-core.md`). The orchestrator call site belongs to B (`03-orchestrator.md`).
>
> **Files read in full (LOC):**
> - **prime-envs** (`deps/prime-envs`, pin `b677502`): `README.md` (126), `AGENTS.md` (127), `CLAUDE.md` (1), `HARBOR.md` (45), `BENCHMARK_PORTING.md` (288), `pyproject.toml` (25), `skills/evaluation/SKILL.md` (214), `scripts/install.sh` (82), `tests/test_envs.py` (86), `judges/README.md` (28), `judges/swe/{eval.toml,prompt.md,rubrics.toml,hints/swesmith-env.md}` (13+28+76+7).
> - **lean_common**: `README`, `pyproject`, `__init__`, `env`, `taskset`, `task`, `prompt`, `sources`, `statements`, `scripts/{eval_hashes,compile_statements}.py`, `tests/test_task.py`, `image/Dockerfile` (about 1,500 LOC).
> - **Representative envs, each read whole:** `math/math_env` (127), `swe/swesmith_env` (399), `swe/r2e_gym` (README and pyproject only), `terminal/terminal_bench_2` (60), `tool_use/tau2_bench` (356), `search/browsecomp_plus` (397), `multimodal/virl39k` (188), `lean/minif2f` (`__init__`, pyproject, taskset head).
> - **Every category README** plus every env's `pyproject.toml` and `__init__.py` (a mechanical scan feeds the catalog table in §3.14).
> - **`registry.json`** (114,118 lines of data): structure verified with a script, not read line by line.
> - **verifiers** (`deps/verifiers`, pin `69cc0f9`): `verifiers/v1/utils/loaders.py` (327), `configs/{env,taskset,task,agent,harness,serve,judge}.py` (165/24/63/80/73/65/61), `configs/cli/{env,init,validate}.py` (59/17/89), the relevant part of `configs/client.py`, `taskset.py` (111), `task.py` (323), `env.py` (411), `cli/{init,validate,resolve}.py` (252/488/107), `serve/server.py` (224), `serve/pool.py` §`env_config_data`/`serve_env` (lines 289–371), `envs/single_agent/*`, `tasksets/harbor/{__init__,env,taskset}.py` (16/86/819), `utils/decorators.py::invoke`, `skills/{create-environments,evaluate-environments}/SKILL.md` (252/208), `docs/v1/{env,tasksets,getting_started,evaluation,harbor}.md` (86/264/16/80/144), `environments/reverse_text/*` (77), `tests/v1/test_taskset.py` (91).
> - **prime-rl**: `src/prime_rl/orchestrator/envs.py` (273), `src/prime_rl/entrypoints/env_server.py` (65), `packages/prime-rl-configs/src/prime_rl/configs/orchestrator.py` (803), `configs/env_server.py` (45), the env parts of `configs/rl.py` and `configs/algorithm.py`, `orchestrator/train_source.py` (85), `orchestrator/curriculum/{base,samplers/*,gates/adv}.py`, `utils/pathing.py` §env (lines 222–247), `entrypoints/rl.py` §env servers, `templates/multi_node_rl.sbatch.j2` §env servers, `pyproject.toml` (357), `docs/configuration.md` §Environments, `skills/{install,eval,configs}/SKILL.md`, `examples/basic/reverse-text/*`, `examples/advanced/qwen3-30b-a3b/{math,swe,tool}.toml` + README, and the env blocks of every other example and config (listed in §4.7).
>
> **Related docs:** 01 (launcher and placement), 02 (config system), 03 (orchestrator), 07 (verifiers core: Task, Trace, scoring), 08 (harnesses, runtimes, env-server protocol), 11 (eval and ops).

---

## 1. Mental model

**An environment is a set of Python classes found by import name.** It is not a registered object or a server image, and nothing is looked up in a catalog. A prime-rl training source carries a verifiers `[env]` block, for example `env.taskset.id = "math-env"`. verifiers turns the id `math-env` into the module name `math_env`, imports it, and reads the module's `__all__`. It takes the one exported `Taskset` subclass, plus an optional exported `Env` subclass and an optional `Harness` subclass (`deps/verifiers/verifiers/v1/utils/loaders.py:70-126`).

A module can be imported only if some package installed it into the venv. In prime-rl that package is a **uv workspace member**. Every directory under `deps/prime-envs/environments/*/*` is a member, and eight verifiers examples are opt-in members (`pyproject.toml:162-182`). These members install only with `uv sync --all-packages` (`README.md:114`).

`deps/prime-envs/registry.json` is **not** an env registry. It is a Harbor *dataset* registry: two datasets whose entries point at task directories in `PrimeIntellect-ai/prime-tasks` (§3.10).

**The v1 package model has five parts:**
- **`TaskData`**: a frozen pydantic row, and the only part that crosses a process boundary. It carries the prompt, system prompt, image, workdir, resources, network policy, artifacts, and any task-specific fields such as the gold answer.
- **`Task`**: the behaviour for that row. It has `setup`/`finalize`/`stage_verifier` hooks, `@vf.reward`/`@vf.metric`/`@vf.stop` methods, `validate`, and `toolsets`.
- **`Taskset`**: a loader, `load() -> Iterable[Task]`, driven by a typed `TasksetConfig`.
- **`Env`**: control flow over agent seats. It defaults to `SingleAgentEnv`, which has one seat named `agent`.
- **`Harness`**: the agent program, run in a runtime (subprocess, docker, prime, or modal).

The taskset decides *what* to solve and how to score it. The harness decides *how the model acts*. The runtime decides *where the code runs*.

AGENTS.md makes this a policy. An environment author ships "a taskset, not a harness", and most tasksets should expose **zero tools**. The model is expected to use a general coding harness such as `bash`, `rlm`, or `codex` (`deps/prime-envs/AGENTS.md:97-101`). Under prime-rl *training* that set is narrower. `TrainClient` refuses every non-chat-completions dialect (`deps/verifiers/verifiers/v1/clients/train.py:342-350`), so `codex` (Responses) and `claude_code` (Anthropic) are **eval-only**. The refusal surfaces only at runtime, as a retryable 502 with the cause in `trace.calls[*].error` (F, 07). Shipped training configs use `null`, `bash`, and `rlm` (`rlm` in `intellect-3.1/rl.toml:66`, `glm-5.3/swe.toml:135`, `glm-4.5-air/search.toml:98,122`, `configs/advanced/nemotron-3-super/swe.toml:89`).

**prime-rl splits ownership across two processes:**
- **The orchestrator owns the taskset.** It imports the package, calls `load()` once, and materializes a finite taskset into a list. It samples tasks (ratio across envs, then a per-env curriculum) and ships each task's `TaskData` as JSON on the run request (`src/prime_rl/orchestrator/envs.py:1-18, 95-119`; `dispatcher.py:599`).
- **The env server owns execution.** It imports the same package and constructs the `Env`, which instantiates the `Taskset` class but never iterates it. For each request it rebuilds `TaskData` and `Task` from the JSON, then runs one episode through harness and runtime (`deps/verifiers/verifiers/v1/serve/server.py:37, 80-92, 97-101`).

Because of this split, workers stay stateless about data, and every process needs the env package importable, including the `rl` launcher, which only validates config (§3.2).

```
TOML  [[orchestrator.train.source]]  env.taskset.id = "r2e-gym"
  │  (config validation, in rl launcher / orchestrator / env-server processes)
  ▼
_import_plugin("r2e-gym") → "r2e_gym" → import r2e_gym → __all__ → R2EGymTaskset  (+ R2EGymConfig via generics)
  │
  ├── orchestrator:  Taskset(config).load() ──list──► Curriculum ──► task.data.model_dump(json) ─┐
  │                                                                                                │ ZMQ run(task_data,…)
  └── env-server:    load_environment(EnvConfig) → Env(taskset, harnesses) → serving()  ◄──────────┘
                      _build_task: DataCls.model_validate(task_data) → TaskCls(data, cfg.taskset.task)
                      → Env.run_slot → harness in runtime → Task.score → Episode ──► orchestrator
```

## 2. Where it runs

| Stage | Process / host | What happens | Cite |
|---|---|---|---|
| Config resolve | `rl` launcher (and again in each spawned process when it re-parses its JSON) | `TrainSourceConfig._resolve_env` narrows `env` to the concrete `EnvConfig` subclass. That narrowing imports the taskset package, the harness package, and any env package. | `packages/prime-rl-configs/src/prime_rl/configs/orchestrator.py:172-176`; `deps/verifiers/verifiers/v1/configs/env.py:96-108`; `configs/agent.py:53-66` |
| Env-server config emit | launcher | For each source whose `serve.address` is unset, writes `envs/<split>/<name>.json` = `{env, serve, address_file, log}`. | `src/prime_rl/utils/pathing.py:230-247`; `src/prime_rl/entrypoints/rl.py:109-113` |
| Env-server spawn (single node) | launcher `Popen(["env-server","@",json])` | Inherits `os.environ + DEFAULT_COMMON_ENV_VARS + [env_vars] + [orchestrator.env_vars]`. Its log goes to `logs/envs/<split>/<name>.log`. A monitor thread watches it. | `src/prime_rl/entrypoints/rl.py:262-291` |
| Env-server spawn (SLURM multi-node) | **on the orchestrator's node**, backgrounded before `uv run orchestrator` | Same command. Binds loopback. | `src/prime_rl/templates/multi_node_rl.sbatch.j2:541-559` |
| Env-server boot | env-server process | `VF_RUN_ID := PRL_RUN_ID or uuid` (`env_server.py:60`). Sandbox labels come from `PRL_RUN_NAME` (`:36-37`). `serve_env(...)` runs either one in-process `EnvServer` or an `EnvServerPool` (elastic by default). Each worker calls `load_environment(config)` → `Env.__init__` → `load_taskset` (constructs, no `load()`) + `load_harness` per seat. It rejects `TaskData` fields with `exclude=True`, then enters `env.serving()`, which starts shared toolsets and interception. | `src/prime_rl/entrypoints/env_server.py:34-61`; `deps/verifiers/verifiers/v1/serve/pool.py:296-371`; `serve/server.py:37-51,193`; `env.py:86-145,360-411` |
| Address publish | env-server thread | The OS-assigned `tcp://127.0.0.1:<port>` is written atomically to `envs/<split>/<name>.address`. | `env_server.py:24-31, 39-47`; `pathing.py:222-227` |
| Orchestrator connect + taskset load | orchestrator | For each env: poll the address file (≤600 s), create an `EnvClient`, wait for health (≤600 s), then call `vf.load_taskset` (so `load()` starts only after that env's server answers health), then either materialize the list (finite) or keep an iterator (infinite). Optional shuffle with seed 42. Envs within a group start concurrently; the train group finishes before the eval group starts. | `src/prime_rl/orchestrator/envs.py:42-62,95-119,222-232`; `orchestrator.py:270-281` |
| Steady state | both | The orchestrator sends `run(task_data, client, model, sampling)`. The server builds the Task and runs one episode. Deltas stream back. | `envs.py:127-153`; `serve/server.py:97-120` |
| Shutdown / crash | launcher | A server crash is caught by the launcher's monitor thread (A owns the policy). Servers spawned by the eval entrypoint are its children and exit with it (`skills/eval/SKILL.md:75`). They inherit only `os.environ + DEFAULT_COMMON_ENV_VARS`; `eval` has no `[env_vars]` (`src/prime_rl/entrypoints/eval.py:166-179`). | `rl.py:283-291` |

An **externally managed** server (`serve.address` set, for example a k8s pod) is not spawned, and its TOML is not written. The orchestrator connects directly (`orchestrator.py:163-164, 797-803`; `rl.py:55-62`). Even then, the orchestrator still imports the env package to load tasks.

## 3. Mechanics

### 3.1 Plugin id → module → class

`_import_plugin(plugin_id, kind, group)` (`deps/verifiers/verifiers/v1/utils/loaders.py:70-96`) works in four steps:
1. **Strip the hub form.** `name = plugin_id.rsplit("/",1)[-1].split("@",1)[0]`, so `owner/name@1.2` becomes `name` (`:73`).
2. **Normalize.** `module = name.replace("-","_").lower()` (`:74`).
3. **Prefer a built-in.** If `importlib.util.find_spec(f"{group}.{module}")` exists, import that. Otherwise import the bare `module` (`:75-78`). `find_spec` on a dotted name imports the parent package, whose `__init__` eagerly imports its built-ins (`tasksets/__init__.py:1-10` pulls harbor, nemo_gym, openenv; `harnesses/__init__.py` pulls all 13). So resolving *any* id needs those built-ins' import-time deps (e.g. `mcp.server.mcpserver` for nemo_gym). The `@version` part is discarded: nothing checks it against the installed package. The groups are `verifiers.v1.tasksets` (built-ins `harbor`, `nemo_gym`, `openenv`, `textarena`), `verifiers.v1.harnesses` (`bash`, `browser_use`, `claude_code`, `codex`, `hermes_agent`, `kimi_code`, `mini_swe_agent`, `null`, `openclaw`, `pi`, `prime_agent`, `rlm`, `terminus_2`), `verifiers.v1.envs` (`agentic_judge`, `best_of_n`, `isolated_verifier`, `shared_agentic_judge`, `single_agent`, `user_sim`), and `verifiers.v1.judges` (`reference`, `rubric`).
4. **On failure**, raise `ModuleNotFoundError` with the hint "any other package must already be installed" (`:91-96`). **verifiers never installs anything** (`deps/verifiers/skills/evaluate-environments/SKILL.md:47`).

`_plugin_class(module, base, kind)` (`loaders.py:99-126`) then reads `module.__all__`. A module with no `__all__` raises `AttributeError`. It filters the exports to strict subclasses of `base`:
- **0 matches** raises `TypeError`, which the fallback callers swallow.
- **More than 1** raises `ValueError`, which stays loud everywhere.

The config type is recovered from generics, not from a registry. `taskset_config_type(id)` = `concrete_type(TasksetCls, TasksetConfig, origin=Taskset)` (`:274-279`). The same holds for the harness (`:282-284`), the env (`:292-299`), and the task class (`:324-327` → `Taskset.task_type()`, `taskset.py:97-99`).

| TOML id | Imported module | Notes |
|---|---|---|
| `reverse-text` | `reverse_text` | a verifiers example env (workspace member) |
| `math-env` | `math_env` | the pyproject name is also `math-env` |
| `r2e-gym` | `r2e_gym` | |
| `swebench-verified` | `swebench_verified` | |
| `terminal-bench-2` / `primeintellect/terminal-bench-2` | `terminal_bench_2` | a hub id is stripped to its last segment (`docs/v1/evaluation.md:6`) |
| `swesmith-env` (not `swesmith`) | `swesmith_env` | `swesmith` would import the *upstream* SWE-smith library, which exports no `Taskset` |
| `harbor` (taskset id) | `verifiers.v1.tasksets.harbor` | built-in, wins over a same-named installed package |
| `null` / `bash` / `rlm` (harness) | `verifiers.v1.harnesses.<id>` | `default` raises a "renamed to `bash`" hint (`loaders.py:80-84`) |

### 3.2 Config narrowing, and why imports happen at config time

prime-rl's `EnvConfig.env` is declared `SerializeAsAny[vf.EnvConfig] = vf.SingleAgentEnvConfig()`. Its `mode="before"` validator calls `vf.resolve_env_field(data, vf.narrowed_env_annotation(cls))` (`packages/prime-rl-configs/src/prime_rl/configs/orchestrator.py:160-176`; `deps/verifiers/verifiers/v1/configs/cli/env.py:20-45`). That call reaches `resolve_env_config(raw)` (`loaders.py:302-321`), which does four things:
1. **Read the ids.** It reads `taskset.id` and the env `id` from the raw dict.
2. **Pick the config class.** `env_config_type(taskset_id, env_id)` → `environment_class(...)` imports the taskset module and looks for an exported `Env` subclass (§3.3), then returns that env's config class. This is `SingleAgentEnvConfig` for a plain taskset (`envs/single_agent/env.py:15-20`).
3. **Validate.** `cls.model_validate(raw)` fires the nested validators:
   - `EnvConfig._refuse_env_level_harness`: a top-level `env.harness` key is an error. The harness belongs to a seat, `env.agent.harness` (`configs/env.py:79-94`).
   - `EnvConfig._resolve_taskset` → `narrow_plugin_field(data, "taskset", taskset_config_type)` validates the taskset block against the taskset's own config class (for example `MathConfig`), so an unknown `env.taskset.foo` is a validation error (`configs/env.py:96-108`; `loaders.py:37-67`).
   - `EnvConfig._merge_role_defaults` deep-merges a partial `env.agent` over the declared default `AgentConfig` instance (`configs/env.py:110-123`).
   - `AgentConfig._resolve_harness` narrows a *pinned* harness to its config class, and an absent harness stays `None` (`configs/agent.py:53-66`). A harness block **without an `id`** (e.g. only `env.agent.harness.tool_timeout = 900`) resolves with `default_id="bash"` (`:65`), **not** the taskset's bundled harness. Setting any harness knob on `browsecomp_plus` or `tau2_bench` without also restating `id` silently swaps in `bash` (verified by execution).
   - `TaskConfig._resolve_judges` resolves each `judges` entry by `id` through the judge loader (`configs/task.py:53-63`; `configs/judge.py:34-46`).

prime-rl never pre-narrows via the CLI (`narrow_config` is only used by `vf-eval`/`vf-gepa`), so `narrowed_env_annotation` returns `None` and every parse takes the `resolve_env_config` path. The same `_resolve_env` validator sits on `EnvConfig` (base of `TrainSourceConfig`/`EvalSourceConfig`, `orchestrator.py:172-176`) and on `EnvServerConfig` (`configs/env_server.py:31-35`). Executed against the pin: `{"taskset":{"id":"math-env","foo":1}}` fails with `extra_forbidden` at `taskset.foo`; an uninstalled id raises a bare `ModuleNotFoundError` (not wrapped in a `ValidationError`); a top-level `harness` key fails the seat check.

**Consequence:** any process that validates a prime-rl config containing `env.taskset.id = X` must be able to `import X`. That covers the `rl` launcher (on the submit host, before any sbatch), the orchestrator, the env server, and `eval`. The trainer and inference never parse an env block. An env installed only "where the env server runs" is not enough (§7 G2).

The resolved env block is serialized with `SerializeAsAny` so the subclass fields survive, then re-narrowed on the far side:
- **For the env server:** the launcher writes `dump_resolved_config(source)["env"]` to the server's JSON (`pathing.py:236-241`).
- **Across pool workers:** `serve_env` ships the dict and rebuilds it with `resolve_env_config` (`serve/pool.py:289-293, 355-363`).

### 3.3 Which Env and which Harness a source gets

- **Env** (`environment_class`, `loaders.py:168-181`). An explicit `env.id` imports `verifiers.v1.envs.<id>` or `<id>`, and a failure raises. Without one, the loader uses the taskset module's exported `Env` subclass. If there is none, it falls back to `SingleAgentEnv`, whose `run` is `await agents.agent.run(task)` (`envs/single_agent/env.py:23-29`).
- **Harness per seat** (`Env.__init__`, `env.py:116-124`, `_agent_harness` `:188-190`). The seat's pinned `harness` wins. Otherwise `default_agent_harness(taskset_id)` applies: the taskset module's exported `Harness` subclass, **else `bash`** (`configs/env.py:159-165`; `loaders.py:153-161`).
- **Runtime per seat.** `AgentConfig.runtime` **defaults to `PrimeConfig()`, a remote Prime sandbox** (`configs/agent.py:30`).

| Package exports | Env | Default harness | Examples |
|---|---|---|---|
| `Taskset` only | `SingleAgentEnv` | `bash` (a bash+edit agent in a Prime sandbox) | `math_env`, `swesmith_env`, `r2e_gym`, `terminal_bench_2`, `virl39k` |
| `Taskset` + `Harness` | `SingleAgentEnv` | the bundled harness | `browsecomp_plus` re-exports `NullHarness` (`deps/prime-envs/environments/search/browsecomp_plus/browsecomp_plus/__init__.py:1-5`) |
| `Taskset` + `Env` | the bundled env | `bash` unless bundled | Lean tasksets export `LeanEnv` (Prime-only, isolated verifier). `deep_swe`, `swebench_pro`, `swe_atlas_*`, and `lab` export `HarborEnv`. `scicode` and `bfcl_v3` export their own. |
| `Taskset` + `Env` + `Harness` | bundled | bundled | `tau2_bench` (`Tau2Env` pins `runtime=SubprocessConfig()` and uses `Tau2Harness`) (`tool_use/tau2_bench/tau2_bench/taskset.py:47-55`, `harness.py:50-128`) |

**Every training example therefore pins both harness and runtime for non-agentic envs**, for example `env.agent.harness.id = "null"` and `env.agent.runtime.type = "subprocess"` (`examples/basic/reverse-text/rl.toml:56-60`; `examples/advanced/qwen3-30b-a3b/math.toml:53-57`). Leaving them unset trains a single-turn math task through the `bash` agent in a remote sandbox.

### 3.4 Taskset loading in the orchestrator

`Env.start()` (`src/prime_rl/orchestrator/envs.py:95-119`) runs these steps:
1. **Resolve the address.** If `self.address is None`, it runs `wait_for_address(address_file, 600)`, which polls every 0.5 s (`:50-62, 98-99`).
2. **Connect.** `EnvClient(address)`, then `wait_for_server_startup(timeout=600)` (`:101-104`).
3. **Construct.** `taskset = vf.load_taskset(self.config.env.taskset)` = `taskset_class(id)(config)` (`loaders.py:190-191`). `Taskset.__init__` reads the optional `config.system_prompt` file (`taskset.py:43-47`). The env server's `Env.__init__` also constructs the taskset (`env.py:104`), so the file must exist on both hosts, though only the orchestrator's copy reaches tasks (it is baked into the shipped `TaskData`).
4. **If `type(taskset).INFINITE`:** shuffle raises. `tasks = iter(taskset)` and `num_tasks = None` (`envs.py:106-110`).
5. **Otherwise it materializes everything.** `list(taskset)` runs in a thread, because iterating pulls datasets. With `config.shuffle` it applies `random.Random(42).shuffle` (`TASKSET_SHUFFLE_SEED = 42`, `:47`), then sets `num_tasks = len` (`:111-117`).

`Taskset.__iter__` wraps `load()`. It applies the file-based system-prompt override per task (`task.with_system_prompt`) and any `head`/`shuffle` view transform (`taskset.py:54-62`). The views are used by `vf-eval` and `vf-validate` (`-n` → `head`, `-s` → `shuffle`, fixed `SEED = 0`, `taskset.py:33, 74-95`). **prime-rl uses neither view.** It always materializes the full finite taskset, even for an eval source with `num_examples`, and slices afterwards (`EvalEnv.start`, `envs.py:187-194`). So the "lazy loaders bound construction under `-n`" property in prime-envs (`deps/prime-envs/README.md:96-102`; `skills/evaluation/SKILL.md:90-94`) does **not** apply in training.

`Envs.start` pre-seeds `datasets.utils.tqdm`'s lock before `asyncio.gather` over every env. Without it, concurrent `load_dataset` calls crash with `AttributeError: ... '_lock'` (`envs.py:222-232`).

### 3.5 The task on the wire

- **Client side (orchestrator):** `task_data=group.task.data.model_dump(mode="json")` (`src/prime_rl/orchestrator/dispatcher.py:599`). There is no `exclude_none`, so every field is shipped. A group is `group_size` separate `run` requests carrying the same `task_data`, dispatched one at a time as `episodes_to_schedule` counts down (`:530-567`). The server therefore builds `group_size` independent Task instances.
- **Server side:** `_build_task(task_data)` = `self.data_cls.model_validate(task_data)`, then `self.task_cls(data, self.env.config.taskset.task)` (`serve/server.py:80-92`). `data_cls` comes from `Taskset.task_type().data_type()` (`:38-39`).
- **At server construction** it refuses to serve if any `TaskData` field is `exclude=True`, because that field would vanish on the wire and be silently rebuilt with its default (`:40-51`).
- **Each run request builds a fresh `Task` instance** (`(slot,) = self.env.slots(self._build_task(req.task_data))`, `:101`). Per-rollout state on the Task, such as `SWESmithTask._patch_bases` keyed by `id(runtime)` (`swe/swesmith_env/swesmith_env/taskset.py:171-175`), is therefore per-episode on the server. The orchestrator-side Task objects are never executed.
- The task config the server uses is **its own** `config.taskset.task`, from the env-server JSON, not the orchestrator's (`server.py:87`). The two come from the same resolved TOML, so they agree unless a server is external. Likewise `Taskset.config` on the server is `env.config.taskset` (`env.py:104`; `taskset.py:43-44`).
- **Round-trip is enforced, not assumed.** The server stamps `trace.task.{key,hash}` from its rebuilt Task (`rollout.py:91`; `env.py:263`). On every returned episode the dispatcher checks `(episode.task.key, episode.task.hash) == (task.key, task.hash)` of the dispatched Task and raises `ValueError("Episode task provenance ... does not match ...")` otherwise (`dispatcher.py:116-120, 712`). That raise escapes the dispatch loop (`:340-347`) and is re-raised by the main loop (`orchestrator.py:561-571`), so a lossy `TaskData` (a float that doesn't JSON-round-trip, a validator that rewrites fields, a `Task.key` depending on non-wire state) **crashes the orchestrator** on its first episode. It never trains silently. Executed: `MathData` round-trips equal; `exclude=True` fields are detected by the server's `field.exclude` scan.

`Task.key` defaults to `Task.hash`, the content hash of `data.model_dump(mode="json", exclude_none=True)`, which includes the autogenerated `idx`. Override it with a durable id when appropriate (`task.py:150-164`). `LeanTask.key = f"{source}/{source_name}"` does this (`packages/lean_common/lean_common/task.py:83-85`). Keys are used by the curriculum, which requires them unique within a train taskset (§3.7); by the provenance check above; on `DispatchFailure` records (`dispatcher.py:657-658`); and by eval resume, which counts owed rollouts per `task.key` (`eval/resume.py:146`; `orchestrator/eval_source.py:85-88`). Eval sources are *not* checked for key uniqueness, so duplicate eval keys merge their resume counts.

### 3.6 Task lifecycle hooks (summary; F owns the full semantics)

All hooks are called through `invoke(fn, available)`, which passes **by parameter name** (`deps/verifiers/verifiers/v1/utils/decorators.py:29-38`). That is why `SWESmithTask.setup(self, runtime)` works even though the base signature is `setup(self, trace, runtime)` (`task.py:175`; `rollout.py:235`). Signal functions may request `task` (bound to `TaskData`), `trace`, and `runtime` (`task.py:216-219`).

| Hook | When | Contract |
|---|---|---|
| `setup(trace, runtime)` | after the runtime is provisioned, before the harness | Prepare files and services. A raise becomes a `TaskError` on the trace. |
| `finalize(trace, runtime)` | after the agent, while the box is alive | Capture artifacts, e.g. `capture_patch` → `trace.info["patch"]` (`swesmith_env/taskset.py:209-212`). |
| `stage_verifier(trace, runtime)` | isolated-verifier envs only, in the fresh box after artifact restore | Stage trusted verifier inputs (`task.py:181-183`; `docs/v1/env.md:64-74`). |
| `@vf.metric` / `@vf.reward(weight=)` / plugged judges | `Task.score` while the runtime is live | Merged with config-plugged `metrics`/`rewards`/`stops` (a plugged entry replaces a same-named method). Sorted by priority, then name, but that fixes only recording order. Three phases run in sequence (all metrics, then all rewards, then judges), and the functions *within* a phase run **concurrently** via `asyncio.gather` (`utils/decorators.py:41-45`). Two rewards that mutate the same box (`git checkout`, test runs) race. Runtime-requiring signals are skipped when scoring offline. A `Mapping` return records multiple keys. The whole phase sits inside `boundary(TaskError, ...)`, so any exception becomes a `TaskError` (`task.py:203-274`; `errors.py:76-89`; `configs/task.py:31-63`). |
| `validate(runtime)` | `vf-validate` only (model-free) | `True`/`False`, or `None` for "no ground-truth check" (`task.py:185-187`). |
| `toolsets(config)` classmethod | Task: one server per rollout. Taskset: one shared server per env-server worker. | See §6.2 (`task.py:309-320`; `taskset.py:101-111`; `env.py:404-411`). |

### 3.7 Mixing envs and sampling tasks

- **One env server per source.** Train sources come first, then eval, keyed by `(split, resolved_name)` (`orchestrator.py:788-803`). `resolved_name = name or env_id` (`:182-184`). `env_id` is `"<env.id>+<taskset.id>"` when an env id is set, else the taskset id (`deps/verifiers/verifiers/v1/configs/env.py:63-68`).
- **Name rules:** names must be unique within the train group and within the eval group (`orchestrator.py:327-335, 379-387`), and `agg` is reserved (`:190-193`). The same taskset can appear twice under different names (`configs/debug/multi-env/rl.toml:20-30`).
- **Env choice per task:** `TrainSource.next_task` draws the env as `rng.choices(env_names, weights=[ratio...])` with a seeded `random.Random(42)`, then pulls `next(curriculum.sampler)` for that env (`src/prime_rl/orchestrator/train_source.py:20-40`). Ratios are relative weights (`orchestrator.py:272-273`). This is a per-task (= per-group) draw, not a per-batch quota. The mixing RNG state is checkpointed with each curriculum (`train_source.py:66-83`), and resume refuses a changed env set. Example: INTELLECT-3.1 uses 0.3/0.2/0.3/0.2/0.2 (`examples/advanced/intellect-3.1/rl.toml`).
- **`StandardSampler` (default, also when `curriculum` is unset, `curriculum/base.py:34-36`):** cycles the finite task list in **source order** with `itertools.cycle`, which is dataset order unless `shuffle = true`. It raises on duplicate `task.key`. Its only state is a `cursor`. On resume it replays `cursor % len` `next()` calls (finite), or `cursor` calls into a fresh generator (infinite). There is no length or key check, so resume correctness assumes the reloaded taskset has the same order (`curriculum/samplers/standard.py:15-49`).
- **`DifficultyPoolSampler`** (`curriculum.sampler.type = "difficulty_pool"`): finite tasksets only. Each task is weighted by the pool its latest group-mean reward (error-free trainable traces only) falls in: `hard` ≤0.25 w=0.2, `normal` ≤0.75 w=1.0, `easy` ≤1.0 w=0.2. Unseen tasks get weight 1. Its own `Random(seed=42)` and the reward map are checkpointed. Each draw rebuilds an O(N) weight list (`orchestrator.py:201-234`; `samplers/pool.py:18-88`).
- **Gates:** there are **no gates by default** (`gates = {}`), so every group is admitted. Tokens with exactly zero advantage are still pruned by the sink (`constant_trainer_batch_size`, 03 §3.10), so the gate matters for groups with small non-zero advantages, zero-advantage groups that carry ce/ref_kl weights, and admission metrics (22 H-06). `curriculum.gates.<name> = {type="advantage_range", reject_min, reject_max}` rejects groups whose trainable advantages all fall in the range. The config default range `[0,0]` drops zero-signal groups, and groups with no advantage stream are admitted (`orchestrator.py:243-265`; `gates/adv.py:15-36`). All gates are AND-ed, every gate sees every group, and the sampler observes the group before the gates run (`curriculum/base.py:50-66`).
- **Per-env overrides:** `sampling` (merged over `orchestrator.train.sampling`), `group_size` (inherits `orchestrator.group_size`), and `algo` (inherits `orchestrator.algo`, then `algo.validate_env(env)` runs, e.g. `hierarchical_grpo` requires `ProposerSolverEnvConfig`) (`orchestrator.py:268-286, 314-325, 627-642, 751-754`; `configs/algorithm.py:193-198, 279-294`).
- **Eval sources:** `num_examples` (-1 = all) and `group_size` are inherited from `[orchestrator.eval]`. `interval` is per source (`orchestrator.py:289-304, 338-419`). The eval set is fixed: the first `num_examples` of the (optionally shuffled) list, reused every eval (`envs.py:187-194`).
- **Global dispatch rate limit:** `orchestrator.tasks_per_minute`, recommended for sandbox envs (`orchestrator.py:573-574`).

### 3.8 Installation

| Mechanism | Detail | Cite |
|---|---|---|
| Workspace members | `deps/prime-envs/environments/*/*` is auto-discovered. The verifiers examples are listed explicitly: `alphabet_sort`, `color_codeword`, `gsm8k`, `kuhn_poker`, `proposer_solver`, `reverse_text`, `wiki_search`, `wordle`. | `pyproject.toml:157-173` |
| Excluded (lock conflicts) | `tool_use/tau2_synth` and `tau3_bench` (conflicting `tau2` pins), `apex_agents` (pins verifiers by git rev) | `pyproject.toml:174-182` |
| Opt-in install | `uv sync --all-extras --all-packages`, or `uv sync --package prime-rl --package <env>`. A plain `uv sync` (or `--all-extras`) installs none of them and **removes** members already installed (uv's exact sync) unless you pass `--inexact`. The bootstrap `scripts/install.sh:141` runs only `uv sync --all-extras`, so a fresh install has **no envs**. | `README.md:114`; `skills/install/SKILL.md:24-34` |
| Shared lock | All members share one `uv.lock`, so there is no per-env venv. On a conflict, add the losing env to `exclude`. Adding a member changes `uv.lock`. The Docker build syncs `--locked`, so the refreshed lock must be committed. | `skills/install/SKILL.md:34`; `Dockerfile.cuda:100` |
| verifiers | editable path dep `deps/verifiers` with the `harbor` extra | `pyproject.toml:32, 259` |
| Hub index | `prime-hub` = `https://hub.primeintellect.ai/primeintellect/simple/`, `explicit = true` (uv consults it only for a package pinned to it in `[tool.uv.sources]`), `exclude-newer = false`. No source references it at the pin. | `pyproject.toml:314-320` |
| Container / SLURM | `uv sync --all-packages ...` runs in `Dockerfile.cuda:100,136`. Every sbatch template syncs `--all-extras --all-packages`: once on the batch node for a shared venv, or per node via `srun` when `shared_fs` is false (`multi_node_rl.sbatch.j2:167-179`; `single_node_rl.sbatch.j2:36`). | |
| Standalone prime-envs dev | `uv pip install -e path/to/env` from the repo root, then `uv run eval <id>` | `deps/prime-envs/README.md:80-92`; `AGENTS.md:62-71` |

### 3.9 Datasets: where each env gets its data

The orchestrator process pulls all data at `start()`. The env server pulls only what `Task`/toolset code needs at run time.

| Pattern | Example | Cache / network | Cite |
|---|---|---|---|
| `datasets.load_dataset(name, subset, split=...)` without a revision pin | `math_env`, `reverse_text`, `swesmith_env` (8 datasets), `r2e_gym` | HF cache; needs `HF_TOKEN` for gated datasets | `math/math_env/math_env/taskset.py:74-93`; `swesmith_env/taskset.py:293-351` |
| `hf_hub_download` of parquet + zip, images read lazily from the zip but **base64-encoded into the prompt at load** | `virl39k` | about 1.8 GB zip | `multimodal/virl39k/virl39k/taskset.py:92-148` |
| Pinned-revision HTTPS fetch without `datasets` | Lean (`hub_rows(repo, revision, file)`, `fetch_github`) | `$LEAN_AGENTIC_CACHE` or `~/.cache/lean-agentic`, atomic temp+rename | `packages/lean_common/lean_common/sources.py:20-97` |
| Harbor CLI `harbor download <dataset> --export` | `terminal_bench_2`, `tmax`, `general_agent`, SWE Harbor wrappers | `~/.cache/harbor/<dataset>[_<sha12 of selectors>]` | `deps/verifiers/verifiers/v1/tasksets/harbor/taskset.py:48, 447-523` |
| git shallow fetch of the upstream repo's `data/` | `tau2_bench` | `$TAU2_DATA_DIR` (default `~/.cache/tau2-bench/data`), `flock`-guarded, revision marker | `tool_use/tau2_bench/tau2_bench/taskset.py:9, 58-121` |
| Corpus + index built at load, mmap-loaded by the tool server | `browsecomp_plus` | `$BROWSECOMP_PLUS_CACHE_DIR` or `~/.cache/browsecomp-plus` | `search/browsecomp_plus/browsecomp_plus/corpus.py:28-63` |
| Procedural (no dataset), `INFINITE = True` | `wikispeedia`, `deshuffle_papers` | none | the scan in §3.14 |

**Splits and filters are taskset config fields, not framework fields.** Examples: `dataset_split` (`math_env`); `languages`, `split`, and `filter_fn` (a Python lambda string evaluated with restricted globals, `swesmith_env/taskset.py:83-107, 282-290`); `category` and `min_pass_rate`/`max_pass_rate` (`virl39k`); `tiers`, `min_difficulty`, `names`, and `dedup` (Lean, `lean_common/taskset.py:74-96`). There is no framework-level train/test split. Pointing an eval source at a different split means setting that taskset's own split field, and its name varies per taskset. The prime-rl config skill's own example `env.taskset.split = "test"` for `reverse-text` is **wrong**: `ReverseTextConfig` names the field `dataset_split`, and `split` fails with `extra_forbidden` (verified by execution; `skills/configs/SKILL.md:58-61`; `deps/verifiers/environments/reverse_text/reverse_text/taskset.py:39-41`).

### 3.10 Harbor task format, `HarborTaskset`, and `registry.json`

**A Harbor task is a directory** with these parts (`deps/verifiers/verifiers/v1/tasksets/harbor/taskset.py:570-660, 805-819`):
- **`task.toml`** holds `[task]` (authors, description, keywords), `[metadata]`, `[agent]` (`timeout_sec`, `network_mode`), `[environment]` (`docker_image` or `environment/Dockerfile`, `workdir`, `cpus`, `memory_mb`, `storage_mb`, `gpus`, `gpu_types`, `env`, `healthcheck`, `mcp_servers`, `skills_dir`, network baseline), `[verifier]` (`timeout_sec`, `env`, `environment_mode`, `environment`, `collect`), and `artifacts`.
- **`instruction.md`** is the prompt.
- **`environment/`** holds assets that are uploaded when `should_upload_environment_dir` is true.
- **`tests/test.sh`** writes `/logs/verifier/reward.json` or the legacy `reward.txt`. The JSON is strictly validated as a float or a non-empty dict of floats. With a `reward` key, that key is the score and the other keys become metrics. **Without** one, every key becomes a separate weight-1 reward, and they all sum into `trace.reward`. Stale reward files are deleted before `test.sh` runs (`:290-317, 319-361`).

`HarborTaskset.load` collects every directory under the downloaded dataset root that has both `task.toml` and `instruction.md`. It then turns each into `HarborData` with `parse_task`:
- **Image:** the image must be pullable. **Dockerfile-only tasks are rejected** unless `ignore_dockerfile` is set, which runs them on the runtime's image with the Dockerfile unbuilt. A task with no environment at all runs on the runtime's default image unless `require_image = true` (`HarborConfig`, `:89-98`; `resolve_image`, `:526-567`).
- **Timeouts:** timeouts are dropped by default (`ignore_timeouts = True`, `:79-83`).
- **Resources:** resources are scaled by `resource_multiplier`.
- **`task_dir`:** it is stored as a **host path** and used by `setup` (tar `environment/`) and `stage_tests` (tar `tests/` into `/tests`) on the env server (`:196-216, 290-317`).

Grading behaviour:
- **Shared grading:** runs `bash /tests/test.sh` in the solver box and reads the reward (`solved`, `:319-361`).
- **Separate grading** (`[verifier].environment_mode = "separate"`): requires `HarborEnv`, an `IsolatedVerifierEnv` that grades in a fresh box in `finalize` (`tasksets/harbor/env.py:48-102`). **Only packages that export `HarborEnv` get it.** Thin wrappers that export only their taskset get `SingleAgentEnv` (§3.3), and a separate-verifier task then raises `TaskError` at scoring (`taskset.py:321-328`), unless `ignore_separate_verifier = true` forces grading in the agent's own box (`:99-103`).

A dataset is selected by `HarborConfig.dataset` (a Hub `org/name[@ref]`, or a bare `name@version` for legacy registries) plus `repo` / `registry_path` / `registry_url` (`taskset.py:63-78`; `deps/prime-envs/HARBOR.md:1-45`). Harbor requires `verifiers[harbor]`. Without it, taskset load crashes with `ModuleNotFoundError: No module named 'harbor.models'` (`deps/prime-envs/skills/evaluation/SKILL.md:23`).

**`registry.json`** is a JSON list of `{name, version, description, tasks:[{name, git_url, git_commit_id, path}]}`. At the pin it holds exactly two datasets:
- `general-agent@2026-06-25`: 4,417 tasks at `prime-tasks@257b678…`.
- `tmax@2026-07-01`: 14,600 tasks at `prime-tasks@27de1c1…`.

Both are used by envs through `repo = "PrimeIntellect-ai/prime-envs@main"`, which is passed to `harbor download <dataset> --export --repo ...` (`terminal/tmax/tmax/taskset.py:10-11, 23-26`; `tool_use/general_agent/general_agent/corpus.py:27-38`; `tasksets/harbor/taskset.py:468-488`). The cache dir is keyed by `sha256` of the *selector strings* only (`cache_dir`, `:447-465`), and an existing dir is reused forever (`dataset_dir`, `:491-523`). A missing Harbor CLI raises a `RuntimeError` whose hint is `uv sync --python 3.12 --extra harbor` (`:49, 436-444`).

### 3.11 Worked example A: `math_env` (single-turn, deterministic + judge fallback)

```
deps/prime-envs/environments/math/math_env/
├── pyproject.toml         name="math-env", version 0.1.1, deps ["verifiers>=0.3.1"], tags ["single-turn","v1"],
│                          hatchling wheel packages=["math_env"]                               (pyproject.toml:1-17)
├── README.md              source dataset, size, changelog                                      (README.md:1-14)
└── math_env/
    ├── __init__.py        from math_env.taskset import MathTaskset; __all__ = ["MathTaskset"]   (:1-3)
    └── taskset.py
```

- **`MathData(vf.TaskData)`** adds `question: str` (raw, for the judge) and `answer: str` (`taskset.py:16-21`). `prompt = INSTRUCTION + question`, where INSTRUCTION = "Solve … put the final answer in \boxed{}" (`:13, 88`).
- **`MathTaskConfig(vf.TaskConfig)`** has `math_verify_timeout: int = 5` and **`judge: vf.ReferenceJudgeConfig | None = default_factory(vf.ReferenceJudgeConfig)`**, so the judge is on by default (`:24-33`).
- **`MathTask.correct`** (`@vf.reward(weight=1.0)`) calls `vf.verify_boxed_math_answer(trace.last_reply, answer, timeout)`. That returns exactly 0.0/1.0: 0 on an unterminated `<think>`, text before the last `</think>` is dropped, the last strict `\boxed{}` is compared, and any parse error or timeout scores 0 (`deps/verifiers/verifiers/v1/utils/score.py:89-116`). If that scores 0 and a judge is configured, it extracts the last strict boxed answer and calls `vf.ReferenceJudge(judge_config).evaluate(...)`, returning `float(result.parsed)`. No box means no judge call (`:36-62`). The judge path extracts from the *whole* reply, without the `</think>` split or the unterminated-think guard (`taskset.py:50`). When reasoning text lands in `content`, a truncated reply that math-verify scored 0 can still be judged correct.
- **`MathConfig(vf.TasksetConfig)`** has `dataset_name="PrimeIntellect/Hendrycks-Math"`, `dataset_subset="default"`, `dataset_split="train"`, `question_key`, `answer_key`, and `task: MathTaskConfig` (`:65-71`).
- **`MathTaskset.load`** is a generator over `load_dataset(...)` that yields `MathTask(MathData(idx=i,...), self.config.task)` (`:74-93`).
- **Default judge endpoint:** `ReferenceJudgeConfig` (`id = "reference"`, `choices = ("yes","no")`) → `JudgeConfig.model = "openai/gpt-5.6-luna"`, `api_key_var = "PRIME_API_KEY"`, `base_url` = `$PRIME_INFERENCE_URL`, else the `prime login` config's `inference_url`, else `https://api.pinference.ai/api/v1`. `X-Prime-Team-ID` comes from `PRIME_TEAM_ID` (`deps/verifiers/verifiers/v1/judges/reference.py:30-50`; `configs/judge.py:14-24`; `configs/client.py:23-57`). The key is read **at call time in the env-server worker** (scoring runs there): first `$PRIME_API_KEY`, then the `prime login` config's `api_key`, else `"EMPTY"` (`configs/client.py:93-104`). All of these were verified by executing `resolve_env_config` at the pin.
- **Disable:** `env.taskset.task.judge = "None"`. TOML has no null, and pydantic-config's `_none_str_to_none` maps the string (`deps/pydantic-config/src/pydantic_config/cli.py:99-108`). Verified: it resolves to `judge=None` and survives the JSON round-trip to the env server.
- **In prime-rl** (`examples/basic/hendrycks-sanity/rl.toml:21-25`): `env.taskset.id = "math-env"`, `env.taskset.dataset_name = ...`, `env.agent.harness.id = "null"`, `env.agent.runtime.type = "subprocess"`. This example does **not** disable the judge, and neither do `configs/basic/hendrycks-sanity/rl.toml`, `configs/ci/nightly-fft/hendrycks-sanity.toml`, or `examples/extra/dynamo/rl.toml` (§7 G5). `math_env` is the only math env with a judge.

### 3.12 Worked example B: `swesmith_env` (sandboxed, multi-turn agentic)

```
swe/swesmith_env/
├── pyproject.toml   name "swesmith-env"; deps verifiers>=0.3.1, swesmith, swebench>=4.1.0,<5;
│                    [tool.uv.sources] swesmith = git SWE-smith@9b74ac0; tags multi-turn,sandbox,v1   (pyproject.toml:1-21)
├── README.md        88,130 tasks over 8 HF datasets; long changelog                                 (README.md:1-24)
└── swesmith_env/{__init__.py (exports SWESmithTaskset), taskset.py (351)}
```

- **Data:** `SWESmithData` has `language`, `row` (the full upstream row, needed by the swesmith profile registry), `instance_id`, `gold_patch`, `fail_to_pass`, and `pass_to_pass` (`taskset.py:158-165`). Each task sets `image=row["image_name"]`, `workdir="/testbed"`, `resources=TaskResources(cpu=4, memory=4, disk=10)`, and `system_prompt=SOLVER_SYSTEM_PROMPT`, a fair-internet-use rule (`:49-63, 323-339`).
- **`NEEDS_CONTAINER = True`** (`:169`), so the subprocess runtime cannot run it.
- **`setup(runtime)`:** `git fetch origin`, `git checkout <instance_id>` (the branch head where F2P tests are removed), a venv symlink for Python, installing `ripgrep` from GitHub (needs network), and recording the pre-agent SHA (`:177-207`).
- **`finalize`:** `capture_patch` writes `trace.info["patch"]` (`:209-212`).
- **`solved`** (`@vf.reward(weight=1.0)`): `git checkout HEAD~1` restores the tests, reverts the test files, writes `/eval.sh` from `profile.get_test_cmd`, runs it, parses the status map, and returns 1 iff every F2P and P2P test passes. An exit code >1 raises (an env error, not reward 0) (`:214-257`).
- **`validate(runtime)`:** applies the gold patch in reverse and requires `solved == 1.0`, for `vf-validate` (`:259-279`).
- **Loading:** it iterates the languages. Rows whose repo has no registered swesmith profile are skipped with an aggregated warning (`:293-351`).
- **Grading visibility:** the category README's grading-visibility table documents, per SWE env, whether grading material is hidden from the live agent (`deps/prime-envs/environments/swe/README.md`, "Grading-material visibility"). This env matches the upstream authors.
- **In prime-rl** (`examples/advanced/qwen3-30b-a3b/swe.toml:51-55` uses `r2e-gym`, the same shape): `env.taskset.id`, `env.agent.harness.id = "bash"`, and `env.agent.runtime.labels = [...]`. The runtime type is left at the **default `prime`**, so Prime sandbox credentials are required (`prime login`, README of that example).

### 3.13 Other representative envs (one line each on what they teach)

- **`terminal_bench_2`** is a 19-line thin Harbor wrapper: `TerminalBench2Config(HarborConfig)` with `dataset: Literal["terminal-bench/terminal-bench-2"]`, and `TerminalBench2Taskset(HarborTaskset, vf.Taskset[HarborTask, TerminalBench2Config])` (`terminal/terminal_bench_2/terminal_bench_2/taskset.py:14-19`). It depends on `verifiers[harbor]` (`pyproject.toml:7-9`).
- **`tau2_bench`** bundles an Env, a Harness, and a Taskset:
  - `Tau2Harness.launch` runs Tau's own orchestrator in a subprocess via `runtime.run_program([sys.executable, "-m", __name__])`. It routes the *evaluated* agent through the verifiers `endpoint` + `secret`, and a *user simulator* (`gpt-4.1`) through Prime inference, falling back to `OPENAI_API_KEY`/`OPENAI_BASE_URL` (`tau2_bench/harness.py:54-128`).
  - The reward is read from `trace.info["tau2"]` (`taskset.py:34-44`).
  - `Tau2Env` pins `SubprocessConfig` so Tau runs with the interpreter that installed it (`:47-55`).
  - `load()` returns a list, so it is eager (`:58-121`).
- **`browsecomp_plus`** shows a **taskset-scoped shared toolset**:
  - `SearchToolsetConfig(vf.SharedToolsetConfig)` sits on `TasksetConfig.tools`, and `Taskset.toolsets(config)` returns `[SearchToolset(config.tools)]`.
  - Tools are registered on the MCP server with `TOOL_PREFIX = "browsecomp_plus"`, so the model sees `browsecomp_plus_search`.
  - Per-rollout state comes from `SearchState(vf.State)`, synced over the state channel and read by the `recall` metric (`browsecomp_plus/taskset.py:115-146`; `servers/search.py:30-106`).
  - `TaskData.network_allow = []` makes the solver default-deny (`taskset.py:96`). `NullHarness` is re-exported as the default harness.
- **`virl39k`** shows a **multimodal prompt**: `prompt=[vf.UserMessage(content=[TextContentPart | ImageUrlContentPart(data:<mime>;base64,...)])]`, built at load (`multimodal/virl39k/virl39k/taskset.py:107-141`). Scoring is a lenient math-verify that strips `<think>` and returns 0 on an unterminated think (`:42-63`).
- **`lean_common`** shows a **shared base package** used through a `[tool.uv.sources]` path dep (`lean/minif2f/pyproject.toml`):
  - Each Lean env subclasses `LeanTaskset`, implements `rows()`, and exports `LeanEnv`. `LeanEnv` is an `IsolatedVerifierEnv` whose config validator rejects any non-Prime runtime (`packages/lean_common/lean_common/env.py:10-26`).
  - The reward refuses to grade anywhere except the staged fresh verifier box (`task.py:149-158`).
  - A grader failure raises instead of scoring 0, so infra errors never become negative examples (`task.py:117-127`; README:52).
  - Training tasksets drop exact-text matches of eval statements (`taskset.py:163-175`, `scripts/eval_hashes.py`).
  - Network is off for both train and eval unless `network = true` (`taskset.py:93-94, 150`).

### 3.14 Catalog (all 102 env packages at the pin)

**Method:** each package's `pyproject.toml` (tags, deps) and `__init__.__all__` exports were scanned mechanically, and the sources were grepped for `NEEDS_CONTAINER = True`, `HarborTaskset`, judge usage, `toolsets`, and `INFINITE`. Reward descriptions come from each category README. "Default harness" follows §3.3: `bash` unless the package exports a harness. Treat per-env details as indicative and read the env before relying on them.

**Legend:**
- **Shape:** ST = single-turn, MT = multi-turn/agentic.
- **Runtime:** "host" means no container is needed (subprocess works). "container" means `NEEDS_CONTAINER` or a per-task image (docker/prime/modal). "Prime-only" means the env rejects other runtimes.
- **Env:** "own" means the package exports an `Env`. `HarborEnv` means isolated-verifier grading is available.

| Group | Env ids (module names) | Shape | Runtime | Env / harness | Reward |
|---|---|---|---|---|---|
| **math** (8) | `aime24`, `aime25`, `aime26`, `apex_shortlist`, `math500`, `math_env`, `i3_math`, `arxivmath` | ST (arxivmath: "bring your own web search") | host | SingleAgent / bash (examples pin `null`) | math-verify on `\boxed{}`. `math_env` adds a ReferenceJudge fallback, on by default. |
| **code** (7) | `humaneval`, `i3_code`, `livecodebench` | ST | in-runtime test execution (host OK) | SingleAgent / bash | hidden tests |
| | `forth_lang` | MT with its own toolset (submit/run/lookup) | sandboxed gforth image | SingleAgent | binary on hidden tests |
| | `nl2repobench` (harbor extra), `programbench_env`, `scicode` (own `SciCodeEnv`) | MT | container | | hidden pytest / tests in sandbox |
| **if** (2) | `ifbench`, `ifeval` | ST | host | SingleAgent | strict/loose rule checks |
| **knowledge** (6) | `hle`, `simpleqa`, `simpleqa_verified` | ST | host | | LLM judge (HLE uses structured output) |
| | `mmlu_pro`, `triviaqa`, `mmmu` (multimodal) | ST | host | | math-verify / alias exact match / MC |
| **multimodal** (4) | `virl39k` (train), `mmk12`, `mmmu_pro`, `charxiv` | ST | host | | math-verify (mmk12 adds a judge fallback), MC, LLM judge (charxiv) |
| **reasoning** (6) | `i3_logic`, `unscramble` | ST | host | | in-process verifiers, difflib |
| | `prolog`, `uuid_ctf`, `deshuffle_papers` (INFINITE) | MT | container | | sandbox-verified. deshuffle blocks egress. |
| | `wikispeedia` (INFINITE) | MT with its own toolset | host | | reaching the target |
| **long_context** (12) | `patterned_needle_in_haystack`, `verbatim_copy`, `clbench` | ST | host | | exact match / LLM judge |
| | `graphwalks`, `longbenchpro`, `longcot_env`, `longcot_mini`, `longpdfs`, `mrcr_v2`, `oolong_pairs`, `oolong_real`, `oolong_synth` | MT (agent reads the document in a sandbox) | container | | set F1, SequenceMatcher, Oolong scoring, judge |
| **science** (4) | `gpqa`, `frontierscience` | ST | host | | letter extraction (optional judge) / LLM judge |
| | `i3_science` | MT | sandbox | | math-verify + judge fallback |
| | `drug_discovery_bench` | MT, Harbor | container, own Env | | gated expert rubrics |
| **search** (10) | `browsecomp`, `deepdive`, `openseeker`, `papersearchqa`, `redsearcher`, `s1_deepresearch`, `wideseek` | answer in chat, bring your own search (`--env.agent.harness.search true` or `rlm` + `forward_env = ["SERPER_API_KEY"]`) | host or sandbox per harness | several export their own `Judge` class | LLM judge (plugged via `task.judges` in the GLM example) |
| | `browsecomp_plus` | MT with a shared BM25 toolset | host | bundled `NullHarness` | HLE-style judge + recall metric |
| | `officeqa_pro_v2`, `pi_officeqa_pro_v2` | MT | Prime VM image (corpus baked in) | | regex / judge |
| **swe** (16) | `r2e_gym`, `swesmith_env`, `scaleswe`, `swelego`, `swerebench_v2`, `multiswe`, `openswe` | MT | container (per-task image, prime by default) | SingleAgent / bash | F2P/P2P test pass (0/1) |
| | `swebench_verified`, `swebench_multilingual`, `senior_swe_bench` | MT, Harbor wrapper | container | SingleAgent (taskset-only export) | Harbor `tests/test.sh` |
| | `swebench_pro`, `swe_atlas_{qna,rf,tw}`, `deep_swe`, `pi_deepswe` (own `DeepSWEEnv`) | MT, Harbor | container | `HarborEnv` (separate verifier) | Harbor verifier |
| **terminal** (5) | `terminal_bench_2`, `pi_terminal_bench_2`, `openthoughts_tblite`, `terminal_lego`, `tmax` (registry-backed, 14,600 tasks) | MT, Harbor | container | SingleAgent / bash | Harbor `test.sh` pass/fail |
| **tool_use** (10) | `bfcl_v3` (own `BFCLEnv`), `enterprise_ops_gym`, `general_agent`, `mcp_atlas` | MT with toolsets / MCP services | host or colocated (`general_agent` on Modal in the example) | | AST/rule, DB state, DB hash, claim coverage |
| | `tau2_bench` (tau2_synth, tau3_bench: **excluded from the workspace**) | MT with a user simulator | subprocess (pinned) | own Env + Harness | official Tau2 evaluation |
| | `pinchbench`, `lab` (`HarborEnv`), `apex_agents` (excluded) | MT | sandbox / Harbor | | hybrid / RewardKit rubrics |
| **lean** (11) | `minif2f`, `proverbench`, `matholympiadbench`, `proofnet`, `cambench`, `formalmath`, `putnambench`, `combibench`, `discover_and_prove` (eval); `kimina`, `numina` (train) | MT, agentic proving | **Prime-only VM** (`LeanEnv`) | `LeanEnv` / bash | kernel check (`proved`), second kernel (nanoda) |
| **judges** (1) | `trace_cheating_recall_500` | ST judge audit | host | | judge recall over 500 SWE traces |

The **agentic-judge** artifacts (`deps/prime-envs/judges/`) are not envs. They are prompt, rubric, and hint files for the built-in `--env.id agentic-judge` env (§6.5; `judges/README.md:1-28`).

## 4. Interfaces & contracts

### 4.1 Package contract (what makes a directory an env)

| Item | Requirement | Cite |
|---|---|---|
| Importable module | A module name equal to the normalized id (`-`→`_`, lowercased, last path segment, `@ver` stripped). It is normally a package installed by the workspace or pip. A single `.py` module on `PYTHONPATH` also works; the verifiers test fixtures load this way. | `loaders.py:70-78`; `deps/verifiers/tests/v1/conftest.py:85-96` |
| `__all__` | Must list exactly one `vf.Taskset` subclass. Optionally one `vf.Env` and one `vf.Harness` subclass. Other names (data/config classes) are allowed. | `loaders.py:99-126`; `skills/create-environments/SKILL.md:52-56` |
| No loader functions | No `load_environment()`, `load_taskset()`, or `load_harness()`. Config types come from the generics `Taskset[TaskT, ConfigT]`, `Task[DataT, StateT, ConfigT]`, and `Env[ConfigT]`. | `skills/create-environments/SKILL.md:56` |
| Do not override `Taskset.__init__` | except for validation that calls `super().__init__`, as `LeanTaskset` does | `skills/create-environments/SKILL.md:102`; `lean_common/taskset.py:109-113` |
| `TaskData` wire-safety | JSON round-trip through `model_dump(mode="json")` → `model_validate`. No `exclude=True` fields. No live handles. | `serve/server.py:40-51, 80-92` |
| `TaskConfig` | every field needs a default | `configs/task.py:31-37` |
| pyproject | hatchling, `[tool.hatch.build.targets.wheel] packages = ["<pkg>"]`, deps `verifiers>=0.3.1` (`verifiers[harbor]` for Harbor), Python `>=3.11` (Lean `>=3.12`). Git deps pin a 7-char rev. | `math_env/pyproject.toml`; `deps/prime-envs/AGENTS.md:33-35` |
| README | Must be kept up to date with a changelog. Category READMEs list the envs. | `deps/prime-envs/AGENTS.md:80`; `swe/README.md` Workflow |

### 4.2 prime-rl source config fields (`[[orchestrator.train.source]]`, `[[orchestrator.eval.source]]`)

| Field | Type / default | Effect | Cite |
|---|---|---|---|
| `env` | `vf.EnvConfig`, default `SingleAgentEnvConfig()`; narrowed by env/taskset id | the whole verifiers `[env]` block (§4.3) | `orchestrator.py:160-176` |
| `serve` | `vf.ServeConfig` | `pool` (elastic by default: `max_workers=None`, `multiplex=128`; or static `num_workers=4`), `address` (set = externally managed), `max_concurrent` (episodes per worker) | `orchestrator.py:163-164`; `deps/verifiers/verifiers/v1/configs/serve.py:13-65` |
| `name` | `str \| None` | display/buffer/metrics key; defaults to `env_id`; unique per group; not `agg` | `orchestrator.py:166-193` |
| `shuffle` | `bool=False` | shuffle a finite taskset once, seed 42 | `orchestrator.py:169-170`; `envs.py:47,114-115` |
| train: `sampling` | `TrainSamplingConfig` | merged over `orchestrator.train.sampling`; `extra_body` may not carry truncation knobs | `orchestrator.py:45-108, 269, 314-325` |
| train: `ratio` | `float>0 = 1.0` | relative env-draw weight | `:272-273` |
| train: `group_size` | `int≥1`, inherits `orchestrator.group_size` | rollouts per task | `:275-277, 751-754` |
| train: `algo` | `AlgoConfig \| None`, inherits `orchestrator.algo` | per-env algorithm; `validate_env` hook | `:279-282, 627-642` |
| train: `curriculum` | `{sampler: standard\|difficulty_pool, gates: {name: advantage_range}}` | task selection and admission | `:259-265, 284-286` |
| eval: `num_examples` | `int=-1`, inherits the group | size of the fixed eval set | `:293-294, 364-365` |
| eval: `group_size` | inherits the group | pass@k | `:296-297` |
| eval (online): `interval` | inherits `orchestrator.eval.interval=100` | step interval | `:300-304, 396-414` |
| `orchestrator.tasks_per_minute` | `int \| None` | global dispatch rate limit | `:573-574` |

### 4.3 verifiers `[env]` block fields used from prime-rl TOML

| Path | Type / default | Cite |
|---|---|---|
| `env.id` | `""`. An empty value uses the taskset's own Env, else single-agent. Set it to pair a reusable env (`best-of-n`, `agentic-judge`, `user-sim`, `isolated-verifier`). | `deps/verifiers/verifiers/v1/configs/env.py:40-42` |
| `env.taskset.id` | required (`validate_env` raises "no env configured") | `configs/taskset.py:13-15`; `orchestrator.py:186-189` |
| `env.taskset.<field>` | the taskset's own config fields | §3.2 |
| `env.taskset.system_prompt` | `Path \| None`: a file whose text overrides every task's `system_prompt` | `configs/taskset.py:18-20`; `taskset.py:43-47,54-62` |
| `env.taskset.task.<field>` | the taskset's `TaskConfig` fields (judge, tools, scoring knobs) | `configs/taskset.py:16-17` |
| `env.taskset.task.{judges,stops,metrics,rewards}` | plugged signals: `judges` is a list of `{id, name, weight, model, base_url, api_key_var, sampling, prompt, ...}`; the others are dicts of `{fn, priority[, weight]}` | `configs/task.py:39-51`; `configs/judge.py:14-24`; `docs/v1/tasksets.md:238-260` |
| `env.agent.harness.{id, env, forward_env, disabled_tools, skills, tool_timeout=600, ...}` | `id` defaults to `bash` when the block is present (even without `id`, §3.2); absent means the taskset default | `configs/harness.py:26-66`; `configs/agent.py:65` |
| `env.agent.runtime.{type, labels, cpu, memory, image, ...}` | default `PrimeConfig()` | `configs/agent.py:30` |
| `env.agent.{model, client, sampling}` | `None` means the run's policy, client, and sampling | `configs/agent.py:34-39` |
| `env.agent.{max_turns, max_input_tokens, max_output_tokens, max_total_tokens}` | `None` means unlimited | `configs/agent.py:41-48` |
| `env.agent.timeout.{setup, rollout, finalize, scoring}` | rollout unset means the task's timeout, else 4 h | `configs/agent.py:13-24` |
| `env.agent.retries`, `env.retries` | per-agent-run retries and whole-episode retries | `configs/agent.py:51`; `configs/env.py:47-49` |
| `env.timeout.{episode, finalize}` | env hooks | `configs/env.py:18-23` |
| `env.max_concurrent_agents` | `1` | `configs/env.py:50-59` |
| `env.interception` | `elastic` / `server` / `static` | `configs/env.py:60-61` |
| `env.verifier.runtime.*`, `env.verifier.retries` | isolated-verifier envs (Harbor/Lean) | `docs/v1/env.md:78-86` |

### 4.4 Files and directories

| Path | Writer → reader | When | Cite |
|---|---|---|---|
| `<output>/<run>/configs/.../envs/<split>/<name>.json` | launcher → env-server | before spawn | `pathing.py:230-247` |
| `.../envs/<split>/<name>.address` | env-server → orchestrator | once the server is bound (atomic replace) | `env_server.py:24-31`; `envs.py:50-62` |
| `logs/.../envs/<split>/<name>.log` | env-server stdout/stderr | runtime | `rl.py:267-280` |
| `~/.cache/huggingface` | orchestrator (`load_dataset`) | `start()` | HF |
| `~/.cache/harbor/<dataset>[_sha12]` | orchestrator (download), then env server (tar `tests/` and `environment/` from `task_dir`) | load / each rollout | `tasksets/harbor/taskset.py:48, 447-523, 196-216, 290-317` |
| `$TAU2_DATA_DIR`, `$BROWSECOMP_PLUS_CACHE_DIR`, `$LEAN_AGENTIC_CACHE` | taskset loaders (and toolsets) | load | §3.9 |

### 4.5 Environment variables and secrets

| Var | Needed by | Cite |
|---|---|---|
| `PRIME_API_KEY`, `PRIME_TEAM_ID` (or `prime login` config) | the default `prime` runtime; default judges (`ReferenceJudgeConfig` → Prime inference); tau2's user simulator | `configs/agent.py:30`; `configs/client.py:29-57`; `tau2_bench/harness.py:76-93` |
| `PRIME_INFERENCE_URL` | overrides the Prime inference base URL for `PRIME_API_KEY` clients | `configs/client.py:41-48` |
| `HF_TOKEN` | gated datasets (gpqa, hle, drug_discovery_bench, officeqa). Keep it evaluator-side. | `deps/prime-envs/tests/test_envs.py:21-27`; `skills/evaluation/SKILL.md:35-38` |
| `SERPER_API_KEY` | bring-your-own-search tasksets. Forward it into the harness with `env.agent.harness.forward_env = ["SERPER_API_KEY"]`. | `examples/advanced/intellect-3.1/rl.toml`; `skills/evaluation/SKILL.md:160` |
| `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` | `runtime.type = "modal"` | `examples/advanced/qwen3-30b-a3b/README.md:24-27` |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` | tau2's user-sim fallback | `tau2_bench/harness.py:81-85` |
| `[verifier.env]` templates `${VAR}` | Harbor verifiers, resolved on the env-server host at scoring time | `tasksets/harbor/taskset.py:162-165, 782-791` |
| `VF_RUN_ID` (from `PRL_RUN_ID`), `PRL_RUN_NAME` | env server: run-scoped limiters, sandbox labels | `env_server.py:36-37, 57-60` |

**Where secrets travel.** Env servers inherit the launcher's environment plus `[env_vars]` and `[orchestrator.env_vars]` (`rl.py:272-277`). Inside the agent box, only what the harness passes through arrives (`HarnessConfig.env` / `forward_env`, read from the env-server worker's `os.environ`; explicit `env` wins; `configs/harness.py:30-33, 64-66`) and what the task sets through `Task.runtime_env()`, which is live-only and not traced (`task.py:171-173`).

### 4.6 CLIs

| Command | What | Cite |
|---|---|---|
| `uv run vf-init <name> [-p dir] [-T] [-H] [--force]` | Scaffolds `<dir>/<pkg>/{pyproject.toml, README.md, <pkg>/{__init__.py, taskset.py[, harness.py][, servers/tool.py]}}`. `pkg` is the name lowercased with `-`→`_`; the distribution name uses the dash form. The pyproject depends on unpinned `verifiers`. The default dir is `./environments`. Its printed "next" (`uv pip install -e`) does not survive a prime-rl `uv sync`. | `deps/verifiers/verifiers/v1/cli/init.py:10-232`; `configs/cli/init.py:7-17` |
| `uv run vf-validate <taskset-id> [--only-setup\|--only-gold] [--runtime.type ...] [-n] [-s] [-c 128]` | Model-free check: per task, `setup` then `validate` (gold) and setup-only. Writes `outputs/<run>/{results.jsonl, summary.json, configs/validate.json}`. The default runtime is **prime**. `subprocess` is refused for container tasks. | `cli/validate.py:327-426, 337-342`; `configs/cli/validate.py:22-89` |
| `uv run vf-eval <taskset-id> [-m] [-n] [-r] [-c] [-s] [--dry-run] [--no-push] [@ toml]` | Standalone verifiers eval. It pushes to the platform by default (`--no-push` keeps it local). | `deps/verifiers/pyproject.toml:151-157`; `skills/evaluate-environments/SKILL.md:20-41`; `docs/v1/evaluation.md:49-52` |
| `uv run eval <taskset-id> -n -r -m --client.base_url ... [--env.<field>]` (**prime-rl's**) | prime-rl's eval: env servers + monitors, the same `EvalEnvs` + `Dispatcher` as online eval (`eval/runner.py:98-117`). `-m/-n/-r` are field aliases; `<taskset-id>`, `--env.*`, and `-c` are rewritten into one JSON `--source` by `expand_shorthands` and cannot be combined with a TOML defining `[[source]]`. Defaults: model `deepseek/deepseek-v4.1-flash` on Prime inference, concurrency pinned at 128. | `pyproject.toml:46`; `src/prime_rl/entrypoints/eval.py:48-84`; `configs/eval.py:48-69`; `skills/eval/SKILL.md:14-26` |
| `uv run env-server @ envs/<split>/<name>.json` | the env server | `pyproject.toml:47`; `env_server.py:54-61` |
| `harbor datasets list/download ... --repo/--registry-path/--registry-url` | Harbor registry | `deps/prime-envs/HARBOR.md` |

### 4.7 Example and config inventory (env blocks)

- **`examples/basic/`:**
  - `reverse-text` (`reverse-text`, null/subprocess)
  - `alphabet-sort` (`alphabet-sort` with `min_turns`/`max_turns`, `task.power_per_turn`)
  - `wiki-search` (`wiki-search`, null/subprocess)
  - `wordle`
  - `hendrycks-sanity` (`math-env` with an alternate dataset; eval `aime24`)
- **`examples/advanced/`:**
  - `qwen3-30b-a3b/{math (i3_math → aime25), swe (r2e-gym, bash, prime → swebench-verified), tool (general-agent, colocated tools, null, modal)}`
  - `glm-4.5-air/{search (openseeker + redsearcher via rlm+search skill, plugged `reference` judges → browsecomp), swe (scaleswe), swe-*-node/budget, terminal (tmax → swebench-verified, terminal-bench-2)}`
  - `glm-5.3/swe*.toml` (r2e-gym, rlm)
  - `intellect-3.1/rl.toml` (5-way mix: r2e-gym 0.3, deepdive 0.2, i3_math 0.3, i3_logic 0.2 with `filter.max`, i3_code 0.2)
- **`examples/extra/`:** `dynamo`, `vlm/{sft,sft-moe}`
- **`configs/`:**
  - `basic/*` (2-GPU twins of the examples)
  - `debug/multi-env/rl.toml` (two named reverse-text sources)
  - `debug/eval/{single-turn, multi-turn, multi-env (tb2 × bash/rlm), resume, aime2026, tb2}.toml`
  - `ci/integration/*` (reverse-text variants, alphabet_sort, gsm8k-eval)
  - `ci/nightly*/*`, `ci/nightly/multimodal_color_codeword.toml`
  - `advanced/{minimax-m2.5, nemotron-3-super}/swe.toml`, `deepseek-v4-flash/*`

## 5. Invariants & assumptions

1. **Import parity:** the launcher, orchestrator, and every env-server worker must import the same env package version. Config validation, task loading, and task execution each import it independently (§3.2). Nothing checks the versions against each other.
2. **Deterministic, stable `load()` order:** the default sampler cycles source order and restores by cursor, so resume assumes an identical task sequence. Unpinned HF datasets (`load_dataset` without `revision`, which most prime-envs use; only a handful pin one) can drift between runs (`standard.py:37-49`).
3. **Unique `Task.key` within a taskset:** enforced by both samplers (`standard.py:23-26`; `pool.py:36-39`). The curriculum also asserts that one group has exactly one key (`curriculum/base.py:54-58`).
4. **`TaskData` is self-sufficient and JSON-roundtrippable**, which the dispatcher enforces per episode via `(key, hash)` provenance and a mismatch crashes the orchestrator (§3.5). Anything the server needs to run or score must be on the data or in the server-side `TaskConfig`. Host-path fields such as `HarborData.task_dir` assume the **orchestrator and env server share a filesystem path**. This holds for launcher-managed servers, which run on the same node (`multi_node_rl.sbatch.j2:541-559`), and is not guaranteed for externally managed ones.
5. **One `Taskset` export per module.** More than one is a hard `ValueError`, even for the fallback callers (`loaders.py:118-125`).
6. **Built-ins shadow installed packages** with the same normalized name in `verifiers.v1.{tasksets,harnesses,envs,judges}` (`loaders.py:75-76`).
7. **Rewards are floats.** Weighted rewards are recorded per key. A `Mapping` return records multiple keys (`task.py:254-263`). Errors during scoring become a `TaskError` on the trace; the rollout is not a reward-0 sample. Envs such as Lean and swesmith deliberately raise on grader/infra failure so that it becomes an error rather than a negative example.
8. **Infinite tasksets** must declare `INFINITE = True`. prime-rl refuses `shuffle` on them and needs `num_examples ≥ 0` for eval (`envs.py:106-110, 190-191`). `DifficultyPoolSampler` refuses them (`pool.py:31-32`).

## 6. Extension points (recipes)

### 6.0 Common scaffolding and registration (every recipe)

1. **Scaffold.** From the prime-rl root: `uv run vf-init my-task -p deps/prime-envs/environments/<group>`. The default `-p` is `./environments`, which is **not** a workspace member path in prime-rl (`cli/init.py:205-232`). Use `-T` for a toolset and `-H` for a harness.
2. **Register.** Placing it under `deps/prime-envs/environments/<group>/<name>/` makes it an auto-discovered workspace member (`pyproject.toml:157-164`). To install it, run `uv sync --all-extras --all-packages` or `uv sync --package prime-rl --package my-task`, using the dash-form distribution name (`skills/install/SKILL.md:24-34`). Either one rewrites `uv.lock`; commit it, because the image builds `--locked`. Outside the workspace, `uv pip install -e path` works, but the next non-`--inexact` `uv sync` removes it. There is **no registry to edit**, and `registry.json` is unrelated.
3. **Dependencies.** Put them in the env's own `pyproject.toml`. **Never** in the repo root or `uv.lock` of prime-envs (`skills/create-environments/SKILL.md:220-225`). In prime-rl, the shared lock means a conflicting pin forces a `[tool.uv.workspace].exclude` entry, which means you cannot train on that env from this venv (§7 G3).
4. **Check that it imports.** `uv run python -c "import my_task"`, then `uv run vf-eval my-task --help`. The help shows the typed `--env.taskset.*` fields, which proves the package is discoverable (`skills/evaluate-environments/SKILL.md:76-84`).
5. **Validate model-free** (if `validate()` is implemented): `uv run vf-validate my-task --runtime.type subprocess` (host tasks) or `--runtime.type docker|prime` (container tasks). `runtime` is a top-level `vf-validate` field (default `prime`), not `--env.agent.runtime`. `subprocess` is refused if any selected task has `NEEDS_CONTAINER` or a `data.image` (`configs/cli/validate.py:22-45`; `cli/validate.py:327-342`).
6. **Smoke eval against an API model.** `uv run vf-eval my-task -n 3 -r 1 --no-rich -v --no-push`. The default model is `deepseek/deepseek-v4-flash` on Prime inference, so `PRIME_API_KEY` or `prime login` is needed. Add `--env.agent.harness.id null --env.agent.runtime.type subprocess` for a non-agentic task (§7 G1). Then run >500 total rollouts for a representative reward (`deps/prime-envs/skills/evaluation/SKILL.md:54-71`). Inspect `outputs/<run>/traces.jsonl`.
7. **Eval through prime-rl's env-server path** against the model you will train. Start `uv run inference --vllm.model <M>`, then `uv run eval my-task -n 32 -r 4 -m <M> --client.base_url http://localhost:8000/v1 --env.agent.harness.id null --env.agent.runtime.type subprocess` (`skills/eval/SKILL.md:14-26`; `docs/eval.md:28-35`). This exercises the orchestrator-side taskset load, the env-server spawn and address handshake, the `TaskData` wire round-trip and provenance check, and scoring. It does **not** exercise the training wire. Eval groups always use the eval client, which is chat completions through the server's chat template (`dispatcher.py:543-552`), not the renderer/`TrainClient` token path. Harness-dialect refusals and renderer or tool-parser problems (F/H) surface only in a real `rl` run.
8. **Train.** Add a `[[orchestrator.train.source]]` block, then `uv run rl @ my.toml --dry-run` to resolve. Config parsing precedes the dry-run exit, so it already imports your package, validates your taskset fields, and writes `envs/train/<name>.json` (`skills/configs/SKILL.md:28`). Then run a short real `rl` (small `max_steps`) to exercise the `TrainClient`/renderer path.

### 6.1 Recipe: single-turn verifiable task

Files: `my_task/{pyproject.toml, README.md, my_task/{__init__.py, taskset.py}}`, following `reverse_text` and `math_env`.

```python
import verifiers.v1 as vf
from collections.abc import Iterator

class MyData(vf.TaskData):
    answer: str                                   # gold; rides the wire, lands on the trace

class MyTaskConfig(vf.TaskConfig):               # every field defaulted
    tolerance: float = 0.0

class MyTask(vf.Task[MyData, vf.State, MyTaskConfig]):
    @vf.stop                                      # hard single-turn guard (reverse_text pattern)
    async def single_turn(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= 1
    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace) -> float:
        return float(vf.verify_boxed_math_answer(trace.last_reply, self.data.answer, timeout_seconds=5))
    @vf.metric
    async def has_box(self, trace: vf.Trace) -> float: ...

class MyConfig(vf.TasksetConfig):
    dataset_name: str = "org/ds"; dataset_split: str = "train"
    task: MyTaskConfig = MyTaskConfig()

class MyTaskset(vf.Taskset[MyTask, MyConfig]):
    def load(self) -> Iterator[MyTask]:
        from datasets import load_dataset
        for i, row in enumerate(load_dataset(self.config.dataset_name, split=self.config.dataset_split)):
            yield MyTask(MyData(idx=i, prompt=row["q"], answer=row["a"]), self.config.task)
```

`__init__.py`: `from my_task.taskset import MyTaskset; __all__ = ["MyTaskset"]`.

The pattern comes from `deps/verifiers/environments/reverse_text/reverse_text/taskset.py:20-60` and `math_env/taskset.py`. For a stable resume identity, override `Task.key` to use a dataset id (`task.py:156-164`).

TOML:
```toml
[[orchestrator.train.source]]
name = "my-task"
env.taskset.id = "my-task"
env.taskset.dataset_split = "train"
env.agent.harness.id = "null"          # tool-less chat loop; do NOT leave unset (defaults to bash)
env.agent.runtime.type = "subprocess"  # do NOT leave unset (defaults to prime)
```

### 6.2 Recipe: multi-turn tool env

Before writing tools, check AGENTS.md. It says prefer a general harness with *no* custom tools (`deps/prime-envs/AGENTS.md:97-101`). When tools are genuinely needed, pick the scope:

- **Per-rollout toolset:** `vf-init my-tool -T` generates `servers/tool.py` with `class MyToolset(vf.Toolset[vf.ToolsetConfig])`, `TOOL_PREFIX`, `@vf.tool` methods (the docstring is the tool description), and `if __name__ == "__main__": MyToolset.run()`. It also generates `MyTaskConfig.tools: vf.ToolsetConfig` and `Task.toolsets(cls, config) -> [MyToolset(config.tools)]` (`cli/init.py:61-134`). By default the server runs host-side in a subprocess. `tools.colocated = true` runs it inside the harness box, so it sees the agent's FS (`skills/create-environments/SKILL.md:182-187`; `examples/advanced/qwen3-30b-a3b/tool.toml:41`).
- **Shared toolset (one per env-server worker):** `vf.SharedToolsetConfig` on `TasksetConfig`, plus `Taskset.toolsets(cls, config)` (`browsecomp_plus`, §3.13). Per-rollout counters live in a `vf.State` subclass passed as `Toolset[Config, State]`. For a heavy index, build it in `load()` (orchestrator) *and* `setup()` (server), because both processes need it (`browsecomp_plus/corpus.py:42-63`; `servers/search.py:50-62`).
- **Remote MCP service:** set `url` on the toolset config (`skills/create-environments/SKILL.md:187`).
- **Harness requirement:** it must support MCP. `null` (a chat loop with MCP tools only) and `bash` do (`deps/prime-envs/skills/evaluation/SKILL.md:160`). A task that declares tools with a non-MCP harness fails the fit check, which is F/G's domain.
- **User in the loop** (scripted user, game engine, LLM user): this is **env control flow**, not a tool. Export an `Env` subclass whose `run()` drives `agents.agent.interaction(task)` / `turn()`, or use `--env.id user-sim` (`skills/create-environments/SKILL.md:189-196`; examples `alphabet_sort`, `wordle`).

### 6.3 Recipe: sandboxed agentic env (SWE-style)

1. **Data.** Put `image` (a pullable ref; for Prime, any pullable ref is built and cached on first use), `workdir`, `resources=vf.TaskResources(cpu, memory, disk)`, optionally `timeout=vf.TaskTimeout(...)`, `network_allow`/`network_block`, and `artifacts` on `TaskData` (`task.py:81-122`).
2. **Behaviour.** `NEEDS_CONTAINER = True`. `setup(runtime)` prepares the checkout. **Hide grading material** until scoring, by host-roundtripping it or restoring it in the reward (`swe/README.md` "Grading-material visibility"). `finalize` captures the patch (`capture_patch`, `resolve_head` from `verifiers.v1`). `@vf.reward` runs the tests in `runtime`. `validate(runtime)` applies the gold patch and checks reward == 1.
3. **If grading must not share the agent's box:** export an `IsolatedVerifierEnv` subclass (the `LeanEnv` pattern) and implement `stage_verifier`. You can also point users at `--env.id isolated-verifier` (`docs/v1/env.md:23-86`).
4. **Validate at scale** before training. No-op runs must fail, gold runs must pass, and repeated passes separate flaky infra from bad tasks (`swe/README.md` Workflow 3). Mirror the images into the Prime registry (`prime images push` / `transfer-bulk`, Workflow 4; image naming `<env>.x86.<task>:latest`, `skills/create-environments/SKILL.md:31-35`).
5. **TOML:** `env.agent.harness.id = "bash"` (or another trainable harness: `rlm`, `pi`, `kimi_code`, `prime_agent`, `hermes_agent` at their defaults; `openclaw` depends on the model name; not `codex`/`claude_code`, which are eval-only because `TrainClient` is chat-completions-only; full table in 08 §3.7), `env.agent.runtime.labels = ["run", "env"]` (the Prime runtime is the default), and consider `orchestrator.tasks_per_minute` and `orchestrator.concurrency.max_inflight` to bound live sandboxes (`orchestrator.py:496-497, 573-574`).

### 6.4 Recipe: Harbor-format tasks (no custom scoring code)

- **Zero code:** `env.taskset.id = "harbor"`, `env.taskset.dataset = "org/name@ref"` (or `registry_path = "deps/prime-envs/registry.json"` + `dataset = "tmax@2026-07-01"`). The built-in module exports `HarborEnv`, so separate verifiers work.
- **A pinned package:** subclass `HarborConfig` with a `Literal` dataset, plus `class T(HarborTaskset, vf.Taskset[HarborTask, TConfig])` (`terminal_bench_2`). **Export `HarborEnv` too** if any task uses `environment_mode = "separate"` (§3.10). Task images must be prebuilt and pullable, because verifiers never builds Dockerfiles.

### 6.5 Recipe: multi-agent / judged episodes

- **Reuse first:**
  - `--env.id best-of-n` (`--env.n`) for pass@k / rejection sampling.
  - `agentic-judge` (a judge agent verifies the solver in its box; see `deps/prime-envs/judges/swe/eval.toml:1-13`, where `solved` carries the weight and the taste criteria are weight-0 metrics, `rubrics.toml:1-76`).
  - `user-sim`.
  - `isolated-verifier`.

  (`skills/evaluate-environments/SKILL.md:55-63`; `docs/v1/env.md:17-21`)
- **Custom:** export `class MyEnv(vf.Env[MyEnvConfig])`. Declare each role as `vf.AgentConfig` **with a default instance** (a bare annotation or `default_factory` raises at class definition, `configs/env.py:125-146`). Write `run(task, agents)`, and optionally `setup(agents)` (e.g. `agents.judge.trainable = False`) and `finalize(task, episode)` for cross-trace rewards (`env.py:147-184`).
- **In lockstep on the prime-rl side:** an algorithm whose credit assignment encodes the episode structure must override `validate_env` (`hierarchical_grpo`, `configs/algorithm.py:279-294`).

### 6.6 Porting an existing benchmark (BENCHMARK_PORTING.md, distilled)

- **Preserve verbatim** (these are benchmark *data*): prompts, rubrics, model choices, sampling parameters, task assets, scoring semantics, and workspace layout, including files that were *removed*. Document any deviation (`BENCHMARK_PORTING.md:23-37, 194-209`).
- **Map to the smallest abstraction:**
  - prompt → answer: a plain single-turn taskset with `@vf.reward`.
  - tools / files / workspace: a sandboxed taskset (`NEEDS_CONTAINER`) + a coding harness (`bash`).
  - a truly custom protocol: the smallest custom `Env`.
  - LLM judge: `vf.ReferenceJudge` (`vf.Judge` with `schema=` for structured output).
  - deterministic + judge: keep them separate and combine only at the final layer.

  (`:59-107, 140-176`)
- **Keep configuration single-path.** Rollout and judge share client config unless upstream requires otherwise (`:89-97, 211-223`).
- **Order of work:** invariants → split data from adapter → map → write gap adapters only → validate behaviour. Validation order: task loading → one rollout → one task per scoring mode → one concurrency check → the full suite. It must show both *correctness* and *use of the intended framework path* (`:109-119, 242-264`).
- **Prompting policy** (for training envs): no role prompts. Keep prompts minimal. **Instructions and reward must match.** Do not instruct behaviours that should generalize; reward them instead (`deps/prime-envs/AGENTS.md:103-127`).

## 7. Gotchas & limitations

- **G1 — `null` + `subprocess` are not defaults.** An unpinned seat runs the `bash` harness in a **Prime sandbox** (`configs/agent.py:30`; `configs/env.py:159-165`). Every non-agentic example pins both. Forgetting this changes the task (tools appear), costs money, and needs `PRIME_API_KEY`.
- **G2 — Import-time coupling.** Config validation imports the taskset, harness, and env packages in *every* process that parses the config (§3.2). An externally managed env server (`serve.address`) does **not** remove the need to install the env in the orchestrator's venv: the orchestrator both validates the config and runs `load()`. For SLURM the `rl` launcher parses the config on the submit host *before* the sbatch's own `uv sync --all-packages`, so the submit host's venv needs the env too.
- **G3 — One lock for all envs.** Envs with conflicting pins cannot coexist. `tau2_synth`, `tau3_bench`, and `apex_agents` are excluded and **cannot be trained from the stock prime-rl venv** (`pyproject.toml:174-182`). A plain `uv sync` without `--all-packages`/`--inexact` uninstalls workspace envs (`skills/install/SKILL.md:34`).
- **G4 — Full materialization in the orchestrator.** Finite tasksets are `list()`ed at startup, including eval sources with a small `num_examples` (`envs.py:111-117, 187-194`). Examples: swesmith's 88k tasks each carry a full upstream `row` dict. virl39k base64-encodes ~39k images into orchestrator RAM *and* ships them on every run request. The prime-envs "lazy under `-n`" guarantee is a `vf-eval` property only.
- **G5 — The math judge fallback is on by default.** `math_env` calls `ReferenceJudge` (gpt-5.6-luna on Prime inference) whenever math-verify misses (`math_env/taskset.py:28-33, 44-62`). In training this makes reward partly an external LLM call: non-deterministic, costly, rate-limited, and needing `PRIME_API_KEY`. Judge calls run in the env-server worker. A missing key resolves to `"EMPTY"` (`configs/client.py:93-104`). `Judge.complete` has no error handling, and `judge_verdict` raises on a verdict matching neither label (`judge.py:102-107, 163-211`). Any such exception escapes inside `Task.score`'s `boundary(TaskError, ...)` and becomes a **`TaskError` on the trace** (an errored rollout, not reward 0) (`task.py:221`; `errors.py:76-89`). That the endpoint actually rejects `"EMPTY"` is runtime behaviour. Disable the judge with `env.taskset.task.judge = "None"` (§3.11). No shipped math-env config does.
- **G6 — The CLI name `eval` is ambiguous.** In prime-rl, `uv run eval` is **prime-rl's** entrypoint (`pyproject.toml:46`), with a different config shape (`[[source]]`, `num_examples`, `group_size`). verifiers' standalone CLI at this pin is `vf-eval` (`deps/verifiers/pyproject.toml:152`). prime-envs docs still say `uv run eval <taskset-id>` with verifiers flags. prime-envs' test harness calls the Python entry point directly because the script was renamed (`deps/prime-envs/tests/test_envs.py:70-73`).
- **G7 — Ids are module names.** Hyphens become underscores and the text is lowercased; the pyproject distribution name is irrelevant. For example, `swesmith` ≠ `swesmith-env` (§3.1). A built-in with the same name (`harbor`, `textarena`, `nemo_gym`, `openenv`) shadows your package.
- **G8 — `__all__` pitfalls.** Re-exporting a *base* taskset class next to the concrete one (two `Taskset` subclasses) is a hard error. The fallback for Env and Harness silently swallows "0 matches", so a typo'd `Env` export silently falls back to `SingleAgentEnv` (`loaders.py:153-181`). Related: a harness block without `id` resolves to `bash`, not the bundled harness (§3.2).
- **G9 — Thin Harbor wrappers lose separate-verifier support** unless they export `HarborEnv`. Such tasks then raise `TaskError` at scoring (`tasksets/harbor/taskset.py:321-328`); `ignore_separate_verifier = true` trades isolation for shared grading.
- **G10 — Host-path task data.** `HarborData.task_dir` (and similar cache paths) is read by the env server from the path the orchestrator produced (§5.4). Launcher-managed servers are co-located. k8s or external servers need the same cache path mounted [cross-seam].
- **G11 — Registry drift.** `tmax` and `general_agent` fetch `registry.json` from `prime-envs@main`, not from the pinned submodule (`tmax/taskset.py:11`; `general_agent/corpus.py:28`). The first download is then cached forever under a selector hash. Task contents are commit-pinned inside the registry, but which tasks are included can change.
- **G12 — Dataset order and resume.** The default sampler walks dataset order (no shuffle). Resume replays the cursor against a *freshly reloaded* taskset, and an unpinned HF dataset that changed breaks the correspondence silently (§5.2).
- **G13 — Network defaults.** `TaskData.network_allow` defaults to `["*"]` (`task.py:104`). Several SWE/math envs re-enabled network because "training rollouts need outbound network" (math/swesmith/r2e/virl changelogs) and rely on a *prompt* to forbid solution lookup (`swesmith_env/taskset.py:49-63`). Lean and browsecomp_plus default-deny. Default-deny needs Docker or a Prime VM. Subprocess and Modal ignore task network policy (`deps/prime-envs/skills/evaluation/SKILL.md:123`).
- **G14 — Reward edge cases in the SWE envs:**
  - Grader infrastructure failures raise, so the result is an errored rollout, not a zero. Examples: swesmith test exit >1, and the Lean verifier crash.
  - An *empty* F2P+P2P list scores 0.0, because `is_resolved` returns False (`swesmith_env/taskset.py:148-155`).
  - An empty status map also scores 0.0.
  - Agents can tamper with anything visible in the live box, hence the grading-visibility audit (`swe/README.md`).
  - `HarborTask.run_verifier` ignores `test.sh`'s exit code. With no valid `reward.json` and an unreadable, missing, or non-float `reward.txt`, it scores 0.0, so a crashed test script is a **negative example**, not an error (`tasksets/harbor/taskset.py:333-361`).
- **G15 — Timeouts.** Harbor tasksets ignore task timeouts by default (`ignore_timeouts=True`), so the agent default is 4 h (`configs/agent.py:18-20`). Set `env.agent.timeout.rollout` for training throughput.
- **G16 — Concurrent `load_dataset` race.** The tqdm lock pre-seed exists for this reason (`envs.py:225-231`). Custom multi-env launchers must replicate it.
- **G17 — Stale docs.** `glm-4.5-air/search.toml:116,140,165` refer to `deps/prime-envs/rubrics/rlm.toml`, which does not exist at the pin (the file is `judges/rlm/rubrics.toml`). `skills/configs/SKILL.md:61` uses `env.taskset.split` for reverse-text, but the field is `dataset_split` (§3.9).
- **G18 — Prime VM images** must come from Docker Hub or the Prime registry. The OCI digest form is rejected for VMs (`deps/prime-envs/skills/evaluation/SKILL.md:42, 153-154`).

## 8. For a custom framework

**Essential design, worth keeping:**
- **Taskset / Task / TaskData split with a typed, serializable row.** Shipping `TaskData` JSON per request makes execution workers stateless, lets the sampler/curriculum live beside the trainer's data path, and makes traces self-describing (the row rides on the trace). Keep it.
- **Hooks with parameter-name injection and decorator-discovered signals** (`@reward/@metric/@stop`, plus config-plugged replacements). This is a small and very expressive authoring surface. Keep the rule that "infra failure raises, task failure scores 0" as a first-class contract. It is what keeps sandbox noise out of the gradient.
- **Harness is orthogonal to taskset.** One taskset can be *evaluated* with `null`, `bash`, `rlm`, or `codex`, but *trained* only through chat-completions harnesses (`null`, `bash`, `rlm` in shipped configs; also `pi`, `kimi_code`, `prime_agent`, `hermes_agent` at their defaults; `codex`/`claude_code` are eval-only; 08 §3.7). Our framework should make the dialect-to-trainability constraint a config-time check, not a runtime 502. This is the main axis of diversity prime-envs exploits (`AGENTS.md:97-101`).
- **Isolated verifier as an env, not a task feature** (`IsolatedVerifierEnv`, `HarborEnv`, `LeanEnv`). Reward hacking in live sandboxes is real, and the design keeps grading code off the agent's box.

**Incidental or costly, to simplify or replace:**
- **Plugin resolution by import side effects at config-parse time.** It gives nice typed CLIs (`--env.taskset.<field>` in `-h`), but it couples every process to every env's dependency closure (G2, G3). For our framework, resolve the **env config schema** in one place (for example, a JSON schema exported by the env package, or a lightweight "config-only" entry point) and let only the loader and executor import the heavy package. Better still, run each env's executor from its **own venv or container** with a narrow RPC, so env dependency conflicts stop being a monorepo lock problem.
- **Orchestrator-side full materialization.** Replace it with a streaming or indexed task source: an index of keys plus lazy row fetch, or sharded iteration with a persisted cursor that is keyed by `task.key` rather than by position. That fixes G4 (memory) and G12 (resume against drift). If you keep materialization, at least hash the task-key list into the checkpoint and refuse to resume on mismatch.
- **Positional cursor + `itertools.cycle` in dataset order.** Replace it with an explicit epoch-seeded permutation keyed on stable task keys.
- **Defaults that pick remote paid infra** (Prime runtime, Prime-inference judges, `bash` harness). For a research framework, default to local/no-op and require an explicit opt-in for sandboxes and judges (G1, G5).
- **Host paths in wire data** (`task_dir`). Make task assets content-addressed and fetchable by the executor (for example, an object store or a hash-named cache) instead of assuming co-location (G10).
- **Two CLIs named `eval`** (G6). Have one front door.

**Coupling points to design explicitly:**
1. **The wire schema of `TaskData`,** versioned and validated on both ends.
2. **The per-source config split:** client-side knobs (sampling, ratio, curriculum, algo) versus server-side knobs (env, serve). prime-rl does this by writing only `env` and `serve` to the server JSON (`pathing.py:230-247`).
3. **The credit-assignment ↔ episode-structure check** (`algo.validate_env`).
4. **Secret propagation boundaries:** launcher env → executor → harness `forward_env` → runtime.

## 9. Open questions

1. **Environments Hub install path.** The in-repo evidence is thin. The loader comment says Hub ids `owner/name[@version]` are "installed as just `name`" (`loaders.py:71-72`). Publishing is `prime env push <name> --visibility ...` (`deps/verifiers/skills/create-environments/SKILL.md:246-249`). The `prime-hub` index is `explicit = true` and no `[tool.uv.sources]` entry names it (`pyproject.toml:314-320`), so no `uv sync` ever pulls from it. The install command is presumably `prime env install owner/name` or `uv add <name> --index prime-hub`, which would add a source pin, but neither is documented in-repo. The `@version` is never checked at load. [UNVERIFIED]
2. **Does `rlm` (nano-rlm) always speak chat completions** to the interception endpoint, and do `pi`/`kimi_code`/`hermes_agent`/`openclaw` at their defaults? This is F/G's scope. Shipped training configs rely on `rlm` (§1). [UNVERIFIED]
3. **Elastic-pool startup errors.** With the default elastic pool, the broker publishes the address and answers health before a worker has built its `Env` (G, 08). A worker-side construction failure (`exclude=True` refusal, bad harness import) therefore may surface as hung runs rather than a clean startup failure. [UNVERIFIED runtime; G's scope]
4. **Lean image internals** (`packages/lean_common/image/{tools,checker}`, 1,624 LOC) were summarized from the README, not read line by line.
5. **The exact version of the `eval` → `vf-eval` script rename.** It is inferred from `tests/test_envs.py:70` ("renamed vf-eval scripts") and the verifiers pyproject. [UNVERIFIED version boundary]
