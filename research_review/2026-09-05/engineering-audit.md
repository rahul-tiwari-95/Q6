# Q6 engineering, reproducibility, and release audit

Scope: current `/Users/rahul/Q6-nwh` checkout, September 5, 2026. Read-only source audit; existing tests and isolated diagnostic scripts run with temporary outputs. No source edited, no dependencies installed. Other agents cover the RL algorithms, No Way Home science, and history.

## Overall judgement

There is a substantial, usable research prototype here. The strongest engineering work is observability and learning from actual failures: structured experiments, replay recording, a static run gallery, real-process orchestrator integration tests, and regression tests for incidents that lost training time. This is meaningfully more than a concept document. It is not yet a self-contained reproducible release: this checkout has historical summaries but no underlying runs, inconsistent documentation, no explicit license, no package/install metadata, and no automated CI workflow.

The first worthwhile release should be a curated research artifact with one runnable experiment and its evidence, not a generic framework promising all possible multi-world research. The orchestrator is also a plausible small independent utility, but should be a side artifact rather than the main research identity.

## Validation performed

- Interpreter: `/Library/Developer/CommandLineTools/usr/bin/python3`, Python 3.9; torch 2.8.0, numpy 2.0.2, pytest 8.4.2 already available.
- Full suite invoked: `python3 -m pytest tests/ -q -o cache_dir=/tmp/q6-pytest-cache`. Result: **253 passed in 272.69s (4m32s)**, then intentionally interrupted inside the slow final `tests/test_train_phase2.py` CPU training smoke. Only that unfinished test was rerun with `torch.set_num_threads(1)`: **1 passed in 186.88s (3m06s)**. Thus all **254 test cases passed across two executions**, with no failures; this was not one uninterrupted full-suite run. No previously passed tests were repeated. Logs: `/tmp/q6-full-suite-interrupted.log`, `/tmp/q6-final-smoke.log`; final-test JUnit artifact: `/tmp/q6-final-smoke.xml`. These tests validate implementation checks and the short training path, not the scientific interpretation of results.
- Source count finds 254 named test functions across 18 test modules; parametrization can alter collected case count. README simultaneously claims 221 tests (README.md:18,172) and 147 (README.md:220), showing documentation drift.
- Isolated subprocess smoke probe confirmed orchestrator supervises an immediately failing child and writes failed status; its own CLI nevertheless returns exit code 0. See details below.
- Isolated scanner probe confirmed a mixed-present/missing optional CSV metric generates non-finite NaN entries incompatible with strict JSON.
- Raw files and dashboard references audited programmatically; results below.
- Git commands in this worktree report `fatal: not a git repository: (null)`; `.git` points to absent `/Users/rahul/Q6/.git/worktrees/Q6-nwh`. Treat as this environment/worktree limitation, not a source defect. Parent verified GitHub is public, with six branches and no license.

## Genuine reusable assets

1. **Experiment recorder and static replay dashboard.** `utils/replay_recorder.py:6-27` documents a small streaming JSONL header/step/footer format. `:83-95`, `:111-145` record states, actions, optional Q-values, rewards, outcomes and seeds. The reader validates structure. This is understandable and extendable by another student without a service or database. `dashboard/scan.py:49-59` and `:85-114` accommodate algorithm-specific diagnostics; `:199-250` preserves historical summaries when raw runs are absent. `dashboard/run.html:205-230` only displays appropriate per-algorithm charts. The grid rendering is Q6-specific, so a reusable release needs a narrow documented schema, sample episodes, and explicit limits, not a claim of generic experiment tracking.

2. **Local checkpoint-resume supervisor.** `orchestrator/orchestrator.py` uses only stdlib imports (:67-79), preserves the original command and appends resume (:196-200), recovers status across restarts (:172-190), reloads config after a crash (:206-279), rate-limits concurrency (:389-402), writes status atomically (:408-415), and handles graceful shutdown (:431-468). Tests include real subprocess success/crash/restart checks (`tests/test_orchestrator.py:179-241`), restored state (:243 onward), signal-driven clean-stop status (:320-337), and changed cwd on restart (:352-431). These are substantive incident-driven tests. Most portable standalone asset, after modest hardening.

3. **Research histories and failure analyses.** The version/articles structure is useful for educational release: failed experiments are documented rather than silently removed. Root README links separate ablation and PPO branches (README.md:7,129-135). A concise index specifying code revision, claim status, raw evidence and rerun command would make this far more useful than adding another long plan.

4. **No Way Home experimental harness.** Other agent owns assessment. Engineering evidence includes an explicit paired-RNG bug record and its fix in `no_way_home/results/README.md:58-65`, and structural invariant testing (:128-136). These are promising reproducibility practices, stronger than the early RL training provenance. Do not confuse simulation-specific invariants with empirical validation of human institutions.

## Artifact availability and reproducibility

`dashboard/data/index.json` is generated August 1, 2026 (:2), referencing `/Users/rahul/Q6` (:3). It contains **40 historical run entries**, **71,272 episodes summed from summary counts**, and **2,910 replay references**. Entries span December 2025 through July 2026. **None of the 40 run paths exists under the current checkout**, and a recursive scan finds **zero CSV, JSONL, or PTH files** here. Seventeen run entries carry `training.git_commit` metadata. This demonstrates a substantial history, but not raw evidence available to a new cloner. Parent may recover raw artifacts from other worktrees or external storage; absence here does not mean the author lost them.

The exclusions are deliberate: `.gitignore:37-44` ignores logs, checkpoints, runs, training_runs, and plots. `dashboard/scan.py:203-220` explicitly explains why old index entries remain after data disappear. This is fine for Git size, but a release needs an artifact archive (small CSV/JSONL directly, large checkpoints separately) with hashes and persistent references. Current replay UI links to raw paths (`dashboard/replay.html:246,466`); the 2,910 referenced replays will not work in this checkout. Summary curves can still render because they are embedded in the index.

Important provenance gap: `train_phase2.py:257-275` logs a hardcoded `epsilon_decay=0.9999`, while the agents take `EPSILON_DECAY` from configuration (`agent/dqn_v2_agent.py:115-117`) and the documented/current value is 0.9994. The run metadata is therefore not an exact configuration capture. Seed is accepted and used in the runner (`train_phase2.py:73,81-82`) but not included in experiment metadata (:257-294); Phase 3 likewise omits the seed and software versions (`train_phase3.py:447-489`). Git revision is present, but dirty state, exact environment, seed/RNG state, and full invocation should be saved from actual values. Other agents assess resume and randomness correctness more deeply.

`requirements.txt:1-5` specifies only lower bounds (torch>=2.0, numpy>=1.24, gymnasium>=0.29, matplotlib>=3.7, pytest>=7.0). There is no Python requirement, lock/snapshot environment, pyproject/setup metadata, or CI workflow. `.github/` contains only copilot instructions. No LICENSE, COPYING, CONTRIBUTING or CITATION.cff was found. Before calling it open source, choose and add a license; before claiming reproducibility, provide a tested environment and a short CI smoke path. A polished package is optional for a research artifact; these small essentials are not months of framework work.

## Confirmed limitations worth fixing before a utility release

- **Supervisor reports shell success even when all jobs failed.** `/tmp` probe launched one `python -c 'raise SystemExit(3)'` child with zero restarts; status JSON was `failed`, outer process return code was 0. `orchestrator/orchestrator.py:551-567` unconditionally returns 0 after `orch.run()`. A caller/CI cannot trust CLI exit status. Return nonzero when any jobs fail, with explicit handling for user stops.
- **Resume detection can pick an old run from append-only logs.** Logs reopen in append mode (:281-282), but `_extract_run_dir` scans the whole file and takes the first match (:303-312). A temp log with `run_dir=/old/previous` followed by `run_dir=/new/current` yields `/old/previous`. This matters when a status file is reset while log history survives. Track launch offsets or latest valid run metadata.
- **Checkpoint-aware is a convention, not checkpoint validation.** `_build_command` appends `--resume` to any parsed path (:196-200); it does not verify a checkpoint exists, is complete, or fits the current command. README appropriately describes the stdout/CLI convention (:76-81,178-183). Keep the public claim precise.
- **Scanner can serialize NaN.** `dashboard/scan.py:103` fills absent optional metrics with float NaN; if at least one metric exists, the full series remains (:113-114). Default json dumping produces `NaN`, which browser JSON.parse rejects. Temp CSV with loss=0.5 then blank reproduces strict serializer rejection. Use null and handle missing values deliberately.
- **Graphs are presentation summaries, not evidence for rare events.** `_downsample` takes one indexed point per interval (:30-34), not aggregate statistics; the viewer ignores the stored episode indices and plots by array index (`dashboard/app.js:61,96-100`; `dashboard/run.html:223-230`). Do not estimate win-rate or tail-risk statistics from these sampled curves.

## Documentation vs actual code

`longer_memory/` is **not implemented long-term neural memory**. Its README explicitly calls it future feature reminders (:3), lists architecture/continual learning ideas (:33-51), and still calls the dashboard planned (:91-101). The spec describes Flask/PostgreSQL (`longer_memory/DASHBOARD_REQUIREMENTS.md:10-15,69-79`); actual dashboard is static HTML/JS and a file scanner (`dashboard/scan.py:8-14`). Archive this as an old plan and point to the current implementation. Do not make it a new mandatory workstream.

Root README remains about Krishna/Hunter, not the current No Way Home branch; branch differences are acknowledged for v7/v8 (README.md:7,129-135) but prospective contributors need a short front-page map distinguishing independent research threads. MPS-first training examples (:21-27) also need an obvious CPU smoke command.

`orchestrator/q6_jobs.json:9,22` hardcodes the author's worktree paths and launches branch-specific scripts; make runnable templates relative or parameterized. Sample configurations should not imply a fresh clone has these paths or checkpoints.

## Recommended release order

1. Pick one primary result and audience with parent agent's scientific assessment. Freeze a small artifact: exact revision + tested environment + one command + raw results + 2-3 illustrative replays + limitation statement. Publish as a reproducible research/teaching lab.
2. Add license, minimal contribution instructions, artifact manifest, and a CI smoke test. Reconcile the README and mark old plans archived. Do not require full package generalization to release one good experiment.
3. Offer the supervisor/recorder as separate examples or tiny utilities only if people ask to reuse them. Fix the confirmed robustness issues above and add focused tests before pitching them as tools.
4. For multi-world efficiency, benchmark the existing simulator first: worlds/sec, agent-steps/sec, peak memory, deterministic seed equivalence under batching. Reuse the logging/replay discipline. Avoid simultaneously committing to a simulator framework, nested-network architecture, and institutional theory.
