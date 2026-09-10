# Independent bank-replication validation — 2026-09-06

This records software and evidence checks separately from the
[scientific interpretation](../experiments/bank_replication_results_v1.md).
The milestone changes collector/support draws while retaining the network,
DDQN, four-action supervision and training budget per fit. Memory is deferred.

## Declared execution

The [protocol](../experiments/bank_replication_protocol_v1.md) was independently
reviewed and pushed at `7ae6f47` before execution. Clean, pushed main source is
`156343273b084777a08cc9206b270940db002462`, including the final pre-run
clarification distinguishing pooled blocked ratios from means of bank ratios.
This is a local declaration, not external registration or peer review.

The single main run uses Python 3.12.13, Torch 2.8.0, NumPy 2.0.2, deterministic
CPU execution, one Torch thread, `PYTHONHASHSEED=0`, `OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1` and `/usr/bin/nice -n 10`. Observed priority was 10, with
approximately one CPU core occupied. The 4-GiB sampled peak-process-RSS guard
measures this process, not total system memory or a hard OS allocation limit.
The declared wall-time admission cap is 1,200 seconds.

All six support arrays are frozen before any fit. Every fit completes before
policy evaluation. Initial online and target weights match the archived
same-seed controls; within-bank paired local sampling schedules match. Final
inference has no optimizer/replay and preserves online/target and file hashes
before, after and around each model-panel evaluation. No intermediate snapshot
is evaluated or selected. Shared collector/panel helper defaults preserve the
historical behavior; old artifacts and captured sources are unchanged.

## Before the main run

The seeded fast lane passed **344 tests, 3 deselected in 43.87 seconds**.
Six new tests plus 22 related historical coverage/equal-support/panel tests
passed together: **28 passed in 4.06 seconds**. New tests cover:

- Reproducible independent collector suffixes and unchanged default streams;
  owned uniform draws and exact matched support size.
- All banks frozen before updates, all fits complete before rollouts, no
  online `observe`/`learn`, paired sampling, support/model hashes and counts.
- Partial collection preserving completed supports, nonfinite-loss rejection,
  and post-aggregation resource-cap disqualification.

All audit scripts, the new runner and tests compile under an actual local
**Python 3.10.20** interpreter. JavaScript syntax and whitespace checks pass.
The three unchanged legacy training-heavy tests remain in the scheduled/manual
full lane, rather than being claimed as part of the fast result.

Final CLI smoke used three pairs on two alternate training maps, one learner
seed and 24 updates per fit. It completed 96 collector episodes / 2,132 steps,
144 updates, 72 learner + 12 reference episodes and 28 saved recordings in
**0.753 seconds**, with **267.78 MB** peak RSS. Support sizes were 477/396/495.
Smoke is explicitly ineligible research evidence.

Independent full and portable smoke audits passed in **0.180 / 0.159 seconds**:
69 manifest files, all collector draws, 5,120 transitions/targets, 480 kernel
crosschecks, six supports, 12 between-bank intersections, 12 snapshots,
sampling/counts and all 28 recordings / 839 steps. The smoke dashboard harness
also passed all new recordings and all 794 prior learner-study recordings.
Source/protocol review completed before the clean main source freeze.

## Single main run and evidence audit

The main comparison completed **12,288 collector episodes / 301,585 steps**,
**18 fits / 540,000 updates**, **34,560,000 state presentations**, and
**27,648 learner + 3,584 shared reference episodes = 31,232 evaluation episodes**.
Support sizes are 59,839 / 59,984 / 59,838, matched within each pair.

Duration was **243.592524 seconds**, with **461,717,504 bytes peak RSS
(0.430 GiB)** across **885,657 resource checks**. All completion, source,
initialization, sampling, support and evaluation-identity checks passed; no
resource stop or protocol deviation occurred. There was one main run.

The pinned-runtime [independent audit](../../scripts/audit_bank_replication.py)
passed in **11.035 seconds**, without research-artifact or source corrections:

- All **147 manifest files**, captured source/protocol/environment, archive
  input identities and clean source revision.
- All **655,360 exact target entries and saved transitions**, with 480
  independent kernel crosschecks; maximum Bellman residual 2.22e-16 and zero
  saved reward discrepancy.
- Every original collector action/RNG draw and terminal decision across
  12,288 episodes; all six support draws, composition tables and 12
  between-bank intersections.
- **36 sampling streams**, local/global row counts/digests, **90 snapshots**,
  initial online/target matches, **18 frozen final models** and their
  **144 model-panel** identity checks; **5,400 raw loss rows**.
- **512 admitted layouts**, two prescribed candidate rejections, training and
  prior-panel exclusions, all raw learner/reference episode keys and summary
  arithmetic, pooled and paired denominator distinctions and thresholds.
- **4,608 greedy paired layout rows**, **81 paired seed rows**, **27 paired
  aggregate rows**, all bank effects, ranges and sign counts.
- All **304 preselected recordings / 4,561 steps**, including reconstructed
  observations, exploration draws, actions, rewards, terminal flags, shortest
  paths, finite-horizon Q* and exact regenerated learned Q outputs.

The audit reconstructs saved collection and recording histories. It does not
collect new training experience, retrain policies or repeat the full learned
policy evaluation. Raw arithmetic validation is not independent reproduction
of every unrecorded episode.

The integrated portable verifier passed this export and all previous studies.
Portable mode skips regenerated neural outputs while retaining saved-action
consistency, dynamics/Q*, hashes, sampling, collection reconstruction and all
raw-table arithmetic. No historical artifact was modified.

## Dashboard scope

Experience coverage defaults to **Bank replications**, with every historical
study preserved. The actual-data harness passed **304 new + 794 prior = 1,098
recordings**, all bank/panel/condition/learner selectors, both modes, playback,
scrubbing, reset/refresh and missing/incomplete/inconsistent/smoke states.

It checks **108 aggregate rows**, **324 seed rows**, **18 shared reference
rows**, **1,800 aggregated loss windows**, map/clock membership counts and
separate per-bank historical threshold counts. Primary cards remain the
predeclared greedy efficiency comparison when exploration mode changes.
There is no intermediate-checkpoint evaluation selector or new competence gate.

HTTP serves the exact **8,143,410-byte** dashboard export, SHA-256
`6da7eedb5a2fd8500fa0fd7f1ec08e4ccbbb7f9d1fdc82dcdbc4917cd833aa53`.
The first map remains the preselected 1020000, including mixed outcomes across
seeds. The active rule remains visible initially.

Native Computer Use could not start its pipe. **Pixel and responsive-layout
QA remain unverified**; DOM/canvas stubs and HTTP checks do not replace them.

## CI and review scope

CI covers Python 3.10 and 3.12 fast tests, audit-script compilation, portable
checks of every archived study, JavaScript syntax and actual dashboard data.
Python 3.12 also builds/installs a wheel away from source and imports
`q6.bank_replication`. The full legacy training suite remains scheduled/manual.
Consult [PR #1 checks](https://github.com/rahul-tiwari-95/Q6/pull/1/checks) for
outcomes at the submitted revision; configuration alone is not a pass.

The result and roadmap are separately reviewed for per-bank reporting,
shared-seed/panel dependence, ratio denominators, reversals, historical gate
preservation and limits of privileged offline evidence. The recommended
map-balanced replay intervention has not run.
