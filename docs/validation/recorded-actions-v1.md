# Recorded-action supervision validation — 2026-09-07

Software and evidence checks are separate from the
[scientific interpretation](../experiments/recorded_actions_results_v1.md).
This study retains original collected states and replay, changing direct
supervision from four action outcomes to distinct actions actually logged.

## Declaration and execution

The [protocol](../experiments/recorded_actions_protocol_v1.md) was independently
reviewed and pushed at `60d6299` before the main run. Clean, pushed execution
source is `5b43def285c92ceb8dc314677cbd47b322273b1e`. This is a local
declaration, not external registration or peer review.

Runtime is Python 3.12.13, Torch 2.8.0, NumPy 2.0.2, deterministic CPU and one
Torch thread. `PYTHONHASHSEED=0`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` and
`/usr/bin/nice -n 10` are set. The process was observed at priority 10 using
approximately one CPU core. The admission cap is 1,200 seconds; the sampled
4-GiB peak-process-RSS guard is not a hard OS allocation or total-machine limit.

All manifest-verified logs and deduplicated tables are prepared before fits;
all supports, controls and prospective panels precede training. Original
local/global replay streams are reconstructed without retraining and matched
to each treatment, including ordered map identities. Optimization receives
only observations, recorded tensors and sampled state indices; it does not
receive exact labels or the exhaustive transition table. Actual target and
successor-query counts are tracked and reconciled with final state counts.

All nine treatment fits precede learned evaluation. Final-policy evaluation
has no optimizer/replay or gradients, and verifies frozen model identities
around every panel. Historical baseline training is labelled separately from
new updates; no collection occurs.

## Pre-execution checks

The seeded fast lane passed **369 tests, 3 deselected in 68.12 seconds**.
All **53 focused and related historical checks passed in 25.89 seconds**.
The 13 new cases cover deterministic deduplication without frequency weighting;
inconsistent reward/end/map/clock/terminal records and missing supported states;
per-state rather than per-action loss normalization; Double DQN online
selection/target evaluation; terminal-only batches with no successor queries;
poisoned absent outcomes; all-four-observed-action equivalence to the existing
objective; preparation and evaluation ordering; exact global/local/map replay;
truthful historical exposure; immutable tensors; resource-cap preservation;
and disqualification after a forced global-stream mismatch.

Actual Python **3.10.20** compilation passed for all auditors, the new runner
and tests. JavaScript syntax and whitespace checks passed. The three unchanged
legacy training-heavy cases remain in the scheduled/manual lane.

Final smoke-v2 completed **72 updates / 4,608 state presentations / 6,493
recorded-action targets** in **5.953 seconds**, peaking at **556,761,088 bytes**.
Targets split into **267 terminal and 6,226 nonterminal**; every recorded
successor query stays within current-state support. It preserves nine
snapshots, 72 learner + 12 reference episodes, and 28 recordings. Its
24-update treatment and archived 30,000-update control budgets are unequal;
the matching historical prefix is checked while full baseline labels remain.
Alternate smoke panels are ineligible scientific evidence.

Independent full/portable smoke audits passed in **6.97 / 7.03 seconds**:
77 manifest files, 52 archive hashes, all **301,585 historical collection
steps / 12,288 episodes**, three recorded-edge tables (**84,305 / 84,091 /
83,933 edges**), 90,000 historical sampler draws, exact local/global/map
prefix matches, snapshots/raw metrics, and all **28 recordings / 712 steps**.
Full mode regenerated learned outputs. The final smoke dashboard harness
passed **28 new + 1,706 prior = 1,734 recordings**, including action counts,
actual versus structural queries, provenance and failed replay checks.

## Main evidence and dashboard

One main run completed without deviations or artifact corrections:
**nine new fits / 270,000 updates / 17,280,000 state presentations**,
**27,648 learner + 3,584 shared reference episodes**, **54 snapshots** and
**304 recordings**. New collection, baseline training and support draws are
zero. Actual treatment targets total **24,273,701**, comprising **1,062,190
terminal and 23,211,511 nonterminal** targets, with zero recorded successor
queries outside support. The historical control has 69,120,000 targets.

Duration is **166.542847 seconds**, peak process RSS **555,122,688 bytes
(0.517 GiB)**, over **572,960 resource checks**. All completion, initialization,
source/support/model/table, sampler, target/query and resource checks passed.

The [independent auditor](../../scripts/audit_recorded_actions.py) passed in
**14.64 seconds**, checking:

- **122 manifest files**, **64 archived input hashes**, captured source,
  protocol, clean revision and runtime; no original artifact changed.
- Every **301,585 historical collector step / 12,288 episode**, reconstructed
  logged sequence and **three deduplicated recorded-action tables**; separate
  termination/truncation identities and exact source/copy hashes.
- **655,360 diagnostic transitions/targets**, **480 kernel crosschecks**,
  all **270,000 historical sampler draws**, and nine exact treatment
  local/global/map replay pairs with their count vectors.
- **54 snapshots**, nine unchanged controls, 18 final policies and
  model-panel identities; initialization, **2,700 loss windows**, immutable
  NumPy tables/optimizer tensors, target counters and recorded query vectors.
- **4,608 map exposure rows**, **90 clock rows**, 18 exposure summaries,
  all action-coverage/target/query reductions, separating actual treatment
  queries from counterfactual structural diagnostics.
- **512 admitted layouts**, three prescribed exclusions, all **31,232 episode
  rows**, unique keys, paired/pooled arithmetic and historical reference
  crossings; **4,608 layout pairs**, **81 seed pairs**, **27 aggregate pairs**.
- All **304 recordings / 4,824 steps**, including observations, exploration
  draws, actions, dynamics, rewards/end flags, shortest paths, Q* and exact
  regenerated learned outputs.

The audit does not retrain, collect or repeat the complete learned-policy
evaluation. Arithmetic validation does not reproduce each unrecorded rollout.
Portable mode omits neural forward regeneration while retaining identities,
recorded-action consistency, dynamics/Q*, sampling, exposure and arithmetic.
The integrated portable verifier passed this export and all earlier studies.

The actual-data dashboard harness passed **304 new + 1,706 prior = 2,010
recordings**. It checks all bank/panel/learner/action-mode controls, playback,
scrubbing, study isolation and missing/incomplete/smoke/inconsistent branches;
108 aggregates, 324 seed rows, 18 references and 900 aggregate loss windows;
all state/action coverage and query displays, exact replay checks and
historical-versus-new budgets. Primary cards remain greedy when lower action
mode changes. Active rules remain visible initially.

The first predetermined map is 1080000: default bank 1 / seed 0 takes three
steps in both arms; bank 2 / seed 0 improves from 26 to seven while seed 1
worsens from three to 17. This mixed trace is retained without selection by
outcome. HTTP serves exact **10,089,421-byte** data, SHA-256
`2f99b6ec9f1ce32ba75aa5492d04adce17a9a5e7b2bc0f1c60079619882d103e`.

Native Computer Use could not start its pipe on this turn. **Pixel and
responsive-layout QA remain unverified**; DOM/canvas stubs and HTTP checks
do not replace them.

## CI scope

The fast matrix covers Python 3.10/3.12 tests, auditor compilation, portable
verification of all stored studies and actual-data dashboard checks. Python
3.12 builds and installs a wheel away from source and imports
`q6.recorded_actions`. Full legacy training remains scheduled/manual.
See [PR #1 checks](https://github.com/rahul-tiwari-95/Q6/pull/1/checks) for the
submitted revision's outcomes; configuration alone is not a pass.
