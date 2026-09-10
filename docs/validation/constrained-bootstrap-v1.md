# Constrained-bootstrap validation — 2026-09-07

Software and evidence checks are separate from the
[scientific interpretation](../experiments/constrained_bootstrap_results_v1.md).
Only training-time successor action selection changes; recorded data, exact
replay, objective and action-target counts remain matched.

## Declaration and execution

The [protocol](../experiments/constrained_bootstrap_protocol_v1.md) was independently
reviewed and pushed at `03fe2f16b39e1b8b8ce7b5259285a18d4cb045ec` before execution.
Clean, pushed main source is `7b70b43aa9df08ddd3e71fe5ef3d54cbc688c7ec`.
This is a local declaration, not external registration or peer review.

Runtime is Python **3.12.13**, Torch **2.8.0**, NumPy **2.0.2**, deterministic
CPU with one Torch thread. `PYTHONHASHSEED=0`, `OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1` and `/usr/bin/nice -n 10` are set. The process was observed
at priority 10 using approximately one CPU core. The admission cap is 1,200
seconds; the sampled 4-GiB peak-process-RSS guard is not a hard OS allocation
or total-machine limit.

Recorded tables reconstruct from original collector logs and match the prior
recorded-actions archive. The frozen controls, their initial weights and
historical replay counts come from that archive. No control is retrained.
All supports, controls, original replay reconstructions, fixed probe rows and
prospective panels precede fitting. All nine fits precede policy evaluation.
Fresh policy selection retains all four actions, with frozen identities checked
around each panel. Checkpoint probes are separate training-state inference;
they do not select models or advance replay, optimization or RNG.

## Pre-execution checks

The seeded fast lane passed **379 tests, 3 deselected in 84.17 seconds**.
All **63 focused and related historical checks passed in 35.88 seconds**.
The ten new tests cover restricted online selection and lagged target
evaluation, including positive deltas and equal-valued ties; empty successor
masks without fallback; terminal-only batches without successor queries;
all-actions-allowed equivalence with unequal current-action counts; per-state
weighting; query-weighted statistics; pure targets/probes; unrestricted fresh
actions; correct control bytes, exposure and exact replay; phase ordering;
partial-artifact preservation at the memory cap; and disqualification after
a forced replay mismatch.

Actual Python **3.10.20** compilation passed for all auditors, the new runner
and tests. JavaScript syntax and whitespace checks passed. Three unchanged
legacy training-heavy cases remain in the scheduled/manual lane. An initial
read-only NumPy-to-Torch probe warning was fixed before final smoke without
changing numerical inputs; the full suite then passed without that warning.

Final smoke-v2 completed **72 updates / 4,608 state presentations / 6,493
recorded-action targets** in **5.962 seconds**, peaking at **559,529,984 bytes**.
Targets split into **267 terminal and 6,226 nonterminal**. Six fixed probes
make **508 separate inference queries**, excluded from optimizer counts.
Smoke preserves nine snapshots, **72 learner + 12 reference episodes**, and
**28 recordings / 843 steps**. Its 24-update treatment and archived
30,000-update control budgets are unequal; historical prefix checks and full
control labels remain truthful. Smoke is ineligible scientific evidence.

Independent full/portable smoke audits passed in **7.42 / 7.87 seconds**,
checking **92 manifest files**, **65 archived input hashes**, all original
collector logs/tables, 90,000 historical sampler draws, three diagnostic
windows, six fixed probes and all recordings. Full mode regenerates both
networks for every probe and the learned predictions in saved replays.
The final smoke dashboard harness passed **2,038 recordings**. A display
defect hiding small signed changes at one decimal place was corrected before
source freeze; bootstrap value displays now retain three decimals.

## Main evidence

One main run completed without deviations or artifact corrections:
**nine new fits / 270,000 updates / 17,280,000 state presentations**,
**27,648 learner + 3,584 shared reference episodes**, **54 snapshots** and
**304 recordings**. Both arms have **24,273,701 action targets**, including
**1,062,190 terminal and 23,211,511 nonterminal targets**. Their successor-query
vectors are identical, with zero queries outside current-state support.
There is no collection, support drawing or new baseline training.

Duration is **174.530803 seconds**, peak process RSS **579,174,400 bytes
(0.539 GiB)**, over **572,974 resource checks**. All completion, initialization,
source/support/model/table, replay, target/query and resource checks passed.

The [independent auditor](../../scripts/audit_constrained_bootstrap.py) passed
in **15.11 seconds**, checking:

- **137 manifest files**, **77 archived input hashes**, captured source,
  protocol, clean revision and runtime; preserved original artifacts.
- All **301,585 historical collector steps / 12,288 episodes**, reconstructed
  RNG and logged sequences, and three deduplicated recorded-action tables.
  Prior recorded-control table/support copies match exactly.
- **655,360 diagnostic transitions/targets**, **480 kernel crosschecks**,
  all **270,000 historical sampler draws**, nine exact treatment replay pairs
  across local/global/map streams and their count vectors.
- **54 snapshots**, nine unchanged controls, 18 final policies and their
  panel identities; initialization, **2,700 loss windows**, immutable tables,
  actual action-target counts and both-arm successor-query vectors.
- **2,700 bootstrap diagnostic windows**, independent query denominators,
  sign partitions, bounds and saved sum/squared-sum arithmetic. This does
  not reproduce evolving window values by retraining.
- All **45 fixed checkpoint probes / 3,770 separate queries**, including
  frozen rows, snapshot identities, both networks' regenerated four-action
  outputs, masks, choices and complete float32 targets.
- **4,608 map exposure rows**, **90 clock rows**, state/action coverage and
  target/query reductions, with historical and new exposure labelled separately.
- **512 admitted layouts**, nine prescribed exclusions, all **31,232 episode
  rows**, unique keys, paired/pooled arithmetic and historical reference
  crossings; **4,608 layout pairs**, **81 seed pairs**, **27 aggregate pairs**.
- All **304 recordings / 5,575 steps**, including observations, exploration
  draws, actions, dynamics, rewards/end flags, shortest paths, Q* and exact
  regenerated learned outputs.

Auditing does not retrain, collect or repeat the complete learned-policy
evaluation. Arithmetic validation does not regenerate every unrecorded
rollout. Portable mode omits neural forward regeneration while retaining
saved-vector probe formulas, identities, dynamics/Q*, recorded-action tables,
replay, exposure and arithmetic. The integrated portable verifier passed this
export and every earlier shipped study.

## Dashboard and CI scope

The actual-data dashboard harness passed **304 new + 2,010 prior = 2,314
recordings**. It exercises bank/panel/learner/action-mode controls, playback,
scrubbing, study isolation and missing/incomplete/smoke/inconsistent branches.
It checks actual equal target/query budgets, recorded-control provenance,
activation/gap/signed-target displays, positive/negative chart ranges and
separation of the 45 probes from optimizer totals. Primary outcome cards remain
greedy when lower replay controls change; active rules remain visible initially.

The first predetermined map is **1100000**: default bank 1 / seed 0 changes
from success in 12 steps to failure at 32, while seed 1 changes from failure
at 32 to success in 14. This unfavorable default trace is preserved alongside
the aggregate improvement. HTTP serves exact **16,991,205-byte** data,
SHA-256 `a4c1afed5e580b5e362395b797a668717875af6f1b9d50cb799165e55ab93c9f`.

Native Computer Use could not start its pipe on this turn. **Pixel and
responsive-layout QA remain unverified**; DOM/canvas stubs and HTTP checks
do not replace them.

The fast CI matrix covers Python 3.10/3.12 tests, auditor compilation,
portable verification of all stored studies and actual-data dashboard checks.
Python 3.12 builds and installs a wheel away from source and imports
`q6.constrained_bootstrap`. Full legacy training remains scheduled/manual.
See [PR #1 checks](https://github.com/rahul-tiwari-95/Q6/pull/1/checks) for the
submitted revision's outcomes; configuration alone is not a pass.
