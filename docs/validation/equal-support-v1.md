# Equal-size bank milestone validation — 2026-09-06

This record separates implementation and evidence checks from the scientific
interpretation in the [result report](../experiments/equal_support_results_v1.md).
The study retains privileged four-action transitions and detached full-bank
successor access. It performs no new collection and does not establish an
online-RL baseline.

## Protocol and implementation scope

The [protocol](../experiments/equal_support_protocol_v1.md) was independently
reviewed, committed and pushed at `03f0ff8` before main execution. It was locally
declared, not externally registered. A shared runner extension supplies bank
selection, provenance and diagnostics; DDQN update mathematics, the network
and the historical coverage entry point remain unchanged.

The main run captured clean, pushed implementation revision
`e52043e2338045718f3939b463b60f996f740df3`. Execution uses Python 3.12.13,
Torch 2.8.0, NumPy 2.0.2, deterministic CPU operation and one Torch thread.
The launch sets `PYTHONHASHSEED=0`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`
and `/usr/bin/nice -n 10`. All fits run sequentially. The 4 GiB sampled
peak-process-RSS guard is not an operating-system allocation limit or a
measurement of other applications' RAM.

The main control requires exact reproduction of the preceding collected arm's
initial/final online weights, final target weights and global sampling digest
and count vector. Equal cardinality additionally requires identical local
sampling digests/count vectors across the new arms. Global states intentionally
differ. These checks establish execution consistency, not independent bank
replication.

## Dashboard scope

The Experience coverage tab now has a study selector. Equal-size banks is the
new default; the previous exhaustive-versus-collected comparison remains
available with its own fresh panel and interpretation. Both conditions' bank
composition is shown together, while a selector reuses the detailed map/time
views. Own-support diagnostics remain separate from full-bank fit gates.

Native browser review was unavailable: Computer Use failed to start its native
pipe. No pixel or responsive-layout verification is claimed. DOM/canvas-stub
checks and HTTP serving checks are separate from visual layout validation.

## Pre-execution checks

The documented seeded fast lane passed **331 tests, 3 deselected in 42.69s**.
The unchanged three legacy training-heavy cases were not repeated. All **six
new equal-size tests plus nine legacy coverage tests passed in 2.21s**. They
check the owned uniform RNG and exact draw, archive support/mask integrity,
three prior-panel exclusions, each arm's own diagnostic mask, equal local
sampling with distinct global rows, absence of collection and post-diagnostic
resource-cap disqualification.

The final separate CLI smoke used two alternate maps per panel and synthetic
every-third-row collected support. It completed **48 updates / 3,072 state
presentations**, with **427 states per arm, zero collection steps**, in
**0.822s**, using **284.49 MB peak process RSS**. It is ineligible for all gates.

The independent audit passed the final smoke in **0.436s** with regenerated
forward predictions and **0.318s** in portable mode. It verified 45 manifest
files, 10,240 exact values, 5,120 transitions, 960 isolated kernel steps,
support selection, local/global sampling, model/prediction hashes and raw
metrics/gates/paired/own-support diagnostics. Historical artifact verification
also passed after generalizing the shared raw-table audit helper.

The dashboard smoke harness checks all 20 new recordings, both supports,
equal-size/overlap counts, own/outside-support diagnostics, seven interpretation
branches, action modes, paired outcomes, study switching and playback stop.
All 510 earlier learned-study recordings and historical controls remain
available. These checks validate code paths and displayed numbers, not pixels.

## Main evidence audit

The complete pinned-runtime audit passed in **7.57s**, without learning-source
or artifact corrections:

- All **75 manifest files**, source/protocol hashes, clean source revision,
  runtime and prior-input identities. Training arrays/transitions match the
  archive; no train/fresh or prior-fresh layout overlaps were found.
- All **819,200 exact target entries**, **655,360 transitions** and **960
  isolated cloned-world steps**. Maximum Bellman residual was 2.22e−16 and
  maximum saved-reward discrepancy zero.
- The historical **4,096-episode / 99,814-step** random collector reconstructed
  from its original action RNG and validated transitions. This verifies an
  archived trace; the new experiment performs **zero collection steps**.
- Both **59,626-row** supports, the exact declared uniform draw, the 21,825-row
  intersection, per-map/clock/goal-near composition and successor counts.
  All **12 local/global sampling streams** and exposure vectors verify;
  paired local schedules match and neither arm directly samples outside its
  own bank. Collected online/target weights and global exposures match history.
- All **30 online/target snapshots**, **12 bit-exact final prediction arrays**,
  **28,800 learner episode rows**, **48,000 state rows**, **1,800 loss windows**,
  **2,240 unique reference episodes**, **960 paired layout rows** and every
  success/fit/efficiency gate. Final own/outside diagnostics reconcile in all
  **12 per-seed and four pooled rows**, using each arm's own mask.

The main run completed **180,000 updates / 11.52 million state presentations**
in **153.73s**, using **666.6 MB peak process RSS (0.62 GiB)** across 200,552
resource checks. No resource limit was reached and no deviations recorded.
All 30 scheduled checkpoint groups finished. The historical collector cost
is a reused input, not a new cost hidden among optimizer presentations.

The [independent audit](../../scripts/audit_equal_support_study.py) imports
captured source and runs no optimizer updates or new learned-policy rollouts.
It reconstructs the old random trace and final dense forward predictions;
intermediate policy outcomes and losses are reconciled to raw saved tables.
General artifact verification passed again on the main export, including this
audit's portable `--skip-forward-inference` mode and all previous studies.
Portable mode omits only regenerated forward predictions, preserving the other
saved-data, graph, sampling, model-hash and metric checks. Neither mode is
external peer review or independent training/bank replication.

## Actual dashboard and report review

The actual export passed all **124 new recordings**, 40 rollout aggregates,
100 full-state aggregates and 600 pooled loss windows; both 256-layout heatmaps,
time views, bank overlap and all 16 own/outside diagnostic rows. All 510 prior
recordings and historical controls remain reachable. Study switching preserves
each panel and stops active playback. The new default correctly shows both
fresh gates passing, with fit/efficiency unmet and no equivalence claim.

A final presentation-only addition explains the same-weight panel sensitivity.
It derives 68.2% and 83.9% from the two studies' own final greedy aggregates,
and shows numbers only with complete/eligible studies, matching archive,
checkpoint/seed identities and successful recorded replication checks. Missing
prior data, archive mismatch, failed replication and smoke show generic context.
Main and smoke harness checks passed; gates and curves remain study-specific.

The report was separately reviewed and approved for numerical accuracy,
primary versus secondary outcomes, paired/pooled blocked-rate denominators,
arm-specific versus common fit distributions, panel variation and limits of
the proposed eight-panel frozen-policy evaluation. No further training,
collection, support draws, checkpoint selection or post-hoc gates followed.

## CI scope

CI runs the 331-test fast lane on Python 3.10 and 3.12, portable artifact audits,
JavaScript syntax and dashboard controls. Python 3.12 also builds/installs a
wheel away from source and imports `q6.equal_support`. The unchanged full
legacy suite remains scheduled/manual. Consult [PR #1 checks](https://github.com/rahul-tiwari-95/Q6/pull/1/checks)
for actual outcomes at the submitted revision; configuration alone is not a pass.
