# Within-map composition validation — 2026-09-07

This records software and evidence checks separately from the
[scientific interpretation](../experiments/within_map_results_v1.md).
Three replacements preserve each archived bank's exact per-map quotas and
original replay schedule. Nine frozen controls are reused without training.

## Declaration and execution

The [protocol](../experiments/within_map_protocol_v1.md) was independently
reviewed and pushed at `20844a5` before execution. Clean, pushed main source is
`aa278609b06da1d0d28912268d695bc0d9450659`. This is a local declaration,
not external registration or peer review.

The pinned runtime is Python 3.12.13, Torch 2.8.0 and NumPy 2.0.2. Execution
uses one Torch thread, `PYTHONHASHSEED=0`, `OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1` and `/usr/bin/nice -n 10`. The process was observed at
priority 10 using approximately one CPU core. The sampled 4-GiB
peak-process-RSS guard measures this process, not total machine memory or a
hard OS allocation limit; the admission cap is 1,200 seconds.

All replacement supports are frozen before updates. Each sorted local slot
retains its map identity, so identical local draws preserve ordered map
sequences and batch diversity within each bank/seed. Historical 30,000-update
sampler streams are reconstructed and checked against archived local/global
digests and counts without retraining. Treatment initialization matches the
archived same-seed online/target weights. All treatment fits finish before
learned evaluation; only final policies are evaluated. Core world, update,
target and rollout sources retain their archived identities.

## Pre-execution checks

The seeded fast lane passed **356 tests, 3 deselected in 57.08 seconds**.
The **40 focused and related historical tests passed in 17.65 seconds**.
The six new tests cover exact quota draws and RNG ownership; empty quotas
and invalid map blocks; ordered local/map pairing including repeated map IDs
within a batch; preparation and fit/evaluation ordering; immutable controls;
no collection or online learning; truthful checkpoint labels; resource-cap
preservation; and post-run disqualification on a failed consistency check.
The three unchanged legacy training-heavy cases remain scheduled/manual.

Actual **Python 3.10.20** compilation passed for all auditors, the new runner
and tests. JavaScript syntax and whitespace checks passed. A report-link
variable-shadowing defect was found and fixed before final smoke/source
freeze; it did not affect learning or sampling.

Final smoke-v2 completed **768 per-map draws → three replacement supports**,
**72 new updates**, **72 learner + 12 reference episodes**, **nine snapshots**
and **28 recordings** in **3.714 seconds**, peaking at **481,804,288 bytes**.
Its 24-update treatments and archived 30,000-update controls have explicitly
unequal training budgets. The paired schedule check uses the matching
24-update historical prefix; full baseline counts remain labelled 30,000.
Smoke uses alternate panels and is ineligible scientific evidence.

Independent full and portable smoke audits passed in **4.30 / 4.18 seconds**:
67 manifest files, 47 archived input hashes, all 768 quota draws, 655,360
transitions/targets, 480 kernel crosschecks, 90,000 historical sampler
updates, three local/map prefix pairings, and all **28 recordings / 614
steps**. Full mode regenerated learned outputs. The dashboard harness passed
all **28 smoke + 1,402 prior recordings**, including separate draw/support
counts and failed-digest displays.

## Main evidence and dashboard

The single main run completed without deviations or artifact corrections:
**768 quota draws → three replacements**, **nine new fits / 270,000 updates /
17,280,000 presentations**, and **27,648 learner + 3,584 shared reference
episodes**. Historical baseline updates total 270,000; new baseline updates
and collection steps are zero. Execution took **146.846956 seconds**, peaking
at **483,835,904 bytes (0.451 GiB)** over **572,547 resource checks**.

The [independent auditor](../../scripts/audit_within_map.py) passed in
**10.564 seconds**, checking:

- **112 manifest files**, **59 archived input hashes**, captured source and
  protocol, clean revision, exact supports/control copies and initialization.
- All **768 independent quota draws**, **270,000 reconstructed historical
  sampler updates**, **27 treatment digests** (local/global/map), and
  **nine exact local/map pairings**, including count arrays.
- **655,360 transitions/targets**, **480 kernel crosschecks**, maximum
  Bellman residual 2.22e-16 and zero reward discrepancy.
- **54 snapshots**, **18 final models**, frozen identities around every
  panel, **2,700 raw loss windows**, **4,608 map exposure rows**, **90 clock
  rows**, and 18 state/category/query summaries.
- **512 selected layouts**, five prescribed exclusions, all **31,232 raw
  episode rows**, unique keys, per-bank/panel/learner and pooled arithmetic,
  **4,608 layout pairs**, **81 seed pairs**, **27 aggregate pairs**, and
  historical reference-level calculations.
- All **304 preselected recordings / 3,498 steps**, with dynamics, exploration
  draws, actions, observations, rewards/end flags, shortest paths, Q* and exact
  regenerated learned Q outputs.

This audit does not train, collect or repeat the full learned-policy
evaluation. Raw table arithmetic is not independent reproduction of every
unrecorded rollout. Portable mode omits neural forward regeneration but keeps
recorded-action consistency, dynamics/Q*, identities, sampling, exposure and
arithmetic. The integrated portable verifier passed all stored studies;
historical artifacts were unchanged.

The actual-data dashboard harness passed **304 new + 1,402 prior = 1,706
recordings**, all bank/panel/learner/action-mode controls, playback/scrubbing,
study switching, and missing/incomplete/smoke/inconsistent states. It checks
108 aggregates, 324 seed rows, 18 references, 900 aggregated loss windows,
every map/clock exposure row, quota counts and full/prefix replay flags.
The primary cards remain greedy when action mode changes. Active rules are
visible initially; identical map curves overlap as intended. On the first
preselected map 1060000, default bank 1 / seed 0 goes from 19 to two steps.

HTTP serves the exact **9,293,204-byte** export, SHA-256
`a7964fcb81cff4323342c6a62307f56f5959ffd3bb8bb678a02ea80f340c2749`.

Native Computer Use failed to start its pipe on this turn. **Pixel and
responsive-layout QA remain unverified**; DOM/canvas stubs and HTTP checks
do not replace them.

## CI scope

CI runs Python 3.10 and 3.12 fast tests, audit-script compilation, portable
verification of all stored studies, JavaScript syntax and the actual-data
dashboard harness. Python 3.12 also builds/installs a wheel away from source
and imports `q6.within_map`. The full legacy training lane remains
scheduled/manual. Consult [PR #1 checks](https://github.com/rahul-tiwari-95/Q6/pull/1/checks)
for outcomes at the submitted revision; configuration alone is not a pass.
