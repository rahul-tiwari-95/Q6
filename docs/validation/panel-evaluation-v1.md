# Frozen-policy panel evaluation validation — 2026-09-06

This record describes software/evidence checks, separate from the
[scientific interpretation](../experiments/panel_evaluation_results_v1.md).
The study evaluates existing policies; no training, collector, support bank,
architecture or historical gate changes.

## Declared execution

The [protocol](../experiments/panel_evaluation_protocol_v1.md) was independently
reviewed, committed and pushed at `5100595` before execution. Clean, pushed
main source is `d2dbb6f56036d2aec36f048522c92022bf2e93df`. It is a local
pre-execution declaration, not external registration or peer review.

The run uses Python 3.12.13, Torch 2.8.0, NumPy 2.0.2, deterministic CPU,
one Torch thread, `PYTHONHASHSEED=0`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`
and `/usr/bin/nice -n 10`. Observed process priority was 10 and CPU usage about
one core. The sampled 4 GiB peak-process-RSS guard measures this process, not
other applications or a hard operating-system memory limit.

The dedicated inference wrapper has no optimizer/replay allocation, disables
gradients, sets evaluation mode and preserves global RNG state. Online weights
choose actions; archived target weights are identity evidence only. World,
competence and supervised rollout source hashes match all four prior input
archives. All panels are admitted before predictions, with no difficulty
balancing or outcome-driven selection.

## Pre-execution checks

The seeded fast lane passed **338 tests, 3 deselected in 47.73 seconds**. The
three unchanged legacy training-heavy cases remain in the scheduled/manual
full lane. All **seven new focused tests passed in 0.90 seconds**, covering:

- Frozen inference without learner/Adam construction, gradients or RNG drift;
  snapshot hash failure detection.
- Training/prior-panel exclusions and cross-panel identity rejection.
- Expected smoke cells, greedy-only pair counts, before/after model identities
  and manifest bytes.
- Raw efficient-success and pooled versus paired blocked-rate denominators.
- Missing/duplicate episode rejection, partial-panel preservation and
  post-aggregation resource-cap disqualification.

The final alternate-panel CLI smoke completed **36 learner + 12 reference
episodes**, with **16 replays**, in **0.165 seconds**, peaking at **199.6 MB**.
It performed zero updates, collection steps or support draws and is explicitly
ineligible main evidence. Pre-main smoke development caught a CSV flush call
typo, which was fixed before source freeze; no main-run corrections occurred.

The independent smoke audit passed in **0.064 seconds**, and portable mode in
**0.058 seconds**: 32 manifest files, four admitted layouts, three immutable
models/six panel checks, raw summaries/pairs and all 16 recordings/156 steps.
Corrupted action, reward and Q-value probes were rejected. The smoke dashboard
harness covered every recording, both action modes, ineligible/missing states,
study switching and all 634 prior learner-study recordings. Historical
artifact verification also passed before main execution.

## Single main run and independent audit

The run completed **13,824 learner episodes + 3,584 references = 17,408 total**
on **512 unique layouts**, with **zero** new training updates, collection
steps or support draws. Duration was **30.542769 seconds**, peak RSS
**242,188,288 bytes (0.226 GiB)**, across **17,934 resource checks**. No resource
limit, source deviation, incomplete episode group or model mismatch occurred.

The pinned-runtime [independent audit](../../scripts/audit_panel_evaluation.py)
passed in **0.881 seconds**, without source/artifact corrections:

- All **38 manifest files**, captured source/protocol hashes, clean revision,
  environment and archived input identities.
- All **512 selected layouts**, 28 cross-panel disjointness relationships and
  training/four-prior-panel exclusions. Candidate **975004** is the sole
  prescribed rejection, matching the prior supervised fresh panel.
- **Nine** final snapshots, online/target hashes and **72 model-panel** checks,
  with source and copied snapshot bytes unchanged before/after evaluation.
- Complete unique keys and invariants for all **13,824 learner** and **3,584
  reference** rows, every per-seed/panel/pooled aggregate, reference distance,
  additive count, reported resource limit and completion count.
- **4,608 greedy paired layout rows**, **81 paired seed rows** (72 per panel
  plus nine all-panel), **27 paired aggregates** (24 per panel plus three
  all-panel), panel ranges/sign counts and descriptive threshold arithmetic.
- All **160 preselected recordings / 1,674 recorded steps**, independently
  reconstructed exploration draws, chosen actions, rewards, terminal flags,
  shortest paths and finite-horizon Q*. Saved learned Q outputs reproduce
  bit-exactly under the pinned runtime.

The audit replays only recorded actions to reconstruct saved observations; it
does not repeat the complete learned-policy evaluation or select new traces.
Raw arithmetic checks are not independent reproduction of every unrecorded
episode. Portable `--skip-forward-inference` skips regenerated neural outputs
while retaining saved-action/value consistency, dynamics/Q*, model hashes,
panel reconstruction and every raw-table/aggregate check.

The integrated portable artifact verifier passed the new main export and all
previous studies after execution. No historical artifact was modified.

## Dashboard and review scope

Experience coverage now defaults to **Panel robustness**, preserving both
earlier coverage studies and all other tracks. It displays 54 rollout
aggregates, panel means/ranges, pooled learner rows, greedy paired contrasts,
historical threshold counts and 160 recordings. It has no optimizer curves,
full-state fit metrics or new competence gate. Epsilon mode has its own episode
summaries; undeclared epsilon paired results are explicitly unavailable.

The actual-data harness passed **all 160 new and 634 prior recordings**, all
three conditions, eight panels, both modes, saved range/sign/threshold values,
incomplete/inconsistent/smoke/empty states, playback and study isolation.
HTTP serves the exact dashboard export: **3,632,967 bytes**, SHA-256
`2375f3110b37f55753f4d26ab3b64dab256f970560ba400e7f00af02c2356f8a`.
HTML, JavaScript and CSS served bytes also matched this checkout.

Native Computer Use could not start its pipe. **Pixel and responsive-layout
QA remain unverified**; DOM/canvas stubs and HTTP checks do not replace them.
The active movement rule remains visible at replay frame zero.

The final report is separately reviewed for counts, paired/pooled denominators,
the distinction between eight panel means and 24 model-panel cells, unchanged
historical gates, privileged offline training and the limits of support-bank
generalization. No extra main evaluation, bank draw or training followed.

## CI scope

The first final-artifact CI run exposed Python 3.11-only starred subscript
syntax in the new auditor on Python 3.10. Explicit tuple/coordinate indexing
replaces it, and CI now compiles all audit scripts even when their main
artifacts are absent. Full and portable saved-evidence audits pass after this
compatibility correction. The experiment runner, captured source, checkpoints
and result artifacts were not changed or rerun.
All audit scripts, the new runner and its tests also compile under an actual
local Python 3.10.20 interpreter after the fix.

CI runs Python 3.10 and 3.12 fast tests, portable audits for every stored study,
JavaScript syntax and the dashboard harness. Python 3.12 also builds and
installs a wheel away from source and imports `q6.panel_evaluation`. The full
legacy training suite remains scheduled/manual. Consult
[PR #1 checks](https://github.com/rahul-tiwari-95/Q6/pull/1/checks) for actual
outcomes at the submitted revision; configuration alone is not a pass.
