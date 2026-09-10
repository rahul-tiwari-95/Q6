# Exact logged-graph validation — 2026-09-07

Software and evidence checks are separate from the
[scientific interpretation](../experiments/logged_graph_results_v1.md).
Exact fixed labels come only from the recorded transition graph; replay,
objective and action-target presentations remain matched to constrained DDQN.

## Declaration and execution

The [protocol](../experiments/logged_graph_protocol_v1.md) was independently
reviewed and pushed at `6c07198` before execution. Clean, pushed main source
is `d5a88e20d3cbe63aea2589b3da394c10fc22824a`. This is a local declaration,
not external registration or peer review.

Runtime is Python **3.12.13**, Torch **2.8.0**, NumPy **2.0.2**, deterministic
CPU with one Torch thread. `PYTHONHASHSEED=0`, `OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1` and `/usr/bin/nice -n 10` are set. The process was observed
at priority 10 using approximately one CPU core. The admission cap is 1,200
seconds; the sampled 4-GiB peak-process-RSS guard is not a hard OS allocation
or total-machine limit.

Original tables reconstruct from archived collector logs and match the
constrained-control archive. All supports, controls, replay reconstructions,
exact float64 labels/float32 casts and prospective panels precede fitting.
All nine fits precede final supported-state inference and every policy rollout.
The graph solver has no exhaustive outcome or full-world Q* argument; the
optimizer receives only observations, masks, fixed labels and sampled rows.

## Pre-execution checks

The seeded fast lane passed **395 tests, 3 deselected in 93.86 seconds**.
All **79 focused and related historical checks passed in 54.02 seconds**.
The 16 new cases cover backward graph values with shuffled clock order;
negative-value branches excluding missing actions; input immutability and
occurrence invariance; missing successors, cross-map edges, clock cycles,
invalid ends and nonfinite rewards; per-state loss/gradient weighting without
successor inference; invalid sampled labels; fit metric denominators and
ranking tolerance; correct controls/replay/counts; all-fit-before-inference
ordering and pure inference; resource-cap preservation during graph preparation
and final diagnostics; and disqualification after a forced replay mismatch.

Actual Python **3.10.20** compilation passed for all auditors, the runner and
tests. JavaScript syntax and whitespace checks passed. Three unchanged legacy
training-heavy tests remain in the scheduled/manual lane. A smoke-interpretation
metadata issue was corrected before smoke-v1: complete smoke remains explicitly
ineligible rather than receiving the main study's descriptive interpretation.

Final smoke-v2 completed **72 updates / 4,608 state presentations / 6,493
action targets**, with zero optimizer neural successor queries, in **6.137
seconds**, peaking at **588,709,888 bytes**. Six final model assessments cover
**359,322 states / 504,658 observed-edge errors**. It preserves nine snapshots,
**72 learner + 12 reference episodes**, and **28 recordings / 637 steps**.
The 24-update treatment versus archived 30,000-update control comparison is
ineligible research evidence; short replay prefixes and full historical labels
are checked separately.

Independent final full/portable smoke audits pass in **7.45 / 7.37 seconds**:
**95 manifest files**, **67 archived input hashes**, independent recorded-only
DP on 252,329 edges, exact casts, all original collector logs/tables and 90,000
historical sampler draws, final-fit metric arithmetic and all recordings.
Full mode regenerates all 359,322 support prediction rows exactly. Final smoke
dashboard checks pass **28 new + 2,314 prior = 2,342 recordings**, including
missing graph/fit diagnostics and distinct label/query accounting.

## Main evidence

One main run completes without deviations or artifact corrections:
**nine new fits / 270,000 updates / 17,280,000 state presentations**,
**27,648 learner + 3,584 shared reference episodes**, **54 snapshots** and
**304 recordings**. Both arms have **24,273,701 action targets**, including
**1,062,190 terminal and 23,211,511 nonterminal-labelled targets**.
The historical control has 23,211,511 neural successor queries; treatment has
zero. There is no new collection, support drawing or baseline training.

Duration is **184.085166 seconds**, peak process RSS **613,695,488 bytes
(0.572 GiB)**, over **574,179 resource checks**. All completion, initialization,
source/support/model/table/label, replay, target/query and resource checks pass.

The [independent auditor](../../scripts/audit_logged_graph.py) passes in
**16.16 seconds**, verifying:

- **140 manifest files**, **79 archived input hashes**, captured source,
  protocol, clean revision and runtime; preserved original artifacts.
- All **301,585 historical collector steps / 12,288 episodes**, collector RNG,
  logged sequences and three deduplicated recorded-action tables, matching
  control support/table copies and separate termination/truncation flags.
- Independent recorded-only DP on **252,329 edges / 241,295 nonterminal
  backups**, all 96 bank/clock accounting rows, exact float32 casts and zero
  float64 Bellman residuals. Another 241,295 residual-validation successor
  lookups are accounted separately; no world oracle supplies graph labels.
- **655,360 diagnostic transitions/targets**, **480 kernel crosschecks**,
  all **270,000 historical sampler draws**, nine exact local/global/map replay
  pairs and counts, matched action targets and zero treatment neural queries.
- **54 snapshots**, nine unchanged controls, 18 final policies and their panel
  identities, initialization, **2,700 loss windows**, immutable graph references,
  casts and optimizer tensors, and historical/new exposure provenance.
- All **18 final support prediction sets / 1,077,966 state rows** regenerated
  exactly, in batches of at most 1,024. Fit metrics reduce **1,513,974 observed
  edge errors**, with state/edge weighting, ranking tolerance, regret and
  outside-mask argmax denominators checked independently.
- **4,608 map exposure rows**, **90 clock exposure rows**, state/action coverage,
  **512 admitted layouts**, 14 prescribed exclusions and all **31,232 episode
  rows**; unique keys, reference crossings, paired/pooled arithmetic,
  **4,608 layout pairs**, **81 seed pairs**, **27 aggregate pairs**.
- All **304 recordings / 5,336 steps**, including observations, exploration
  draws, actions, dynamics, rewards/ends, shortest paths, Q* and regenerated
  learned outputs.

Final-fit inference takes **4.338810 seconds**, making **1,062 online batches /
4,311,864 output values**. Sampled process peak during that phase is
**583,172,096 bytes**; this is process memory, not incremental fit allocation.
All source/model/RNG checks remain unchanged. No target-network inference or
new checkpoint probes occur in these diagnostics.

The audit does not retrain, collect or repeat complete policy evaluation.
Portable mode omits neural forward regeneration while retaining independent
graph construction, casts, saved-prediction fit formulas, identities,
dynamics/Q*, replay, exposure and arithmetic. It does not regenerate each
unrecorded rollout. The integrated portable verifier passes this export and
every earlier shipped study.

## Dashboard and CI scope

The actual-data harness passes **304 new + 2,314 prior = 2,618 recordings**,
including bank/panel/learner/action-mode controls, playback, scrubbing,
study isolation and missing/incomplete/smoke/inconsistent branches. It checks
graph preparation/residuals, full-support fit cards and per-seed values,
exact replay, equal labels and unequal neural queries. Primary outcome cards
remain greedy when replay controls change; rules remain visible initially.

The first predetermined map is **1120000**. Default bank 1 / seed 0 changes
from success in 12 steps to failure at 32; the same map includes treatment
improvements in bank 2 / seed 0 and bank 3 / seed 2. HTTP serves exact
**10,651,587-byte** data, SHA-256
`4422a9a3e3307971fbb3be12c0d40a01eef2a131d7d19d74b68b683b613a5258`.

Native Computer Use could not start its pipe on this turn. **Pixel and
responsive-layout QA remain unverified**; DOM/canvas stubs and HTTP checks
do not replace them.

The fast CI matrix covers Python 3.10/3.12 tests, auditor compilation, portable
verification of all stored studies and actual-data dashboard checks. Python
3.12 builds/installs a wheel away from source and imports `q6.logged_graph`.
Full legacy training remains scheduled/manual. See
[PR #1 checks](https://github.com/rahul-tiwari-95/Q6/pull/1/checks) for submitted
revision outcomes; configuration alone is not a pass.
