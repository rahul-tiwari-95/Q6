# Equal-map replay validation — 2026-09-07

This records software and evidence checks separately from the
[scientific interpretation](../experiments/map_replay_results_v1.md).
The study reuses three collected supports and nine archived control policies,
training nine treatments with a different replay design. No new collection,
baseline training, support draw, memory or historical gate change occurs.

## Declaration and execution

The [protocol](../experiments/map_replay_protocol_v1.md) was independently
reviewed and pushed at `1bd8628` before execution. Clean, pushed main source is
`9104aa723d6858c5c70ccb14419724b6bf31ca68`. This is a local declaration,
not external registration or peer review. There was one main run, with no
protocol deviation, artifact correction or additional tuning run.

The pinned runtime is Python 3.12.13, Torch 2.8.0 and NumPy 2.0.2. Execution
uses deterministic CPU operations, one Torch thread, `PYTHONHASHSEED=0`,
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` and `/usr/bin/nice -n 10`. The process
was observed at priority 10 with approximately one CPU core occupied. The
sampled 4-GiB peak-process-RSS guard measures this process, not total machine
memory or a hard OS allocation limit. The admission cap is 1,200 seconds.

All supports and baseline inputs are verified before updates. Treatment
initial online/target hashes match the archived same-seed controls. Separate
map and within-map generators preserve identical ordered map schedules across
banks without coupling future map draws to different state-count bounds.
Every treatment fit finishes before any learned-policy evaluation. Final
inference allocates no optimizer/replay and preserves all control/treatment
source/copy/module identities around each panel. World, core learning,
fixed-target and rollout source identities match the archived baseline.

## Pre-execution checks

The seeded fast lane passed **350 tests, 3 deselected in 52.01 seconds**.
All **six new focused tests passed in 6.58 seconds**, and **28 related historical
coverage/equal-support/panel/bank tests passed in 4.25 seconds**.

Focused tests check independent map schedules under different support sizes
and global RNG activity, exact map exposure when every map is selected,
missing/invalid support rejection, fixed supports and copied controls before
updates, all fits before rollouts, no online `observe`/`learn` or collection,
truthful baseline/treatment checkpoint labels, exposure denominators, saved
manifest bytes, nonfinite-update preservation, preparation caps and final
post-aggregation disqualification. The three unchanged legacy training-heavy
cases remain in the scheduled/manual full lane.

Actual local **Python 3.10.20** compilation passed for all auditors, the new
runner and tests after final implementation. JavaScript syntax and whitespace
checks passed. A transient shared-helper default-name error was caught during
preflight collection and fixed before the passing regression run; historical
default aggregation behavior is preserved.

Final smoke-v2 reuses all three shipped supports and all 256 training maps,
with 24 treatment updates per bank and archived 30,000-update seed-0 controls.
It completed **72 updates / 4,608 presentations**, **72 learner + 12 reference
episodes**, **nine snapshots** and **28 recordings** in **2.466 seconds**,
peaking at **477,675,520 bytes**. Its unequal treatment/control budgets and
alternate panels are explicit, and smoke is ineligible scientific evidence.

Independent full and portable smoke audits passed in **2.693 / 2.720 seconds**:
64 manifest files, 45 archive-input hashes, all 655,360 transitions/targets and
480 kernel crosschecks, exact support/control reuse, 12 ordered sampling
digests, one paired map-schedule group across three banks, 1,536 map / 30 clock exposure rows,
three loss windows, and all 28 saved recordings / 667 steps. The full audit
regenerated learned Q outputs exactly. Final smoke dashboard checks passed
all 28 new and 1,098 prior recordings. A percentage-formatter scope issue was
fixed before final UI validation and source freeze.

## Single main run and independent audit

The main run completed **nine new treatment fits / 270,000 updates /
17,280,000 state presentations**, **27,648 learner episodes**, and **3,584
shared references = 31,232 evaluation episodes**. It reused nine final
controls and exact support sizes **59,839 / 59,984 / 59,838**. Historical
baseline updates total 270,000; new baseline updates, collection steps and
support draws are all zero.

Duration was **178.149670 seconds** and peak process RSS **481,869,824 bytes
(0.449 GiB)**, across **301,780 resource checks**. All completion, initialization,
source/support/model, sampler, loss and resource checks passed.

The pinned-runtime [independent auditor](../../scripts/audit_map_replay.py)
passed in **20.879 seconds**, checking:

- **109 manifest files** and **57 archive-input hashes**, captured source,
  protocol, environment and clean revision; archived supports and controls
  unchanged, including historical local/global sample counts.
- **655,360 exact targets and transitions**, 480 independent kernel
  crosschecks, maximum Bellman residual 2.22e-16 and zero reward discrepancy.
- **36 ordered sampling digests** (map, within-map rank, support-local and
  global rows for nine treatments), all counts, and three paired map-schedule
  groups. These are digest streams, not 36 independent RNG generators.
- **54 snapshots**, nine frozen archived controls, 18 final models and
  144 model-panel identity checks; exact initial online/target matches and
  **2,700 finite raw loss windows**.
- All **4,608 map exposure rows**, **90 clock rows**, 18 state/category/query
  summaries, fixed support membership and zero direct off-support samples.
- **512 selected layouts**, six prescribed candidate rejections, training/
  prior-panel exclusions and complete unique keys for every learner/reference
  episode; all per-seed/panel/bank/equal-bank arithmetic and denominators.
- **4,608 greedy layout pairs**, **81 seed-pair rows**, **27 aggregate-pair
  rows**, all effects, sign/range counts and historical threshold arithmetic.
- All **304 preselected recordings / 5,259 steps**, including observations,
  external exploration draws, selected actions, rewards, termination,
  shortest paths, finite-horizon Q* and exact regenerated learned Q outputs.

The audit does not retrain, collect or repeat the complete learned-policy
evaluation. Raw arithmetic checks are not independent reproduction of each
unrecorded episode. Portable mode omits regenerated neural outputs while
retaining recorded-action consistency, dynamics/Q*, all identities, sampling,
exposure and table arithmetic. The integrated portable verifier passed this
new export and every historical study. No old artifact was modified.

## Dashboard and review scope

Experience coverage defaults to **Map-balanced replay**, retaining all earlier
studies. Actual-data checks passed **304 new + 1,098 prior = 1,402 recordings**,
all bank/panel/condition/learner controls, both action modes, playback,
scrubbing, missing/incomplete/smoke/inconsistent states and study isolation.
The primary cards remain greedy when exploration mode changes.

The harness checks **108 aggregates**, **324 seed rows**, **18 references**,
**900 aggregated loss windows**, every map/clock exposure row, normalized map
shares, map selection/highlighting, actual versus historical update counts,
state repetition/category/query diagnostics and all threshold values. Support
membership and query-weighted exposure fractions have distinct labels. The
preselected first map remains 1040000 with mixed outcomes across learners.
Active rules remain visible initially; no intermediate-evaluation selector.

HTTP serves the exact **10,329,483-byte** export with SHA-256
`d59b31fe2a9b0efae3495c4e82e685a164756958f35f1b1a34885242c25fc1dc`.
Native Computer Use could not start its pipe. **Pixel and responsive-layout
QA remain unverified**; DOM/canvas stubs and HTTP checks do not replace them.

The report, README and roadmap are separately reviewed for mixed effects,
shared-bank/seed/panel dependence, exposure versus support, ratio denominators,
unchanged historical gates and limits of privileged offline evidence.

## CI scope

CI runs Python 3.10 and 3.12 fast tests, audit-script compilation, portable
verification of every stored study, JavaScript syntax and the actual-data
dashboard harness. Python 3.12 also builds and installs a wheel away from
source, importing `q6.map_replay`. The full legacy training lane remains
scheduled/manual. Consult [PR #1 checks](https://github.com/rahul-tiwari-95/Q6/pull/1/checks)
for outcomes at the submitted revision; configuration alone is not a pass.
