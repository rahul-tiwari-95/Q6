# Experience-coverage milestone validation — 2026-09-06

This record separates software and artifact checks from the scientific
interpretation in the [result report](../experiments/coverage_results_v1.md).
Both conditions retain privileged four-action transitions and detached
successor queries from the full training bank. This is an offline coverage
comparison, not an online-RL baseline.

## Protocol and software checks

The [protocol](../experiments/coverage_protocol_v1.md) was committed and pushed
at `552a7f7` before main execution, following a separate agent review with no
blocking methodological finding. It was locally declared, not externally
registered. The implementation reuses the previous DDQN update unchanged;
the exhaustive arm must reproduce the previous main DDQN weights and sampling
at every final seed.

Main execution captured clean, pushed implementation revision
`8070ed14ca78016f69759df21a0e77bf0676390b`. The launch set
`PYTHONHASHSEED=0`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` and used
`/usr/bin/nice -n 10`. Conditions run sequentially on one CPU thread in
Python 3.12.13, Torch 2.8.0 and NumPy 2.0.2. The sampled 4 GiB process-RSS
stop guard is not an OS allocation limit or a measure of total machine RAM.

Nine focused tests passed in **1.83s**. They check actual collector actions and
transitions, isolated random-number streams, deduplication, local/global batch
identities, unchanged exhaustive DDQN updates, detached outside-support
successor access, coverage denominators, both previous fresh-panel exclusions,
archived smoke integrity, preserved interrupted collection and explicit failure
when unique support is smaller than one batch.

The documented seeded fast lane passed **325 tests, 3 deselected in 43.82s**.
The three unchanged legacy training-heavy cases were not repeated for this
addition; their earlier validation remains archived.

The separate CLI smoke completed **32 collection episodes, 606 actual steps,
396/1,280 unique current states and 48 optimizer updates** in **0.893s**, with
**274.8 MB peak process RSS**. Its alternate layouts and small learning budget
make it ineligible for all research gates.

The independent audit passed the complete smoke in **0.55s** with regenerated
forward predictions, and **0.42s** in portable `--skip-forward-inference` mode.
Both reconstructed all original collection action draws, episodes, support,
local/global sampling and saved metrics. Smoke validation supplies no research
gate evidence.

## Dashboard scope

Before main execution, the DOM/canvas-stub harness passed all **20 smoke
recordings**, 16 rollout aggregates, 40 state aggregates, both conditions and
action modes, two layout cells, five clock-bucket rows, seven interpretation
branches, final outcome cards, paired differences, pre-action Q labels,
playback and refresh. All five previous tracks remain available. This checks
code paths and displayed numerical values, not browser pixels.

Native visual review was unavailable: Computer Use failed to start its native
pipe. No pixel or responsive-layout verification is claimed for this track.

On the actual main export, the harness passed all **124 recordings**, **256
layout cells**, 40 rollout aggregates, 100 state aggregates and 600 pooled loss
windows. The default banner shows exhaustive-only fresh competence, identifies
collected seeds 0 and 2 as missing that gate, and retains both unmet every-seed
fit and efficiency diagnostics. All five historical tracks continue to pass.
The final presentation check additionally verifies exact winnable, goal-near
and outside-support edge denominators, plus every per-layout efficient-success
and no-op-rate delta. These presentation edits change no learning artifact.
The real local HTML and data are served over HTTP; native visual review remains
unavailable as described above.

## Main artifact audit

The full independent audit passed in **6.88s**, without source or main-artifact
corrections:

- All **72 manifest files**, captured source/protocol hashes and historical
  input hashes. No train/fresh or prior-fresh layout overlap; the captured
  implementation was clean and runtime matched the declared environment.
- All **819,200 exact target entries** and **655,360 compact transitions**, plus
  **960 isolated cloned-world transitions**. Maximum Bellman residual is
  2.22e−16 and maximum saved-reward discrepancy is zero.
- The entire original **4,096-episode, 99,814-step** collection reconstructed
  from its declared RNG, including action draws, current/successor rows and
  deduplicated **59,626-state** support. Map, clock, goal-near and successor
  coverage reconcile to the raw logs.
- All **12 local/global sampler streams**, counts and ordered digests.
  Collected training has zero direct samples outside the declared support.
  All 30 online/target snapshots and **12 bit-exact final prediction arrays**
  verify. Exhaustive training arrays, transitions, final online/target weights
  and sampling digests match the preceding DDQN archive.
- All **28,800 learner episode rows**, **48,000 state rows**, **1,800 loss
  windows**, **2,240 unique reference episodes** and **960 paired layout rows**,
  including efficient-success deltas and every per-seed gate.

Main execution completed all 180,000 updates in **156.64s**, using **637.9 MB
peak process RSS (0.59 GiB)** across 304,461 resource checks. No limits were
reached and no deviations recorded. Both arms share initial weights and
update counts; their current-state supports and actual minibatches intentionally
differ. This is not the identical-batch design of the previous target study.

The [audit](../../scripts/audit_coverage_study.py) imports captured source
snapshots and independently reconstructs the original random collection.
It runs no optimizer updates or new learned-policy rollouts. Intermediate
learned-policy outcomes and training losses are reconciled to saved raw logs,
not reproduced through another training run. Repeating the exhaustive arm is
an execution-consistency control, not independent task-bank replication.

General artifact verification also passed, including the new audit in
`--skip-forward-inference` mode and all earlier studies. The portable mode
retains saved-data, transitions, collection, sampling, model hashes and raw
metric/gate checks; only regenerated final forward predictions are omitted.
This permits dependency-range CI, while the pinned-runtime default audit
provides the bit-exact forward check. Neither mode is external peer review.

Two agents independently recomputed the supplemental visited/unvisited
agreement from saved prediction arrays and obtained identical numbers. This
post-hoc analysis performs no new inference or learning, adds no gate, and is
stored outside `pilot_v1`. The result report was separately reviewed for
numerical accuracy, paired versus pooled denominators, full-bank versus
sampled-support fit, fresh-panel differences and limits of the next control.
Final general verification also recomputes this supplemental artifact and
checks its input hashes, counts and numerical results without writing files.

## CI scope

CI runs the 325-test fast lane on Python 3.10 and 3.12, general artifact
verification, JavaScript syntax and the six-track dashboard harness. Python
3.12 additionally builds and installs a wheel away from source and imports
`q6.coverage`. The full legacy suite remains scheduled/manual. Consult
[PR #1 checks](https://github.com/rahul-tiwari-95/Q6/pull/1/checks) for the submitted
revision and actual check outcomes; workflow configuration alone is not a pass.
