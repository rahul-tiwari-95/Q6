# Exact-target milestone validation — 2026-09-06

This record covers the supervised runner, shipped study and **Exact targets**
dashboard. The [result report](../experiments/supervised_results_v1.md) states
the scientific outcome: useful fresh-layout behavior under exact supervision,
with the stricter training-fit diagnostic unmet. Passing software and artifact
checks is not an independent training replication or evidence of online-RL
competence.

## Runtime and software checks

- Main run and local tests: Python 3.12.13, Torch 2.8.0, NumPy 2.0.2, macOS
  arm64 and one Torch CPU thread. The study captures the resolved environment.
- The documented seeded fast lane passed **306 tests, 3 deselected in 38.90s**.
  The three unchanged legacy training-heavy tests passed in the earlier preview
  milestone and were not repeated for this addition.
- Seven focused supervised tests passed in **1.23s**. Coverage includes exact
  Bellman targets checked against actual world transitions, all-four-action
  loss, enumeration and split identity, sampled exposure, unchanged replay and
  target-network state, evaluation isolation, metric denominators, and gates
  for smoke, incomplete checkpoints and admission-cap interruption.
- A CLI smoke completed 24 optimizer updates and 1,536 state presentations
  in 0.74s on separate smoke layouts. It is explicitly ineligible for research
  gates and is not the shipped result.

The protocol was committed at `7902cbd` before main training; it was locally
declared, not externally registered. Main execution captured clean source
revision `83bf7e66db1da7f536b4b37f5d3484d01aa6ed0b`. The world and learner source
hashes match the prior competence study, as do all three initial policy hashes.
Later edits add results, documentation and stronger read-only verification;
they do not change captured training sources or prior study artifacts.

## Main artifact checks

`python scripts/verify_pilot_artifacts.py` passed on the complete main study:

- All **45 manifest files**, source/protocol hashes, dataset array hashes and
  dashboard/result byte equality.
- All **20 rollout aggregate groups**, complete seed/checkpoint/map/repetition
  cells, full-state additive metrics, final dense-prediction errors/quantiles,
  and per-seed gates against raw evidence.
- Zero training/fresh layout and complete-observation intersections; all
  163,840 training rows sampled by each seed; **90,000 updates and 5,760,000
  state presentations** with complete scheduled checkpoint accounting.
- Prior competence, adaptation, diagnostic and provenance manifests and
  aggregates still pass their existing checks.

The main study contains 14,400 learner episode rows, 24,000 per-map/time-bucket
state rows, 2,816 unique reference episodes, 900 loss windows, 15 scheduled
seed/checkpoint groups, 18 inference models and 70 recordings. It completed
in 80.13 seconds with no declared deviations. The weights are inference
snapshots, not resumable optimizer checkpoints.

## Separate agent audit

A second agent reviewed the protocol and implementation, then recomputed
the archived evidence using the portable
[`audit_supervised_study.py`](../../scripts/audit_supervised_study.py). A full
read-only audit from another working directory passed in **6.38s**:

- All **819,200 target action values** satisfy an independently implemented
  finite-horizon Bellman recurrence; maximum residual was 2.22e−16. Another
  **960 cloned-world transitions** check movement, collection, timeout and
  shaping against the actual world implementation.
- All 90,000 minibatch draws reconstruct exactly, including ordered digests
  and per-row sampling counts. All 18 model parameter hashes pass; all six
  final dense prediction arrays reproduce bit-for-bit from saved weights.
- Raw evaluation/state/reference cell sets, reductions, final prediction
  quantiles and per-seed gates reconcile. The historical exposure audit
  independently regenerates 17,222 task IDs and confirms 3/3/4 overlapping
  layouts for the three historical policies.
- Reported scores and the post-hoc successful-episode step breakdown match
  the saved CSVs. No blocking discrepancy remains.

This audit uses the captured source snapshots and recorded library environment.
It performs forward inference and isolated transition checks, without optimizer
updates or policy rollouts. It does not reproduce every training loss or rerun
intermediate policy evaluations; their saved raw records and aggregates are
reconciled. Separate agent review is not external peer review or independent
training replication.

## Dashboard checks

`node scripts/check_dashboard.mjs` passed against the actual main export:

- All **70 recordings** are reachable: 60 supervised, six historical RL,
  and two each for random and shortest-path references.
- All 20 rollout aggregates, 50 exhaustive-state aggregates, 300 pooled loss
  windows, 14 policy/mode/panel combinations and six interpretation branches
  render, including fresh-gate-pass/training-fit-unresolved.
- Pre-action Q labels, play/pause, automatic end, reset, scrub, stop on tab
  switch and refresh pass. Epsilon changes rollout metrics while leaving
  exhaustive action agreement and logged optimization loss unchanged.
- Missing-data and smoke paths were checked separately. All 192 preceding
  competence recordings and older adaptation/provenance controls still pass.

The harness uses DOM/canvas stubs, so these are code-path checks. Native Chrome
loaded the actual new study, and the desktop screenshot showed the four-stage
explanation, correct unresolved-fit banner and dataset/budget cards without
overflow. Native scroll/capture automation became unreliable
(`noWindowsAvailable`); complete pixel review of the lower charts/replay and
responsive layouts was not established. Harness coverage is not a substitute
for that uncompleted visual coverage.

## Continuous integration

The workflow runs the fast lane on Python 3.10 and 3.12, artifact integrity,
dashboard syntax and the dashboard control harness. The Python 3.12 job builds
and installs a wheel away from source, including a supervised-runner import.
Implementation revision `83bf7e6` passed both push and PR CI. Consult
[PR #1 checks](https://github.com/rahul-tiwari-95/Q6/pull/1/checks) for the precise
submitted result revision. The bit-exact deep audit above uses the recorded
environment and is separate from CI's broader dependency compatibility checks.
