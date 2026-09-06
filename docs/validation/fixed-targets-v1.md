# Fixed-data target milestone validation — 2026-09-06

This record separates software and evidence checks from the scientific
interpretation in the [result report](../experiments/fixed_targets_results_v1.md).
The comparison uses privileged offline coverage and does not establish an
online-RL baseline or independent task-bank replication.

## Protocol, runtime and software checks

The [protocol](../experiments/fixed_targets_protocol_v1.md) was committed at
`c253fcb` before main training, following a separate agent review with no
blocking methodological finding. It was locally declared, not externally
registered. Main execution captured clean implementation revision
`14bb9571921ba5fc55c7a2a8bb2f7c92d52416be`.

- Python 3.12.13, Torch 2.8.0, NumPy 2.0.2, macOS arm64, deterministic CPU
  execution and one Torch thread. Full package versions are archived.
- The documented seeded fast lane passed **316 tests, 3 deselected in 39.14s**.
  The three unchanged legacy training-heavy cases passed in the earlier
  preview milestone and were not repeated for this addition.
- **Ten focused tests passed in 1.71s**: exhaustive compact-transition versus
  actual-kernel checks including all 32 clocks; terminal masking; online
  selection versus target evaluation; detached labels; post-update soft-target
  timing; exact-update parity; paired sampling; layout exclusion; efficiency
  denominators; platform-correct RSS units; and preserved ineligible results
  for consistency failures, partial checkpoints and post-aggregation caps.
- The final CLI smoke used alternate layouts and completed 48 total updates
  and 3,072 state presentations in 0.646s, with 269.3 MB peak process RSS.
  It was explicitly ineligible for all research gates and is not shipped
  as the main dataset.

Main training was launched with `PYTHONHASHSEED=0`, `OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1` and `/usr/bin/nice -n 10` around the recorded Python command.
The lower scheduler priority leaves room for other applications. The 4 GiB
peak-process-RSS guard is sampled, not an OS allocation limit or a measure
of all machine memory. Conditions run sequentially.

## Audit scope

The complete main study passed the full independent audit in **5.25s**, with
no source or artifact corrections required:

- All **65 manifest files**, captured source/protocol hashes and historical
  input hashes; zero train/fresh observation or layout overlap, and zero
  overlap with the preceding supervised fresh panel.
- All **819,200 exact target entries** and **655,360 saved transitions** checked
  by independent reconstruction. Maximum Bellman residual was 2.22e−16 and
  maximum saved-reward discrepancy was zero; **960 isolated cloned-world
  transitions** cross-check the actual kernel.
- All six sampling digests/count vectors, 30 online/target snapshots and
  **12 bit-exact final prediction arrays**. Training arrays and all three
  exact-arm final weights/digests match the prior supervised study.
- All **28,800 learner episode rows**, **48,000 state rows**, **1,800 loss
  windows**, **2,240 unique reference episodes**, **960 paired layout rows**
  and per-seed success/fit/efficiency calculations.

Main execution completed all 180,000 updates in **145.67s**, using **579.6 MB
peak process RSS (0.54 GiB)** across 200,551 resource checks. It stayed within
its declared limits, with no deviations. Both fresh-success gates pass; both
all-seed fit and efficiency diagnostics remain unmet. A separate report review
confirmed the numbers and distinctions between pooled fractions, per-seed
paired differences and successful-only episode subsets.

The portable [independent audit](../../scripts/audit_fixed_targets_study.py)
uses captured source snapshots, not mutable checkout training modules. Before
main execution both modes passed on the complete smoke: full forward checking
in 0.45s and `--skip-forward-inference` in 0.29s. Smoke validation checks
structure and numerical invariants; it cannot satisfy research gates.

The default audit independently reconstructs finite-horizon exact values and
compact transitions, checks isolated actual-world transitions, regenerates
sampling digests, verifies online/target model hashes and reproduces final
dense predictions. It also reconciles raw rollout/state/loss rows, paired
outcomes, success/fit/efficiency gates and historical exact-arm identity.
It runs no optimizer updates or policy rollouts. Recorded training losses
and intermediate policy outcomes are reconciled to their raw logs rather
than independently reproduced by another training run.

General artifact verification invokes this audit in a separate process with
`--skip-forward-inference` for dependency-range CI. That mode retains all
saved-data, transition, sampling, model-hash and metric checks and explicitly
reports that bit-exact regenerated forward predictions were omitted. The
default pinned-runtime audit provides that additional check. Neither mode
constitutes external peer review or independent training replication.

## Dashboard and CI scope

The dashboard harness uses DOM/canvas stubs. Before main execution it checked
all 20 smoke recordings, both conditions and action modes, paired metrics,
pre-action Q values, initial visible rules, playback, scrub, reset, tab-stop,
refresh and seven interpretation branches. Consistency failures show their
literal stop reason, including when no aggregates exist. Missing data stay
empty. The 70 earlier supervised and 192 competence recordings, plus
adaptation/provenance controls, continue to pass.

On the actual main export, the harness passed all **124 recordings**, **40
rollout aggregates**, **100 exhaustive-state aggregates** and **600 pooled loss
windows**. The actual initial banner correctly shows both fresh-success gates
passing while fit/efficiency remain unresolved, including DDQN's per-seed fit
split. The first preselected fresh map, 940000, is a four-step success for both
conditions in every seed; it is not presented as a contrasting example.
Post-run presentation adds prominent final fresh success, efficient-success and
mean-step cards for both conditions. The harness matches their displayed values
to saved greedy and epsilon-mode aggregates and verifies that changing mode
leaves the declared greedy gates unchanged. No training source or artifact was
edited for this presentation change.

Native browser review was unavailable in this milestone: Computer Use failed
to start its native pipe, including after a session reset. No pixel or
responsive-layout verification is claimed for the new track. Serving the
actual files and exercising their controls is checked separately from layout.

CI runs the 316-test fast lane on Python 3.10 and 3.12, general artifact
verification, JavaScript syntax and the dashboard harness. Python 3.12 also
builds and installs a wheel away from source and imports `q6.fixed_targets`.
The full legacy suite remains scheduled/manual. Consult
[PR #1 checks](https://github.com/rahul-tiwari-95/Q6/pull/1/checks) for the precise
submitted revision; workflow configuration alone does not establish a pass.
