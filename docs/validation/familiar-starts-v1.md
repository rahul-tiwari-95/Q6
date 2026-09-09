# Familiar-start validation

The single main experiment ran on September 7, 2026 EDT (September 8 UTC).
Documentation and publication resumed September 9 after an interruption;
the experiment was not repeated. Scientific interpretation is in the
[result report](../experiments/familiar_starts_results_v1.md).

## Declaration and execution

The [protocol](../experiments/familiar_starts_protocol_v1.md) was independently
reviewed and pushed at `fc89597`. Amendment `48de7e8` explicitly identifies
the original-start prediction slice as exploratory analysis of already
available predictions, documented before the new rollout outcomes.
The evaluation population and primary interaction were declared earlier.
This is a local declaration, not external registration or peer review.

Clean pushed execution source is
`77ebcb968bab142eb5b4e8b0470bd6ef89e38f7a`. Its GitHub
[push checks](https://github.com/rahul-tiwari-95/Q6/actions/runs/34175379918)
and [PR checks](https://github.com/rahul-tiwari-95/Q6/actions/runs/34175385692)
pass. Publication-head checks are linked from PR #1.

Runtime: Python 3.12.13, Torch 2.8.0, NumPy 2.0.2, deterministic CPU,
one Torch thread, `PYTHONHASHSEED=0`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`
and nice priority 10. The main process was observed near one CPU core.
The 1,200-second admission cap and sampled 4-GiB peak-process-RSS guard are
not hard OS or total-machine limits.

## Software and smoke checks

Before main execution, the seeded fast lane passed **415 tests, three
deselected, in 105.62 seconds**.
Final focused verification passed **20 tests in 1.77 seconds**, after adding
the explicit identity check between all 256 supported clock-32 rows and
original starts. The unchanged long training checks remain in the scheduled
or manually dispatched lane. Actual Python 3.10 compilation passed for all
auditors, the new evaluator and its tests; JavaScript syntax and whitespace
checks passed.

The new tests cover return-independent success reachability; last-move success;
absent outcomes and occurrence invariance; invalid graph closure, clocks,
maps, flags and indices; guard propagation; lowest-label allowed ties and
invalid predictions; off-mask choices that stay supported; exits, reentries
and terminal handling; frozen parameters without gradients; masked closure;
paired interactions and pooled denominators; partial-step preservation;
duplicate state identities; a compact run forbidding learner/optimizer
construction; duplicate or missing evaluation/reference cells; interrupted-run
ineligibility; and per-state error-offset decomposition.

Final smoke-v2 completed in **1.628 seconds**, peaking at **417,873,920 bytes**:
six frozen models, 24 learner plus eight reference episodes, 32 recordings,
313 steps and 60 archived prediction slices. It performed no training or
collection and is explicitly ineligible research evidence.

Independent full and portable smoke audits passed in **1.645 / 1.669 seconds**.
Both verify 55 manifest files, 25 archived inputs, recorded-only graph values
and reachability, all steps and paired reductions. Full mode additionally
regenerates 276 saved neural recording decisions. The smoke dashboard harness
passes **32 new plus 2,618 prior = 2,650 recordings**.

## Main evidence audit

One main run completes without deviations: **18 frozen models**, **9,216 learner
plus 1,024 reference episodes**, **146,963 step records**, **180 prior-prediction
slices** and **320 recordings**. New training updates, collection, support draws
and fitted labels are all zero. Duration is **12.504479 seconds**, sampled
peak process RSS **448,446,464 bytes (0.418 GiB)**, with **148,906 resource checks**.

The [independent auditor](../../scripts/audit_familiar_starts.py) passes in
**6.250 seconds**, checking:

- All 67 manifest files and 37 archived inputs, including original logs,
  copied predictions and all final model/source/copy/online/target identities.
- All 163,840 state rows and 655,360 independently reconstructed transition
  entries, with 480 direct kernel-clone checks.
- All 252,329 distinct recorded edges and backward values, plus independent
  success reachability on 179,661 supported states across three banks.
- Original spawn/clock identities, graph ceilings, reference returns and
  all 180 saved-prediction slices with their correct denominators.
- Every one of 146,963 trace records: state/action/reward/terminal identity,
  masks, support exits/reentries, reachability loss, episode reductions,
  occupancy counts and paired comparisons.
- All 320 recordings / 4,188 steps, regenerating **3,997 selected neural
  decisions**. This does not repeat the complete learned-policy evaluation.
- All **11,520 paired-start rows, 405 seed rows and 135 bank rows**, including
  the interaction and its two absolute masking effects.

The integrated portable verifier passes across every shipped study, including
this archive. No prior research artifact or dashboard data file was changed.
Portable verification skips selected forward regeneration while retaining
hashes, recorded-graph reconstruction and raw-evidence arithmetic.

### Linux audit portability correction

The first publication CI run exposed a float32 reduction-order difference
in the auxiliary `mean_unrestricted_prediction_gap` scalar. The saved Mac
mean was `0.0917019471526146`; Linux recomputed `0.09170196205377579`, a
1.49e-8 difference just beyond the generic 1e-8 absolute check. All tests
and the preceding archived studies passed; this was an auditor portability
failure, not a changed policy outcome.

For this scalar only, the auditor now computes a canonical reference with
`math.fsum` over the saved float32 per-state gaps and permits **two float32
rounding units at the reference magnitude**. Every one of the 180 saved slices
was checked; the largest discrepancy is 1.518 units. Exact zero remains
exact, invalid inputs fail, and larger discrepancies are rejected. This is
an explicit archival tolerance, not a universal error bound for arbitrary
float32 summations. No common tolerance, runner, snapshot or artifact changes.

The added 11 regression cases accept the actual Mac and Linux reductions of
the same archived values, reject discrepancies outside the budget, and cover
zero, nonfinite, negative, empty and wrong-precision inputs. Together with
the evaluator tests, **31 focused tests pass in 1.77 seconds**. Actual Python
3.10 compilation passes. The complete seeded suite then passes **426 tests,
three deselected, in 92.05 seconds**. The corrected auditor passes full and portable main
checks in **6.410 / 6.278 seconds**, preserving every prior evidence check.
The main experiment was not rerun. Final publication CI is recorded on PR #1.

## Dashboard and interpretation checks

The final actual-data DOM/canvas harness passes **2,938 distinct recordings**:
320 new and 2,618 prior. The familiar view has 336 selector paths because
the shared planner is reachable from each bank. Checks cover all four cells,
paired effects, graph ceilings, support denominators, all 180 prediction
slices, status/empty branches, playback/refresh and every earlier study.
Original-start slices are labelled exploratory; rules remain initially visible.

After the run, UI wording was corrected to show that a positive interaction
means a more positive or less negative masking effect. Both absolute effects
and the remaining masked-model gap appear beside it. These presentation and
harness changes do not alter the captured execution source or artifacts.

HTML, JavaScript, CSS and the new data export matched localhost:8080 byte for
byte. The export is **12,332,621 bytes**, SHA-256
`695db2a1b5afadabc886f4cb58dc760f08cbfcf0b6eee9dfabdce5f43fdba0a4`.
Native browser automation failed at startup, so these are functional and HTTP
checks; no pixel or responsive-layout verification is claimed.

These checks establish internal evidence consistency, not independent
experimental replication or broad RL competence. Familiar logged masks remain
privileged diagnostics. License remains undecided and PR #1 stays open.
