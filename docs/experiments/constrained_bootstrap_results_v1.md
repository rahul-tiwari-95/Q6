# Logged-successor bootstrap constraints — results v1

**Changing only the training bootstrap action set more than doubles efficient
success: 22.66% → 49.52%.** Effects are **+27.80, +25.72 and +27.08 percentage
points** across the three original collected banks. All 24 bank-panel means
and all 72 bank/panel/learner cells improve in efficiency. Overall success
improves less consistently, **66.17% → 69.49%**; some learners and panels worsen.

Both arms receive the same recorded outcomes, replay and **24,273,701 action
targets**. The intervention changes only which successor action the online
network may select when constructing a DDQN target. Fresh-map policies still
choose among all four actions. The result supports a material effect of the
backup action set under these sparse logs; it does not prove overestimation
or unsupported-action extrapolation as the sole cause.

The [protocol](constrained_bootstrap_protocol_v1.md) was reviewed and pushed at
`03fe2f16b39e1b8b8ce7b5259285a18d4cb045ec` before the single main run. Clean,
pushed execution source is `7b70b43aa9df08ddd3e71fe5ef3d54cbc688c7ec`.
There were no deviations, artifact corrections or subsequent tuning runs.
This is a local declaration, not external registration or peer review.

Inspect [raw results](../../experiments/constrained_bootstrap/pilot_v1/results.json),
[paired episodes](../../experiments/constrained_bootstrap/pilot_v1/paired_layouts.csv),
[training diagnostics](../../experiments/constrained_bootstrap/pilot_v1/bootstrap_diagnostics.json),
[fixed probes](../../experiments/constrained_bootstrap/pilot_v1/bootstrap_probes.json),
and [validation](../validation/constrained-bootstrap-v1.md). In the
[dashboard](http://127.0.0.1:8080/dashboard/lab.html#coverage), select
**Experience coverage → Constrained bootstrap**.

## What changed, and what was held fixed

The original banks contain **59,839 / 59,984 / 59,838 states** from the same
256 training maps. All **301,585 historical collector steps / 12,288 episodes**
are retained and validated. Deduplication yields **84,305 / 84,091 / 83,933
recorded state/action edges**. About 69% of collected states have only one
logged action. Neither condition receives outcomes for unrecorded actions.

The control is nine archived final **recorded_actions** policies, reevaluated
here without retraining. It is not the earlier four-action control. Nine new
fits start from identical seed-specific online/target weights and use the
same 20,420-parameter network, Adam settings, 64-state batches, per-state
mean SmoothL1 objective, clipping, soft target update and 30,000-update budget.
All local/global/map replay streams, counts and map exposure match exactly.

For each observed nonterminal edge, the treatment selects the online
network's highest-valued action **among actions logged at that successor**,
then evaluates that action with the target network. Ties use the lowest
allowed action label. Every queried successor has a nonempty recorded mask;
an empty mask is an error, with no unrestricted fallback. Terminal targets
use the logged reward without a successor query. The optimizer receives
observations, recorded tensors and sampled rows; exhaustive outcomes and
full-world exact Q* are not training inputs.

| Budget or access | Recorded-action control, historical | Constrained treatment, new |
| --- | ---: | ---: |
| Updates | 270,000 | 270,000 |
| State presentations | 17,280,000 | 17,280,000 |
| Recorded action targets | 24,273,701 | 24,273,701 |
| Terminal targets | 1,062,190 | 1,062,190 |
| Nonterminal target queries | 23,211,511 | 23,211,511 |
| Queries outside current-state support | 0 | 0 |
| Training successor choices | All four predictions | Logged-action predictions |
| Fresh-map choices | All four predictions | All four predictions |

Target totals by bank are **8,115,408 / 8,078,271 / 8,080,022**. Both-arm
successor-query vectors are identical, not merely equal in total. Recorded
tables and optimizer tensors remain unchanged. There are no new collection
steps, support draws or baseline updates.

## Behavior on new common panels

Eight prospective 64-map panels scan from `1100000 + 1000 × panel`, excluding
training layouts and the earlier panels specified in the protocol. Nine
prescribed collisions are skipped, including one training layout; original
spawns are preserved. Selection uses no outcomes. All fits finish before
policy evaluation on **512 accepted layouts**: **27,648 learner + 3,584
shared reference episodes = 31,232**. Only final policies are evaluated.

Efficient success means reaching the goal within twice its shortest-path
length, with failures included. Each greedy bank/condition has 1,536 episodes.
Cells show recorded-action control / constrained treatment.

| Bank | Success | Efficient success | Efficiency Δ | Mean steps, all episodes |
| --- | ---: | ---: | ---: | ---: |
| 1 | 65.23% / 69.79% | 22.14% / 49.93% | +27.80 pp | 20.35 / 14.77 |
| 2 | 67.84% / 69.53% | 23.05% / 48.76% | +25.72 pp | 19.95 / 14.81 |
| 3 | 65.43% / 69.14% | 22.79% / 49.87% | +27.08 pp | 19.92 / 14.77 |

Equal-bank means equal episode pooling here because counts are equal:

| Greedy outcome | Recorded-action control | Constrained treatment |
| --- | ---: | ---: |
| Success | 66.17% (3,049/4,608) | 69.49% (3,202/4,608) |
| Efficient success | 22.66% (1,044/4,608) | 49.52% (2,282/4,608) |
| Mean steps, all episodes | 20.07 | 14.79 |
| Mean steps, successes only | 13.98 | 7.23 |
| Blocked steps / total steps | 68.17% (63,061/92,503) | 71.36% (48,620/68,130) |

There are **1,238 more efficient successes** but only **153 more successes**.
Much of the benefit is shorter routes. Successful-only averages compare
different successful subsets. Absolute blocked steps decrease, but their
share increases by **3.192 points**; the mean of bank ratios rises **3.201**
points and the mean of bank/learner ratios rises **3.221**. The denominator
matters: do not describe the blocked-step ratio as improved.

| Panel | Bank 1 efficiency Δ | Bank 2 efficiency Δ | Bank 3 efficiency Δ |
| --- | ---: | ---: | ---: |
| 0 | +36.46 pp | +27.60 pp | +28.65 pp |
| 1 | +23.44 pp | +17.19 pp | +29.17 pp |
| 2 | +26.04 pp | +22.40 pp | +21.35 pp |
| 3 | +29.17 pp | +29.17 pp | +31.25 pp |
| 4 | +32.29 pp | +30.21 pp | +20.31 pp |
| 5 | +27.60 pp | +27.08 pp | +19.27 pp |
| 6 | +17.19 pp | +22.92 pp | +35.94 pp |
| 7 | +30.21 pp | +29.17 pp | +30.73 pp |

Efficiency improves in **24/24 bank-panel means**, **72/72 learner-panel
cells**, and **9/9 whole-panel learner comparisons**. Whole-panel learner
effects by seed 0 / 1 / 2 are bank 1 **+32.03 / +26.95 / +24.41**, bank 2
**+20.90 / +32.42 / +23.83**, and bank 3 **+24.41 / +25.20 / +31.64** points.

Success is less consistent: **18/24 bank-panel means improve, five decline
and one ties**; **44/72 learner-panel cells improve, 20 decline and eight
tie**. Eight of nine whole-panel learner comparisons improve, with bank 2 /
seed 2 losing 0.59 points. These crossed cells share training maps, banks,
initializations and panels; they are not 72 independent replications.

Historical all-learner 70% success/above-random reference crossings occur on
**1/8, 1/8, 0/8** control panels and **1/8, 2/8, 2/8** treatment panels.
No bank/condition reaches the all-learner 80% efficiency reference on any
panel. These are descriptive references, not new competence gates.

Separate epsilon-0.1 diagnostic success / efficiency is **72.52% / 22.99%**
control and **77.42% / 50.36%** treatment. Shared random references succeed
**40.59% (1,247/3,072)**; the planner succeeds on all 512 maps, averaging
**3.68 steps**. Older privileged policies were not evaluated on these panels;
their earlier scores do not establish a matched ranking against this treatment.

## What the training diagnostics say

Diagnostics compare restricted and unrestricted alternatives under the
**same pre-update treatment weights and recorded successor query**. The
unrestricted alternative never enters the loss. Across **23,211,511 optimizer
queries**, unrestricted argmax lies outside the logged set **13,951,582 times
(60.11%)**. The mean online-value gap is **0.07157**; the mean change in the
complete DDQN target is **−0.06680**, ranging from **−1.40554 to +0.50766**.

| Target change | Queries | Fraction of all optimizer queries |
| --- | ---: | ---: |
| Negative | 13,154,243 | 56.67% |
| Positive | 797,336 | 3.44% |
| Zero, within 1e-12 | 9,259,932 | 39.89% |

The online and lagged target networks can rank actions differently, so a
restricted online argmax need not lower the target. Signed differences use
subtraction of the complete float32 targets, preserving actual rounding.
Online gaps are nonnegative and can be zero when a tie changes the action.
Statistics are query-weighted, not unweighted averages of differently sized
windows. All **2,700 hundred-update windows** retain raw sums, squared sums,
extrema and sign counts. No historical-control diagnostic curve is invented.

Fixed probes provide an independently reproducible view across training.
Each bank/seed uses its first original sampler batch, reconstructed and frozen
without advancing actual replay. The same **754 nonterminal edge presentations across nine
fits** are probed at each checkpoint:

| Update | Unrestricted argmax outside logged set | Mean online gap | Mean target change |
| --- | ---: | ---: | ---: |
| 0 | 63.40% | 0.06315 | −0.06126 |
| 1,000 | 60.34% | 0.03815 | −0.02869 |
| 3,000 | 56.37% | 0.05778 | −0.05261 |
| 10,000 | 58.22% | 0.06775 | −0.06531 |
| 30,000 | 57.16% | 0.07278 | −0.06792 |

The **45 probes / 3,770 probe queries** are inference only and excluded from
optimizer and policy totals. They leave parameters, RNG and replay/update
counters unchanged. Full auditing regenerates both networks' saved outputs
for every probe. Window arithmetic is audited separately; it is not reproduced
by retraining. The continuing activation shows the constraint remains active,
not that the historical unrestricted control had these same diagnostic values.

Full-world Q* error along greedy rollouts increases from **0.18958 to
0.52994**, despite the efficiency benefit. The arms visit different states and
now use different backup operators; this is not a fixed-data calibration
comparison. Better routes do not imply more accurate full-world values.

## Decision and next bounded control

Use constrained recorded-action DDQN as the next offline baseline. Its route
efficiency benefit repeats across all banks and panels, with matched outcomes
and target counts. Task completion remains unreliable, and the constraint
changes the backup operator: useful but unlogged actions may be excluded.
Unobserved current-action outputs remain unsupervised, and fresh actions remain
unrestricted. This is not online-RL competence or evidence for adding memory.

**Next: compare these bootstrapped targets with exact targets computed only
from the logged transition graph.** Remaining time strictly decreases on each
recorded nonterminal edge, and each successor has logged actions. Backward
dynamic programming can therefore compute the exact value of the same
restricted action graph, using only recorded rewards, ends and successors.
Missing actions stay excluded; full-world Q* and oracle outcomes do not enter
training.

Keep current-state supports/masks, recorded data, exact replay, network,
initialization, per-state loss, action-target presentations and update budget
fixed. Reuse these frozen constrained policies as controls on new common
panels. Compare fit and action ranking against the logged-graph values as
separate diagnostics. Static target preparation replaces evolving neural
successor queries, so query/compute counts will differ and must be reported
separately rather than claimed equal.

This holds the restricted Bellman operator fixed while testing difficulty
from moving approximate targets. Exact logged-graph values are not optimal
values for the full world, and this comparison alone cannot assign a remaining
fresh-map gap solely to optimization or coverage. It requires its own protocol
and has not run. Memory and continuous online collection remain deferred.

## Watch and reproduce

The dashboard adds **304 preselected recordings**, preserving **2,010 older
recordings**, plus training activation/gap/signed-target curves. Rules remain
visible initially. The first predetermined map, **1100000**, has mixed outcomes: default bank 1 / seed 0 changes from success in 12 steps to failure at
32; seed 1 changes from failure at 32 to success in 14. Bank 2 / seed 1 improves
from 14 to six. No favorable example replaces that first map.

The main run took **174.53 seconds (2.91 minutes)** on one CPU thread at reduced
priority, peaking at **579,174,400 bytes (0.539 GiB)**. Independent audit passed
all **304 recordings / 5,575 steps**, including regenerated learned outputs,
without retraining or repeating the complete policy evaluation. See validation
for software checks and the remaining pixel/responsive-QA limitation.

Use a fresh output directory with Python 3.12, Torch 2.8.0 and NumPy 2.0.2:

```bash
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 python -m q6.constrained_bootstrap \
  --output experiments/constrained_bootstrap/my-reproduction \
  --protocol-file docs/experiments/constrained_bootstrap_protocol_v1.md
python scripts/audit_constrained_bootstrap.py --study experiments/constrained_bootstrap/my-reproduction
```

Smoke uses short treatments, archived 30,000-update controls and alternate
panels; it validates execution only. License remains undecided; Q6 remains
a public research preview.
