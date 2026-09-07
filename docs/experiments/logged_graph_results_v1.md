# Exact targets on the recorded graph — results v1

**More accurate recorded-graph values produced much worse fresh behavior.**
State-balanced value MAE falls **0.2473 → 0.1134**, but greedy efficient
success falls **48.70% → 18.64%** and success **68.84% → 44.36%**.
Both behavior metrics decline in all 24 bank-panel means and all 72
bank/panel/learner cells. Exact targets are not automatically better policy
training targets under this limited recorded experience.

The recorded data, action masks, exact replay, network, loss and **24,273,701
action-target presentations** stay fixed. Only evolving constrained-DDQN
labels become fixed exact values of the same recorded transition graph.
Those values exclude unrecorded actions and are not full-world optimal Q*.
Fresh policies still choose freely among all four outputs.

The [protocol](logged_graph_protocol_v1.md) was independently reviewed and
pushed at `6c07198` before the single main run. Clean, pushed execution source
is `d5a88e20d3cbe63aea2589b3da394c10fc22824a`. No deviations, artifact
corrections or subsequent tuning runs occurred. This is a local declaration,
not external registration or peer review. See [raw results](../../experiments/logged_graph/pilot_v1/results.json),
[graph construction](../../experiments/logged_graph/pilot_v1/logged_graph.json),
[fit diagnostics](../../experiments/logged_graph/pilot_v1/fit_diagnostics.json),
[paired episodes](../../experiments/logged_graph/pilot_v1/paired_layouts.csv),
and [validation](../validation/logged-graph-v1.md). The
[dashboard](http://127.0.0.1:8080/dashboard/lab.html#coverage) defaults to
**Experience coverage → Exact logged graph**.

## What “exact” means here

Original supports contain **59,839 / 59,984 / 59,838 states** on training
maps 300000–300255. All **301,585 historical collector steps / 12,288 episodes**
and the recorded-action tables are reconstructed and verified against their
archives. Repeated visits do not increase a state/action pair's loss weight.
No new collection, support draws or unrecorded outcomes are introduced.

Every nonterminal logged successor remains in the same bank's support, on the
same map, with remaining time reduced by one and at least one logged action.
This permits backward dynamic programming:

```text
Q_log(s,a) = logged reward                              if ended
Q_log(s,a) = logged reward + 0.97 × max_logged Q_log(s') otherwise
```

Absent actions remain NaN and are excluded before maxima or loss operations.
Terminal edges never query a successor. Float64 reference values are frozen
before fitting; training uses their saved float32 casts. An independent solver
reproduces all labels using only recorded edges. Full-world Q* and exhaustive
outcomes remain separate integrity/rollout diagnostics and do not enter the
graph solver or optimizer.

| Bank | Supported states | Recorded edges | Terminal edges | Nonterminal graph backups |
| --- | ---: | ---: | ---: | ---: |
| 1 | 59,839 | 84,305 | 3,684 | 80,621 |
| 2 | 59,984 | 84,091 | 3,676 | 80,415 |
| 3 | 59,838 | 83,933 | 3,674 | 80,259 |

Construction covers **252,329 edges**, including **241,295 successor backups**,
over 32 clock levels per bank. Maximum float64 Bellman residual is zero in all
three saved graphs. Residual checking makes another 241,295 recorded successor
lookups, counted separately. These are static graph operations, not neural
optimizer queries. Preparation, casts and integrity work are measured separately
from fitting; the overall run includes their cost.

## Matched training, different query costs

The control is nine archived final **constrained_bootstrap** policies,
reevaluated without retraining. Nine new exact-label fits use matching initial
online/target weights, the same 20,420-parameter network, Adam settings,
64-state batches, gradient clipping and 30,000-update budget. The loss remains
mean SmoothL1 over each state's distinct recorded actions, then mean over
states. Target-network soft updates remain in place, but their predictions
are unused by fixed-label optimization.

| Measurement | Constrained control, historical | Exact logged graph, new |
| --- | ---: | ---: |
| Updates | 270,000 | 270,000 |
| State presentations | 17,280,000 | 17,280,000 |
| Recorded action targets | 24,273,701 | 24,273,701 |
| Terminal-labelled targets | 1,062,190 | 1,062,190 |
| Nonterminal-labelled targets | 23,211,511 | 23,211,511 |
| Optimizer neural successor queries | 23,211,511 | **0** |

All nine local/global/map replay streams and count vectors match exactly.
Target totals by bank are **8,115,408 / 8,078,271 / 8,080,022**. Every support
state is sampled and none outside support is directly sampled. Tables,
references, casts and optimizer tensors retain their hashes. Equal update and
target counts do not mean equal compute: graph preparation replaces repeated
neural target-query work; final-fit inference is separate diagnostic work. No speedup benchmark is
claimed from runs with different evaluation trajectories and machine load.

## Fit to the recorded graph

After all training, both arms are assessed on their own complete supported
states, including states with no successful path through the recorded graph.
The **18 final assessments** preserve all four online predictions for
**1,077,966 state presentations / 1,513,974 observed-edge errors**. Inference
uses 1,062 batches of at most 1,024 states, takes **4.34 seconds**, and changes
no weights, RNG or optimizer state. There are no new checkpoint probes, fit
gates, model selection or further training.

Each condition has 538,983 state and 756,987 observed-edge presentations.
State metrics average over logged actions within each state, then states;
the displayed pool weights model summaries by their state counts. Edge MAE
uses its distinct edge denominator; maxima range over all assessed models.

| Final recorded-graph metric | Constrained control | Exact-label treatment |
| --- | ---: | ---: |
| State-balanced MAE | 0.247286 | **0.113436** |
| State-balanced MSE | 0.102807 | **0.030898** |
| Edge-pooled MAE | 0.251011 | **0.114839** |
| Maximum absolute observed-action error | 1.314093 | 1.250497 |
| Restricted action agreement | **97.51%** | 97.05% |
| Mean graph regret | 0.009657 | 0.009545 |
| Unrestricted argmax outside logged set | 65.16% | 65.04% |

MAE improves **54.13%** and MSE **69.95%**; every bank/seed improves numerical
fit. Edge-pooled error confirms this is not a state-weighting artifact. The
network still does not reproduce graph values exactly, and improved value
error does not improve restricted action agreement. Graph regret changes little.

Agreement selects the lowest-label predicted argmax among recorded actions,
then checks its graph value against the graph maximum with absolute tolerance
1e-6 and zero relative tolerance. **123,538 of 179,661 supported states
(68.76%) have only one recorded action**, making their restricted agreement
automatic. High agreement is therefore not evidence of near-perfect decision
making. The unrestricted argmax still chooses an unrecorded action on about
65% of familiar supported states in either arm. These choices have no
recorded-graph target; restricted fit measures only part of the deployed choice.
Similar outside-set frequency alone cannot explain the behavior difference.

## Behavior on new common panels

Eight prospective 64-map panels scan from `1120000 + 1000 × panel`.
Fourteen prescribed collisions are skipped, including two training layouts
and one already selected layout. Original spawns are retained and selection
uses no outcomes. All fits and graph-fit diagnostics finish before policy
rollouts. Final policies are evaluated on **512 accepted layouts**, yielding
**27,648 learner + 3,584 shared reference episodes = 31,232**.

Efficient success means reaching the goal within twice its shortest-path
length, including failures. Each greedy bank/condition has 1,536 episodes.
Cells show constrained control / exact-label treatment.

| Bank | Success | Efficient success | Efficiency Δ | Mean steps, all episodes |
| --- | ---: | ---: | ---: | ---: |
| 1 | 70.51% / 42.90% | 49.93% / 17.97% | −31.97 pp | 14.58 / 23.44 |
| 2 | 68.29% / 44.86% | 47.20% / 19.92% | −27.28 pp | 15.13 / 23.02 |
| 3 | 67.71% / 45.31% | 48.96% / 18.03% | −30.92 pp | 15.03 / 23.35 |

Equal-bank averages equal episode pooling for these equal-sized groups:

| Greedy outcome | Constrained control | Exact-label treatment |
| --- | ---: | ---: |
| Success | 68.84% (3,172/4,608) | 44.36% (2,044/4,608) |
| Efficient success | 48.70% (2,244/4,608) | 18.64% (859/4,608) |
| Mean steps, all episodes | 14.91 | 23.27 |
| Mean steps, successes only | 7.18 | 12.32 |
| Blocked steps / total steps | 72.14% (49,580/68,725) | 70.72% (75,829/107,231) |

There are **1,128 fewer successes and 1,385 fewer efficient successes**, with
**8.36 more steps** per episode. Successful-only means compare different
successful subsets. The pooled blocked-step share decreases **1.427 points**,
but absolute blocked steps increase by 26,249. Mean bank ratios decrease
1.417 points and mean bank/learner ratios 1.450; these denominators remain
separate. A lower blocked-step fraction does not establish better routes.

| Panel | Bank 1 efficiency Δ | Bank 2 efficiency Δ | Bank 3 efficiency Δ |
| --- | ---: | ---: | ---: |
| 0 | −34.38 pp | −18.75 pp | −23.44 pp |
| 1 | −29.17 pp | −22.40 pp | −29.69 pp |
| 2 | −35.94 pp | −33.33 pp | −36.46 pp |
| 3 | −27.08 pp | −29.69 pp | −32.29 pp |
| 4 | −31.25 pp | −38.02 pp | −31.77 pp |
| 5 | −37.50 pp | −27.60 pp | −29.69 pp |
| 6 | −32.81 pp | −31.77 pp | −32.29 pp |
| 7 | −27.60 pp | −16.67 pp | −31.77 pp |

Both success and efficiency decline in **24/24 bank-panel means**,
**72/72 learner-panel cells**, and **9/9 whole-panel learner comparisons**.
There are no ties or improvements at those levels. Efficiency effects across
whole-panel learners range from −32.42 to −25.39 points. Shared banks, maps,
initializations and panels mean these are not 72 independent replications.

Historical all-learner success-reference crossings occur on **2/8, 2/8,
1/8** control panels and zero treatment panels. Neither arm reaches the
all-learner 80% efficient-success reference on any panel. These are descriptive
references, not new gates. Separate epsilon-0.1 success / efficiency is
**77.07% / 49.70%** control and **53.58% / 19.90%** treatment.
Shared random success is **43.46% (1,335/3,072)**; the planner succeeds on all
512 maps, averaging **3.61 steps**. Other historical policies were not evaluated
on these panels and cannot supply a matched ranking here.

## Decision and next bounded diagnostic

Retain constrained recorded-action DDQN as the stronger behavioral baseline.
Keep the exact graph as a useful reference and preserve this negative result.
Moving approximate labels were not an obstacle whose removal improved fresh
behavior in this experiment. This does not establish why they helped, prove
implicit regularization, or show that exact labels generally harm learning.
The graph omits actions, numerical fit remains imperfect, and deployed choice
uses outputs that did not receive direct supervision.

**Next: evaluate frozen policies on familiar starts, with and without logged
action masks, before another training experiment.** Use every original
training map/spawn at clock 32, shared within each bank/seed comparison and
chosen without outcome filtering. Both frozen arms receive unrestricted
all-four greedy evaluation and logged-mask greedy evaluation. The masked
trajectory must stay inside the closed graph; record off-mask choices while
supported and first support exit separately for unrestricted trajectories.

Include an exact logged-graph policy reference to show what its recorded
actions permit. That policy is optimal for the declared graph return; do not
call it a universal success ceiling without separately verifying reachability.
Keep this familiar-state diagnostic separate from the current fresh-map scores.
The mask is privileged familiar-state diagnostic access and cannot be supplied
to fresh-world policies as an assumed deployment improvement.

A disproportionate rescue of exact-label policies would support a mismatch
between supervised actions and unrestricted choice. Weak masked behavior
would direct attention to graph policy/ranking limitations before blaming
fresh-map transfer. Either outcome would narrow the next training question
without collecting data or fitting another network. It would not uniquely
identify the cause of every fresh-map failure. This diagnostic has not run
and requires its own protocol. Memory and online collection remain deferred.

## Watch and reproduce

The dashboard adds **304 preselected recordings**, preserving **2,314 prior
recordings**, and shows graph preparation, final fit, and distinct label/query
budgets. Rules remain visible initially. On first predetermined map **1120000**,
bank 1 / seed 0 changes from success in 12 steps to failure at 32; seed 1 changes
from success in three to failure at 32. The same map also contains exceptions:
bank 2 / seed 0 changes from failure at 32 to success in 19, and bank 3 / seed 2
from failure at 32 to success in three. No example was selected by outcome.

The run takes **184.09 seconds (3.07 minutes)** on one CPU thread at reduced
priority, peaking at **613,695,488 bytes (0.572 GiB)**. Independent audit passes
all 18 sets of saved support predictions and **304 recordings / 5,336 steps** without
retraining or repeating complete policy evaluation. See validation for test
coverage and the pixel/responsive-QA limitation.

Use a fresh directory with Python 3.12, Torch 2.8.0 and NumPy 2.0.2:

```bash
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 python -m q6.logged_graph \
  --output experiments/logged_graph/my-reproduction \
  --protocol-file docs/experiments/logged_graph_protocol_v1.md
python scripts/audit_logged_graph.py --study experiments/logged_graph/my-reproduction
```

Smoke uses short treatments, archived 30,000-update controls and alternate
panels; it validates execution only. License remains undecided; Q6 remains
a public research preview.
