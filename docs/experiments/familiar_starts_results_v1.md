# Familiar starts: restricting choices does not repair the regression

**Keep constrained DDQN. The exact-label regression appears on familiar maps
and survives restriction to logged actions.** Masking improves the exact
model's success rate, but does not recover efficient routes. It also removes
useful choices from constrained DDQN.

This [locally declared diagnostic](familiar_starts_protocol_v1.md) evaluates
18 frozen final policies from the [exact logged-graph study](logged_graph_results_v1.md)
on all 256 original training maps/spawns, with 32 moves remaining. Each policy
chooses either among all four actions or only actions recorded at its current
state. Three banks × three learner seeds × two models × two action sets give
9,216 learner episodes. No weights, training data or training procedure change.
The bank-specific masks are privileged familiar-state diagnostic information.

## Same starts, four outcomes

Each cell contains **2,304 episodes**. Efficient success means reaching the
goal within twice the full-world shortest route, with failures included.

| Frozen model | Action choices | Success | Efficient success | Mean steps, all episodes |
| --- | --- | ---: | ---: | ---: |
| Constrained DDQN | All four | 86.72% (1,998) | **68.49% (1,578)** | 9.51 |
| Constrained DDQN | Logged only | 82.20% (1,894) | 40.76% (939) | 13.98 |
| Exact logged labels | All four | 66.93% (1,542) | 25.82% (595) | 19.08 |
| Exact logged labels | Logged only | 75.61% (1,742) | 24.05% (554) | 17.78 |

With unrestricted choices, exact-minus-DDQN differences are **−19.79 success
points and −42.66 efficiency points**. With logged choices, they remain
**−6.60 and −16.71 points**. All three bank means and all nine whole-population
learner comparisons decline in both metrics under either action set.

The primary efficiency interaction—exact's masking effect minus DDQN's—is
**+25.95 points**. This is a relative difference, **not an efficiency rescue**:
masking costs DDQN **27.73 points**, versus **1.78 points** for exact labels.
Exact success improves by 8.68 points under masking, while DDQN success loses
4.51 points. Report the absolute effects alongside the interaction.

| Bank | DDQN efficiency: all → logged | Exact efficiency: all → logged | Efficiency interaction |
| --- | ---: | ---: | ---: |
| 1 | 70.96% → 38.67% | 25.78% → 23.44% | +29.95 pp |
| 2 | 68.75% → 40.49% | 28.39% → 25.65% | +25.52 pp |
| 3 | 65.76% → 43.10% | 23.31% → 23.05% | +22.40 pp |

Each bank/cell has 768 episodes. The interaction is positive in all 24
bank-by-32-map-block means and 71/72 learner-block cells; one is negative.
Exact efficiency trails DDQN in all 24 block means under both action sets;
the corresponding learner-block counts are 72/72 unrestricted declines and
70 declines, one gain, one tie with masks. Blocks are disjoint within a
bank/seed; maps are reused across banks and learner seeds. These descriptive
cells are not independent trials.

## What the logs can and cannot support

An independent backward reachability calculation finds a successful logged
route on **744/768 bank/start pairs (96.875%)**. Only **468/768 (60.9375%)**
contain a logged route meeting the full-world efficiency threshold.

| Bank | Logged success ceiling | Logged efficient-success ceiling |
| --- | ---: | ---: |
| 1 | 248/256 = 96.88% | 151/256 = 58.98% |
| 2 | 249/256 = 97.27% | 153/256 = 59.77% |
| 3 | 247/256 = 96.48% | 164/256 = 64.06% |

The 768 exact recorded-Q reference episodes attain both ceilings on these
starts. This agreement is checked; return optimality was not assumed to imply
success optimality. Their mean length is 9.05 steps. The 256 shared full-world
planner episodes all succeed, averaging 3.73 steps.

Unrestricted DDQN's **68.49% efficient success exceeds the 60.94% ceiling of
logged-only routes**. Departures from the logs therefore sometimes provide
useful shortcuts. Missing recorded outcomes do not make an action intrinsically
bad. At the same time, both masked networks fall substantially below what the
existing graph permits: there is also learned decision-making headroom.

## Investigating the regression

**Novel maps are not required.** These are the original training starts, and
the exact model still loses efficiency under both action sets. This does not
quantify how much of the earlier fresh-map gap each mechanism explains; those
512 fresh maps remain separate evidence and were not evaluated again.

**Off-mask selection is only part of the problem.** Unrestricted DDQN leaves
support in 1,291/2,304 episodes (56.03%); exact does so in 1,743 (75.65%).
Off-mask choices occur on 7,227/16,273 supported DDQN decisions (44.41%) and
15,496/30,562 exact decisions (50.70%). Off-support occupancy is 25.76% versus
30.47% of their respective steps. These are different visitation distributions,
not matched-state error comparisons. An off-mask choice can land within support;
successful termination is not an exit. Every masked trajectory has zero exits,
zero off-mask choices and a verified logged outcome at every step.

Each masked arm has 72 episodes whose original start has no successful logged
path. Its remaining failures occur after losing an available path through a
recorded choice: **338 for DDQN versus 490 for exact labels**. Initial ranking
mistakes need not be immediately fatal; later decisions and detours also matter.

**The regression includes choices that were supervised.** Archived predictions
on original clock-32 states have logged-action ranking agreement **89.28% →
79.43%**, despite value MAE **0.2392 → 0.1156**. Each arm contributes 2,304
state presentations, all with multiple logged choices. Initial unrestricted
argmax is outside the mask only **1.52% → 1.09%** of the time. This original-start
slice was added after inspecting existing predictions and before this rollout;
it is explicitly exploratory prior-data analysis, not a new unseen-data test.

Across all supported states, about 69% have one logged action, whose restricted
agreement is automatic. On the 168,369 multiple-action presentations per arm,
agreement falls **92.03% → 90.55%**. Full-support averages hide decision-relevant
differences at the start of an episode.

**Better value levels are not necessarily better action ordering.** Decompose
each state's mean squared observed-action error into its squared mean error
and the residual after subtracting that state's mean error. The reduction of
the first component accounts for **99.56% of the overall MSE improvement**.
This is an arithmetic decomposition of squared error, not of MAE, and not a
claim that one global constant explains the models. At original starts,
centered MAE slightly worsens **0.06466 → 0.06613**, and centered MSE worsens
**0.008653 → 0.008855**, while signed bias improves **−0.20350 → −0.03724**.
Correcting value levels explains why numerical fit can improve without better
decisions; it does not uniquely explain the training dynamics that caused them.

The [independent audit](../../scripts/audit_familiar_starts.py) found no source,
reward, transition, label-recurrence or frozen-weight mismatch. Exact labels
solve the same recorded-action Bellman problem used by constrained DDQN.
These checks support a real behavioral regression; they do not prove that
every possible implementation issue has been excluded.

## Watch a two-step task become a detour

Open **Experience coverage → Familiar starts**, bank 1 / seed 0, block 0.
On preselected map **300000**, the goal is two moves away. DDQN succeeds in two
steps with either action set. The exact model takes **10 steps unrestricted
and 15 with a mask**. At the initial state all four actions are logged, yet it
chooses down while the logged reference favors up or left; the mask cannot
correct that ranking mistake. Bank 3 / seed 2 also preserves a success rescue:
the exact model changes from failure at 32 to success in four masked steps.

All eight replay maps were selected before outcomes. The dashboard contains
**320 new recordings and 2,938 total**, with prior studies and initially visible
movement rules preserved. The interaction cards show both absolute masking
effects so a positive interaction cannot be confused with an absolute gain.

## Decision and next bounded experiment

**Retain constrained DDQN with unrestricted evaluation. Change collection next,
while keeping its learning procedure fixed.** Compare the uniform-random
collector with one declared mixture of random and frozen-DDQN-guided complete
episodes, across three collection draws on the same training maps. Use equal
episode allocations per map, report realized interaction counts separately,
and disclose the frozen collector's historical training cost. Fix the mixture
and collector identity before collection; no sweep or outcome-based selection.

Measure logged shortest successful routes and the efficient-route ceiling,
state/action composition, then train the same DDQN network under the same
update budget and evaluate on common new fresh panels. This tests the total
effect of more goal-directed experience; route length, coverage and realized
steps may change together. A higher graph ceiling alone is not success: the
learned policy must benefit. The current ranking gaps remain useful diagnostics,
but tuning a new loss mainly to rescue the weaker exact-label arm is a less
direct next step than improving the baseline's experience. This next experiment
has not run. Continuous online feedback and memory remain deferred.

## Reproduce and inspect

Protocol commits: `fc89597`, then the exploratory-slice clarification `48de7e8`.
Clean pushed execution source: `77ebcb968bab142eb5b4e8b0470bd6ef89e38f7a`.
One main run: **12.504 seconds**, **448,446,464 bytes (0.418 GiB)** sampled
process peak RSS, one CPU thread at nice priority 10, 148,906 resource checks.
It preserves **146,963 raw step records**, 18 frozen snapshots and 180
prior-prediction slices, with zero updates, new collection or fitted labels.
This is one observed local run, not a cross-machine performance claim.

```bash
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 \
  python -m q6.familiar_starts \
  --output /tmp/q6-familiar-reproduction \
  --protocol-file docs/experiments/familiar_starts_protocol_v1.md

python scripts/audit_familiar_starts.py --study /tmp/q6-familiar-reproduction
# Portable audit: add --skip-forward-inference.
# Execution-only smoke: add --smoke to the runner and --allow-smoke to its audit.
```

Use Python 3.12, Torch 2.8.0 and NumPy 2.0.2; the run uses Python 3.12.13.
Uncommitted source or changed runtime/budget is recorded as a deviation, so
use a clean checkout for eligible reproduction. The [validation record](../validation/familiar-starts-v1.md)
separates tests from evidence audit. [Raw results](../../experiments/familiar_starts/pilot_v1/results.json),
[episodes](../../experiments/familiar_starts/pilot_v1/episodes.json),
[paired comparisons](../../experiments/familiar_starts/pilot_v1/paired_differences.json)
and the [manifest](../../experiments/familiar_starts/pilot_v1/manifest.json)
preserve the evidence. License remains undecided; PR #1 remains open.
