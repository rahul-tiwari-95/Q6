# Exact-target supervision — result v1

**The unchanged network learned useful fresh-layout behavior under exact
supervision.** Final greedy success was **82.81%, 85.94%, and 87.50%** for
seeds 0, 1 and 2. Every seed exceeded the declared 70% supervised fresh-layout
threshold and the random reference. The separate, stricter training-fit
diagnostic remained unmet. This establishes useful capacity under the supplied
targets and coverage; it does not establish an online-RL solution or identify
a single cause of the earlier RL failure.

The [protocol](supervised_protocol_v1.md) was committed before main training,
locally declared rather than externally registered. One run completed all
90,000 optimizer updates and 15 scheduled seed/checkpoint evaluations, with no
protocol deviations. Reported total time was **80.13 seconds**, within the
900-second admission cap. No extra training or checkpoint selection followed.

## Primary outcomes

The 20,420-parameter network, 92-value observation, world, rewards and three
initial weight states match the preceding competence study. Training used
30,000 Adam updates per seed, 64 states per batch, and fixed exact Q* targets
for all four actions. No demonstrations or oracle actions were supplied during
the network's rollouts.

| Seed | Training-bank greedy success | Training-state optimal-action agreement | Fresh-layout greedy success | Training-fit diagnostic | Supervised fresh gate |
| --- | ---: | ---: | ---: | --- | --- |
| 0 | 89.45% | 85.71% | 82.81% | Not met | Passed |
| 1 | 90.63% | 83.55% | 85.94% | Not met | Passed |
| 2 | 92.19% | 85.39% | 87.50% | Not met | Passed |

The training-fit diagnostic requires both 90% training-bank rollout success
and 90% optimal-action agreement over winnable training states, in **every**
seed. All seeds fall short on action agreement; seed 0 also falls short on
rollout success. The fresh gate is independent of that diagnostic. Passing
it does not imply exact value fitting or optimal routes.

| Policy on the same 64-layout evaluation panel | Greedy success | Epsilon-0.1 success |
| --- | ---: | ---: |
| Exact-supervised network | 85.42% | 85.94% |
| Historical frozen RL stream policies | 21.35% | 31.77% |
| Uniform random actions | 44.27% reference | — |
| Shortest-path controller | 100% reference | — |

Learned-policy greedy scores pool 192 episodes (64 layouts × three seeds).
Epsilon and random scores use 384 episodes; the deterministic planner uses
64. The historical networks received their original 120,000 RL transitions
per seed, and no additional training. They are **not matched for target access,
state coverage, training data or computation**. Their earlier 19.79% result
was on a different evaluation panel; it is not used as this comparison's score.

The historical policies had previously encountered the wall-and-goal layout
of **3, 3 and 4** of these 64 cases, respectively, possibly with different
starts. This overlap is preserved and reported; fresh selection was not
conditioned on it. All 64 layouts are unseen by the new supervised bank.

## Prediction quality and inefficient behavior

| Exhaustive state panel, pooled across seeds | Four-action MAE | RMSE | 95th percentile of per-state MAE | Optimal actions among winnable states |
| --- | ---: | ---: | ---: | ---: |
| Training | 0.0333 | 0.0783 | 0.1419 | 84.89% of 452,286 states |
| Fresh | 0.0395 | 0.0985 | 0.1791 | 79.29% of 112,581 states |

These denominators count state evaluations across three networks, not three
independent datasets. Impossible states remain in value-error metrics but are
excluded from action agreement. Their action values can all tie trivially.
The corresponding winnable-state mean action regrets are 0.00833 and 0.01417.
Remaining-time buckets and per-map records are preserved; low mean error does
not guarantee the correct ranking between close action values.

**Successful collection is not yet efficient navigation.** Across all fresh
greedy episodes, the supervised policy took **14.31 steps on average**, versus
**3.67** for the planner. **62.61% of its visited steps were blocked actions.**
That statistic weights long episodes more heavily and measures policy-visited
states, unlike the exhaustive uniform-state panel above. Optimal-action
agreement along the still-winnable portions of these rollouts was only 29.63%.

A post-hoc breakdown of the saved CSV, without new rollouts, confirms that
this is not solely a consequence of failed episodes: the 164 successful fresh
episodes averaged **11.29 steps**, with **975 / 1,851 steps (52.67%)** blocked.
The 28 failed episodes lasted 32 steps each. This breakdown was not a declared
primary outcome and does not establish why the policy wastes steps.

## Coverage, budget and evidence

The training bank contains 256 distinct wall-and-goal layouts, seeds
300000–300255. Enumerating 20 agent positions and 32 remaining times gives
**163,840 hypothetical valid states**. Some are unreachable from the original
sampled start at that elapsed time. The 64 fresh layouts, seeds 930000–930063,
provide 40,960 states. No candidate required replacement, and both layout
and complete-observation intersections are zero.

Each learner received **1,920,000 state presentations**, approximately 11.72
dataset-equivalent passes, with all four exact targets per presentation.
All 163,840 training rows were sampled at least once by every seed. These
5,760,000 total presentations are supervised examples, not RL interactions.
Three initializations share one bank and one panel; ranges are descriptive,
not confidence intervals or independent task-bank replications.

The main run used clean source revision
`83bf7e66db1da7f536b4b37f5d3484d01aa6ed0b`, Python 3.12.13, Torch 2.8.0,
NumPy 2.0.2 and one CPU thread on macOS arm64. Dataset/reference preparation
took 10.49 seconds, optimization 17.50 seconds, learner evaluation 44.96 seconds,
and reference rollouts 6.12 seconds. These machine-specific observations are
not a simulator throughput benchmark.

The [artifact directory](../../experiments/supervised/pilot_v1/) contains
14,400 learner rollout rows, 24,000 map/time-bucket state-metric rows, 2,816
unique reference episodes, 900 loss windows, 70 saved recordings, 15 supervised
checkpoints and three copied historical checkpoints. Dataset arrays, final
dense predictions, sample counts/digests, source snapshots, historical
provenance and a [45-file manifest](../../experiments/supervised/pilot_v1/manifest.json)
are included. Inference checkpoints are not resumable optimizer snapshots.

```bash
# From a Python 3.12 checkout environment; choose a new output directory.
python -m pip install 'torch==2.8.0' 'numpy==2.0.2'
python -m q6.supervised --output experiments/supervised/my-reproduction \
  --protocol-file docs/experiments/supervised_protocol_v1.md

# Inspect shipped evidence without training or policy rollouts.
python scripts/verify_pilot_artifacts.py
python scripts/audit_supervised_study.py
```

The [validation record](../validation/supervised-v1.md) distinguishes software
checks, artifact recomputation and browser coverage from the scientific result.

## Decision after this milestone

Prioritize the **learning procedure and data coverage** before increasing
network complexity. Useful fresh-layout behavior is possible with the current
representation, although its value fit and routes remain imperfect. Exact
supervision and broad coverage changed together, so neither should receive
sole credit for the improvement.

The next bounded comparison should hold a fixed transition dataset, sampled
state-action pairs, network, initializations, loss reduction and update budget
constant, and compare **exact Q* targets with bootstrapped one-step targets**.
Use the same all-four-action loss in both conditions, so action coverage does
not change with the target method. Evaluate both on a newly declared panel.
If bootstrapped fitting also works with fixed coverage, investigate online
collection/replay next. If it fails while exact-target fitting works, focus
on the target/optimization procedure under that controlled dataset. Such an
offline comparison would still not establish online-RL competence. It has
not run here. Keep memory and A → B → A extensions paused until that separate
competence requirement is met.
