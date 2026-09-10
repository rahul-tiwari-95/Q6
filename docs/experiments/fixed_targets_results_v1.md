# Fixed data, exact versus bootstrapped targets — result v1

**Both target procedures learned useful fresh-layout behavior under the same
offline coverage.** Final greedy success was **84.38% with exact targets** and
**86.46% with Double DQN targets**. Every seed in both conditions passed the
declared 70% fresh-success gate and exceeded the shared random reference.
Bootstrapping is therefore viable in this setting; online collection and
replay coverage are the next diagnostic priority. This does not establish
online-RL competence or a general advantage for one target method.

Neither condition passed the every-seed training-fit or efficient-success
diagnostic. Double DQN produced shorter episodes and more efficient successes,
while retaining substantial wasted moves. The locally declared
[protocol](fixed_targets_protocol_v1.md) was committed before execution.
All 180,000 updates and 30 scheduled seed/condition/checkpoint groups completed
in **145.67 seconds**, with **579.6 MB peak process RSS (0.54 GiB)** and no
protocol deviations. No additional training or checkpoint selection followed.

## Matched conditions and primary outcome

Both conditions used the same 20,420-parameter network, observations, world,
rewards, three initializations, 163,840 training states, ordered minibatches,
all four action labels, Smooth L1 reduction, Adam optimizer and 30,000 updates
per seed. The sole intended procedural change was fixed exact Q* labels
versus detached one-step Double DQN labels with post-update soft target refresh
at tau 0.01. Equal update counts do not mean equal computation.

| Seed | Exact fresh success | Double DQN fresh success | Paired difference, DDQN − exact | Exact efficient success | DDQN efficient success |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 87.50% | 87.50% | 0.00 pp | 45.31% | 64.06% |
| 1 | 81.25% | 82.81% | +1.56 pp | 34.38% | 67.19% |
| 2 | 84.38% | 89.06% | +4.69 pp | 37.50% | 67.19% |
| Mean / pooled equal-size panels | **84.38%** | **86.46%** | **+2.08 pp** | **39.06%** | **66.15%** |

Each seed uses the same 64 fresh starts. Each learned-policy pooled greedy
score covers 192 episodes: exact succeeded in 162, DDQN in 166. The small
four-episode difference does not establish superiority or equivalence. Three
initializations share a bank and panel; seed ranges are descriptive, not
confidence intervals or independent task-bank replications.

The fresh random reference succeeds in **41.15%** of 384 episodes; shortest
path succeeds in **100%** of 64 episodes. Epsilon-0.1 evaluation gives exact
**84.11%** and DDQN **89.58%**, each over 384 episodes. These are secondary
measurements; the declared gates use final greedy behavior only. Historical
RL scores from earlier panels are not comparison arms here.

## Route efficiency and incomplete training fit

Efficient success means an episode both collects and uses at most twice
**its own layout's shortest-path length**. All fresh episodes, including
failures, are in the denominator. The declared secondary criterion is 80%
in every seed. DDQN's 66.15% exceeds exact's 39.06% by **27.08 percentage
points**, with improvement in each seed, but both fail this criterion.

| Final fresh greedy behavior | Exact targets | Double DQN targets | Shortest path |
| --- | ---: | ---: | ---: |
| Mean steps, all episodes | 17.13 | 11.32 | 3.75 |
| Mean steps, successful episodes only | 14.38 | 8.08 | 3.75 |
| Blocked steps / all visited steps | 2,149 / 3,289 (65.34%) | 1,463 / 2,174 (67.30%) | 0% |
| Blocked fraction within successful episodes | 58.78% | 52.98% | 0% |

The successful subsets differ, so their mean lengths alone are not a paired
route comparison. The all-episode paired step reduction is **5.81 steps**,
and the per-layout efficient-success definition supplies a common criterion.
DDQN reduces absolute blocked steps while its overall blocked **fraction is
higher**. Long looping failures and denominator changes matter; do not describe
this as a blanket improvement in blocked-action rate. The dashboard's paired
no-op delta averages per-seed fractions, which differs from pooling all steps.

| Seed | Exact training success / winnable-state agreement | Exact fit | DDQN training success / winnable-state agreement | DDQN fit |
| --- | ---: | --- | ---: | --- |
| 0 | 89.45% / 85.71% | Not met | 90.23% / 89.82% | Not met |
| 1 | 90.63% / 83.55% | Not met | 94.53% / 90.77% | Met |
| 2 | 92.19% / 85.39% | Not met | 93.36% / 90.62% | Met |

Training fit requires both quantities to reach 90% in every seed. DDQN's pooled
training agreement is 90.40%, but seed 0 remains below threshold. Pooling must
not conceal that missed all-seed gate. Exact fails on agreement in every seed.

## Value prediction and action ranking

| Exhaustive panel, final predictions pooled across seeds | Four-action MAE | RMSE | 95th percentile of per-state MAE | Winnable optimal-action agreement |
| --- | ---: | ---: | ---: | ---: |
| Exact, training | 0.0333 | 0.0783 | 0.1419 | 84.89% |
| DDQN, training | 0.0694 | 0.1184 | 0.2735 | 90.40% |
| Exact, fresh | 0.0387 | 0.0955 | 0.1736 | 79.57% |
| DDQN, fresh | 0.0731 | 0.1245 | 0.2852 | 82.63% |

Value accuracy and action ranking point in different directions: exact targets
produce lower error to Q*, while DDQN achieves higher optimal-action agreement
and shorter episodes here. The fresh winnable-state action regrets are 0.01388
and 0.01195, respectively. Fresh signed Q bias is −0.00066 for exact and
−0.02271 for DDQN. These observations do not establish why the target procedure
changes action ranking.

The pooled panels contain 491,520 training and 122,880 fresh state evaluations
across three networks; their winnable denominators are 452,286 and 112,392.
Impossible states remain in regression error but are excluded from optimal
action agreement. Uniform exhaustive states and policy-visited rollout states
have different distributions. Reported training loss is pre-update minibatch
Smooth L1; DDQN's labels move, so it is not a common ruler of error to Q*.

## Integrity, resources and reproduction

All training-array hashes match supervised v1. All three exact-arm final
online weight hashes and sampling digests reproduce that archived run exactly.
Both new arms have identical initial weights, ordered batch digests and
per-row exposure counts within each seed. This is an execution-consistency
check, not independent scientific replication.

The new fresh panel is seeds **940000–940063**. No candidate needed replacement;
training/fresh layout and observation overlap, and overlap with the preceding
supervised fresh panel, are zero. Exact's prior 85.42% fresh score came from
the 930000 panel; its current 84.38% score is on this new panel despite
identical final weights. The training bank has already informed research;
the primary test panel was selected independently of outcomes.

There are **655,360 saved training transitions**, indexed compactly into the
same bank for nonterminal successors. They cover hypothetical valid states
and all four actions. This exhaustive offline access exceeds what a standard
trajectory-only learner observes. Total training consumed 11,520,000 state
presentations / 46,080,000 action targets, not that many world interactions.

The clean training source is `14bb9571921ba5fc55c7a2a8bb2f7c92d52416be`, with
Python 3.12.13, Torch 2.8.0, NumPy 2.0.2 and one CPU thread. Runs were sequential
at niceness 10. Dataset/transition preparation took **11.24s**, exact-arm
optimization **17.53s**, DDQN optimization **29.92s**, learner evaluation
**80.54s**, and shared reference rollouts **4.46s**. The runner performed
200,551 resource checks and stayed below its sampled 4 GiB RSS guard and
1,200-second admission cap. These are machine-specific costs, not a simulator
throughput benchmark or a measure of total system memory.

The [artifact archive](../../experiments/fixed_targets/pilot_v1/) includes a
[65-file manifest](../../experiments/fixed_targets/pilot_v1/manifest.json),
30 online/target inference snapshots, six final prediction files, datasets,
transitions, provenance, sampling counts/digests, 28,800 learner episode rows,
48,000 state-metric rows, 2,240 unique reference episodes, 1,800 loss windows,
960 paired layout rows and 124 recordings. The checkpoints are not resumable
optimizer snapshots. The dashboard export byte-matches the saved result.

```bash
# Python 3.12 checkout environment; choose a new output directory.
python -m pip install 'torch==2.8.0' 'numpy==2.0.2'
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 python -m q6.fixed_targets \
  --output experiments/fixed_targets/my-reproduction \
  --protocol-file docs/experiments/fixed_targets_protocol_v1.md

# Read-only verification; no training or policy rollouts.
python scripts/verify_pilot_artifacts.py
python scripts/audit_fixed_targets_study.py
node scripts/check_dashboard.mjs
```

The [validation record](../validation/fixed-targets-v1.md) documents independent
artifact recomputation, test results, CI and the native-browser review limit.

## Decision after this milestone

Prioritize **state coverage and the online collection/replay procedure**.
Double DQN can learn useful fresh-layout behavior with this network when
supplied broad, fixed coverage. This weakens the explanation that bootstrapped
targets alone prevent learning in Q6. It leaves online distribution, replay,
action coverage and feedback between policy and collected data unresolved.

The next bounded comparison should bridge these settings carefully: collect
a fixed bank of states with a declared exploratory trajectory policy, then
compare it with the exhaustive state bank using the **same DDQN all-action
update** and matched optimization budget. Supplying all four transitions at
visited states keeps four-action coverage per sampled state fixed while testing
state coverage; that
counterfactual transition access must remain explicit. If visited-state replay
works offline, test the online collection/update feedback next. If it fails
while broad coverage works, diagnose coverage before changing memory or network
architecture. Declare the collection budget, fresh panel and gates separately.

This follow-up has not run. Route efficiency remains a tracked requirement;
the original A → B → A direction still awaits a competent online baseline.
