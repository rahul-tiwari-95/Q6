# Exhaustive versus collected state coverage — result v1

**The same DDQN procedure transfers less reliably when trained on the states
visited by a fixed random collector.** Exhaustive training reaches **80.21%
fresh-layout greedy success**, compared with **68.23%** from collected unique
states. All three exhaustive seeds pass the declared fresh-success gate;
collected seeds 0 and 2 miss it. Collected training nevertheless reaches
**97.14% success from familiar training starts**, versus exhaustive's 92.71%.
Learning the familiar starts well does not establish broad competence.

This identifies current-state support and its induced sampling distribution
as a consequential part of this offline setup. It does not isolate the number
of unique states from which states were encountered, nor establish that online
exploration or replay weighting is the sole cause of earlier RL failures.
The next useful control holds support size fixed before changing collection,
DDQN target mechanics, or memory.

The locally declared [protocol](coverage_protocol_v1.md) was committed at
`552a7f7` before execution. The single main run completed all **180,000 updates**
in **156.64 seconds**, using **637.9 MB peak process RSS (0.59 GiB)**, with no
protocol deviations. No extra training, collection or checkpoint selection
followed.

## What changed and what stayed fixed

Both conditions use the same 256 training layouts, visible rule, observations,
20,420-parameter network, three paired initializations, DDQN targets, all-four-
action Smooth L1 loss, Adam optimizer and 30,000 updates per seed. Each update
samples 64 distinct current states. Both retain the full transition table for
detached successor queries. Exact Q* labels are diagnostic references, not
training targets.

- **Exhaustive:** sample uniformly from all 163,840 current states.
- **Collected unique:** sample uniformly from the 59,626 unique pre-action
  states visited by 16 random episodes per layout. Repeated visits remain in
  collection logs but do not increase replay weight.

Actual minibatches intentionally differ. Each network receives 1,920,000
current-state presentations and 7,680,000 action targets. Every support row is
sampled at least once. The smaller bank therefore receives more updates per
unique row on average. Maps with more collected rows also receive more total
sampling mass; this is not a pure support-size intervention.

The exhaustive arm's initial and final online weights, final target weights,
training arrays, transitions and sampling digests reproduce the previous
[fixed-target DDQN arm](fixed_targets_results_v1.md) exactly. It is the same
learned control evaluated on a new panel, not an independent training
replication. The 64 fresh layouts begin at seed 950000 and exclude training
layouts and both prior fresh panels. All current comparisons use this new
panel; the previous 86.46% DDQN score came from different layouts.

## Primary outcome and route efficiency

| Seed | Exhaustive fresh success | Collected fresh success | Collected − exhaustive | Exhaustive efficient success | Collected efficient success |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 75.00% | 65.63% | −9.38 pp | 60.94% | 31.25% |
| 1 | 82.81% | 70.31% | −12.50 pp | 70.31% | 46.88% |
| 2 | 82.81% | 68.75% | −14.06 pp | 65.63% | 45.31% |
| Mean / pooled equal-size panels | **80.21%** | **68.23%** | **−11.98 pp** | **65.63%** | **41.15%** |

The learned conditions succeed in 154/192 and 131/192 final fresh greedy
episodes respectively. Each seed shares the same 64 layouts and starts.
The fresh random reference succeeds in **150/384 episodes (39.06%)**, and the
shortest-path reference succeeds in **64/64 (100%)**. Collected seed 1 alone
meets the 70% fresh threshold; all collected seeds exceed random. Three learner
initializations share one collection bank and evaluation panel. These are
descriptive paired results, not independent collection replications or a
significance claim.

Efficient success requires collection within twice **that layout's own
shortest-path length**, with failures included in the denominator. Neither
condition meets the declared 80%-in-every-seed efficiency criterion.

| Final fresh greedy behavior | Exhaustive | Collected unique | Shortest path |
| --- | ---: | ---: | ---: |
| Mean steps, all episodes | 12.02 | 17.84 | 3.94 |
| Mean steps, successful episodes only | 7.09 | 11.25 | 3.94 |
| Blocked steps / all visited steps | 1,552 / 2,308 (67.24%) | 2,524 / 3,426 (73.67%) | 0% |
| Blocked fraction within successful episodes | 43.96% | 57.39% | 0% |

Collected training adds **5.82 steps** per fresh episode and reduces efficient
success by **24.48 percentage points**. Successful-only subsets differ, so
their route lengths are secondary descriptions. The mean paired per-seed
blocked-fraction increase is **6.68 pp**; pooling all steps instead gives
**6.43 pp**, a different denominator.

Epsilon-0.1 fresh evaluation yields **84.38%** success for exhaustive and
**73.70%** for collected, each over 384 episodes. These secondary outcomes do
not replace the predeclared greedy gates or select a different checkpoint.

## What the collector actually covered

The collector completed **4,096 episodes and 99,814 real environment steps**,
with 1,696 successes. It used uniform random actions, original starts and full
clocks, with a separate declared RNG per episode. Collection finished before
learning and never used Q* or learned-policy outputs.

| Current-state subset | Visited / exhaustive states | Unique coverage |
| --- | ---: | ---: |
| All states | 59,626 / 163,840 | 36.39% |
| Winnable states | 53,979 / 150,762 | 35.80% |
| Within two shortest-path moves of goal, all clocks | 13,938 / 52,704 | 26.45% |
| Remaining steps 1–8 | 15,010 / 40,960 | 36.65% |
| Remaining steps 9–16 | 16,012 / 40,960 | 39.09% |
| Remaining steps 17–24 | 16,531 / 40,960 | 40.36% |
| Remaining steps 25–32 | 12,073 / 40,960 | 29.48% |

Per-layout coverage ranges from **1.25% to 50.16%**. A fixed episode count
therefore does not imply equal layout coverage or replay mass. Goal-near
coverage and clock coverage were declared diagnostics computed after
collection; their deficits are not proven causes of the performance gap.

Of the 226,736 nonterminal four-action edges from unique visited current
states, **79,004 (34.84%) lead outside collected support**. They reach 47,246
distinct outside-support destinations. DDQN still queries those successors
through the full observation bank, with detached targets, but none enters
collected current-state sampling. This counterfactual transition access is
privileged: the comparison is not trajectory-only or continuous online RL.

## Training fit and value diagnostics

| Seed | Exhaustive training success / full-bank winnable agreement | Exhaustive fit | Collected training success / full-bank winnable agreement | Collected fit |
| --- | ---: | --- | ---: | --- |
| 0 | 90.23% / 89.82% | Not met | 96.48% / 78.57% | Not met |
| 1 | 94.53% / 90.77% | Met | 97.27% / 84.45% | Not met |
| 2 | 93.36% / 90.62% | Met | 97.66% / 77.53% | Not met |

The common fit diagnostic requires 90% in both quantities in every seed.
Both conditions miss it. Its agreement denominator is the **full exhaustive
training bank**, including states never directly sampled in the collected
condition. Failure of this gate alone is not evidence that the network failed
to fit its collected subset. Familiar-start success is also insufficient to
establish full-state fit.

A separate **post-hoc diagnostic**, computed only from saved final predictions,
finds collected-arm winnable agreement of **84.36% on visited states** and
**77.86% on unvisited states**. Visited agreement is 82.90%, 88.30% and 81.89%
across seeds; the exhaustive arm reaches 90.39% on that same visited subset.
Thus the weakness also exists within sampled support and cannot be described
solely as extrapolation to missing states. This adds no gate and does not
identify a target-fitting mechanism. The [supplemental artifact](../../experiments/coverage/analysis_v1/support_diagnostic.json)
records input hashes, per-seed counts and results separately from the immutable
main study. Its per-seed denominators are 53,979 visited winnable states and
96,783 unvisited winnable states.

| Final exhaustive-state evaluation, pooled across seeds | Four-action MAE to Q* | RMSE | Winnable optimal-action agreement |
| --- | ---: | ---: | ---: |
| Exhaustive, training | 0.0694 | 0.1184 | 90.40% |
| Collected, training | 0.1152 | 0.1784 | 80.19% |
| Exhaustive, fresh | 0.0752 | 0.1271 | 81.61% |
| Collected, fresh | 0.1238 | 0.1900 | 69.86% |

Here the common value-error and action-ranking diagnostics both favor the
exhaustive condition. Impossible states remain in Q error and are excluded
from winnable agreement. Training loss uses moving DDQN targets and different
sample distributions; it is not a common measure of error to exact Q*.

## Decision and next bounded control

**Compare the collected bank with a uniformly selected subset of exactly
59,626 states from the exhaustive bank.** Keep DDQN, four-action loss, paired
initialization and update budget unchanged, and declare the subset RNG and a
new fresh panel before execution. This equal-size control tests whether the
trajectory-induced composition matters beyond simply having fewer unique
current states. It does not separately identify map imbalance, clocks,
goal proximity or successor-query coverage; those remain measured components
of composition. Transition-graph structure also changes: a uniformly scattered
subset can have more outside-support successor queries than trajectory states.

If a uniform subset transfers better, test a specified coverage-balancing
intervention next. If both similarly sized subsets struggle relative to the
exhaustive control, investigate support size and repeated exposure before
claiming that more random episodes are the remedy. Two weak sparse subsets
would not prove that state count caused the gap. In either case, independent
collection/subset replications are needed before generalizing. This follow-up
has not run. Memory and continuous online feedback remain deferred while the
offline data question has a testable explanation.

## Evidence, reproduction and visual inspection

See the [complete artifact](../../experiments/coverage/pilot_v1/results.json),
[raw collection](../../experiments/coverage/pilot_v1/collection_steps.csv),
[paired outcomes](../../experiments/coverage/pilot_v1/paired_differences.json)
and [validation record](../validation/coverage-v1.md). Captured source revision
is `8070ed14ca78016f69759df21a0e77bf0676390b`; the manifest preserves 72 files.
The single-thread run includes collection, dataset construction, training,
evaluation and aggregation within its 1,200-second admission cap. The 4 GiB
RSS limit is a sampled stop guard, not an operating-system allocation limit.
All six fits total 11.52 million current-state presentations and 46.08 million
action targets. The 99,814 actual collector steps are a separate cost.

```bash
# Python 3.12; use the captured environment for strict reproduction.
python -m pip install 'torch==2.8.0' 'numpy==2.0.2'
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 \
  python -m q6.coverage --output experiments/coverage/my-reproduction \
  --protocol-file docs/experiments/coverage_protocol_v1.md

# No training; verifies the archived main artifact.
python scripts/audit_coverage_study.py

# Recompute the separate post-hoc support diagnostic from saved arrays.
python scripts/analyze_coverage_support.py
```

Serve the checkout and open **Experience coverage** in
[the local lab](http://127.0.0.1:8080/dashboard/lab.html#coverage). The heatmap
shows every training layout's coverage, the clock bars show which times were
encountered, and the outcome cards compare success, efficient success and
steps. All 124 saved recordings are reachable. The first fresh map, 950000,
was selected before outcomes: exhaustive fails in all three seeds, while
collected seed 2 succeeds in six steps. This individual map runs against the
pooled comparison; inspect full-panel measurements alongside replays.

The result supports a bounded offline coverage finding. It does not establish
online competence, retention, memory benefits, or a broadly superior collection
strategy. Earlier artifacts and the undecided license remain unchanged.
