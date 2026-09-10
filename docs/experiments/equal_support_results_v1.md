# Equal-size collected versus uniform state banks — result v1

**At equal bank size, uniform states improve route efficiency but do not
improve the primary greedy-success score in this pilot.** Collected training
reaches **83.85% fresh success**, versus **82.81%** for the uniform subset.
Uniform training produces **67.71% efficient fresh successes**, versus collected
training's **44.79%**, and reduces mean episode length from **15.45 to 11.30**.
All three seeds in both arms pass the declared fresh-success gate. Neither arm
passes the stricter every-seed training-fit or efficiency diagnostic.

The primary difference is only two episodes out of 192, with mixed directions
across learner seeds. This is not evidence of equivalence. The efficiency
advantage appears in every seed, conditional on these two fixed banks and this
panel. Equal row counts cannot explain that between-arm efficiency difference;
which states enter training matters here. The result does not identify goal
proximity, map balance or successor-graph structure as the cause.

A second finding changes the next priority: the **bit-identical collected
networks scored 68.23% on the preceding fresh panel and 83.85% on this one**.
That is panel variation, not improved learning. Before changing training again,
evaluate frozen policies on multiple new, fixed panels to test how reliably
the behavioral differences repeat.

The [protocol](equal_support_protocol_v1.md) was locally declared and pushed at
`03f0ff8` before execution. One main run completed all **180,000 updates** in
**153.73 seconds**, using **666.6 MB peak process RSS (0.62 GiB)**, one CPU
thread and **zero new collection steps**. No deviations, additional draws,
training extensions or checkpoint selection followed.

## What the equal-size control establishes

Both conditions use **59,626 current states** from the same 163,840-row bank,
the unchanged 20,420-parameter network, three paired initializations, DDQN
updates, four-action Smooth L1 loss, optimizer and 30,000 updates per seed.

- **Collected unique:** the exact archived support from the preceding random
  collector, reused without new exploration or visitation-frequency weighting.
- **Uniform subset:** one declared, sorted, uniformly drawn subset without
  replacement, shared by all three learner initializations.

Each model receives 1,920,000 state presentations / 7,680,000 action targets.
Local batch-index digests and count vectors match exactly across arms. Mapping
those local ranks to the different supports intentionally changes global rows;
a matching rank is not the same physical state. Every row in each bank was
sampled. Support size and average presentations per row are therefore matched.

Collected initial/final online weights, final target weights, global sampling
digests and count vectors reproduce the previous coverage study exactly.
Training arrays and all transitions also match. This is an execution-consistency
control, not independent training replication.

Both arms retain all four counterfactual transitions per sampled state and
full-bank detached successor queries. No state outside an arm's own support
receives direct sampling. This remains privileged offline evidence, not
trajectory-only learning or online-RL competence. One collected bank and one
uniform draw are not three independent bank replications.

## Primary outcome and efficient routes

| Seed | Collected fresh success | Uniform fresh success | Uniform − collected | Collected efficient success | Uniform efficient success |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 84.38% | 87.50% | +3.13 pp | 42.19% | 76.56% |
| 1 | 81.25% | 79.69% | −1.56 pp | 50.00% | 65.63% |
| 2 | 85.94% | 81.25% | −4.69 pp | 42.19% | 60.94% |
| Mean / pooled equal-size panels | **83.85%** | **82.81%** | **−1.04 pp** | **44.79%** | **67.71%** |

Collected succeeds in **161/192** final fresh greedy episodes and uniform in
**159/192**. The shared random reference succeeds in **188/384 (48.96%)** and
shortest path in **64/64 (100%)**. Every learner seed reaches 70% and exceeds
random on this panel. No confidence interval, superiority or equivalence test
was declared; seed ranges are descriptive.

Efficient success means collecting within twice **that layout's own planner
length**, with failures included in the denominator. Uniform adds **44 efficient
successes out of 192 (+22.92 percentage points)**. The separate 80%-in-every-seed
criterion remains unmet for both arms.

| Final fresh greedy behavior | Collected unique | Uniform subset | Shortest path |
| --- | ---: | ---: | ---: |
| Mean steps, all episodes | 15.45 | 11.30 | 3.58 |
| Mean steps, successful episodes only | 12.26 | 7.01 | 3.58 |
| Blocked steps / all visited steps | 1,942 / 2,966 (65.48%) | 1,423 / 2,170 (65.58%) | 0% |
| Blocked fraction within successful episodes | 55.93% | 45.24% | 0% |

Uniform reduces all-episode mean length by **4.15 steps**, in the same direction
for every seed. Successful-only subsets differ. Absolute blocked steps fall,
but the pooled blocked **fraction is slightly higher**, so this is not a blanket
no-op-rate improvement. Averaging paired per-seed blocked fractions gives a
−0.21 pp delta, a different denominator from pooling steps.

Epsilon-0.1 evaluation gives **85.68% collected success versus 88.02% uniform**,
each over 384 fresh episodes. These secondary results do not replace the
declared greedy outcomes or select another action mode after the fact.

## Why the fresh-panel change matters

The new 64-layout panel scans from **960000**, excluding all training layouts
and all prior 930000, 940000 and 950000 fresh panels. No candidate was rejected
in this run. Both arms are compared on exactly the same new layouts and starts.

| Measurement on separate fresh panels | Previous coverage panel, 950000 | Current equal-size panel, 960000 |
| --- | ---: | ---: |
| Same collected policies, greedy success | 68.23% | 83.85% |
| Random reference success | 39.06% | 48.96% |
| Mean shortest-path length | 3.94 | 3.58 |

The collected score changes by **15.63 pp without a weight change**, and its
all-seed fresh gate changes from unmet to met. The references also differ;
these are descriptive signs that the panels differ, not proof that planner
length alone explains the shift. The previous within-panel exhaustive-versus-
collected result remains valid for that panel. This observation cautions against
treating a single 64-map gate outcome as a general competence label.

The previous exhaustive score of 80.21% came from panel 950000. It is not an
exhaustive comparison arm on the current panel, and must not be ranked against
this study's 82.81% uniform score. The dashboard keeps studies and panels separate.

## Equal cardinality, different composition

Both banks cover **36.39%** of the full state space and include states from all
256 maps. They share **21,825 rows**: 36.60% of each bank, with a union of 97,427
and Jaccard overlap 22.40%.

| Included-state diagnostic | Collected unique | Uniform subset |
| --- | ---: | ---: |
| Unique current states | 59,626 / 163,840 | 59,626 / 163,840 |
| Included rows per map, range | 8–321 of 640 | 200–266 of 640 |
| Winnable states | 53,979 / 150,762 (35.80%) | 54,833 / 150,762 (36.37%) |
| Goal-near states, all clocks | 13,938 / 52,704 (26.45%) | 19,121 / 52,704 (36.28%) |
| Remaining steps 1–8 | 15,010 / 40,960 (36.65%) | 14,805 / 40,960 (36.15%) |
| Remaining steps 9–16 | 16,012 / 40,960 (39.09%) | 14,995 / 40,960 (36.61%) |
| Remaining steps 17–24 | 16,531 / 40,960 (40.36%) | 15,025 / 40,960 (36.68%) |
| Remaining steps 25–32 | 12,073 / 40,960 (29.48%) | 14,801 / 40,960 (36.14%) |

Goal-near means physical shortest-path distance at most two moves, irrespective
of clock. These labels were computed after support selection and did not guide
it. Uniform selection is more even in these measured dimensions, but graph
coverage differs in the other direction:

| Edges from unique supported current states | Collected unique | Uniform subset |
| --- | ---: | ---: |
| All four-action edges | 238,504 | 238,504 |
| Terminal edges | 11,768 | 15,097 |
| Nonterminal edges | 226,736 | 223,407 |
| Nonterminal edges outside own support | 79,004 (34.84%) | 142,133 (63.62%) |
| Distinct outside-support destinations | 47,246 | 77,899 |

Outside-support successor queries are retained and detached. The uniform arm's
better route efficiency despite more such queries shows that this edge fraction
alone does not order the observed outcomes. It does not show that successor
coverage is irrelevant or establish a causal role for any other single feature.

## Fit and value diagnostics

| Seed | Collected training success / full-bank agreement | Uniform training success / full-bank agreement |
| --- | ---: | ---: |
| 0 | 96.48% / 78.57% | 89.45% / 89.66% |
| 1 | 97.27% / 84.45% | 92.58% / 87.26% |
| 2 | 97.66% / 77.53% | 93.36% / 85.26% |

All six fit diagnostics fail the requirement for both quantities to reach 90%.
Pooled familiar-start success is **97.14% for collected and 91.80% for uniform**.
Higher familiar-start success therefore does not imply more efficient fresh
behavior or better exhaustive-state action agreement.

| Common full-state panel, pooled across seeds | Four-action MAE to Q* | RMSE | Winnable action agreement |
| --- | ---: | ---: | ---: |
| Collected, training | 0.1152 | 0.1784 | 80.19% |
| Uniform, training | 0.0840 | 0.1357 | 87.39% |
| Collected, fresh | 0.1192 | 0.1839 | 71.08% |
| Uniform, fresh | 0.0876 | 0.1415 | 79.54% |

The declared final own/outside-support breakdown uses saved predictions:

| Condition | Own-support winnable agreement | Outside-own-support agreement | Own MAE | Outside MAE |
| --- | ---: | ---: | ---: | ---: |
| Collected unique | 84.36% | 77.86% | 0.11514 | 0.11519 |
| Uniform subset | 87.80% | 87.16% | 0.08240 | 0.08494 |

Each bank has 59,626 states and its complement has 104,214. Winnable own/outside
counts per seed are 53,979/96,783 for collected and 54,833/95,929 for uniform.
These arm-specific subsets contain different states; the common full-bank
metrics above supply the shared evaluation distribution. Own-support agreement
is descriptive and does not create a new gate. Repeated seed evaluations are
not independent state banks. Impossible states stay in errors and outside
winnable agreement; tie tolerance is absolute 1e−6, relative zero.

## Next bounded milestone

**Evaluate the frozen collected, uniform and archived exhaustive policies on
eight new, fixed, disjoint 64-layout panels.** Declare panel selection,
exclusions, metrics and aggregation before execution. Keep the final checkpoints,
banks and policies fixed; perform no training, support redraw or tuning.

Report panel-wise paired success, efficient success and episode length, along
with pooled per-learner outcomes and unchanged-policy variation across panels.
This checks whether the efficiency advantage repeats and how sensitive the
competence readout is to panel choice. It does not retrospectively replace
these studies' gates or replicate independent training/support banks.
Independent bank replications can follow if the behavioral distinction persists.
This evaluation milestone has not run. Memory and online feedback remain
deferred while measurement and data questions remain unresolved.

## Artifacts, reproduction and visual inspection

The [main artifact](../../experiments/equal_support/pilot_v1/results.json),
[supports](../../experiments/equal_support/pilot_v1/supports.npz),
[composition](../../experiments/equal_support/pilot_v1/coverage.json),
[own/outside diagnostics](../../experiments/equal_support/pilot_v1/support_diagnostics.json)
and [validation record](../validation/equal-support-v1.md) preserve the evidence.
Clean source revision is `e52043e2338045718f3939b463b60f996f740df3`; the manifest
covers 75 files. The full independent audit passed without corrections.

```bash
# Python 3.12; match the captured environment for strict reproduction.
python -m pip install 'torch==2.8.0' 'numpy==2.0.2'
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 \
  python -m q6.equal_support --output experiments/equal_support/my-reproduction \
  --protocol-file docs/experiments/equal_support_protocol_v1.md

# Audit only; no optimization or new learned-policy rollouts.
python scripts/audit_equal_support_study.py
```

Open [the local lab](http://127.0.0.1:8080/dashboard/lab.html#coverage), then
**Experience coverage → Equal-size banks**. Compare both banks' composition,
inspect the selected bank's map/time views, and use outcome cards, paired tables
and 124 replays alongside the full-state diagnostics. The previous coverage
study remains selectable. First fresh map 960000 was chosen before outcomes:
seed 0 succeeds in 28 collected-policy steps versus five uniform-policy steps;
seed 1 times out in both; seed 2 succeeds with collected and times out with
uniform. These mixed examples are preserved without choosing a favorable map.

All earlier artifacts remain unchanged. This public research preview still
has no selected license.
