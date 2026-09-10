# Frozen-policy panel robustness — results v1

**The uniform-bank route-efficiency advantage repeats across all eight new
panel means.** It is smaller than on the preceding single panel, but remains
positive: **+15.30 percentage points** pooled, with a **+7.81 to +21.88-point**
range across panels. Mean episode length is lower on all eight panels.
Success favors uniform on seven panels, reverses on one, and improves by only
2.54 points pooled. No network was trained or changed in this experiment.

The [protocol](panel_evaluation_protocol_v1.md) was locally declared, reviewed
and pushed at `5100595` before execution. The single main run used clean,
pushed source `d2dbb6f56036d2aec36f048522c92022bf2e93df` and completed without
deviations. See [raw results](../../experiments/panel_evaluation/pilot_v1/results.json),
[paired rows](../../experiments/panel_evaluation/pilot_v1/paired_layouts.csv),
[validation](../validation/panel-evaluation-v1.md), and the
[dashboard](http://127.0.0.1:8080/dashboard/lab.html#coverage), selecting
**Experience coverage → Panel robustness**.

## What was held fixed

Reuse final 30,000-update networks for learner seeds 0/1/2: collected and
uniform-subset policies from the equal-size archive, exhaustive policies from
the preceding coverage archive. All have the same 20,420-parameter feedforward
architecture. Inference-only loading creates no optimizer or replay, disables
gradients, and preserves online/target weights and source/copy file hashes
before, after and around each model's panel evaluation.

Choose eight disjoint panels of 64 layouts before predictions. Exclude training
layouts and all four earlier fresh panels, preserving the original spawn and
ascending candidate scan. One candidate, **975004**, matched the preceding
supervised panel and was rejected by the declared rule; panel 5 therefore ends
at 975064. The remaining panels use 970000–970063 through 977000–977063 at
1,000-seed intervals. No difficulty balancing or outcome-based replacement.

Every policy receives the same maps and external exploration draws: one greedy
episode and two epsilon-0.1 episodes per map/model. There are **13,824 learner
episodes** and **3,584 shared reference episodes**, covering **512 layouts**.
No new training updates, collection steps or support draws. Exact Q* along
rollouts measures action quality; it does not choose learned actions.

## Pooled greedy behavior

Each condition contributes 1,536 episodes: 512 maps × three saved learners.
Efficient success means success within twice that map's planner length, with
**all episodes**, including failures, in the denominator.

| Outcome | Collected unique | Uniform subset | Exhaustive |
| --- | ---: | ---: | ---: |
| Success | 78.19% (1,201/1,536) | 80.73% (1,240/1,536) | 84.90% (1,304/1,536) |
| Efficient success | 46.42% (713/1,536) | 61.72% (948/1,536) | 67.38% (1,035/1,536) |
| Mean steps, all episodes | 15.42 | 12.51 | 11.15 |
| Mean steps, successful episodes only | 10.80 | 7.85 | 7.45 |
| Blocked steps / total steps | 67.63% | 68.38% | 66.53% |
| Panel-mean success range | 69.27–84.90% | 74.48–87.50% | 78.12–91.15% |
| Panel-mean efficient-success range | 39.06–55.21% | 53.65–69.27% | 58.85–75.52% |

Uniform produces **235 more efficient successes**, but only **39 more total
successes**, than collected. Its routes improve even when considering
successful episodes alone, though the successful subsets differ between
conditions. This does not imply every aspect of behavior
improves: uniform's pooled blocked fraction is slightly higher, **13,135/19,210
versus 16,021/23,689**. Fewer total blocked steps and a smaller blocked fraction
are different claims. The paired mean-seed blocked-fraction delta is +0.765
points; the pooled fraction delta is +0.745 points.

Random references succeed in **40.40%** of 3,072 episodes. The planner succeeds
on all 512 maps and averages **3.62 steps**. These references belong to this
study's selected layouts, not an interchangeable earlier panel.

## All eight panels

Each cell pools three fixed learners on 64 maps. Percentage-point differences
are uniform minus collected; every panel is shown, including the reversal.

| Panel | Collected success | Uniform success | Exhaustive success | Collected efficient | Uniform efficient | Exhaustive efficient | Uniform efficiency Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 80.21% | 84.38% | 84.90% | 50.52% | 66.15% | 71.88% | +15.62 pp |
| 1 | 78.65% | 79.69% | 85.42% | 45.83% | 57.81% | 63.02% | +11.98 pp |
| 2 | 75.52% | 76.04% | 78.12% | 44.79% | 60.94% | 58.85% | +16.15 pp |
| 3 | 69.27% | 74.48% | 80.73% | 39.06% | 53.65% | 64.58% | +14.58 pp |
| 4 | 84.90% | 85.94% | 91.15% | 47.40% | 69.27% | 75.52% | +21.88 pp |
| 5 | 75.00% | 81.25% | 86.46% | 42.19% | 60.42% | 69.79% | +18.23 pp |
| 6 | 82.29% | 87.50% | 88.54% | 55.21% | 63.02% | 71.35% | +7.81 pp |
| 7 | 79.69% | 76.56% | 83.85% | 46.35% | 62.50% | 64.06% | +16.15 pp |

Uniform's success differences range **−3.12 to +6.25 points**; episode length
falls by **1.80–4.17 steps** across panel means. Its efficiency advantage is
positive in **22 of 24 panel/learner cells**, not every cell. The exceptions
are panel 4/seed 1 (−3.12 points) and panel 6/seed 0 (−1.56 points).

| Saved learner | Collected success / efficient | Uniform success / efficient | Exhaustive success / efficient |
| --- | ---: | ---: | ---: |
| Seed 0 | 77.15% / 43.95% | 80.86% / 64.06% | 83.59% / 63.87% |
| Seed 1 | 80.47% / 55.86% | 81.45% / 59.57% | 84.38% / 70.70% |
| Seed 2 | 76.95% / 39.45% | 79.88% / 61.52% | 86.72% / 67.58% |

All-panel uniform efficiency differences are +20.12, +3.71 and +22.07 points
for the three saved learners. These repeated evaluations are not new training
replications or new collector/uniform-bank draws.

Exhaustive exceeds uniform success on every panel, with a pooled **+4.17-point**
advantage. Its efficiency advantage is **+5.66 points** pooled, positive on
seven panels and negative on panel 2. Its larger bank and lower average
presentations per row remain contextual differences; this is not a pure
state-count intervention.

## Variation, reference lines and exploration

The earlier collected score changed from 68.23% to 83.85% across two panels
with identical weights. This prospective evaluation again shows substantial
variation: collected panel means span **15.63 points**. Old results and gates
remain unchanged; the new pooled score does not retrospectively replace them.

Descriptively, every learner crosses the historical 70% success level and
exceeds that panel's random reference on **7/8 collected panels**, **8/8 uniform
panels** and **8/8 exhaustive panels**. No condition has every learner reach
80% efficient success on any panel. These are reference-line counts, **not a
new competence gate**. Full-state training fit was not reevaluated.

The separately reported epsilon-0.1 diagnostic gives success/efficient success
of **81.35% / 46.91%** collected, **85.61% / 62.76%** uniform and
**87.92% / 67.58%** exhaustive. It does not replace the declared greedy contrast.
No exploratory paired table or new preferred action mode was selected afterward.

## What this changes next

The route-efficiency difference survives prospective panel variation for these
fixed policies. Equal unique-state count alone therefore does not explain the
observed equal-bank contrast. However, map, clock, goal-near and successor-graph
composition changed together, and all three learner seeds share one collected
bank and one uniform draw. This does not isolate a composition component,
establish statistical significance, or show that any collector is generally
inferior to any uniform sampler.

**Next recommended control: independent support-bank replications.** Generate
three new fixed-budget exploratory banks; for each, draw a uniform subset with
the same realized unique-state count. Keep the network, DDQN, learner seeds,
loss and update budget fixed, and declare evaluation panels before training.
Report each bank pair separately before pooling. Matching within each pair
controls state count; collector size and composition can still vary across
pairs, so disclose both. This tests whether the result belongs to these bank
instances before designing a collection/replay intervention. It has not run.

Memory remains deferred. Privileged four-action offline training still limits
the conclusion; this evaluation does not establish online-RL competence or
readiness to interpret A → B → A forgetting.

## Inspect and reproduce

The dashboard adds all **160 outcome-independent recordings**, preserving
634 earlier learner-study recordings and historical controls. Its first map
is deliberately unremarkable for most policies: on map 970000 all collected
and uniform learners finish in two steps, while exhaustive seed 0 takes 30
greedy steps (29 with epsilon-0.1 exploration).
An individual trace can reverse the pooled impression; use the panel charts
and change the panel/controller selectors. Rules remain visible initially.

The single main run took **30.54 seconds**, peaked at **242.2 MB process RSS
(0.226 GiB)**, used one CPU thread at reduced priority and performed 17,934
resource checks. All nine online/target identities remained unchanged across
72 model-panel checks. The full independent audit passed all raw arithmetic
and 160 saved replays, including 1,674 recorded steps with exact reproduced
network outputs. It did not repeat the complete policy evaluation.

Use a new output directory with the pinned runtime and shipped input archives:

```bash
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 python -m q6.panel_evaluation \
  --output experiments/panel_evaluation/my-reproduction \
  --protocol-file docs/experiments/panel_evaluation_protocol_v1.md
python scripts/audit_panel_evaluation.py --study experiments/panel_evaluation/my-reproduction
```

Python 3.12, Torch 2.8.0 and NumPy 2.0.2 match the main run; remaining packages
are recorded in its environment file. Smoke uses separate panels and remains
ineligible evidence. License remains undecided: Q6 is a public research preview.
