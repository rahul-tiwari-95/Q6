# Equal-map replay on fixed collected supports — results v1

**Map balancing worked mechanically, but its behavior benefit was mixed.**
Efficient-success differences are **−3.91, +7.16 and +4.56 percentage points**
across the three reused banks. The equal-bank mean improves **42.17% → 44.77%**,
but bank 1 becomes worse on six of eight panel means. This does not justify
adopting map-balanced replay as a reliable replacement for the control.

The [protocol](map_replay_protocol_v1.md) was independently reviewed and pushed
at `1bd8628` before execution. The single main run used clean, pushed source
`9104aa723d6858c5c70ccb14419724b6bf31ca68` and completed without deviations.
This is a local declaration, not external registration or peer review.
See [raw results](../../experiments/map_replay/pilot_v1/results.json),
[exposure](../../experiments/map_replay/pilot_v1/exposure.json),
[paired episodes](../../experiments/map_replay/pilot_v1/paired_layouts.csv),
[validation](../validation/map-replay-v1.md) and the
[dashboard](http://127.0.0.1:8080/dashboard/lab.html#coverage), selecting
**Experience coverage → Map-balanced replay**.

## What changed

Reuse the exact three collected supports from the bank-replication archive:
**59,839 / 59,984 / 59,838 unique states**, on the same 256 training layouts.
No new collection, support draw or added training state occurs. Baseline
`collected_unique` policies are the nine archived final 30,000-update models,
reevaluated on this study's panels without retraining. Their historical counts
and weights remain byte-identical.

Train nine `map_balanced` policies from the same seed-specific initial online
and target weights, using the same 20,420-parameter network and DDQN update,
four-action SmoothL1 loss, Adam settings and 30,000-update budget. At every
update choose **64 distinct maps uniformly**, then one uniformly selected
supported state per chosen map, retaining order. Separate owned generators
`[seed,99301]` and `[seed,99302]` keep map schedules identical across banks
without letting different within-map support sizes alter future map draws.

Baseline replay sampled 64 distinct unique-state rows globally. The treatment
therefore changes both map exposure and within-batch map diversity. It is a
practical replay-design intervention, not a pure estimate of map weighting.
All four transitions and detached full-bank successor observations remain
privileged. Exact labels do not select replay states or learned actions.

All **nine treatment fits / 270,000 new updates** finish before learned-policy
evaluation. There are **17,280,000 state presentations**, **45 treatment
snapshots** and **nine copied baseline finals**. Only final policies are
evaluated; no intermediate checkpoint or new full-state fitting sweep is used.

Eight prospectively selected 64-map panels exclude training, all four earlier
fresh panels, the preceding frozen-policy and bank-replication panels, and
already accepted layouts. Six prescribed exclusions occur: 1040016, 1042009
and 1043049 match training; 1042016 matches an earlier selected layout;
1044024 matches the bank-replication panel; 1047061 matches the equal-size
panel. Selection uses walls + pellet + rules identity, preserves original
spawns and never uses outcomes. Every model receives one greedy and two
epsilon-0.1 episodes per map with paired external exploration draws:
**27,648 learner + 3,584 shared reference episodes** on **512 layouts**.

## Behavior by bank

Each greedy bank/condition has 1,536 episodes. Cells show control / balanced.
Efficient success means reaching the goal within twice the map's planner
length, counting failures in the denominator.

| Bank | Success: control / balanced | Efficient: control / balanced | Efficiency Δ | Mean steps: control / balanced |
| --- | ---: | ---: | ---: | ---: |
| 1 | 79.62% / 77.93% | 44.01% / 40.10% | -3.91 pp | 15.63 / 16.42 |
| 2 | 77.21% / 78.32% | 39.71% / 46.88% | +7.16 pp | 16.91 / 15.46 |
| 3 | 78.12% / 81.77% | 42.77% / 47.33% | +4.56 pp | 15.90 / 14.95 |

The equal-bank mean efficient-success gain is **+2.60 points**; success improves
by **+1.02 points**, and mean length falls **0.54 steps**. Equal evaluation
counts make these equal-bank means numerically equal to episode pooling.
There are 4,608 greedy episodes per condition:

| Outcome | Archived collected control | Map-balanced replay |
| --- | ---: | ---: |
| Success | 78.32% (3,609/4,608) | 79.34% (3,656/4,608) |
| Efficient success | 42.17% (1,943/4,608) | 44.77% (2,063/4,608) |
| Mean steps, all episodes | 16.15 | 15.61 |
| Mean steps, successes only | 11.76 | 11.34 |
| Blocked steps / total steps | 65.35% (48,626/74,406) | 63.80% (45,893/71,928) |

The aggregate contains 120 more efficient successes and 47 more total
successes, but does not remove the bank-1 reversal. Successful-only route
lengths compare different successful subsets. Pooled blocked-fraction change
is **−1.55 points**, versus **−1.49 points** for the mean of within-bank pooled
ratios and **−1.38 points** for the mean of bank/learner ratios. Bank 1's blocked
fraction falls even while its episodes grow longer: that ratio alone would
give a misleading impression of improvement.

## Every panel and learner

| Panel | Bank 1 efficiency Δ | Bank 2 efficiency Δ | Bank 3 efficiency Δ |
| --- | ---: | ---: | ---: |
| 0 | -1.04 pp | +13.02 pp | +6.25 pp |
| 1 | +0.00 pp | +7.29 pp | -5.73 pp |
| 2 | -11.46 pp | +7.29 pp | +3.12 pp |
| 3 | -5.73 pp | +4.17 pp | +8.85 pp |
| 4 | -4.69 pp | +9.38 pp | +10.42 pp |
| 5 | -5.73 pp | +7.81 pp | +6.25 pp |
| 6 | -3.65 pp | +3.65 pp | +4.69 pp |
| 7 | +1.04 pp | +4.69 pp | +2.60 pp |

Efficient success improves on **16/24 bank-panel means**, declines on seven
and ties on one. Bank 1 has 1 positive / 6 negative / 1 tied panel, bank 2
has eight positive panels, and bank 3 has 7 positive / 1 negative. The overall
panel-effect range is **−11.46 to +13.02 points**. Success improves on 17/24
bank-panel means, declines on six and ties on one.

Across all panels, **seven of nine bank/learner efficiency effects are
positive**. Bank 1's seed effects are **+9.57, −10.16 and −11.13 points**;
bank 2's are **+9.38, +10.35 and +1.76**; bank 3's are **+2.15, +5.66 and
+5.86**. At the finer bank/panel/learner level, 47/72 efficiency effects are
positive, 23 negative and two tied. These crossed measurements share banks,
initializations and panels; they are not 72 independent replications.

Historical 70% success/above-panel-random reference crossings occur for every
learner on **6/8, 5/8, 7/8 control panels**, versus **7/8, 5/8, 8/8 balanced
panels**. No bank/condition has every learner reach 80% efficient success on
any panel. These remain descriptive reference levels, not new competence
gates. Existing gate and full-state fit results are unchanged.

The separate epsilon-0.1 diagnostic gives success / efficient success of
**82.06% / 42.94%** control and **83.17% / 45.66%** balanced. It does not replace
the primary greedy contrast. Shared random references succeed in
**42.02% (1,291/3,072)**; the planner succeeds on all 512 maps, averaging
**3.63 steps**.

## Did the intervention actually balance experience?

Yes. Treatment map schedules and counts match across banks for a given learner
seed, and all supports remain identical to the archive. Both arms directly
sample every available supported state; neither samples outside its support.

| Exposure measurement | Control | Map-balanced |
| --- | ---: | ---: |
| Map presentation-share CV, range across fits | 0.249–0.258 | 0.00997–0.01011 |
| Individual map's share of presentations, overall range | 0.0104–0.5592% | 0.3785–0.4026% |
| Largest per-state presentation count in each fit, range | 56–61 | 858–1,117 |
| Available states never directly sampled | 0 | 0 |

The equal-map reference is 1/256 = 0.390625% of presentations. Balancing maps
with only 7–9 supported states necessarily repeats those states more often;
it does not create new experience. The average remains approximately 32
presentations per supported row, while the within-support distribution changes.
The dashboard exposes counts, quantiles and individual map shares rather than
showing support membership as though it were exposure.

Goal-near states receive about **23.3% of control presentations**, versus
**26.4–26.9%** under balancing (bank means across learners). These are shares
of training presentations, not the earlier percentage of all goal-near states
covered. Query-weighted successor access outside support rises from roughly
**34.43–34.56% to 34.98–35.35%**. Support membership, clock exposure, state
repetition and successor-query weighting are all recorded separately.

This rules out a failed balancing implementation as the explanation for the
mixed result. It does not establish that repetition, clock distribution,
goal-near exposure or any other single component causes the reversals.

## Decision and next bounded step

Keep the original replay as the baseline. Do not promote equal-map replay
based on its pooled gain: it helps two banks, harms one and varies strongly
with initialization. The present panels did not evaluate uniform-subset
policies, so comparing its older score with this study would not measure a
matched-panel gap closure.

**Next: replace states within each map while preserving its exact collected
state quota and original replay schedule.** For each bank/map, draw once from
that map's exhaustive valid states the same number of unique rows as its
collected support contains. Use owned bank/map RNG streams, shared across
learner seeds and fixed before predictions. Preserve sorted map blocks and
the original uniform-over-unique-state sampler, network, initializations,
loss and update budget.

Because every map keeps its exact quota, identical local row draws map to
identical ordered map IDs at every update, preserving both map exposure and
within-batch map diversity. Verify that ordered-map digest against the archived
baseline. The selected within-map states change: positions, clocks and their
successor structure remain a combined composition intervention. This does not
isolate goal-near or clock coverage alone.

Nine new replacement-support fits can be compared with the original archived
controls on prospectively declared common panels. A repeatable gain would
support within-map state selection as a useful next collection target; mixed
results would leave interactions unresolved. This recommendation has not run
and needs its own protocol. After this bounded control, prioritize the bridge
to training from actually recorded actions/transitions over an open-ended
series of balancing variants.

Memory remains deferred. These are reused support banks and privileged offline
transitions, not evidence of online-RL competence or interpretable forgetting
under A → B → A. No extra training or evaluation followed the fixed budget.

## Inspect and reproduce

The dashboard adds **304 preselected recordings** while preserving **1,098
prior recordings**, with an initially visible movement rule, bank/learner/map
exposure controls and final-policy panel comparisons. The first preselected
map is 1040000: bank 1 / seed 0 improves from 30 to 15 steps, while bank 1 /
seed 2 changes from a 14-step success to failure; bank 3 / seed 2 changes from
failure to a four-step success. No favorable trace was substituted.

The single main run took **178.15 seconds (2.97 minutes)** and peaked at
**481,869,824 bytes (0.449 GiB)**, using one CPU thread at reduced priority.
It performed zero new collection steps, support draws or baseline updates.
The independent audit passed all saved evidence and **304 recordings / 5,259
steps** with exact forward outputs, without retraining or repeating the full
policy evaluation. See the validation record for the scope and limitations.

Use a fresh output directory with Python 3.12, Torch 2.8.0 and NumPy 2.0.2:

```bash
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 python -m q6.map_replay \
  --output experiments/map_replay/my-reproduction \
  --protocol-file docs/experiments/map_replay_protocol_v1.md
python scripts/audit_map_replay.py --study experiments/map_replay/my-reproduction
```

Smoke uses shipped supports, tiny treatment budgets and alternate panels. Its
archived 30,000-update controls and 24-update treatments are explicitly unequal
and ineligible research evidence. License remains undecided; Q6 is a public
research preview.
