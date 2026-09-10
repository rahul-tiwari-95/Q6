# Better recorded routes did not make a better learner

The fixed collection mixture **reduced fresh-map success and efficiency**.
Retain constrained DDQN trained on the original 16 random episodes per map.
The mixed banks contain many more efficient successful routes, but replacing
half the exploration with repeated greedy journeys also narrows experience.
This experiment does not isolate which change caused the regression.

One main run completed on September 9, 2026, under the independently reviewed
[protocol](guided_collection_protocol_v1.md), pushed at `a668b3b` before clean
execution source `2a1fa03df615a3254c8f0a5e9933386b8a1eb2a5`. There are no declared
deviations, mixture sweeps or main reruns. This is a locally declared bounded
comparison, not external registration or independent replication.

## Fresh behavior: keep the random-collection baseline

Both arms use constrained DDQN, matching initial weights, the same
20,420-parameter network, loss, optimizer and 30,000-update budget per fit.
Nine archived random-control policies and nine newly fitted mixed policies
are evaluated after all fits on the same eight new 64-map panels. Fresh
policies choose among all four actions; they receive no logged masks.

| Bank | Random success | Mixed success | Random efficient | Mixed efficient | Efficiency difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 69.01% | 62.30% | 50.39% | 42.19% | −8.20 pp |
| 2 | 71.94% | 62.57% | 49.15% | 44.21% | −4.95 pp |
| 3 | 67.64% | 66.60% | 47.85% | 43.55% | −4.30 pp |
| Equal-bank mean | **69.53%** | **63.82%** | **49.13%** | **43.32%** | **−5.82 pp** |

Each bank/arm contains 1,536 greedy episodes: three learner seeds × 512 maps.
Across 4,608 episodes per arm, successes fall **3,204 → 2,941** and efficient
successes **2,264 → 1,996**. Efficiency means success within twice the full-world
shortest route, counting failures. Mean steps across all episodes rise
**14.79 → 16.47**. Successful-only steps rise **7.26 → 7.67**, but those means
refer to different successful subsets.

Efficiency declines in **21 of 24 bank-panel averages**, with three gains;
learner-panel cells have **53 declines, 14 gains and five ties**. Eight of nine
whole-panel bank/learner comparisons decline in efficiency; bank 3 / seed 1
gains 0.78 points. Success declines in all nine whole-panel learner comparisons
and 19 of 24 bank-panel averages (four gains, one tie). The epsilon-0.1
diagnostic also declines: success **77.50% → 71.82%**, efficiency
**50.55% → 44.72%**, over 9,216 episodes per arm.

All three mean effects fail the declared direction for adopting this mixture.
These crossed cells share maps, initializations and panels; their counts are
descriptive consistency checks, not independent replication or significance.
Scores from earlier panels remain separate context.

## Collection improved journeys while narrowing the bank

Every map receives 16 complete episodes in each arm. Treatment retains random
slots 0–7 exactly and replaces slots 8–15 with greedy episodes from one frozen
constrained-DDQN collector, bank 1 / seed 0 / final 30,000 updates, chosen by
index before this run. The collector is unchanged throughout.

| Bank | Random interactions | Mixed interactions | Random unique states | Mixed unique states | Random recorded actions | Mixed recorded actions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 100,878 | 67,981 | 59,839 | 38,619 | 84,305 | 47,737 |
| 2 | 100,326 | 67,763 | 59,984 | 38,646 | 84,091 | 47,596 |
| 3 | 100,381 | 67,369 | 59,838 | 38,256 | 83,933 | 47,209 |
| Total across banks | **301,585** | **203,113** | **179,661** | **115,521** | **252,329** | **142,542** |

State/action totals count each bank separately, including shared experience.
Both arms have **12,288 complete episodes**. Mixed collection uses **32.7%
fewer interactions**, but contains **35.7% fewer distinct states** and **43.5%
fewer distinct recorded state/action pairs**. Collection success rises
**40.61% → 64.81%**. This is not an equal-interaction or equal-total-history
comparison.

Each bank's guided component has 2,048 episodes, **17,208 interactions** and
1,832 successes. Eight deterministic repetitions reproduce the same route on
each map: **256 distinct guided routes**, also identical across banks. All
repetitions count toward interaction cost. Deduplicated replay gives repeated
visits no extra loss weight. Retained random interactions are
**50,773 / 50,555 / 50,161** and match every corresponding archived step.

The collector's prior work remains part of the account: **100,878 historical
interactions, 30,000 updates, 1,920,000 sampled states and 2,705,715 action
targets** (118,412 terminal; 2,587,303 nonterminal). Its knowledge includes the
bank-1 random episodes replaced by guidance here. These three draws therefore
measure variation in the random half conditional on one shared collector.

## The logs permit faster routes; the students still regress

Independent backward reachability uses only actually recorded transitions.
At the original training starts, the best possible logged efficient-success
rate rises **468/768 = 60.94% → 632/768 = 82.29%**. By bank, it changes
**58.98% → 81.64%**, **59.77% → 82.42%**, and **64.06% → 82.81%**.
The overall logged success ceiling slightly falls:
**744/768 = 96.88% → 737/768 = 95.96%**. Faster available routes do not imply
that every map retains a successful recorded path.

These are graph ceilings on familiar starts, not measured learner performance
or ceilings on unrestricted fresh behavior. They establish that the mixture
improved the availability of efficient logged routes. They do not establish
why the newly trained networks deteriorated.

Learning remains 270,000 updates / 17,280,000 sampled states per arm. Actual
supervised action targets change **24,273,701 → 21,322,723**, including
**1,062,190 → 960,030** terminal targets and **23,211,511 → 20,362,693**
nonterminal successor queries. All queries remain inside the relevant bank's
closed recorded support. Different bank sizes change realized state/map replay;
neither minibatch identity nor target count is claimed to match.

## Watch the difference

Open [the local lab](http://127.0.0.1:8080/dashboard/lab.html#coverage) and choose
**Experience coverage → Guided collection**. The movement rule appears initially.
The first predetermined collection example, map **300000 / slot 8**, takes
**26 → 2** steps in bank 1 and **20 → 2** in bank 3; bank 2 takes two in both
arms. Each bank already contains a two-step logged route for this map, so this
example illustrates episode quality, not a newly enabled path. Slot 0 is
identical between arms.

The first fresh recording also cautions against judging by one replay:
map **1140000 / bank 1 / seed 0** changes from a 32-step failure to a 23-step
success, yet both are inefficient against the four-step planner reference and
the full comparison favors random collection. Every predetermined recording
is retained: **96 collection + 304 fresh = 400 new**, **3,338 total**.

## Decision and next controlled comparison

**Retain constrained DDQN with the original 16-random-episode banks.** This
specific mixture fails its intended fresh-behavior test despite better route
availability. Reduced exploration, changed support/action coverage, replay
composition and target count are plausible contributors that changed together;
the result does not prove that guided collection generally harms learning.

Next, derive a bank containing **only the same eight retained random episodes**
from each existing log, with no new collection. Train nine policies using the
same constrained-DDQN procedure and budget, then compare them with both frozen
current arms on common new panels. That supplies the missing comparison:

`mixed − random16 = (mixed − random8) + (random8 − random16)`.

The first contrast measures the effect of adding these guided records to the
retained experience; the second measures dropping the other eight random
episodes. Both still include the resulting replay/target-composition changes.
Do not sweep mixture ratios or add memory before resolving that distinction.
Continuous online collection/learning remains a later milestone.

## Reproduce and inspect

The run took **186.30 seconds** on one CPU thread, sampled peak process RSS
**601,538,560 bytes (0.560 GiB)**. It completed 27,648 learner and 3,584 shared
reference episodes, with no admission or memory interruption. Full and portable
audits validate saved evidence without another fit or complete policy evaluation.
See the [validation record](../validation/guided-collection-v1.md).

```bash
# Execution check only; choose a new output directory.
python -m q6.guided_collection \
  --output /tmp/q6-guided-smoke \
  --protocol-file docs/experiments/guided_collection_protocol_v1.md --smoke

# Reproduce the declared main with the pinned runtime and a new output path.
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 \
  python -m q6.guided_collection \
  --output /tmp/q6-guided-reproduction \
  --protocol-file docs/experiments/guided_collection_protocol_v1.md

python scripts/audit_guided_collection.py \
  --study experiments/guided_collection/pilot_v1
```

[Raw results](../../experiments/guided_collection/pilot_v1/results.json),
[collection logs and snapshots](../../experiments/guided_collection/pilot_v1),
[paired outcomes](../../experiments/guided_collection/pilot_v1/paired_layouts.csv)
and the [manifest](../../experiments/guided_collection/pilot_v1/manifest.json)
are preserved. Earlier artifacts remain unchanged. The license remains
undecided and PR #1 remains open.
