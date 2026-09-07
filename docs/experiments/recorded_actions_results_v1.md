# Recorded-action supervision with fixed state replay — results v1

**Removing counterfactual action outcomes substantially reduces performance.**
Greedy efficient success falls **43.03% → 21.29%**, with effects of **−25.20,
−19.01 and −21.03 percentage points** across the three original collected
banks. All 24 bank-panel means decline. Success falls **78.43% → 64.00%**;
the learner still succeeds more often than the shared random reference, but
its routes become longer and less efficient.

The original states, exact replay schedule, network and optimizer budget are
fixed. Treatment receives only outcomes for actions actually logged at those
states. This identifies a material dependence on the previous broader
supervision, without isolating target count, missing action coverage, changed
action weights or successor bootstrapping as a single cause.

The [protocol](recorded_actions_protocol_v1.md) was reviewed and pushed at
`60d6299` before the single main run. Clean, pushed execution source is
`5b43def285c92ceb8dc314677cbd47b322273b1e`; no deviation or subsequent tuning
run occurred. This is a local declaration, not external registration or peer
review. See [raw results](../../experiments/recorded_actions/pilot_v1/results.json),
[action coverage](../../experiments/recorded_actions/pilot_v1/action_coverage.json),
[actual target/query exposure](../../experiments/recorded_actions/pilot_v1/action_exposure.json),
[paired episodes](../../experiments/recorded_actions/pilot_v1/paired_layouts.csv),
[validation](../validation/recorded-actions-v1.md), and the
[dashboard](http://127.0.0.1:8080/dashboard/lab.html#coverage), selecting
**Experience coverage → Recorded actions**.

## What the learner actually received

Both arms use the same original collected supports: **59,839 / 59,984 /
59,838 states** on maps 300000–300255. No new collection, support draw or
baseline training occurs. The control consists of nine archived final
30,000-update four-action policies. The synthetic within-map replacements
from the previous experiment are not training inputs here.

Manifest-verified collection logs retain **301,585 historical steps from
12,288 episodes**. Repeated state/action records are checked for identical
reward, successor and separate termination/truncation flags, then deduplicated.
Visit frequency is preserved as a diagnostic, not a loss weight. All supported
states have at least one observed action.

| Bank | States | Distinct recorded actions | Fraction of four-action possibilities | States with 1 / 2 / 3 / 4 observed actions |
| --- | ---: | ---: | ---: | ---: |
| 1 | 59,839 | 84,305 | 35.22% | 41,040 / 14,093 / 3,745 / 961 |
| 2 | 59,984 | 84,091 | 35.05% | 41,283 / 14,208 / 3,580 / 913 |
| 3 | 59,838 | 83,933 | 35.07% | 41,215 / 14,103 / 3,568 / 952 |

Approximately **69% of states have only one recorded action**. The treatment
loss averages SmoothL1 over each sampled state's distinct observed actions,
then over the same 64 sampled states. A global mean over observed state/action
pairs would overweight states with more actions and is deliberately not used.

Rewards, ends and successor choices come only from logs. At each recorded
nonterminal successor, ordinary DDQN selects the online network's highest
predicted action value across all four actions and evaluates it with the
target network. It can therefore select an action without a logged outcome
at that successor. Predicting its value does not supply its actual outcome.
Terminal targets use logged rewards without a successor query. Exact Q*
and exhaustive transitions remain integrity/rollout diagnostics; the optimizer
does not receive them.

The unchanged 20,420-parameter network, seed-specific initial online/target
weights, Adam settings, clipping, soft target update and 30,000-update budget
are preserved. All nine fits finish before learned evaluation. The run makes
**270,000 new updates / 17,280,000 state presentations**, preserving 45
treatment snapshots and nine copied control finals.

| Supervision/query measurement | Four-action control, historical | Recorded-action treatment, new |
| --- | ---: | ---: |
| State presentations | 17,280,000 | 17,280,000 |
| Supervised action targets | 69,120,000 | 24,273,701 |
| Recorded terminal targets | — | 1,062,190 |
| Recorded nonterminal targets/queries | — | 23,211,511 |
| Nonterminal queries outside current support | Approximately 34.4–34.6% by bank | 0 |

Treatment receives **35.12% of the control's action-target presentations**.
Its bank totals are **8,115,408 / 8,078,271 / 8,080,022**. Actual optimizer
counters and saved successor-query vectors agree with independent reductions
from the state presentation counts and logged tables. Every nonterminal
logged successor appears as a subsequent current state in the complete
collection history, explaining the verified zero outside-support queries.
Counterfactual control successors do not have that property. Thus successor
query composition changes along with action supervision.

All original local/global replay digests and count vectors reconstruct exactly;
all nine treatment local/global/map streams match them. Current-state support,
map/clock/category exposure and batch diversity remain identical within each
pair. Every supported state is sampled, with no direct off-support samples.
Recorded NumPy tables and optimizer tensors retain their original hashes.

## Behavior on the same prospective panels

Eight new 64-map panels scan from `1080000 + 1000 × panel`, excluding training
and the prior panels named in the protocol. Three prescribed rejections occur:
1082040 matches the coverage panel; 1082052 and 1085001 match within-map
composition panels. Original spawns are preserved and selection never uses
outcomes. Only final policies are evaluated: **27,648 learner and 3,584 shared
reference episodes**, totaling **31,232** on 512 layouts.

Each greedy bank/condition has 1,536 episodes. Efficient success means success
within twice the map's shortest-path length, including failures in the
denominator. Cells show four-action control / recorded-action treatment.

| Bank | Success | Efficient success | Efficiency Δ | Mean steps, all episodes |
| --- | ---: | ---: | ---: | ---: |
| 1 | 78.52% / 64.00% | 46.09% / 20.90% | −25.20 pp | 15.17 / 20.44 |
| 2 | 77.73% / 63.80% | 39.52% / 20.51% | −19.01 pp | 16.95 / 20.89 |
| 3 | 79.04% / 64.19% | 43.49% / 22.46% | −21.03 pp | 15.77 / 20.15 |

The equal-bank mean loses **21.74 efficiency points** and **14.43 success
points**, with **4.54 more steps** per episode. Equal counts make these
means numerically equal to episode pooling, with 4,608 greedy episodes per arm:

| Outcome | Four-action control | Recorded-action treatment |
| --- | ---: | ---: |
| Success | 78.43% (3,614/4,608) | 64.00% (2,949/4,608) |
| Efficient success | 43.03% (1,983/4,608) | 21.29% (981/4,608) |
| Mean steps, all episodes | 15.96 | 20.50 |
| Mean steps, successes only | 11.55 | 14.02 |
| Blocked steps / total steps | 64.19% (47,209/73,543) | 68.58% (64,769/94,441) |

There are **1,002 fewer efficient successes** and **665 fewer total successes**.
Successful-only means compare different successful subsets. The blocked-step
ratio rises **4.389 points** when pooled, versus **4.406** for the mean of
bank ratios and **4.464** for the mean of bank/learner ratios. These distinct
denominators remain visible in the saved results.

## Panels, learners and reference levels

| Panel | Bank 1 efficiency Δ | Bank 2 efficiency Δ | Bank 3 efficiency Δ |
| --- | ---: | ---: | ---: |
| 0 | −22.92 pp | −11.46 pp | −26.04 pp |
| 1 | −17.71 pp | −13.02 pp | −15.10 pp |
| 2 | −27.60 pp | −21.88 pp | −22.92 pp |
| 3 | −29.17 pp | −22.92 pp | −20.31 pp |
| 4 | −29.17 pp | −19.79 pp | −22.40 pp |
| 5 | −29.69 pp | −25.00 pp | −22.92 pp |
| 6 | −22.92 pp | −19.79 pp | −23.96 pp |
| 7 | −22.40 pp | −18.23 pp | −14.58 pp |

All **24/24 bank-panel means** decline in both efficiency and success.
Efficiency effects range from **−29.69 to −11.46 points**. At the finer
bank/panel/learner level, **71/72 decline and one ties** for each metric,
with different tied cells: efficiency bank 2 / panel 0 / seed 0; success bank
1 / panel 3 / seed 1. There are no positive cells for either metric.

All nine whole-panel learner effects are negative. Efficiency changes by seed
0 / 1 / 2 are bank 1 **−24.22 / −26.56 / −24.80**, bank 2 **−16.80 / −21.88 /
−18.36**, and bank 3 **−17.77 / −24.02 / −21.29** points. These crossed cells
share banks, maps, initializations and panels; they are not 72 independent
replications or a significance test.

Historical all-learner 70% success/above-random reference crossings occur on
**6/8 panels for each control bank**, versus **0/8, 0/8 and 1/8** for recorded
actions. No bank/condition has all learners reach 80% efficient success on any
panel. These are descriptive references, not new competence gates; earlier
training-fit and gate results remain unchanged.

With epsilon 0.1, separate diagnostic success / efficiency is **82.03% /
43.47%** control and **70.11% / 22.03%** treatment. Exploration helps some
episodes but does not replace the primary greedy comparison. Shared random
references succeed **42.12% (1,294/3,072)**; the planner succeeds on all 512
maps, averaging **3.64 steps**. This is a sharp decline, not total failure
to learn. Previous synthetic-support policies were not evaluated on these
panels; their old scores cannot supply a matched comparison here.

## Decision and next bounded control

Keep recorded-action learning as the baseline for the next actual-experience
investigation, while preserving the privileged control as a diagnostic. Do
not add memory or declare online competence. The access restriction reveals
a substantial weakness that was less visible with four-action supervision.

**Next: restrict bootstrap action selection to actions logged at each recorded
successor.** Compare against the new recorded-action DDQN policies. Keep
current-state masks, recorded rewards/successors, exact state replay, per-state
loss, network, initialization, target count and update budget fixed. Change
only the online argmax used to construct nonterminal targets: select among
the successor's nonempty recorded-action set, then evaluate that action with
the target network. Every queried successor has such a mask in these logs.

Fresh-world evaluation must still choose greedily across all four predicted
actions, exactly as here; fresh states do not have logged masks. Record how
often unrestricted successor argmax chooses an unrecorded action, and the
signed restricted-versus-unrestricted target difference under the same
weights. Do not assume targets always decrease: online selection and target
evaluation may rank actions differently.

This keeps outcome access restricted and tests sensitivity to the bootstrap
action set. It changes the backup operator and can exclude genuinely useful
actions; it does not establish extrapolation as the sole cause or constrain
unobserved current-action predictions. A decline would not rule out
extrapolation as a contributor. This proposal has not run and needs its own
protocol. No further collection, memory or additional comparison was executed
within this milestone.

## Inspect and reproduce

The dashboard adds **304 preselected recordings**, preserving **1,706 older
recordings**. It shows observed-action distributions, equal state exposure,
unequal action-target budgets, actual versus structural successor queries,
frozen-control labels and exact replay checks. Active rules remain visible
initially. Recordings use the first accepted map per panel, fixed before
outcomes; no favorable example was substituted.
On the first preselected map, 1080000, default bank 1 / seed 0 takes three
steps in both arms. Bank 2 / seed 0 improves from 26 to seven, while seed 1
worsens from three to 17. Individual improvements coexist with the consistent
decline in panel averages.

The main run took **166.54 seconds (2.78 minutes)** and peaked at
**555,122,688 bytes (0.517 GiB)** on one CPU thread at reduced priority.
Independent auditing passed all **304 recordings / 4,824 steps**, including
exact regenerated learned outputs, without retraining or repeating the full
policy evaluation. See validation for audit scope and UI review limits.

Use a fresh output directory with Python 3.12, Torch 2.8.0 and NumPy 2.0.2:

```bash
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 python -m q6.recorded_actions \
  --output experiments/recorded_actions/my-reproduction \
  --protocol-file docs/experiments/recorded_actions_protocol_v1.md
python scripts/audit_recorded_actions.py --study experiments/recorded_actions/my-reproduction
```

Smoke uses short treatments, archived 30,000-update controls and alternate
panels; it validates execution only and is ineligible scientific evidence.
License remains undecided; Q6 remains a public research preview.
