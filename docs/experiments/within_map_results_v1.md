# Within-map composition with fixed replay — results v1

**Changing the states inside each map improves efficiency while map exposure
and batch diversity remain exactly fixed.** Efficient-success gains are
**+19.47, +28.65 and +20.70 percentage points** across the three bank pairs.
The equal-bank mean rises **45.75% → 68.68%**, with positive effects on all
24 bank-panel means and all 72 bank/panel/learner cells.

This strengthens the evidence for within-map experience composition. It
does not isolate position, clock, goal proximity or successor structure,
which change together. Both arms still receive privileged four-action
transitions; this is not an online-RL competence result.

The [protocol](within_map_protocol_v1.md) was reviewed and pushed at
`20844a5` before the single main run. Execution used clean, pushed source
`aa278609b06da1d0d28912268d695bc0d9450659`, with no deviations or subsequent
tuning run. This is a local declaration, not external registration or peer
review. See [raw results](../../experiments/within_map/pilot_v1/results.json),
[exposure](../../experiments/within_map/pilot_v1/exposure.json),
[paired episodes](../../experiments/within_map/pilot_v1/paired_layouts.csv),
[validation](../validation/within-map-v1.md), and the
[dashboard](http://127.0.0.1:8080/dashboard/lab.html#coverage), selecting
**Experience coverage → Within-map states**.

## The control

Start with the same three archived collected banks: **59,839 / 59,984 /
59,838 unique states** across 256 training maps. For each bank/map, draw once
uniformly without replacement from that map's exhaustive states, retaining
exactly its collected state count. Owned generators
`SeedSequence([bank_id,map_seed,105301])` produce **768 per-map draws and
three replacement supports**, shared across learner seeds and frozen before
training. Natural overlap is allowed: replacements retain 38.86%, 39.01%
and 38.70% of their respective original supports. No new trajectories are
collected.

Both supports are sorted in identical-length map blocks. The unchanged
64-distinct-state sampler, with `SeedSequence([seed,66301])`, selects the
same ordered local slots in each pair. Therefore every batch has exactly
the same ordered map identities and map diversity. Historical local/global
digests and counts are reconstructed for all nine archived 30,000-update
controls, and all treatment local/map streams match. This is a within-bank
pairing; different banks need not have identical map schedules.

Reuse the nine archived final control policies without retraining. Train nine
replacements from identical seed-specific online/target initializations,
keeping the **20,420-parameter network**, DDQN update, four-action SmoothL1
loss, optimizer and **30,000 updates per fit** fixed. That is **270,000 new
updates / 17,280,000 state presentations / 69,120,000 action targets**.
All fits finish before learned evaluation; only final policies are evaluated.
Forty-five treatment snapshots and nine copied control finals are preserved.

Eight prospectively selected 64-map panels start at `1060000 + 1000 × panel`.
They exclude training and every prior panel specified in the protocol by
walls, pellet and rules identity, retaining original spawns and never
consulting outcomes.
Five candidates are excluded: 1060002 and 1067022 match bank-replication
layouts, 1061037 matches map-replay, 1061047 matches training, and 1063005
matches equal-support. Every model receives one greedy and two epsilon-0.1
episodes per map, with paired external exploration draws: **27,648 learner
and 3,584 shared reference episodes = 31,232 evaluations** on 512 layouts.

## Behavior by bank

Each greedy bank/condition has 1,536 episodes. Efficient success means reaching
the goal within twice the map's shortest-path length; failures remain in the
denominator. Cells show original collected control / within-map replacement.

| Bank | Success | Efficient success | Efficiency Δ | Mean steps, all episodes |
| --- | ---: | ---: | ---: | ---: |
| 1 | 83.53% / 87.63% | 46.55% / 66.02% | +19.47 pp | 14.74 / 10.99 |
| 2 | 82.03% / 87.30% | 43.42% / 72.07% | +28.65 pp | 15.66 / 9.93 |
| 3 | 84.38% / 88.41% | 47.27% / 67.97% | +20.70 pp | 14.53 / 10.54 |

The equal-bank differences are **+22.94 efficiency points**, **+4.47 success
points**, and **−4.49 steps**. Equal evaluation counts make these means
numerically equal to episode pooling. Each condition has 4,608 greedy episodes:

| Outcome | Collected control | Within-map replacement |
| --- | ---: | ---: |
| Success | 83.31% (3,839/4,608) | 87.78% (4,045/4,608) |
| Efficient success | 45.75% (2,108/4,608) | 68.68% (3,165/4,608) |
| Mean steps, all episodes | 14.97 | 10.48 |
| Mean steps, successes only | 11.56 | 7.49 |
| Blocked steps / total steps | 61.83% (42,662/69,003) | 61.69% (29,799/48,307) |

The paired aggregate contains **1,057 more efficient successes** and **206
more total successes**. Successful-only means compare different successful
subsets. The blocked-step fraction barely changes, even though total blocked
steps fall substantially; in banks 1 and 2 the fraction rises while episodes
become shorter. This ratio alone would miss much of the behavioral improvement.
Its pooled change is −0.140 points, versus −0.148 for the mean of within-bank
pooled ratios and −0.046 for the mean of bank/learner ratios.

## Panels and initialization

| Panel | Bank 1 efficiency Δ | Bank 2 efficiency Δ | Bank 3 efficiency Δ |
| --- | ---: | ---: | ---: |
| 0 | +25.52 pp | +27.08 pp | +23.96 pp |
| 1 | +18.75 pp | +31.77 pp | +11.98 pp |
| 2 | +19.79 pp | +32.29 pp | +27.60 pp |
| 3 | +17.19 pp | +26.56 pp | +22.92 pp |
| 4 | +17.19 pp | +29.17 pp | +19.27 pp |
| 5 | +23.44 pp | +27.60 pp | +17.71 pp |
| 6 | +20.31 pp | +22.40 pp | +15.62 pp |
| 7 | +13.54 pp | +32.29 pp | +26.56 pp |

All **24/24 bank-panel efficiency means** improve, ranging from **+11.98 to
+32.29 points**. All **72/72 bank/panel/learner effects** improve, ranging
from +3.12 to +42.19 points. Across all panels, the nine learner effects are
also positive: bank 1 **+27.54 / +19.53 / +11.33**; bank 2 **+27.73 / +28.91 /
+29.30**; bank 3 **+16.21 / +20.12 / +25.78** points. These crossed cells share
banks, initializations and panels; they are not 72 independent replications.

Success is less uniform: 22/24 bank-panel means improve, one declines and
one ties; 55/72 learner-panel cells improve, 11 decline and six tie. All
nine whole-panel learner success effects are positive. Historical 70%
success/above-random reference crossings occur for all learners on **7/8,
7/8, 8/8** control panels and **8/8 for every replacement bank**. Neither
condition has all learners reach 80% efficient success on any panel. These
are descriptive historical reference levels, not new gates; previous strict
full-state fit results remain unchanged.

The separate epsilon-0.1 diagnostic gives success / efficient success of
**85.83% / 46.09%** for controls and **90.72% / 68.24%** for replacements.
It does not replace the primary greedy contrast. Shared random references
succeed **44.95% (1,381/3,072)**; the planner succeeds on all 512 maps,
averaging **3.50 steps**.

## What stayed fixed, and what changed?

All **768 quotas**, **nine ordered local/map stream pairs**, map counts and
local-slot presentation counts match exactly. Both arms sample every supported
state, with zero direct off-support samples. Map presentation-share CV remains
**0.249–0.258** in both arms; largest per-state presentation counts remain
**56–61**, with the entire within-support count distribution preserved in
each pair. No balancing or additional map diversity explains this contrast.

State identities differ. Goal-near states receive approximately **23.3% of
control presentations versus 30.9–31.2%** for replacements, as bank means
across learners. Query-weighted nonterminal successors outside the current
support rise from **34.43–34.56% to 60.99–61.36%**. These are presentation and
query shares, not percentages of all possible states covered. Clock, position
and successor structure also change. The dashboard separates these quantities.

The result shows a within-map composition effect under fixed map exposure
and replay, and supports broader within-map experience as a useful future
collection target. It does not establish which component supplies the gain,
or that an affordable collector can obtain the replacement states. Prior
uniform and map-balanced policies were not evaluated on these new panels;
their old scores cannot measure relative performance or gap closure here.

## Next: remove counterfactual action supervision

The next bounded bridge should compare **all-four-action supervision versus
only actions actually recorded**, using the same collected supports and
original replay schedule. The archived collection logs retain state/action,
reward, successor and termination records, so another collection run is not
needed to construct this control.

After verifying repeated state/action records agree, deduplicate them into
an observed-action mask. Average DDQN loss over recorded actions per sampled
state, then over the same 64 sampled states; retain the network, initialization,
optimizer and 30,000-update budget. Query successors only through recorded
transitions. Predicting all action values at those observed successors remains
ordinary DDQN bootstrapping and can still suffer offline extrapolation error.
The number of supervised action targets necessarily changes and must be
reported; state-presentation and optimizer budgets stay matched.

Nine archived four-action controls can be compared with nine new masked-action
fits on prospective common panels. This removes counterfactual supervision
without introducing another state-coverage or online-collection intervention.
It remains offline, deduplicated state replay rather than raw trajectory replay.
This recommendation has not run and needs its own protocol. Memory remains
deferred until actual-experience learning supports a competent baseline.

## Inspect and reproduce

The dashboard adds **304 preselected recordings**, preserving **1,402 prior
recordings**. The rule is visible initially; controls expose bank, learner,
panel, map quotas, clock exposure and state repetition. Overlapping map curves
are expected because map exposure is identical. Recorded cases were selected
before outcomes, not substituted afterward.
On the first preselected map, 1060000, bank 1 / seed 0 improves from 19 to
two steps; bank 3 / seed 0 improves from 14 to five. Switch learners to
compare routes on the same map before inspecting panel averages.

The single main run took **146.85 seconds (2.45 minutes)** and peaked at
**483,835,904 bytes (0.451 GiB)** on one CPU thread at reduced priority.
Independent auditing passed **304 recordings / 3,498 steps**, including exact
regenerated learned outputs, without retraining or repeating the complete
policy evaluation. Validation records the remaining UI review limits.

Use a fresh output directory with Python 3.12, Torch 2.8.0 and NumPy 2.0.2:

```bash
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 python -m q6.within_map \
  --output experiments/within_map/my-reproduction \
  --protocol-file docs/experiments/within_map_protocol_v1.md
python scripts/audit_within_map.py --study experiments/within_map/my-reproduction
```

Smoke uses 24-update treatments, archived 30,000-update controls and alternate
panels. Its matched-prefix checks validate execution only; it is ineligible
scientific evidence. License remains undecided; Q6 is a public research preview.
