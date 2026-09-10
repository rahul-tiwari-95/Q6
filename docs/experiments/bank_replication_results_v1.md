# Independent support-bank replications — results v1

**The uniform-support efficiency advantage repeats in all three new bank
pairs: +17.12, +27.67 and +23.31 percentage points.** Equal-bank mean efficient
success is **65.47% versus 42.77%** for collected support. Mean episode length
falls from **15.90 to 11.51 steps**, and success rises from **78.54% to 83.57%**.
This supports testing a concrete replay intervention on collected experience.
It does not yet identify which composition difference causes the advantage.

The [protocol](bank_replication_protocol_v1.md) was locally declared, reviewed
and pushed at `7ae6f47`; the single main run used clean, pushed source
`156343273b084777a08cc9206b270940db002462`, including a pre-run denominator
clarification. This is not external registration or peer review. The run
completed without deviations. See [raw results](../../experiments/bank_replication/pilot_v1/results.json),
[paired episodes](../../experiments/bank_replication/pilot_v1/paired_layouts.csv),
[validation](../validation/bank-replication-v1.md), and the
[dashboard](http://127.0.0.1:8080/dashboard/lab.html#coverage), selecting
**Experience coverage → Bank replications**.

## What was compared

Three independent random-action histories use the same 256 archived training
layouts, with 16 complete episodes per map per bank. Each resulting set of
unique pre-action states is paired with a newly drawn uniform subset of the
163,840-state exhaustive archive containing exactly the same number of rows.
Collection uses owned streams `[map_seed, repetition, 77301, bank_id]`; uniform
selection uses `[88301, bank_id]`. All six supports are frozen before training.
Repeated visits are provenance, not extra training weight.

Each support trains the same 20,420-parameter feedforward network with DDQN,
three shared learner initializations, batch size 64, four-action SmoothL1 loss,
Adam and 30,000 updates. Local row-sampling schedules match within each
bank/learner pair; the global states differ. Every arm retains privileged
four-action transitions and detached successor queries into the full archive.
No online interaction updates or memory are introduced.

All **18 fits / 540,000 updates** finish before any learned-policy evaluation. The
90 saved snapshots document training; only the final snapshot is evaluated,
without checkpoint selection or a new full-state fitting sweep.

Evaluation uses eight prospectively selected 64-map panels, excluding training,
all four earlier fresh panels, the preceding 512-map robustness evaluation and
previously accepted new layouts. Candidate 1020010 duplicates training layout 300140; candidate 1025052
duplicates already accepted layout 1020030, under walls + pellet + rules
identity. Panels 0 and 5 therefore end at 1020064 and 1025064. Panels
1/2/3/4/6/7 use 64 consecutive seeds starting at 1020000 + 1000 × panel ID. No outcome-based
selection or difficulty balancing occurs.

Each model receives one greedy and two epsilon-0.1 episodes per map with paired
external exploration draws. There are **27,648 learner episodes** and **3,584
shared references** on **512 layouts**. The three support pairs are the bank
replications. Shared learner seeds and panels are crossed measurements, not
additional independent bank draws or world-family replications.

## Primary result by bank

Each greedy condition/bank has 1,536 episodes. Cells below show collected /
uniform. Efficient success means reaching the goal within twice that map's
planner length, with failures included in the denominator.

| Bank | States in each arm | Success: collected / uniform | Efficient: collected / uniform | Uniform efficiency Δ | Mean steps: collected / uniform |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 59,839 | 79.88% / 82.88% | 44.73% / 61.85% | +17.12 pp | 15.33 / 12.31 |
| 2 | 59,984 | 77.02% / 84.90% | 40.76% / 68.42% | +27.67 pp | 16.50 / 10.80 |
| 3 | 59,838 | 78.71% / 82.94% | 42.84% / 66.15% | +23.31 pp | 15.86 / 11.41 |

The equal-bank mean difference is **+22.70 points** in efficient success,
**+5.03 points** in success and **−4.39 steps** per episode. Because every bank
has equal evaluation counts, these three equal-bank means also equal their
pooled episode summaries. There are 4,608 greedy episodes per condition:

| Outcome | Collected unique | Uniform subset |
| --- | ---: | ---: |
| Success | 78.54% (3,619/4,608) | 83.57% (3,851/4,608) |
| Efficient success | 42.77% (1,971/4,608) | 65.47% (3,017/4,608) |
| Mean steps, all episodes | 15.90 | 11.51 |
| Mean steps, successes only | 11.50 | 7.48 |
| Blocked steps / total steps | 65.00% (47,620/73,264) | 66.95% (35,500/53,023) |

Uniform yields 1,046 more efficient successes and 232 more total successes.
Successful-route means compare different successful subsets. Its blocked-step
fraction is higher despite fewer total blocked steps. Pooled blocked-fraction
difference is +1.95 points; the mean of within-bank pooled differences is
+1.91 points, and the mean of bank/learner differences is +1.92 points. These
ratios have different denominators and should not be interchanged.

## Does the result survive the individual panels and learners?

Uniform efficiency improves on **24/24 bank-panel means**. All are shown here;
these are descriptive repeated measurements on shared panels.

| Panel | Bank 1 efficiency Δ | Bank 2 efficiency Δ | Bank 3 efficiency Δ |
| --- | ---: | ---: | ---: |
| 0 | +18.75 pp | +35.94 pp | +27.08 pp |
| 1 | +11.98 pp | +18.23 pp | +23.96 pp |
| 2 | +17.71 pp | +32.29 pp | +19.79 pp |
| 3 | +19.79 pp | +27.60 pp | +28.12 pp |
| 4 | +18.23 pp | +31.77 pp | +15.62 pp |
| 5 | +17.19 pp | +26.56 pp | +20.83 pp |
| 6 | +18.75 pp | +26.04 pp | +30.73 pp |
| 7 | +14.58 pp | +22.92 pp | +20.31 pp |

The bank-panel efficiency range is **+11.98 to +35.94 points**. Mean steps fall
on all 24 bank-panel means, by **2.04–7.80 steps**. Success improves on 20/24
bank-panel means and reverses on four, spanning **−3.12 to +17.19 points**.

At the finer bank/panel/learner level, efficiency improves in **71/72 cells**.
The exception is bank 1 / panel 5 / seed 2: efficiency −1.56 points, success
−3.12 points and mean steps +0.81. Across all panels, efficiency improves in
all nine bank/learner pairs, but success reverses for bank 1 / seed 2
(−0.59 points). A positive bank mean does not imply every learner improves.
No significance test or population-wide ranking is claimed.

Descriptively, every learner crosses the historical 70% success level and
exceeds the panel's random reference on **6/8, 7/8 and 7/8 collected panels**
versus **8/8 for each uniform bank**. No bank/condition has every learner cross
80% efficient success on any panel. These are reference-line counts, not new
competence gates; old gates and full-state fit results remain unchanged.

The separately reported epsilon-0.1 diagnostic gives success / efficient
success of **81.90% / 43.14%** collected and **86.19% / 65.17%** uniform. It does
not replace the primary greedy contrast. Shared random references succeed in
**42.51% (1,306/3,072)**; the planner succeeds on all 512 maps, averaging
**3.59 steps**.

## What the collected banks contain

| Bank | Collection steps | Unique states | Collected states per map, range | Uniform states per map, range | Collected / uniform goal-near coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 100,878 | 59,839 | 9–329 | 206–265 | 26.51% / 36.52% |
| 2 | 100,326 | 59,984 | 9–331 | 197–271 | 26.49% / 36.48% |
| 3 | 100,381 | 59,838 | 7–326 | 200–284 | 26.53% / 36.62% |

Each collector completes 4,096 episodes; total collection is **12,288 episodes
and 301,585 steps**. Matched banks contain 36.52–36.61% of exhaustive states.
The near-goal statistic covers states within two shortest-path moves over all
remaining clocks; it measures membership, not the frequency of actual visits.

Collection repeatedly underrepresents clocks 25–32 (29.15–29.53% coverage)
relative to its middle clocks and to uniform support (36.41–36.62%). Its
nonterminal four-action edges leave the current-state support about
34.42–34.59% of the time, versus 63.43–63.48% for uniform. Uniform performs
better despite more out-of-support successor queries; that statistic alone
cannot explain the efficiency deficit.

Within-pair Jaccard overlap is 0.222–0.224. Independent collected banks share
about 56% of their rows (Jaccard 0.390–0.393), versus roughly 36–37% for uniform
banks. Independent draws need not be disjoint. Stable map, clock and goal-near
skews describe this collector on these training layouts; they do not isolate
a single causal component.

## Next bounded experiment

**Compare map-balanced replay against uniform-over-unique-state replay on
these same three archived collected supports.** A concrete candidate is 64
distinct uniformly selected training maps per update, then one uniformly
selected supported current state per map. Every map has support, so this
requires no new rows or oracle-selected states. Preserve the network, DDQN,
three initializations, four-action loss and 30,000-update budget; declare new
evaluation panels before fitting and evaluate the archived baseline on those
same panels.

This directly tests whether a practical replay change improves use of the
experience already collected. It changes both map weighting and within-batch
map diversity, so it is a replay-design intervention rather than a pure
weighting estimate. Report actual row exposure and each bank separately. A
gain would show that missing support is not the whole explanation; failure
would leave within-map composition and missing states unresolved. Clock and
goal-near balancing remain separate hypotheses, avoiding several simultaneous
changes. This recommendation has not run and needs its own declared protocol.

Memory remains deferred. The evidence still uses privileged offline action
access and the same training world family. It does not establish online-RL
competence or make A → B → A forgetting interpretable yet.

## Inspect and reproduce

The dashboard exposes every bank separately, equal-bank primary effects,
map/clock coverage, all eight evaluation panels, loss windows and **304
preselected recordings**, preserving **794 earlier learner-study recordings**.
The first preselected map is mixed: bank 1 / seed 0 takes 23 collected steps
versus 6 uniform steps, but uniform seeds 1 and 2 fail where collected
succeeds. No cleaner example was substituted. The active movement rule is visible initially. Final-policy replay selectors
do not imply evaluation of intermediate snapshots.

The single main run took **243.59 seconds (4.06 minutes)** and peaked at
**461,717,504 bytes / 0.430 GiB process RSS**, using one CPU thread at reduced
priority. It stayed within the declared 1,200-second / 4-GiB sampled guard.
The independent audit passed all raw arithmetic, collector draws, support
sampling, snapshots and all 304 saved replays / 4,561 steps, including exact
network outputs. It did not retrain or repeat the complete policy evaluation.

Use a new output directory and the pinned runtime with shipped archives:

```bash
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 python -m q6.bank_replication \
  --output experiments/bank_replication/my-reproduction \
  --protocol-file docs/experiments/bank_replication_protocol_v1.md
python scripts/audit_bank_replication.py --study experiments/bank_replication/my-reproduction
```

Python 3.12, Torch 2.8.0 and NumPy 2.0.2 match the main run. The environment,
source, protocol, raw tables, supports and checkpoints are preserved in its
archive. Smoke runs use separate maps and remain ineligible research evidence.
License remains undecided: Q6 is a public research preview.
