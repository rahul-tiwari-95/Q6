# No Way Home — Learning Citizen: Does It Discover Lineage-Awareness?

Trained on 50 episodes (seeds 2000-2049), evaluated frozen (epsilon=0, pure exploitation) on 20 held-out seeds disjoint from training. Primary metric: need-shortfall/10k ticks, lower is better.

| Policy | Mean shortfall/10k (± std) |
|---|---:|
| C3c_lineage_aware_heuristic (fixed script) | 43526 ± 427 |
| greedy_state_oracle (cheats: sees hidden state) | 44318 ± 318 |
| C3b_lineage_naive_heuristic (fixed script) | 57295 ± 8856 |
| Learner (raw_rate + unique_rate features) | 60864 ± 3340 |
| Learner (raw_rate feature only) | 63615 ± 310 |
| never_mitigate | 112990 ± 315 |

## Learned Q-table: raw_rate-only learner

State bin = (raw_rate bin). Bin edges: [0.0, 0.1, 0.3, 0.6, 1.0, 2.0, inf]

```
  bin=(0,)  Q(mitigate)=  -0.01  Q(don't)=  -0.07  -> prefers MITIGATE
  bin=(1,)  Q(mitigate)=  -1.61  Q(don't)=  -2.40  -> prefers MITIGATE
  bin=(2,)  Q(mitigate)=  -2.73  Q(don't)=  -3.27  -> prefers MITIGATE
  bin=(3,)  Q(mitigate)=  -8.08  Q(don't)=  -9.38  -> prefers MITIGATE
  bin=(4,)  Q(mitigate)= -60.13  Q(don't)= -63.15  -> prefers MITIGATE
  bin=(5,)  Q(mitigate)= -95.30  Q(don't)=-102.34  -> prefers MITIGATE
  bin=(6,)  Q(mitigate)= -51.79  Q(don't)= -44.63  -> prefers don't
```

## Learned Q-table: both-features learner

State bin = (raw_rate bin, unique_rate bin). Bin edges: [0.0, 0.1, 0.3, 0.6, 1.0, 2.0, inf]

```
  bin=(0, 0)  Q(mitigate)=  -0.00  Q(don't)=  -0.03  -> prefers MITIGATE
  bin=(1, 1)  Q(mitigate)=  -0.61  Q(don't)=  -2.31  -> prefers MITIGATE
  bin=(2, 1)  Q(mitigate)=  -0.26  Q(don't)=  -1.11  -> prefers MITIGATE
  bin=(2, 2)  Q(mitigate)=  -8.24  Q(don't)= -10.71  -> prefers MITIGATE
  bin=(3, 1)  Q(mitigate)=  -0.21  Q(don't)=  -0.13  -> prefers don't
  bin=(3, 2)  Q(mitigate)= -10.11  Q(don't)= -11.04  -> prefers MITIGATE
  bin=(3, 3)  Q(mitigate)= -20.12  Q(don't)= -29.29  -> prefers MITIGATE
  bin=(4, 1)  Q(mitigate)=  -0.72  Q(don't)=  -0.05  -> prefers don't
  bin=(4, 2)  Q(mitigate)= -18.28  Q(don't)= -25.33  -> prefers MITIGATE
  bin=(4, 3)  Q(mitigate)= -42.08  Q(don't)= -53.86  -> prefers MITIGATE
  bin=(4, 4)  Q(mitigate)= -82.21  Q(don't)= -92.93  -> prefers MITIGATE
  bin=(5, 3)  Q(mitigate)= -95.21  Q(don't)=-103.83  -> prefers MITIGATE
  bin=(5, 4)  Q(mitigate)= -91.64  Q(don't)= -99.54  -> prefers MITIGATE
  bin=(5, 5)  Q(mitigate)= -97.92  Q(don't)=-105.10  -> prefers MITIGATE
  bin=(6, 5)  Q(mitigate)= -52.09  Q(don't)= -44.42  -> prefers don't
```

## Verdict

Forward-storm-signature bins visited during training (high raw_rate, low unique_rate): [(3, 1), (4, 1)].

**The both-features learner correctly learned NOT to mitigate in every forward-storm bin it visited** -- it discovered the lineage-aware distinction on its own, from reward feedback alone, with nothing hand-coding which feature to trust.

Outcome comparison: both-features learner scores 60864, vs. the hand-coded aware heuristic's 43526 and the hand-coded naive heuristic's 57295. The raw-only learner (structurally blind to lineage, same handicap as the naive heuristic) scores 63615.

**The residual gap is real and worth naming, not glossing over.** Despite reliably discovering the lineage-aware distinction, the both-features learner still scores well above the hand-coded aware heuristic. The most likely reason: its state features are only [raw_rate, unique_rate] -- it has no notion of current wealth/budget at all. `results/smoke_test_v1.md` already found that spending DISCIPLINE (not just correct information) is a separate skill in this world -- `greedy_state_oracle`, with perfect hidden-state information, was beaten by a purely-reactive heuristic because it spent its wealth greedily. This learner has the same blind spot for the same reason: nothing in its state tells it when spending is or isn't affordable, so it can learn WHEN evidence is real, but not WHETHER it can currently afford to act on it. Adding wealth as a third feature is the natural next step, not a new problem.
