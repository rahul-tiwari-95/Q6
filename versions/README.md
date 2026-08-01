# Q6 Version History Index

Each version documents: thesis → architecture → results → failure mode → decision for next version.

| Version | Phase | Status | Key Result | Failure Mode |
|---|---|---|---|---|
| [v1](v1_phase1_baseline.md) | Phase 1 · Single-agent | Complete | avg100 ~+15 vs scripted Hunter | Overfit to deterministic pathfinding |
| [v2/v3](v2_v3_phase2_selfplay_init.md) | Phase 2 · Self-play init | Complete | avg100 ~+4, self-play stable | Pool fills with hard Hunters, collection drops |
| [v4](v4_phase2_bimodal.md) | Phase 2 · Hardened | Complete | avg100 +4.9 @ ep 5300 | **Bimodal collapse** — pure evasion, 0 collection |
| [v5](v5_phase3_gop_cher.md) | Phase 3 · GOP+CHER | Complete | 2567 pellets, first 2-pellet ep | CHER-dependent, reward asymmetry unresolved |
| [v6](v6_planned.md) | Phase 3 v2 · Reward fix | Planned | target avg100 > 15 | — |
| v7 | Phase 3 v2 · Reward redesign + bugfix | Complete, not separately written up | 0 Krishna wins / 12k eps, avg100 negative throughout | Confounded — weighted-CHER change likely undermined the mechanism it was meant to strengthen; see Q6.md §4 |
| [v7-ablations (ablation 1)](v7_ablation1_rectified.md) | Phase 3 · Rectified opponent sampling | Complete | cher_dependency declined for the first time (0.315→0.160) | Mid-training vulnerability window (ep 1000–4000, ~50% Hunter win rate) as the flagged rectification risk materialized |
| [v7-ablations (ablation 1b)](v7_ablation1b_floor_tuning.md) | Phase 3 · Floor tuning + rectification warmup | Complete — hypothesis rejected | mean_hard_tier_score trough improved 2.4x (0.049→0.115) but missed the >0.25 bar; pellet-touch rate rose overall but declining by run's end | Window not closed — windowed Hunter win rate got worse (47.7%→57.7%), and the cher_dependency decline reversed (0.167→0.237); joint-mode win rate stayed elevated through ep 6000, not just the target window |

## The Research Thread

```
v1 → "Can DQN play this game?" → Yes, but against scripted opponent only
v2/v3 → "Can it generalize?" → Bug-fixed self-play works, but pool dynamics break collection
v4 → "More training = more collection?" → No. Bimodal collapse is fundamental
v5 → "Dual heads + CHER = collection?" → Yes! But still CHER-dependent
v6 → "Fix the reward function?" → Implemented as v7 (see below), confounded
v7 → "Reward redesign fixes the asymmetry?" → Never cleanly tested — a second change (weighted CHER) undermined the first
v7-ablations → "Is collection paralysis also a training-distribution problem?" → Partly — CHER dependency finally declines, but rectification creates its own mid-training vulnerability window
v7-ablations (1b) → "Can the vulnerability window be closed without losing the CHER-dependency win?" → No — raising the floor and ramping it in widened hard-tier exposure (a real, measured effect) but made the windowed Hunter win rate worse, not better, and reversed the CHER-dependency decline; the two changes were confounded together, so which one (if either) is actually responsible is still open
```

## Dashboard

Browse all runs and replays at: `http://localhost:8080/dashboard/versions.html`
