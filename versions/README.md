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
| [v7-ablations (ablation 1b)](v7_ablation2_floor_tuning_planned.md) | Phase 3 · Floor tuning + rectification warmup | Planned | target: window closed while cher_dependency decline preserved | — |

## The Research Thread

```
v1 → "Can DQN play this game?" → Yes, but against scripted opponent only
v2/v3 → "Can it generalize?" → Bug-fixed self-play works, but pool dynamics break collection
v4 → "More training = more collection?" → No. Bimodal collapse is fundamental
v5 → "Dual heads + CHER = collection?" → Yes! But still CHER-dependent
v6 → "Fix the reward function?" → Implemented as v7 (see below), confounded
v7 → "Reward redesign fixes the asymmetry?" → Never cleanly tested — a second change (weighted CHER) undermined the first
v7-ablations → "Is collection paralysis also a training-distribution problem?" → Partly — CHER dependency finally declines, but rectification creates its own mid-training vulnerability window
v7-ablations (1b) → "Can the vulnerability window be closed without losing the CHER-dependency win?" → Planned — raise the floor, decouple the rectification/easy-warmup schedules
```

## Dashboard

Browse all runs and replays at: `http://localhost:8080/dashboard/versions.html`
