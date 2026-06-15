# Q6 Version History Index

Each version documents: thesis → architecture → results → failure mode → decision for next version.

| Version | Phase | Status | Key Result | Failure Mode |
|---|---|---|---|---|
| [v1](v1_phase1_baseline.md) | Phase 1 · Single-agent | Complete | avg100 ~+15 vs scripted Hunter | Overfit to deterministic pathfinding |
| [v2/v3](v2_v3_phase2_selfplay_init.md) | Phase 2 · Self-play init | Complete | avg100 ~+4, self-play stable | Pool fills with hard Hunters, collection drops |
| [v4](v4_phase2_bimodal.md) | Phase 2 · Hardened | Complete | avg100 +4.9 @ ep 5300 | **Bimodal collapse** — pure evasion, 0 collection |
| [v5](v5_phase3_gop_cher.md) | Phase 3 · GOP+CHER | Complete | 2567 pellets, first 2-pellet ep | CHER-dependent, reward asymmetry unresolved |
| [v6](v6_planned.md) | Phase 3 v2 · Reward fix | Planned | target avg100 > 15 | — |

## The Research Thread

```
v1 → "Can DQN play this game?" → Yes, but against scripted opponent only
v2/v3 → "Can it generalize?" → Bug-fixed self-play works, but pool dynamics break collection
v4 → "More training = more collection?" → No. Bimodal collapse is fundamental
v5 → "Dual heads + CHER = collection?" → Yes! But still CHER-dependent
v6 → "Fix the reward function?" → TBD
```

## Dashboard

Browse all runs and replays at: `http://localhost:8080/dashboard/versions.html`
