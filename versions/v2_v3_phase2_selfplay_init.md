# v2/v3 — Phase 2 Early: Self-Play Init + Critical Bug Fixes

**Date:** May–June 2026  
**Episodes:** ~3,000 each  
**Status:** Complete (superseded by v4)

---

## Thesis

Introduce Fictitious Self-Play (FSP): maintain a pool of historical Hunter
snapshots and train both agents simultaneously. Both agents see diverse opponents,
preventing over-fitting to any single policy. Hypothesis: Krishna develops a
robust evasion+collection strategy because it faces many Hunter skill levels.

## Architecture Changes from v1

| Component | Change |
|---|---|
| Environment | New `SelfPlayGridworld` — both agents step simultaneously |
| Hunter agent | DQNv2Agent — Hunter now learns alongside Krishna |
| FSP pool | FIFO queue of 20 Hunter snapshots, saved every 100 eps |
| Training split | 70% joint (latest Hunter) / 30% FSP (pool sample) |
| State encoding | Unchanged: 6-channel binary (6,25,25) |

## v2 → v3 Bug Fixes

v2 shipped with two critical bugs discovered during first training runs:

1. **Epsilon not decaying**: `decay_epsilon()` was not called in the training
   loop after each episode. Result: Krishna never stopped exploring, avg100 stayed
   at ~−20 for the entire 3000-episode run.

2. **Position update race condition**: In `SelfPlayGridworld.step()`, both agents'
   new positions were computed simultaneously but written sequentially, creating a
   state where both agents could occupy the same cell without triggering the catch
   mechanic. This made Hunter wins impossible — Hunter could walk through Krishna.

v3 fixed both bugs. Training became functional.

## Key Files

- `environment/selfplay_env.py` — SelfPlayGridworld
- `agent/dqn_v2_agent.py` — DQNv2Agent (Hunter)
- `train_phase2.py` — Phase 2 training loop
- `agent/frozen_agent.py` — FrozenAgent (wraps checkpoint as read-only opponent)
- `utils/environment_wrapper.py` — SelfPlayEnv wrapper

## Results

| Metric | v2 | v3 |
|---|---|---|
| avg100 at end | ~−20 (ε never decayed) | ~+4 |
| pellets/episode | ~0 | ~0.2 |
| self-play stable | ✗ | ✓ |

## What Worked

- FSP architecture functioned after bug fixes
- Hunter and Krishna both show learning curves
- Pool filling worked correctly

## Failure Mode

**Pool fills too fast with hard opponents.** By ep ~1500, the pool contains 15+
snapshots of increasingly capable Hunters. The 5 easy early-training Hunters are
evicted. Once pool=20 all-hard, the optimal Krishna strategy shifts to pure
evasion: attempting to collect a pellet (which requires slowing down near pellet
positions) risks encountering a hardened Hunter. The expected reward is:

```
E[r_collect] = P(survive) × (+50) + P(caught) × (−80) ≈ −5 at pool=20
E[r_evade] = P(survive) × (+0.5/step × 1000) ≈ +2 at pool=20
```

Evasion dominates. Collection drops to near zero.

## Decision for v4

The bug fixes and architecture are correct. Run longer (6000 eps) to see if
collection eventually breaks through with a more mature policy.

→ Also note: the single policy head cannot learn both "evade" and "collect"
simultaneously under adversarial pressure. The two behaviors require conflicting
gradient directions. This is the root problem that v4 will confirm and v5 will solve.
