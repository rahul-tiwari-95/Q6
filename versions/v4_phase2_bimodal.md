# v4 — Phase 2 v4: Self-Play Hardened (Bimodal Collapse)

**Date:** June 2026  
**Episodes:** 6,000  
**Duration:** 21.3 hours  
**Run ID:** `20260606_175631_phase2_v4_nowall`  
**Status:** Complete — confirmed bimodal collapse hypothesis

---

## Thesis

With bug-fixed self-play and a full FSP pool, and given enough training time
(6000 episodes), Krishna will converge to a balanced evasion+collection policy.
Hypothesis: collection paralysis is a transient early-training artifact that
resolves with longer training.

## Architecture (same as v3)

| Component | Detail |
|---|---|
| Network | CNNDuelingQNetwork (unchanged) |
| FSP pool | 20 slots FIFO |
| Training | 6000 eps, ε: 1.0 → 0.05 |
| Wall penalty | k_wall=0 (fixed: non-zero wall penalty caused corner-hugging) |
| Warm start | None — trained from scratch |

## Key Hyperparameters

```python
LEARNING_RATE   = 1e-4
GAMMA           = 0.99
TAU             = 0.001
BATCH_SIZE      = 64
BUFFER_SIZE     = 100_000
EPSILON_DECAY   = 0.9994   # reaches 0.05 at ~ep 5000
```

## Results

| Metric | Value |
|---|---|
| best avg100 | +4.9 |
| best checkpoint ep | 5300 |
| total pellets collected | ~30 |
| Krishna wins (4 pellets) | 6 |
| Hunter wins | ~380 |
| timeout rate | ~94% |
| late-game pellet rate | < 0.5% |

**Dashboard:** [Run details](run.html?id=20260606_175631_phase2_v4_nowall) · [Replay](replay.html?run=20260606_175631_phase2_v4_nowall)

## What Worked

- Self-play training stable for 6000 episodes without divergence
- Hunter became genuinely capable (not scripted — actual learned policy)
- Episode logging, checkpoint saving, dashboard all working
- avg100 > 0 sustained — Krishna is not just dying constantly

## Failure Mode — Bimodal Strategy Collapse (Confirmed)

Collection paralysis did **not** resolve with more training. By ep 4000:

- Krishna collected 0 pellets in 95%+ of episodes
- avg100 plateaued at +4–6 against pool=20
- Reward is entirely from survival (1000 steps × small positive reward)

**Root cause analysis:**

The single policy head receives conflicting gradient signals:
- To collect: move toward pellet positions (reduces distance)
- To evade: move away from Hunter (maximize distance)

When Hunter is hardened (pool=20), these are almost always in opposition. The
network converges to the local optimum that maximizes expected reward under this
constraint: **never collect, always evade**.

The reward asymmetry makes this optimal:
```
r_die   ≈ −80   (Hunter catch penalty × 3 lives lost)
r_pellet ≈ +50   (collect reward)
r_step  ≈ +0.5  (survival reward per step)
```

At pool=20, P(survive collection attempt) ≈ 0.4 → E[collect] ≈ −12.
P(survive evasion) ≈ 0.99 → E[1000 steps] ≈ +495.

The agent is making the mathematically correct decision given the reward function.
**The problem is the reward function, not the learning algorithm.**

## Decision for v5

Three architectural changes are needed simultaneously:

1. **Dual-head network**: Separate "evade" and "collect" policy heads with a
   learned gate (context-conditioned: hunter_dist, pellets_remaining, lives).
   The gate blends the two heads. This prevents gradient interference.

2. **Counterfactual HER (CHER)**: Post-episode, scan for timesteps where the
   Hunter was far and a pellet was nearby. Inject a counterfactual transition
   with the optimal collection action and +30 reward. This gives the collect head
   a training signal without requiring risky real-world attempts.

3. **Hierarchical opponent pool**: 5 permanent easy slots + 15 FIFO hard slots.
   p_easy=0.25. This guarantees Krishna always has safe practice time with weak
   opponents where collection is viable.

The v4 best checkpoint (`krishna_best_ep5300.pth`) is used as warm-start for v5,
giving the collect head a pre-trained evasion foundation to build on.
