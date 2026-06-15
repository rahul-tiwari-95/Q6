# v5 — Phase 3: Gated Option Policies + Counterfactual HER

**Date:** June 2026  
**Episodes:** 12,000  
**Duration:** 53.5 hours (2.24 days)  
**Run ID:** `20260612_184130_phase3_v2_resume_enabled`  
**Status:** Complete

---

## Thesis

A dual-head gated architecture (evade head + collect head, context-conditioned
gate) combined with counterfactual hindsight experience replay (CHER) will break
the v4 bimodal collapse. The key insight from HER (Andrychowicz et al., 2017):
sparse reward environments can be made sample-efficient by relabeling failed
trajectories as successes for alternative goals. CHER extends this: instead of
relabeling goals, it injects optimal *actions* the agent could have taken.

**Hypothesis**: Separating evade and collect into independent heads with a learned
blending gate allows both behaviors to train without gradient interference. CHER
provides the collect head with a dense training signal that doesn't require
risky real-world collection.

## Architecture Changes from v4

### GatedOptionNetwork (`model/gated_option_network.py`)

```
Input: state (6,25,25) + context (3,) = [hunter_dist_norm, pellets_norm, lives_norm]

CNN backbone:
  Conv2d(6→32, k=3, pad=1) → ReLU → MaxPool(2)
  Conv2d(32→64, k=3, pad=1) → ReLU → MaxPool(2)
  → flatten → FC(256)

Three heads:
  evade_head:   FC(256→128) → FC(128→4)   [Q-values for evasion]
  collect_head: FC(256→128) → FC(128→4)   [Q-values for collection]
  value_head:   FC(256→128) → FC(128→1)   [dueling value]

Gate MLP (context-conditioned):
  context(3) → FC(16) → ReLU → FC(1) → sigmoid
  → gate=1.0: pure evasion | gate=0.0: pure collection

Blended output:
  Q_blend = gate × Q_evade + (1−gate) × Q_collect + value
```

### CounterfactualHER (`utils/cher.py`)

Post-episode scan. For each timestep where:
- `hunter_manhattan_dist > 6` (Hunter is safe distance away)
- `nearest_pellet_dist ≤ 4` (pellet is reachable)
- Krishna did NOT move toward the pellet

Inject counterfactual transition: same state/next_state, but with the
**optimal collection action** and reward `r_real + 30.0`.

Capped at 50 injections per episode to prevent buffer flooding.

### HierarchicalOpponentPool (`utils/hierarchical_pool.py`)

```
easy_tier: 5 permanent slots (never evicted)
hard_tier: 15 FIFO slots

Sampling: p_easy=0.25, p_hard=0.75
         Within hard tier: p_latest=0.70 (recent opponents dominate)
```

### Training Additions

- `ContextReplayBuffer`: stores context alongside (s, a, r, s', done)
- `SelfPlayGridworld._get_info()`: enriched with `hunter_manhattan_dist`, `nearest_pellet_dist`
- Resume support: `last_state.json` checkpoint every 100 eps
- SIGTERM/SIGINT handler: saves checkpoint on Mac sleep/kill

## Key Hyperparameters

```python
# Unchanged from v4
LEARNING_RATE  = 1e-4
GAMMA          = 0.99
TAU            = 0.001
BATCH_SIZE     = 64
BUFFER_SIZE    = 100_000

# New for v5
epsilon_start  = 0.15    # warm-start: near-floor
epsilon_min    = 0.05
cher_safe_dist = 6
cher_reach     = 4
cher_bonus     = 30.0
p_easy         = 0.25
easy_slots     = 5
hard_slots     = 15
```

## Results

| Metric | Value | vs v4 |
|---|---|---|
| Total pellets collected | 2,567 | +8,456% |
| Episodes with ≥1 pellet | 2,284 (19.0%) | +3,800% |
| Episodes with ≥2 pellets | 265 | first time |
| Episodes with ≥3 pellets | 16 | first time |
| Krishna wins (4 pellets) | 2 | −67% |
| Late avg100 mean (last 3000) | 8.58 | +75% |
| Late avg100 max | 15.69 | +220% |
| Gate range | 0.22–0.80 | — |
| Gate at end | 0.545 | — |
| CHER injections total | 237,511 | — |
| Best reported avg100 | 41.22 @ ep 5 | (inflated — ep 5, rolling window = 5 samples) |

**Dashboard:** [Run details](run.html?id=20260612_184130_phase3_v2_resume_enabled) · [Replay](replay.html?run=20260612_184130_phase3_v2_resume_enabled) · [Architecture](architecture.html)

## Training Dynamics

### Gate evolution

```
ep 1–2000:   gate ~0.6   (evasion-leaning — warm-start bias)
ep 2000–4000: gate 0.6→0.5  (settling)
ep 4000–4430: gate 0.5→0.22 (CHER floods buffer, collect head dominates)
ep 4430–5000: gate 0.22→0.43 (self-corrects, avg100 recovers)
ep 5000–12000: gate ~0.44–0.55 (stable equilibrium)
```

The gate oscillation at ep 4000–4430 is a design risk: CHER injected so many
collect transitions that the collect head overfit, Krishna started over-reaching
for pellets against hardened Hunter, avg100 tanked to 0–2. The gate then
self-corrected as the evade head's gradient dominated again.

### avg100 by phase

```
Early (1–4000):    mean  7.58  max 41.22 (inflated: weak Hunter, small window)
Mid   (4001–8000): mean  6.24  max 13.10 (gate oscillation trough)
Late  (8001–12000): mean  8.46  max 15.69 (stable equilibrium)
```

## What Worked

- **Bimodal collapse eliminated**: gate never locked to 1.0 (pure evasion)
- **First 2-pellet episode** (ep 10170, r_k=101.70): collect head demonstrated multi-pellet capability
- **3-pellet episodes achieved** (16 total): shows collect head CAN plan multi-step collection
- **Training stability**: 12,000 episodes without divergence or crash
- **Resume feature**: survived multiple Mac sleeps via SIGTERM checkpoint saves

## Failure Mode — CHER Dependency / Reward Asymmetry Unresolved

Against pool=20 hardened Hunter (late training), collection rate drops back to
~3–5% per episode. The root issue:

**The reward asymmetry was not fixed.** Krishna's rational strategy is still
to avoid risk:
```
E[collect vs hardened Hunter] ≈ P(survive) × +50 + P(caught) × −80 ≈ −15
E[evade for 1000 steps]       ≈ 0.97 × ~10 ≈ +9.7
```

CHER teaches collection in simulation but doesn't change the live-game incentive
structure. When CHER fires (cf=50), avg100 temporarily rises. When CHER finds
no safe windows (cf=0), collection drops to near-zero.

The agent is **CHER-dependent** — it hasn't internalized autonomous collection.

## Decision for v6

The dual-head architecture works. The problem is the reward function itself:

1. **Fix reward asymmetry**: Add survival bonus (+0.1/step → max +100 over 1000
   steps). This makes survival AND collection both positive, closing the gap.

2. **Proximity reward**: +0.05 × (1 − nearest_pellet_dist/25) per step.
   Creates a continuous gradient toward pellets, not a sparse binary reward.

3. **Gate regularization**: L2 penalty on |gate − 0.5| during first 2000 eps.
   Prevents the wild oscillation observed at ep 4000–4430.

4. **2-phase curriculum**: Phase A (ep 0–1000): p_easy=1.0 — build collection
   habit against weak Hunters before hardening. Phase B: normal FSP.

5. **Weighted CHER**: inject with weight ∝ hunter_dist/safe_dist instead of
   uniform +30. Risky opportunities get lower weight, teaching risk calibration.

6. **ε floor 0.05 → 0.08**: more exploration needed against hardened Hunter.
