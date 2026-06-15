# v6 — Phase 3 v2: Reward Reshaping + Curriculum (Planned)

**Status:** Planned  
**Hypothesis date:** June 2026

---

## Thesis

The v5 dual-head GOP+CHER architecture is structurally correct but the reward
function contains a fundamental asymmetry that no architectural change can fully
overcome. The fix must happen at the reward level: make the agent **want** to
collect by changing the cost-benefit calculus, not by teaching it via
counterfactuals.

**Hypothesis**: Adding a survival bonus (+0.1/step) and a continuous proximity
gradient (+0.05 × pellet_proximity/step), combined with a collect-first curriculum
(weak Hunter only for first 1000 eps), will cause Krishna to internalize collection
as a primary behavior. CHER injections should decline naturally (cf → 0) as the
agent develops autonomous collection — this is the key success indicator.

## Planned Architecture Changes

### 1. Reward Reshaping (in `environment/selfplay_env.py`)

```python
# Add to step() reward calculation:
r_survival   = 0.1 * alive     # +0.1 every step Krishna is alive
r_proximity  = 0.05 * (1.0 - nearest_pellet_dist / grid_size)  # continuous pull

# Resulting reward range:
# Full evasion (1000 steps): +100 survival + proximity ≈ +125
# Collection (1 pellet):     +50 collect + ~30 proximity approach + survival ≈ +185
# Death:                     −80 catch (unchanged)
# New E[collect] >> E[evade] for moderate Hunter skill
```

### 2. 2-Phase Curriculum (in `train_phase3.py`)

```python
# Phase A: ep 0–1000  
p_easy = 1.0   # Krishna only faces easy tier Hunter
# Goal: build collection habit without threat pressure

# Phase B: ep 1001+
p_easy = 0.25  # Normal FSP sampling resumes
```

### 3. Gate Regularization (in `agent/gated_dqn_agent.py`)

```python
# Add to GatedDQNAgent.learn():
if self.episode < 2000:
    gate_reg = 0.01 * (gate_output - 0.5).pow(2).mean()
    loss = td_loss + gate_reg
# Prevents wild gate oscillation in early training (observed in v5: ep 4000–4430)
```

### 4. Weighted CHER (in `utils/cher.py`)

```python
# Replace uniform collect_bonus with distance-weighted:
cf_reward = r_real + collect_bonus * (hunter_dist / safe_dist)
# hunter_dist=6 (just safe): full bonus
# hunter_dist=12 (very safe): 2× bonus (extra incentive to collect when Hunter far)
# Teaches: "the safer the window, the bigger the reward for collecting"
```

### 5. Hyperparameter Changes

| Param | v5 | v6 |
|---|---|---|
| ε floor | 0.05 | 0.08 |
| cher_bonus | 30.0 | 50.0 |
| survival_reward/step | 0 | +0.1 |
| proximity_reward/step | 0 | +0.05 × proximity |
| gate_reg_coeff | 0 | 0.01 (first 2000 eps) |
| p_easy (phase A) | 0.25 | 1.0 (ep 0–1000) |

## Success Criteria

| Metric | v5 achieved | v6 target |
|---|---|---|
| Late avg100 mean | 8.58 | > 15 |
| Pellet rate (last 3000 eps) | ~3–5% | > 10% |
| 2-pellet eps | 265 | > 500 |
| CHER cf=0 rate (last 1000 eps) | ~40% with 0 | < 20% (agent finds own windows) |
| Gate oscillation | 0.22–0.80 swing | max swing < 0.2 |
| Krishna wins | 2 | > 20 |

## Key Diagnostic: CHER Dependency Index

**The primary success signal for v6 is cf declining over time.** If CHER is
doing its job as a teacher, the student (Krishna) should eventually need it less.

```
cf_dependency = rolling100(cf_injected) / max_cf_per_ep
Target: cf_dependency < 0.2 in last 2000 episodes
v5 result: cf_dependency ≈ 0.4–0.6 throughout (never declined)
```

If cf_dependency stays high in v6, the reward reshaping was insufficient and
we need to reconsider the collect reward magnitude or the curriculum length.

## Risk Assessment

| Risk | Mitigation |
|---|---|
| Survival bonus makes evasion too easy | Monitor: if avg100 > 30 from ep 1, survival bonus too high |
| Proximity reward causes wall-hugging near pellets | Cap proximity bonus: only if Hunter dist > 3 |
| Phase A curriculum creates easy-Hunter overfitting | Phase A limited to 1000 eps (8% of total) |
| Gate reg prevents useful adaptation | Only applied for first 2000 eps |

## Files to Modify

- `environment/selfplay_env.py` — add survival + proximity rewards
- `train_phase3.py` — add p_easy curriculum schedule
- `agent/gated_dqn_agent.py` — add gate regularization to learn()
- `utils/cher.py` — add distance-weighted bonus
- `config.py` — add new reward constants
