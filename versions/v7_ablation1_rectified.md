# v7-ablations (Ablation 1) — Fitness-Rectified Opponent Sampling

**Date:** 2026-07-29 → 2026-07-31
**Episodes:** 6,000
**Duration:** 45.5 hours (163,829s), `--device mps`
**Run ID:** `20260729_194141_v7_ablations_run`
**Branch:** `v7-ablations` @ `2dc45af`
**Status:** Complete

---

## Thesis

v7's reward redesign (see Q6.md §4) was never cleanly tested — the weighted-CHER change likely undermined the exact mechanism (CHER) that was doing v5's work, in the same commit that made the live timeout penalty harsher. Rather than re-running that confounded combination, this run isolates a different, specific hypothesis: **is Krishna's collection paralysis partly a training-distribution problem, not just a live-reward problem?**

The FIFO hierarchical opponent pool samples hard-tier snapshots by pure recency, with no awareness of how Krishna is actually doing against any given snapshot. Borrowed from PSRO's rectified Nash response (Balduzzi et al., 2019): reweight hard-tier sampling so Krishna trains more against snapshots it's currently beating or tying, and less against ones dominating it — the opposite instinct from standard FSP, which pulls toward the hardest available opponent.

**Hypothesis**: if collection paralysis is (partly) caused by the replay buffer being saturated with evasion-only transitions against opponents Krishna can't currently handle, shifting training mass toward winnable matchups should let real collection experience accumulate, and CHER's synthetic teaching signal should become less necessary over time (`cher_dependency` declining) as Krishna internalizes collection on its own.

**Known risk, stated at design time**: a pure floor-based rectifier could let Krishna retreat into an easy-opponent-only niche and never develop robustness against hard snapshots. A `rectify_floor=0.1` was added specifically to keep hard opponents in rotation at a minimum rate, not exclude them entirely.

## Architecture Changes from v7

Only the FSP sampling distribution changed. Reward function, network architecture, and CHER mechanics are otherwise identical to v7.

### `HierarchicalOpponentPool` (rectified mode, `utils/hierarchical_pool.py`)

```
Per-snapshot EMA score (ema_alpha=0.1), updated after every FSP episode:
  score = 1.0 if Krishna won, else pellets_collected / TARGET_PELLETS

Hard-tier sampling weight:
  weight_i = max(score_i - baseline, 0) + floor
  baseline = mean score across current hard tier (recomputed at sample time)
  floor    = 0.1
```

Ablation 2 (frozen-policy anchor) was **not** used in this run (`anchor_weight=0.0`) — isolating ablation 1's effect cleanly. See Q6.md §6.1 for why: the available Phase-1 anchor checkpoint is under-trained (250 episodes) and not representative enough to draw conclusions from yet.

## Key Hyperparameters

Unchanged from v7: `learning_rate=1e-4`, `gamma=0.99`, `tau=0.001`, `batch_size=64`, `buffer_size=100_000`, `epsilon_start=0.15`, `epsilon_min=0.05`, `epsilon_decay=0.9994`, `K_TIMEOUT_ZERO_PELLETS=-20`, `easy_warmup_eps=1000`.

New: `pool_rectified=True` (default), `pool_ema_alpha=0.1`, `rectify_floor=0.1`.

## Results

| Metric | Value | vs v7 (12k eps) |
|---|---:|---|
| Krishna wins | 2 / 6,000 (0.033%) | v7: 0 / 12,000 |
| Hunter wins | 2,104 / 6,000 (35.07%) | v7: 29.4% |
| Timeouts | 3,894 / 6,000 (64.9%) | v7: 70.6% |
| Total pellets | 1,024 | — |
| Episodes with ≥1 pellet | 928 (15.5%) | — |
| Late avg100 mean (last 3,000) | −27.85 | v7: ≈ −14 to −37 (same range — reward structure unchanged from v7, not comparable to v5's positive numbers) |
| Gate range | 0.399 – 0.641, final 0.450 | v7: ends 0.63, trending toward evasion |
| CHER injections | flat-to-declining raw count (~21 → ~15/episode) against **rising** opportunity counts (~90 → ~125/episode) | — |

## Training Dynamics

| episodes | pellet-touch rate | mean hard-tier score | CHER dependency | Hunter win rate |
|---|---:|---:|---:|---:|
| 0–500 (easy warmup) | 26.3% | 0.0 (empty tier) | 0.315 | 22.2% |
| 500–1000 | 16.6% | 0.50 | 0.240 | 39.4% |
| 1000–2000 | 13.2% | **0.05** | 0.235 | **51.6%** |
| 2000–3000 | 12.9% | 0.049 | 0.225 | 49.5% |
| 3000–4000 | 14.7% | 0.053 | 0.198 | 42.0% |
| 4000–5000 | 15.0% | 0.039 | 0.173 | 24.6% |
| 5000–6000 | 15.7% | **0.029** | **0.160** | **11.9%** |

### What worked: CHER dependency declined for the first time in Q6's history

The v6 plan's primary success criterion — `cher_dependency` declining over training, meaning Krishna needs the synthetic teaching signal less as it internalizes collection — was met here: 0.315 → 0.160, roughly halved. This is not an artifact of shrinking opportunities: `cher_opportunities` held steady (70–90/episode) through the middle of training and *rose* to 120+/episode by the end (longer surviving episodes give more steps for opportunities to arise), while the raw injection count stayed flat-to-declining. v5 (12k episodes) found this index "never declined" (0.4–0.6 throughout); v7 didn't track it cleanly due to the confound described in Q6.md §4. This run is the first clean measurement showing it move.

### The predicted risk materialized concretely: a mid-training vulnerability window

Right as the easy-warmup curriculum ends (ep 1000) and rectified sampling takes full effect, `mean_hard_tier_score` collapses from 0.50 to 0.05 and stays there through ep 4000 — meaning Krishna is losing to hard-tier opponents badly and consistently. This coincides with Hunter's win rate spiking to ~50% for roughly 3,000 episodes (vs 22–39% before and 12–25% after). This is exactly the risk flagged at design time: rectification systematically *downweights* the opponents currently beating Krishna, which — combined with the easy-warmup curriculum ending at the same moment — produces a real, measurable window where Krishna is under-practiced against strong opponents and gets caught much more often than in any prior version.

### It self-corrects, but pellet collection stays flat throughout

By episodes 5000–6000, Hunter's win rate falls back to 11.9% (timeout 88%) — better than v7's overall 29.4% — while `mean_hard_tier_score` is still low (0.029), suggesting the recovery is driven by something other than actually solving the hard-tier matchups (plausibly the live joint-mode Hunter co-adapting, or epsilon reaching its floor and behavior stabilizing). Pellet-touch rate never collapses to near-zero (unlike v4's bimodal collapse) but also never grows past ~15–16% — collection remained a minority behavior throughout, just a stable one.

## Decision for next thread

1. **The CHER-dependency decline is worth trusting and building on** — it's the cleanest positive signal Q6 has produced. Worth checking whether it continues declining with more episodes, and whether it holds up under ablation 2 (frozen anchor) layered on top.
2. **The mid-training vulnerability window needs a direct fix before the next real run**: either raise `rectify_floor` (keep more hard-opponent pressure even under rectification), decouple the easy-warmup schedule from the rectification kick-in (stagger them rather than both changing at ep 1000), or track `mean_hard_tier_score` as a live signal and adapt `rectify_floor` dynamically rather than fixing it.
3. **Pellet collection staying flat around 15% (not growing, not collapsing) suggests ablation 1 alone changes *who Krishna trains against*, not *the underlying incentive to collect*** — consistent with the reward-asymmetry diagnosis never actually being tested cleanly (v7's confound). The natural next experiment is ablation 1 + a properly isolated, non-confounded version of v7's reward fix (or v8's PPO comparison), not ablation 1 alone at a larger scale.
4. **This result is not directly comparable to v5's positive avg100 numbers** — this run inherits v7's harsher `K_TIMEOUT_ZERO_PELLETS=-20` reward structure, which alone drags avg100 negative regardless of what the opponent pool does. Any write-up should compare against v7, not v5, on that specific metric.
