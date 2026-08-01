# v7-ablations (Ablation 1b) — Floor Tuning + Rectification Warmup (Planned)

**Status:** Planned
**Hypothesis date:** 2026-07-31

---

## Thesis

Ablation 1 (`versions/v7_ablation1_rectified.md`, complete) validated the core
bet behind rectified opponent sampling: `cher_dependency` declined for the
first time in Q6's history (0.315 → 0.160 over 6,000 episodes) — the cleanest
positive signal the project has produced. But it also found a clear, causal
side effect: `rectify_floor` was a fixed 0.1 from episode 1, and it fully
engages at the exact episode (1,000) that the unrelated `easy_warmup_eps`
curriculum switches `p_easy` from 1.0 to 0.25. The combination produces a
~3,000-episode window (ep 1000–4000) where `mean_hard_tier_score` collapses
from ~0.5 to ~0.05 and Hunter's win rate spikes to ~50% (vs 22–39% before and
12–25% after) — the agent is under-practiced against strong opponents at
precisely the moment it starts seeing them for real.

**Hypothesis**: this is a training-distribution *scheduling* problem, not a
flaw in rectification's core mechanism. Two independent, decoupled, additive
changes — raising the steady-state floor, and ramping rectification in
gradually instead of switching it on abruptly — should close the
vulnerability window (measured via `mean_hard_tier_score` staying materially
above ablation 1's ~0.05 trough during ep 1000–4000, and Hunter's windowed
win rate staying inside the 22–39% range seen elsewhere in that run) without
undoing the CHER-dependency decline that made ablation 1 worth building on.

## Planned Changes

### 1. Raise the floor (`utils/hierarchical_pool.py`, `train_phase3.py`)

`rectify_floor` was already a `HierarchicalOpponentPool` constructor
parameter (default 0.1) but was not exposed as a CLI flag on
`train_phase3.py` — it could only be changed by editing the call site. Added
`--rectify-floor` (default 0.1, unchanged, for backward compatibility with
the completed ablation-1 run's exact recorded config). This run uses 0.3 —
3x the original floor, chosen to meaningfully soften the
`max(score_i - baseline, 0) + floor` skew toward beatable opponents without
flattening rectification into uniform sampling and losing the CHER-dependency
effect entirely.

### 2. Decouple the two schedules (`utils/hierarchical_pool.py`, `train_phase3.py`)

Added `rectify_warmup_eps` (CLI: `--rectify-warmup-eps`, default 0 = off),
mirroring the naming and mechanics of `easy_warmup_eps` and
`anchor_decay_eps`. When set, `train_phase3.py`'s per-episode loop linearly
ramps the pool's *effective* floor from a new class constant,
`HierarchicalOpponentPool.UNIFORM_WARMUP_FLOOR` (1.0), down to the configured
`rectify_floor`, calling the new `pool.set_rectify_floor()` hook once per
episode — the same "call a setter every episode, training loop owns the
schedule" pattern already used for `anchor_weight`'s decay
(`krishna.set_anchor_weight(...)` in the same loop).

```python
# utils/hierarchical_pool.py
UNIFORM_WARMUP_FLOOR = 1.0   # class constant: scores/baseline in [0,1], so
                              # floor=1.0 bounds the fitness term to <=50%
                              # of any weight -- a bounded "near uniform"
                              # starting point (true uniform needs floor->inf)

def set_rectify_floor(self, floor: float) -> None:
    self.rectify_floor = float(floor)   # read fresh by hard_tier_weights()

# train_phase3.py, inside the per-episode loop, unconditional on mode:
if rectify_warmup_eps > 0:
    warmup_progress = min(1.0, ep / max(1, rectify_warmup_eps))
    start_floor = HierarchicalOpponentPool.UNIFORM_WARMUP_FLOOR
    effective_floor = start_floor + (rectify_floor - start_floor) * warmup_progress
    pool.set_rectify_floor(effective_floor)
```

With `rectify_floor=0.3` and `rectify_warmup_eps=4000`: at ep 1,000 (when
`easy_warmup_eps` ends and hard-tier traffic actually starts), the effective
floor is still ≈0.825 — close to uniform — and it tightens to the full 0.3
only by ep 4,000, which is roughly where ablation 1's own vulnerability
window ended and Hunter's win rate started falling back down on its own.
This directly targets the mechanism ablation 1's write-up diagnosed: instead
of both schedules changing behavior at the same moment, rectification is
still mostly "off" (floor high) when the curriculum first exposes the agent
to real hard-tier traffic, and only tightens once the agent has had time to
build up real EMA scores against those opponents.

Both `--rectify-floor` (default 0.1) and `--rectify-warmup-eps` (default 0)
default to values that exactly reproduce ablation 1's completed run when
left unset — no existing recorded config is invalidated by this change.

## Job Config

New job `v7_ablations_v2` in `orchestrator/q6_jobs_v2.json` (not launched):
same scale as the completed ablation-1 run (6,000 episodes, `--device mps`),
with `--rectify-floor 0.3 --rectify-warmup-eps 4000` added. `easy_warmup_eps`
(1,000, unchanged) and ablation 2 (frozen anchor, still off —
`anchor_weight=0.0`) are left exactly as ablation 1 ran them, so this stays a
clean, single-hypothesis follow-up rather than a second confound.

## Success Criteria

| Metric | Ablation 1 (achieved) | Ablation 1b (target) |
|---|---:|---|
| `mean_hard_tier_score`, ep 1000–4000 (trough) | 0.05 | > 0.25 — no collapse to near-zero |
| Hunter win rate, ep 1000–4000 (windowed) | ~50% | < 40% — inside the 22–39% range seen elsewhere in the ablation-1 run |
| `cher_dependency`, last 2,000 eps | 0.160 | ≤ 0.20 — preserve ablation 1's decline, don't regress it while fixing the window |
| Pellet-touch rate, overall | ~15–16% (flat) | ≥ 15% — no regression from softening rectification |
| Krishna win rate | 2 / 6,000 | no strong prior; report as-is (no version has moved this metric meaningfully yet) |

**Primary diagnostic**: `mean_hard_tier_score` and windowed Hunter win rate
specifically during ep 1000–4000 — this is the exact window ablation 1
flagged, so it's the most direct test of whether these two changes fix the
mechanism they were designed to fix, not just a global average that could
mask a persisting-but-shifted window.

## Risk Assessment

| Risk | Mitigation |
|---|---|
| Floor raised too high (0.3 still low, but a much higher floor would) could flatten rectification toward uniform sampling and erase ablation 1's `cher_dependency` win | 0.3 is 3x the original floor, not 10x; `cher_dependency` at ep 5000–6000 is tracked explicitly as a success criterion specifically to catch this regression |
| `rectify_warmup_eps=4000` could just delay the same abrupt transition rather than smooth it, if the ramp is too front-loaded near ep 4000 | Ramp is linear over the full window (not a step function at ep 4000), and 4000 was chosen to line up with ablation 1's own observed recovery point, not arbitrarily |
| `rectify_warmup_eps` too long could mean full-strength rectification only holds for the last ~2,000 episodes of a 6,000-episode run, too little time for `cher_dependency` to visibly re-decline within this run's budget | Same 6,000-episode / 2,000-episode "late window" scale as ablation 1, which already showed measurable decline over its full run; if this proves too short, the fix is a straightforward `--episodes` increase on a follow-up, not a code change |
| Raising the floor and adding the warmup ramp are two simultaneous changes — could reintroduce a v7-style confound where it's unclear which one (if either) fixed the window | Both are independent, single-purpose CLI flags (`--rectify-floor`, `--rectify-warmup-eps`); either can be held at its ablation-1 value in a fast, cheap single-flag follow-up run to isolate credit, without touching code |
| Compute is currently reserved for the v8 PPO port (higher priority, see `Q6.md` §3.3) | This document and the accompanying job config are prep only — `v7_ablations_v2` is written to `orchestrator/q6_jobs_v2.json`, a new file, and is not added to the active `q6_jobs.json` or launched |

## Files to Modify

- `utils/hierarchical_pool.py` — `UNIFORM_WARMUP_FLOOR` class constant, `set_rectify_floor()` method (both already implemented)
- `train_phase3.py` — `--rectify-floor`, `--rectify-warmup-eps` CLI flags and per-episode ramp logic (already implemented)
- `orchestrator/q6_jobs_v2.json` — new `v7_ablations_v2` job (ready, not launched)
- `tests/test_hierarchical_pool.py`, `tests/test_train_phase3_rectify_warmup.py` — unit/smoke coverage for the new floor/schedule logic
