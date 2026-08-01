# v7-ablations (Ablation 1b) — Floor Tuning + Rectification Warmup

**Status:** Complete — hypothesis rejected
**Hypothesis date:** 2026-07-31
**Run date:** 2026-08-01
**Run ID:** `20260731_222534_v7_ablations_v2_floor_tuning`
**Branch:** `v7-ablations` @ `8735e9e`

The section below this line, up to "## Results," is the pre-registered plan
exactly as written on 2026-07-31, before the run started. It is preserved
unedited — including the predictions it turned out to get wrong — because
that's the point of writing it down first.

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

Job `v7_ablations_v2` in `orchestrator/q6_jobs_v2.json`: same scale as the
completed ablation-1 run (6,000 episodes, `--device mps`), with
`--rectify-floor 0.3 --rectify-warmup-eps 4000` added. `easy_warmup_eps`
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

---

## Results

**The hypothesis was wrong.** Raising the floor and ramping it in gradually
did not close the vulnerability window — it made the window worse on the
primary diagnostic, and it reversed the CHER-dependency decline that made
ablation 1 worth building on in the first place. This is a real result, not
a bug: the implementation was independently code-audited against the exact
commit that ran (`8735e9e`) and does precisely what this document specifies
above — `set_rectify_floor()` is called every episode regardless of mode,
`hard_tier_weights()` reads the floor fresh at sample time, and the ramp
value at ep 1,000/4,000 matches the ≈0.825 / 0.3 this document predicted by
hand.

### At a glance

| | |
|---|---|
| Episodes | 6,000 |
| Wall-clock duration | 12.9 hours (46,421s) — much faster than ablation 1's 45.5h; infrastructure/load difference between the two runs, not a truncation (both completed all 6,000 episodes) |
| Hardware | MacBook, Apple Silicon, `--device mps` |
| Run ID | `20260731_222534_v7_ablations_v2_floor_tuning` |
| Branch | `v7-ablations` @ `8735e9e` |
| Code changed vs. ablation 1 | `--rectify-floor 0.3 --rectify-warmup-eps 4000` only; everything else (reward, network, CHER, `easy_warmup_eps=1000`, ablation 2 off) held identical |

### Scorecard against the pre-registered criteria

| Criterion | Target | Ablation 1 | **Ablation 1b** | Verdict |
|---|---|---:|---:|---|
| `mean_hard_tier_score` trough, ep 1000–4000 | > 0.25 | 0.049 | **0.115** | **FAIL** — 2.4x better than ablation 1, but under half the bar |
| Hunter win rate, windowed ep 1000–4000 | < 40% | 47.7% | **57.7%** | **FAIL** — worse than the thing this was meant to fix |
| `cher_dep_idx`, last 2,000 eps | ≤ 0.20 | 0.167 | **0.237** | **FAIL** — the flagship metric regressed |
| Pellet-touch rate, overall | ≥ 15% | 15.5% | **24.0%** | PASS, but see caveat below |
| Krishna win rate | report as-is | 2/6,000 | 2/6,000 | unchanged |

Full per-bucket comparison (mirrors the table in `v7_ablation1_rectified.md`):

| Episodes | Hunter win % | `mean_hard_tier_score` | `cher_dep_idx` | Pellet-touch % |
|---|---:|---:|---:|---:|
| 0–500 (warmup) | 19.8 | 0.001 | 0.310 | 25.0 |
| 500–1000 | 52.0 | 0.500 (default-prior artifact, see below) | 0.317 | 29.2 |
| 1000–2000 | 62.9 | 0.140 | 0.381 | 29.6 |
| 2000–3000 | 59.4 | 0.120 | 0.367 | 28.4 |
| 3000–4000 | 50.8 | **0.115 (trough)** | 0.320 | 25.7 |
| 4000–5000 | 42.7 | 0.072 | 0.260 | 18.0 |
| 5000–6000 | 35.6 | 0.041 | 0.214 | 14.9 |

Two data-quality notes carried over from ablation 1's own review, both
checked and both apply the same way here: the 500–1000 bucket's constant
`mean_hard_tier_score = 0.5` is `HierarchicalOpponentPool`'s default-prior
placeholder for snapshots with no recorded outcome yet (`easy_warmup_eps`
is still 1,000, so the hard tier is barely sampled before then) — not a
measurement of anything. And `cher_dep_idx` is the raw, un-windowed
`cf_injected / cher_opportunities` ratio logged every episode
(`train_phase3.py`) — the same formula ablation 1 reported, so the
comparison above is apples to apples.

### What actually happened

The floor-raise worked exactly as designed on *sampling breadth*: forcing
near-uniform hard-tier sampling early means every snapshot actually gets
exercised and scored, instead of ablation 1's low-floor regime where
sampling concentrated on whichever snapshots were already beatable. That's
most of why `mean_hard_tier_score` roughly tripled at its trough (0.049 →
0.115) — genuine progress on the thing this ablation targeted.

But it wasn't enough, and it cost more than it bought. Splitting Hunter's
win rate by mode tells the real story:

| | Ablation 1, ep1000–4000 | Ablation 1b, ep1000–4000 |
|---|---:|---:|
| fsp mode (opponent pool touched) | 6.4% | **18.0%** |
| joint mode (live Hunter, pool never consulted) | 63.9% | **73.2%** |

Forcing Krishna into near-uniform exposure to the hard tier — including
snapshots it has had zero practice against — means it loses those specific
matchups more often, right when the design doc predicted the opposite. That
part is a direct, mechanistic consequence of the change and is *mostly
transient*: fsp-mode Hunter win rate falls back to ≈7% by ep 5000–6000,
close to ablation 1's ≈5%.

What's more surprising, and not something this document anticipated: the
aggregate win-rate failure is actually **dominated by joint mode**, which
never samples the opponent pool at all, and joint-mode Hunter win rate stays
elevated (62.3% overall for the whole run, vs. 47.0% in ablation 1) all the
way through episode 6,000 — not just inside the target window.
That reads like a downstream co-training destabilization effect: whatever
is happening in the fsp-mode matchups is bleeding into how the two live
agents adapt to each other, not staying contained to the mechanism that
caused it. With one seed, one run, I can't cleanly separate that from
ordinary chaotic divergence in adversarial self-play — but the effect size
(a consistent 10–30 point gap across nearly every 500-episode bucket, not a
single noisy spike) is large relative to what seed noise alone typically
produces.

The `cher_dep_idx` regression is partly a downstream reflection of the same
thing rather than a fully independent third failure. Episodes end faster
when Hunter catches Krishna more often (mean episode length in the last
2,000 episodes: 801 steps here vs. ≈890–940 in ablation 1), which shrinks
`cher_opportunities` — the ratio's denominator — while CHER injections per
episode stay flat-to-higher. That mechanically inflates `cher_dep_idx` on
top of whatever real behavioral shift is also happening. I haven't
separated how much of the 0.167→0.237 regression is "real" dependency vs.
this denominator effect, and I don't want to claim a precision I don't have
here.

**One honest caveat on the one metric that improved.** Pellet-touch rate's
overall rise (15.5% → 24.0%) is real and it's the cleanest positive number
in this run — but the per-bucket table shows it's front-loaded and
*declining* by the end (29.6% at ep1000–2000 down to 14.9% at ep5000–6000,
actually slightly below ablation 1's flat ~15–16% band), not a sustained
improvement. And `mean_hard_tier_score` is an unweighted mean over whatever
currently occupies the hard tier — broader sampling changes *what's being
averaged*, not only how well Krishna plays those matchups, so part of the
0.049→0.115 rise is a measurement-composition effect, not pure competence
gain. Neither caveat flips a FAIL to a PASS anywhere in the scorecard, but
both are reasons not to oversell the parts of this run that look good.

### Is the underlying hypothesis dead, or just this instantiation of it?

Unclear, and that's the honest answer. Two changes shipped together in this
run — the higher terminal floor (0.3) and the warmup ramp's near-uniform
early phase — and the risk table above flagged exactly this confound before
running: *"could reintroduce a v7-style confound where it's unclear which
one (if either) fixed the window."* Right now neither can be credited or
blamed individually. The joint-mode persistence through episode 6,000, in
particular, isn't obviously explained by either single-parameter story on
its own, which suggests the real mechanism here may not be fully
characterized yet.

## Decision for Next Version

Do not tune the floor/warmup magnitudes further before isolating credit.
The design doc's own risk table already named the fix: hold one change at
ablation 1's value and vary only the other —

- `--rectify-floor 0.3` with no warmup (`--rectify-warmup-eps 0`), vs.
- `--rectify-floor 0.1` (ablation 1's value) with `--rectify-warmup-eps 4000`

— to find out whether the higher terminal floor, the warmup ramp's early
near-uniform phase, or their interaction is driving the regression. This run
took 12.9h wall-clock; two isolation runs at the same scale are cheap
relative to the compute currently reserved for the v8 PPO port, and should
run before any further magnitude tuning on this mechanism.
