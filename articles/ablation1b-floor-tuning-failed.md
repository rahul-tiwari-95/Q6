# I pre-registered a fix for my last failure mode. It failed too — and not in the way I predicted.

*Ablation 1b on the Q6 opponent pool. 6,000 episodes, 12.9 hours, one MacBook.*

---

## The finding, up front

Last time, [rectified opponent sampling](ablation1-rectified-sampling.md) — training more against opponents you're currently beating — produced the first sustained decline in my core diagnostic metric, and also opened a three-thousand-episode window where my agent got badly outclassed, right where the mechanism predicted it would. I wrote down a specific fix before running it: raise the sampling floor, and ramp rectification in gradually instead of switching it on abruptly, so the agent isn't thrown at strong opponents cold.

I ran it. It didn't fix the window — the window got *worse* (Hunter's win rate during the target episodes went from 47.7% to 57.7%) — and it undid the one clean win from last time, reversing the metric decline I'd spent the whole previous post establishing. That part, I can construct a story for. The part I didn't predict at all: the damage wasn't contained to the mechanism I changed. It showed up in a mode of play that never touches the opponent pool, and it stayed elevated all the way to the end of training, not just inside the window I was trying to fix.

## What I changed, and why I expected it to work

Ablation 1's opponent pool reweights sampling toward snapshots Krishna currently beats: `weight = max(score - baseline, 0) + floor`, with `floor = 0.1` fixed from episode 1. That floor fully engages at the exact episode (1,000) my unrelated easy-opponent warmup curriculum hands the agent real hard-tier traffic for the first time — two independently-designed schedules pivoting on the same moment, which I didn't notice until I was staring at the data after the run finished.

The fix I pre-registered: raise the floor 3x (0.1 → 0.3) so hard opponents keep more sampling pressure even under rectification, and decouple the two schedules by ramping the *effective* floor down from a near-uniform 1.0 to the target 0.3 over the first 4,000 episodes — so rectification is still mostly "off" exactly when the warmup curriculum first exposes the agent to real hard-tier opponents, and only tightens once it's had time to build up real practice against them.

I want to be specific about what I predicted, because it turned out to be wrong in a way worth being precise about: I expected the vulnerability window to close, `mean_hard_tier_score` to stay meaningfully above last time's 0.05 trough, and the CHER-dependency decline to survive mostly intact. I set numeric targets for all three before running anything, specifically so I couldn't retroactively call a partial result a win.

## What happened

| Criterion | Target | Ablation 1 | **This run** | |
|---|---|---:|---:|---|
| Hard-tier score, trough (ep 1000–4000) | > 0.25 | 0.049 | 0.115 | missed — 2.4x better, still under half the bar |
| Hunter win rate, windowed (ep 1000–4000) | < 40% | 47.7% | **57.7%** | worse than the thing I was fixing |
| CHER dependency, last 2,000 eps | ≤ 0.20 | 0.167 | **0.237** | regressed |
| Pellet-touch rate | ≥ 15% | 15.5% | 24.0% | met, see caveat below |

Three of five targets missed, including the two I'd flagged as primary. The floor-raise did work at the thing it mechanically does — force broader exposure to the hard tier instead of concentrating on whatever's currently beatable, which is most of why the trough score roughly tripled. It just wasn't enough, and it cost more than it bought: forcing near-uniform exposure to opponents the agent has had zero practice against means it loses to them more, right when I'd built the schedule specifically to prevent that.

That part I predicted the shape of, even though I got the magnitude wrong. The part I didn't see coming is what happened when I split Hunter's win rate by the two modes of play in this environment — `fsp`, where Krishna faces a frozen snapshot sampled from the pool, and `joint`, where both agents are live and the pool is never consulted at all:

| | Ablation 1 | This run |
|---|---:|---:|
| fsp mode (pool touched) | 6.4% | 18.0% |
| joint mode (pool never touched) | 63.9% | **73.2%** |

The fsp-mode damage is real and roughly where I'd expect it, and it's mostly transient — it falls back to ~7% by the end of training, close to last time's ~5%. But the aggregate failure is actually *dominated* by joint mode, which structurally cannot be affected by anything the opponent pool does, and joint-mode Hunter win rate stays elevated all the way through episode 6,000 — not contained to the window I built this fix to target. Whatever instability the sampling change introduces in fsp-mode matchups appears to be bleeding into how the two live, co-adapting agents settle into their dynamic with each other. I have one seed and one run, so I can't fully rule out ordinary chaotic divergence in adversarial self-play — but the gap holds at 10–30 points across nearly every 500-episode bucket, not as a single noisy spike, which is a larger and more consistent effect than I'd expect from seed noise alone.

The CHER-dependency regression has a partly mechanical component worth naming honestly: episodes end faster when Hunter wins more, which shrinks the "opportunity" count the dependency ratio divides by, while injections per episode stay flat. Some of the 0.167 → 0.237 move is that denominator shrinking, not necessarily the agent leaning on the synthetic signal more. I haven't cleanly separated how much is which, and I'd rather say that than round it up to "the mechanism failed" or down to "it's just measurement noise."

And one caveat on the metric that did pass: pellet-touch rate's overall rise looks good in aggregate but is front-loaded and declining by the run's end (29.6% early in the window down to 14.9% by the last bucket — slightly *below* ablation 1's flat band by the finish). It's a real number, not a sustained trend.

## Was this worth running?

Yes, and I want to be direct about why, rather than let a failed hypothesis read as wasted compute. I pre-registered specific, falsifiable numbers before this run started, the run falsified three of them cleanly, and the code that produced those numbers has been independently audited against the exact commit that ran — this isn't a bug or a metric artifact, it's rectified sampling's floor-and-warmup fix genuinely not doing what I designed it to do. Ruling that out is worth exactly as much as confirming it would have been. The alternative — not writing this up because the result is negative — is how a research thread quietly loses its own falsifiability.

What I don't have yet is credit assignment. This run changed two things at once — a higher terminal floor and a gradual warmup ramp — and my own risk notes before running it named this exact confound: I won't know whether the higher floor, the ramp's early near-uniform phase, or their interaction is responsible until I isolate them. That's the next run, not a bigger version of this one: hold one change at last time's value, vary only the other, twice. At 12.9 hours per run on a MacBook, that's a cheap way to actually know something, instead of tuning magnitudes on a result I can't yet attribute.

Full run data for `20260731_222534_v7_ablations_v2_floor_tuning`, and the complete pre-registration document with this run's results appended below it, are on `v7-ablations` at commit `8735e9e`.
