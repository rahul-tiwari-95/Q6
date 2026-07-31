# Rectified opponent sampling moved my one real metric — and broke exactly where I said it would

*Ablation 1 on the Q6 opponent pool. 6,000 episodes, 45.5 hours, one MacBook, no datacenter.*

---

## The finding, up front

I reweighted which past-opponent snapshots my agent trains against — more sampling weight for opponents it's currently beating or tying, less for opponents currently dominating it, with a floor so hard opponents never disappear entirely. That single change produced the first sustained decline in my core diagnostic metric, `cher_dependency`, in this project's multi-month history: 0.315 → 0.160 over the run, roughly halved, and I checked hard for the obvious way this number lies to you (more on that below — it isn't).

In the same run, a failure mode I wrote down *before* I hit go materialized almost exactly as predicted: a three-thousand-episode stretch, right at the seam between two schedules I hadn't thought to decouple, where my opponent's win rate more than doubled and my agent got caught constantly. I can point to the two lines of code whose interaction caused it, because I already knew where to look — I'd flagged the mechanism at design time, just not the timing.

Neither half of this cancels the other out. This is the actual post: a real signal and a real, specifically-predicted failure, tangled up in the same six thousand episodes.

## At a glance

| | |
|---|---|
| Episodes | 6,000 |
| Wall-clock duration | 45.5 hours (163,829s) |
| Hardware | MacBook, Apple Silicon, `--device mps` |
| Run ID | `20260729_194141_v7_ablations_run` |
| Branch | `v7-ablations` @ `2dc45af` |
| Code changed | one file, `utils/hierarchical_pool.py` — reward function, network, CHER mechanics untouched from v7 |

No cloud GPU, no cluster. This is the same machine I do everything else on, and that's deliberate — it's part of what I'm trying to prove about this project, not an excuse for a small run.

## The question I'm actually chasing

Q6 is a small self-play gridworld: Krishna collects four pellets, a Hunter chases Krishna, both learn. The question underneath all the version numbers is this: when a reward function says "complete the objective" but the training distribution says "the objective is dangerous," does the agent learn to do the objective, or does it learn to avoid the danger and call that a win?

Formally, that's the difference between a **payoff-dominant** equilibrium (better for everyone, if you can coordinate on it) and a **risk-dominant** one (safer under uncertainty about what the opponent will do, and so more likely to be where training actually lands) — Harsanyi and Selten's distinction, forty years old, illustrated by the Stag Hunt. My agent's "just survive until the clock runs out" strategy is the hare. Collecting all four pellets against a Hunter that's actively hunting is the stag. Nothing about the evasive strategy is a bug. It's the mathematically rational move under the incentives training created. My job is to figure out whether that's fixable, and how.

Every version of this project — v1 through v7 — has rediscovered some flavor of this collapse. v4 collapsed bimodally: pure evasion, zero collection, no in-between. v5 (dual-head architecture plus a synthetic teaching signal called CHER, more below) broke the bimodal collapse but never got the agent off training wheels. v7 tried to fix the live reward function directly and came back *worse* than v5 on every headline metric — and it turned out that result was confounded, not a clean falsification (I've written about that separately). This post is the first of two cheap, targeted follow-ups designed to cut through that confound without another multi-day combined-change run.

## The metric I've been staring at for months: CHER dependency

CHER — Counterfactual Hindsight Experience Replay — is a training-wheels mechanism from v5 (`utils/cher.py`). After each episode, it scans the trajectory for moments where the Hunter was a safe distance away (`hunter_manhattan_dist > 6`), a pellet was reachable (`nearest_pellet_dist ≤ 4`), and Krishna didn't move toward it. At those moments it injects a synthetic transition into the replay buffer: same state, but with the optimal collection action and a bonus reward, capped at 50 injections per episode. It's a teacher showing the agent what it should have done in a moment it judges to have been genuinely safe.

The whole point of a teacher is that the student eventually stops needing one. `cf_dependency` (logged every episode, rolling over 100) is supposed to measure exactly that — how much the agent still leans on the synthetic signal instead of finding safe collection windows on its own. v6's plan, back in June, named this the primary success criterion for the whole reward-redesign thread: *"the key diagnostic: cf_dependency declining over time... target < 0.2 in the last 2,000 episodes."*

v5 got 0.4–0.6, flat, for all 12,000 episodes. It never moved. v7's version of this run was confounded by an unrelated bug (a weighted-CHER formula that quietly undershot the flat bonus it replaced in the most common distance range, landing in the same commit as a harsher timeout penalty — two reasonable changes fighting each other), so it never gave a clean read on this number either.

This run is the first time I've watched `cher_dependency` actually go down.

## What I changed: rectified opponent sampling

Everything else — reward function, network architecture, CHER mechanics — is identical to v7. The only change is *which past-Hunter snapshot Krishna faces* in a given self-play episode.

The existing pool (`utils/hierarchical_pool.py`) is a two-tier design: a permanent easy tier (5 slots, weak early-training snapshots, never evicted) and a rolling hard tier (15 slots, FIFO, most recent opponents). Before this run, hard-tier sampling was pure recency — `p_latest` controls how often you get the newest snapshot, otherwise uniform. It has no idea whether Krishna is currently winning or losing against any given snapshot in the tier.

My hypothesis was specific: if collection paralysis is partly a *training-distribution* problem — the replay buffer saturated with evasion-only transitions against opponents Krishna currently can't beat — rather than purely a live-reward problem, then shifting training mass toward matchups Krishna can actually win should let real collection experience accumulate, and CHER should become less necessary as the agent internalizes collection on its own.

The mechanism I borrowed is PSRO's rectified Nash response (Balduzzi et al., 2019): train more against opponents you're currently beating or tying, not the hardest available one — the opposite instinct from standard fictitious self-play, which pulls toward the strongest opponent. Concretely, after every FSP episode I record an EMA outcome score per snapshot (`record_outcome`, `ema_alpha=0.1`):

```
score = 1.0 if Krishna won, else pellets_collected / TARGET_PELLETS
```

(`train_phase3.py:396–400`; `TARGET_PELLETS = 4`, `config.py:36` — so a loss with 2 pellets collected scores 0.5, not 0.)

Then hard-tier sampling weight is (`utils/hierarchical_pool.py:244–258`):

```
weight_i = max(score_i - baseline, 0) + floor
baseline = mean(score) across current hard tier, recomputed at sample time
floor    = rectify_floor = 0.1
```

Everything below the current hard-tier average gets floored to the minimum weight rather than driven toward zero; everything above average gets extra weight proportional to how far above average it is. Weights are normalized and sampled from (`_sample_hard_rectified`, same file, lines 220–230). Ablation 2, a separate frozen-policy anchor mechanism, was deliberately switched off (`anchor_weight=0.0`) to keep this test to one variable — the only Phase-1 checkpoint available to anchor to is under-trained (250 episodes) and not something I trust yet.

## The risk I wrote down before I hit run

Here's the part I want to be honest about having predicted, not discovered after the fact. From the experiment's own design notes, written before training started:

> **Known risk, stated at design time**: a pure floor-based rectifier could let Krishna retreat into an easy-opponent-only niche and never develop robustness against hard snapshots. A `rectify_floor=0.1` was added specifically to keep hard opponents in rotation at a minimum rate, not exclude them entirely.

The mechanism is obvious once you say it out loud: rectified weighting, by construction, *systematically downweights the opponents currently beating you*. That's the entire point of the technique — it's supposed to concentrate training on winnable matchups. But "currently beating you" and "the ones you most need practice against" are often the same set of opponents. A floor of 0.1 was my mitigation, not a proof it was sufficient.

I want to flag one more thing I only noticed while re-checking the code for this post, because it's the kind of detail that turns "we knew this was a risk" into "we knew almost exactly when it would bite." The easy-warmup curriculum (`train_phase3.py:138, 304`) sets `p_easy = 1.0` for the first 1,000 episodes and drops it to `0.25` after that. Before episode 1,000, Krishna is essentially never facing the hard tier at all — which also means rectified weighting isn't really doing anything yet, because there's nothing in the hard tier to weight. Both schedules — the warmup curriculum ending, and rectified sampling actually taking effect for the first time — pivot on the exact same episode by construction, not coincidence. I didn't design it that way on purpose; I just didn't notice the two schedules were coupled until the data made me go look.

## Results

| Metric | This run (6,000 eps) | v7 (12,000 eps) |
|---|---:|---|
| Krishna wins | 2 / 6,000 (0.033%) | 0 / 12,000 |
| Hunter wins | 2,104 / 6,000 (35.07%) | 29.4% |
| Timeouts | 3,894 / 6,000 (64.9%) | 70.6% |
| Total pellets collected | 1,024 | — |
| Episodes with ≥1 pellet | 928 (15.5%) | — |
| Late avg100 mean (last 3,000 eps) | −27.85 | ≈ −14 to −37 |
| Gate range | 0.399–0.641, ends 0.450 | ends 0.63, trending toward evasion |
| CHER injections vs. opportunities | injections flat-to-declining (~21→~15/ep); opportunities steady-to-rising (~90→~125/ep) | not cleanly measurable (v7 confound) |

One comparability note, stated plainly so I don't accidentally mislead anyone skimming the table: this run inherits v7's harsher `K_TIMEOUT_ZERO_PELLETS = -20` live penalty, which alone drags `avg100` well into negative territory regardless of what the opponent pool does. It is not comparable to v5's positive `avg100` numbers (v5 peaked at late-avg100 +8.58). The right comparison for this run is v7, not v5, on that specific metric — and by that comparison, things aren't better on `avg100`, they're the same shape of bad. The metric that actually moved is `cher_dependency`, which v5 and v7 never touch cleanly at all.

Here's the full training-dynamics table, in 500–1,000-episode buckets, because the aggregate numbers above hide the entire story:

| Episodes | Pellet-touch rate | Mean hard-tier score | CHER dependency | Hunter win rate |
|---|---:|---:|---:|---:|
| 0–500 (easy warmup) | 26.3% | 0.0 (empty tier) | 0.315 | 22.2% |
| 500–1000 | 16.6% | 0.50 | 0.240 | 39.4% |
| 1000–2000 | 13.2% | **0.05** | 0.235 | **51.6%** |
| 2000–3000 | 12.9% | 0.049 | 0.225 | 49.5% |
| 3000–4000 | 14.7% | 0.053 | 0.198 | 42.0% |
| 4000–5000 | 15.0% | 0.039 | 0.173 | 24.6% |
| 5000–6000 | 15.7% | **0.029** | **0.160** | 11.9% |

## What worked: CHER dependency declined for the first time

`cher_dependency` goes from 0.315 in the first 500 episodes to 0.160 in the last thousand. That's not a huge absolute number, but it's the first time this specific metric has moved in the right direction across four architectural generations of this project.

The obvious objection to any declining ratio: maybe the denominator shrank, not the numerator — maybe there were simply fewer moments where CHER *could* have fired, and the agent looks less dependent only because it had less opportunity to be dependent. I checked this directly, because it's exactly the kind of thing that makes a metric lie to you if you don't. `cher_opportunities` — the count of moments meeting the safe-distance-plus-reachable-pellet criteria, independent of whether Krishna acted on them — held steady at 70–90 per episode through the middle of the run and *rose* to 120+ per episode by the end (longer-surviving episodes mechanically produce more steps for opportunities to arise). Meanwhile the raw injection count — the number of times CHER actually had to step in because Krishna didn't act on its own — stayed flat and then declined, from roughly 21 to roughly 15 per episode. Opportunities went up; interventions went down. That's not a shrinking denominator. That's the agent taking the correct action itself, in real opportunity windows, more often than it used to.

I don't want to overclaim what this means. It doesn't mean collection is solved, or even that it's growing — see below. It means the specific thing CHER exists to fix is happening less. That's worth trusting, and worth building on.

## What didn't work — and what I got wrong

I want to give this its own section with real weight, not bury it as a caveat at the end, because it's not a footnote: it's roughly half of what this experiment produced.

Right at episode 1,000 — precisely where the easy-warmup curriculum ends and rectified sampling starts operating on a non-trivial hard tier for the first time — `mean_hard_tier_score` collapses from 0.50 to 0.05 and stays there through episode 4,000. That means Krishna is losing to hard-tier opponents badly and consistently for three thousand episodes. The Hunter's win rate tracks this almost exactly: it spikes from 22–39% in the warmup period to a sustained ~50% (peaking at 51.6% in the 1000–2000 bucket, still 49.5% and 42.0% through 3000, dropping only after 4000), compared to 12–25% in the episodes before and after this window. For three thousand episodes, my agent got caught roughly one game in two — worse than at any other point in this run, and worse than the equivalent stretch in any prior version I have clean data for.

This is not a mystery I had to reverse-engineer after the fact. It is *the exact risk I wrote down before running the experiment*, playing out on a specific, identifiable schedule I hadn't anticipated. Rectified sampling downweights whoever is currently beating you. For the first thousand episodes, "currently beating you" was nobody in the hard tier, because the hard tier was barely being sampled at all under the warmup curriculum. The moment the curriculum handed the hard tier real sampling weight, rectification immediately started steering *away* from whichever snapshots were currently strongest against Krishna — which is a reasonable thing to optimize for in the medium run, and a bad thing to do to an agent that has had essentially zero practice against hard opponents up to that exact instant. The floor (`rectify_floor = 0.1`) was supposed to prevent full abandonment of hard opponents, and it did — Krishna wasn't fully isolated in an easy-opponent niche — but 0.1 was not enough to prevent a three-thousand-episode stretch of getting badly outclassed.

What I got wrong, specifically: I identified the *directional* risk (rectification could create an easy-niche retreat) but didn't think through the *timing* interaction with a second, independent schedule (the warmup curriculum) that happens to end at the same episode. Two mechanisms I designed separately, for separate reasons, turned out to be coupled at exactly the moment that mattered most, and I didn't notice until I was staring at the per-bucket table after the run finished, not before.

The run does partially recover — by 5000–6000, Hunter's win rate falls to 11.9%, better than v7's overall 29.4%. But `mean_hard_tier_score` is *still* low at that point (0.029, actually lower than mid-run), which means the recovery isn't Krishna actually solving the hard-tier matchups. It's something else — possibly the live Hunter co-adapting in a way that happens to help, possibly epsilon reaching its floor and behavior simply stabilizing. I don't have a clean causal story for the recovery, and I'd rather say that plainly than invent one that sounds tidier than the data supports.

## Why pellet collection stayed flat

Pellet-touch rate never collapses toward zero the way v4's bimodal collapse did — it holds in a 12.9%–16.6% band the entire run, ending at 15.7%. But it also never grows past that band. Whatever is happening with `cher_dependency` and the vulnerability window, the underlying rate at which Krishna even attempts collection is flat.

My read: this ablation changes *who Krishna trains against*, not *the underlying incentive to collect*. The reward asymmetry that made evasion the rational choice in the first place — the one v6 was designed to fix, the one v7 never cleanly tested because of its own confound — is still fully present here. Reweighting the opponent pool can change how much practice Krishna gets against winnable matchups, and apparently that's enough to reduce reliance on CHER's synthetic nudges in the moments collection was already somewhat safe. It is not, on its own, enough to make collection the dominant strategy. That's consistent with what I'd expect if the risk-dominance diagnosis is right: you can improve the training distribution all you want, but if the live payoff still favors survival over the objective, the agent has no reason to stop being risk-dominant just because it's had more practice being risk-dominant against opponents it can beat.

## Hardware, compute, cost

Stated plainly, because I think it matters for anyone trying to replicate or extend this: 6,000 episodes, 45.5 hours of wall-clock time, on a single MacBook running Apple's MPS backend, not a rented GPU and not a cluster. This project runs on consumer hardware by design, and I want that to stay legible in every post, not just the ones where the number is flattering. Two days of one laptop's time bought me one clean ablation and a concretely-diagnosed failure mode. That's a fine trade.

## What's next

Two threads, both cheap:

1. **Fix the vulnerability window directly before the next real run.** Three candidates, in rough order of how cheap they are to try: raise `rectify_floor` above 0.1 so hard opponents keep more sampling pressure even under rectification; decouple the easy-warmup schedule from rectification's kick-in — stagger them instead of letting both pivot on episode 1,000; or track `mean_hard_tier_score` as a live signal during training and adapt the floor dynamically instead of fixing it as a constant. I'd start with decoupling the schedules, since that's the piece I actually got wrong, not just under-tuned.

2. **Get v8 (the independent PPO port) finished.** This is the more important thread structurally. Everything in this post is DQN with a hundred-thousand-transition replay buffer holding experience from Hunter policies that, in a real sense, no longer exist by the time it's sampled. v8 asks whether the risk-dominant collapse this whole project keeps rediscovering is a property of *that specific staleness problem*, or a property of the game's payoff structure that would show up under PPO too, where the on-policy training loop can't accumulate the same kind of stale-opponent confusion. If the collapse shows up under both algorithms, that's a materially stronger claim than anything this project has produced so far — evidence I've been looking at something structural, not an artifact of the specific learning rule I happened to start with.

Also worth checking, once there's headroom: whether the `cher_dependency` decline in this post continues if I let a run go longer, and whether it survives being layered on top of ablation 2 (the frozen-policy anchor) once I have a Phase-1 checkpoint I actually trust.

Full run data and the per-episode dashboard for `20260729_194141_v7_ablations_run` are in the repo's `dashboard/` app if you want to look at anything in this post more closely than the aggregated tables above. Code referenced above: `utils/hierarchical_pool.py`, `utils/cher.py`, `train_phase3.py`, `config.py` — all on the `v7-ablations` branch at commit `2dc45af`.
