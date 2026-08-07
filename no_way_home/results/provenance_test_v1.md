# No Way Home — Provenance / Lineage-Awareness Test

Follow-up to `smoke_test_v1.md`, testing invariant I-7 directly: *"Ten forwards of one test
are one evidence lineage, not ten independent tests."* World extended to 4 localities (24
agents total) with a message-lineage layer (`no_way_home/messages.py`): a locality reports an
outbreak when its local sick count crosses a threshold, and existing reports get randomly
forwarded — biased 5x toward one "hub" locality, the high-reach-source scenario the source
design's own worked example names directly.

Two policies compared, same decision rule, same numeric threshold (0.30 pieces-of-evidence per
tick over a 30-tick window), differing only in what counts as one piece of evidence:

- **`C3b_lineage_naive_heuristic`** — counts every message (original report or forward) as one
  unit of evidence.
- **`C3c_lineage_aware_heuristic`** — deduplicates by origin before counting; ten forwards of one
  report count as one.

## Direct check first

Before trusting anything built on top of it, the counting mechanism itself: ten independent
reports from ten different localities vs. one report forwarded ten times.

| | raw message rate | unique-origin rate |
|---|---:|---:|
| 10 independent reports | 1.0 | 1.0 |
| 1 report forwarded 10x | 1.1 | 0.1 |

Raw counting genuinely cannot tell these apart (1.0 vs. 1.1). Unique-origin counting gets it
exactly right (1.0 vs. 0.1, a full 10x difference). Locked in as
`test_raw_vs_unique_message_rate_distinguishes_copies_from_independent_reports`.

## Full-world result, 30 seeds

| Policy | Mean shortfall/10k ticks | Std | Seeds fooled into a pre-shift false mitigation |
|---|---:|---:|---:|
| `C3c_lineage_aware_heuristic` | 43,578 | 344 | **0 / 30** |
| `C3b_lineage_naive_heuristic` | 58,211 | 7,851 | **24 / 30** |

"Fooled" means the policy mitigated during the pre-shift (low-blight, no real crisis) period —
purely because a forward storm pushed raw message volume over the trigger threshold, not because
anything was actually wrong. `lineage_aware` never does this, across every seed tested.
`lineage_naive` does it on 80% of seeds, and when it happens the false-mitigation count is large
(commonly 200+ out of 800 pre-shift ticks) because once a popular report is circulating, more
forwards of it keep coming, so the false alarm doesn't just fire once — it can persist.

This shows up as two separate costs, not one: `lineage_naive`'s mean outcome is 33% worse, but
its **standard deviation is 23x larger** (7,851 vs. 344). It isn't just worse on average — it's
unpredictable, because whether a given run gets fooled depends on random forwarding luck, not on
anything the world or the agent's information actually tells it. `lineage_aware` is both better
and far more consistent, because it's responding to real signal instead of viral noise.

## Reading this honestly

This is a toy world with hand-picked parameters (report threshold, forward probability, hub
bias), not a claim about real misinformation dynamics. What it does establish: the I-7
distinction is not just a good methodological principle to write down, it's mechanically
checkable, and in a world with even a mild high-reach-source asymmetry, ignoring it costs real,
measurable, and highly variable performance — using nothing more exotic than a hidden regime
shift, a fixed forwarding bias, and two counting functions that differ by three lines of code.

**What this doesn't yet test:** whether a *learned* policy would discover the unique-origin
distinction on its own, given only raw message features as input — that's the actual research
question underneath "epistemic delegation," and it's still two increments away (need a learner,
and need to not hand it the dedup logic for free the way `lineage_aware_heuristic` does). This
result is scaffolding for that question, not an answer to it.
