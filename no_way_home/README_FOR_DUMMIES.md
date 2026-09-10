# No Way Home — For Dummies

> **2026-09-06 release update:** This document records the earlier project state. Read the [errata](../docs/experiments/provenance-errata.md) before using its research claims: beta monotonicity is structurally supplied, the old test ignored seed pairing, and the learner receives deduplication as a feature. The [new calibrated-control pilot](../docs/experiments/provenance-results.md) supersedes the old next-step plan. Original as-run result reports are preserved.

*A living document. This explains the project in plain language with concrete examples,
not formal notation. It will be updated as we run more experiments and the world grows.
For the formal spec, see [`PREREGISTRATION.md`](PREREGISTRATION.md). For the full redesign
this doc is based on, see [`ENVIRONMENT_REDESIGN.md`](ENVIRONMENT_REDESIGN.md). For the
active work queue, see [`ENGINEERING_NWH_PHASE_PLAN.md`](ENGINEERING_NWH_PHASE_PLAN.md).*

---

## What is this, in one paragraph

A small world with several localities, each with agents who can get sick. A hidden
"blight" regime occasionally gets worse, which drops food yield and raises sickness risk.
Localities that notice a sickness spike can send a `REPORT`; other agents can `FORWARD` a
report they heard about. The world has a real, scarce economy (food and medicine stocks
that spoil, a shared wealth pool that pays for mitigation). An elected institution decides,
term by term, whether to spend the collective budget on mitigation, based on how "loud"
the reports look. The central research question: **can a system built out of dumb parts
(noisy voters, simple report-counting) reliably make good decisions, and what specifically
breaks that?**

## Why "channels" instead of one big bucket of state

The naive way to build this world is one big bag of numbers — `food_stock`, `wealth`,
`blight_high`, a list of messages — and every agent just reads whatever it wants out of
the bag. That's roughly what the code still looks like today.

The redesign's idea: stop treating the environment as one undifferentiated bucket, and
instead treat it as a handful of distinct **channels**, where each channel gets its own
profile. Think of it like a radar chart instead of a single dial — each channel is its own
axis, with its own answer to four questions:

1. **Who emits it, and how?** — *autogenic* (emitted just by existing, no one had to act)
   vs. *allogenic* (emitted because an agent actively processed something)
2. **How fakeable is it?** — *cue* (a pure side-effect, nobody designed it to
   communicate) vs. *index* (mechanistically tied to the real thing, hard to fake) vs.
   *signal* (deliberately produced to inform someone, which is exactly why it can be
   gamed unless it's costly)
3. **Is it fresh evidence, or a repeat?** — *Initiates* (creates a brand-new evidential
   fact) vs. *Happens-only* (just references a fact that already exists)
4. **Does it fade, and how fast?** — every channel has its own decay behavior

## The four questions, with real-world examples first

- **Autogenic vs. allogenic**: a tree emits shade just by existing — autogenic. A beaver
  emits a dam by actively cutting down trees and building something — allogenic. In our
  world: `blight_high` and the wealth/stock numbers are autogenic (nobody "sends" them,
  they just are). A `REPORT` or `FORWARD` is allogenic — an agent chose to act.

- **Cue vs. index vs. signal**: a footprint in mud is a *cue* — the animal wasn't trying
  to tell you anything, you're just reading a side-effect. A deer's antler size is an
  *index* — you can't fake big antlers without actually being healthy enough to grow them,
  so it's a hard-to-game indicator of the real thing. A peacock's tail or a warning call is
  a *signal* — actively produced specifically to communicate, and because it's produced on
  purpose, it's the one that needs a cost attached or it gets abused.

- **Initiates vs. Happens-only**: if you personally witness a car accident and call it in,
  that call *initiates* a fact — it's new evidence. If you then tell three friends what you
  heard, and they each tell three more people, none of those retellings are new evidence —
  they're all *happens-only* references to the one thing you actually saw. The rumor can
  spread to a hundred people without the underlying evidence growing at all.

- **Decay**: news of an outbreak matters a lot the day it happens and less a month later —
  unless it's still ongoing. A signal's usefulness fades, and how fast it fades is a real
  design choice, not an afterthought.

## Walking through it with our own world

Say Locality 2 has an outbreak. Agent A there notices 3+ sick neighbors and files
`REPORT(origin_id=42)`. That's allogenic (Agent A acted), a signal (deliberately produced
to inform), and it **Initiates** — origin_id 42 is a brand-new piece of evidence, with its
own decay clock starting now.

Now 5 different agents near the "hub" locality each hear about report 42 and forward it.
Every one of those forwards is **Happens-only** — same origin_id, no new observation, just
a relay. None of those 5 agents actually checked locality 2's sickness count themselves.
A forward is also tied to its parent's decay clock, not its own — it can't "refresh" a
stale report just by being repeated, or a report that's actually gone quiet could look
artificially fresh forever purely from people passing it around.

If a policy naively counts *all* messages, it sees 6 and thinks the signal is 6x as strong
as it actually is. That's the **forward storm** — and we've already proven, with a real
30-seed experiment, that a policy which counts unique origins instead of raw messages is
fooled by this 0/30 times, while a policy that counts raw messages is fooled 24/30 times.
This "Initiates vs. Happens-only" distinction is just making that difference explicit as a
rule about the world's own channels, instead of leaving it as an implicit convention in
one policy's code.

## The channel table, in plain language

| Channel | In plain words |
|---|---|
| **Blight Exposure** | The hidden truth about how bad things are. Nobody can fake it, nobody emits it on purpose — it just is what it is. |
| **Resource Ledger** | The wealth/stock numbers. A side-effect of the economy running, not a message to anyone. |
| **REPORT** | A deliberate, fresh claim: "I'm seeing sickness here, right now." |
| **FORWARD** | A cheap relay of someone else's claim. Adds reach, adds zero new evidence. |
| **CORROBORATION** *(new — doesn't exist in code yet)* | A *second, independent* observation of the same situation — someone else who actually went and checked, not someone who just repeated what they heard. |
| **Mandate/Ballot Instrument** | The institution's own current state: who's authorized to spend, and under what term. |

**CORROBORATION is the one genuinely new idea here.** Right now the world can only tell the
difference between "one person said it" and "one person said it and got echoed a lot" — it
has no way to represent "two people independently confirmed it," which is a real and
different thing from an echo chamber. Adding it is the one part of this redesign that isn't
just a more careful relabeling of what already exists.

## Why any of this matters

Once REPORT and FORWARD are classified this precisely, a natural next experiment falls
out: what if a policy doesn't have to choose between fully trusting unique-origin counting
or fully trusting raw counting — what if it can partially discount forwards? We can dial a
single number β from 0 (pure raw count, gets fooled a lot) to 1 (pure unique-origin count,
never fooled) and ask: is there a "knee" where even a modest discount already recovers most
of the protection, or is deduplication genuinely all-or-nothing? That's the concrete,
falsifiable experiment this whole redesign is aimed at — see `ENVIRONMENT_REDESIGN.md` §3
for the full pre-registered design, and `ENGINEERING_NWH_PHASE_PLAN.md` for what's actually
being built right now.

## A note on how we work

This project deliberately tries to keep every claim checkable: metrics are pure functions
over an append-only event log (`metrics.py`), world physics and policy decisions get
independent random-number streams so "same seed" reliably means "same physical trajectory"
(`world.py`), and negative or awkward findings get written down and kept, not quietly fixed
and erased (`results/README.md` has the full history, including two real bugs found and
fixed along the way). The redesign above follows the same instinct: simple concepts, kept
small enough to actually verify, are worth more than an impressive-sounding apparatus that
can't be checked.
