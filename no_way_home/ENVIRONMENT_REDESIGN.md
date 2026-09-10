# No Way Home — Environment Redesign: Merged Proposal

**Status:** Proposal, not yet decided or implemented. Produced by a 6-thread research sweep
(biology's environment-mediated coordination, signaling theory, formal reward specification,
event-driven architecture, information aggregation/attention economics, 2025-26 environment-design
work — full citations in `CITATIONS.md`), two independent redesigns from that research, adversarial
critique of both, and this merged synthesis. If the migration path in §4 is approved, Increment 1
is the next concrete PR; nothing here has been built yet.

*Verified against the live repo (`/Users/rahul/Q6-nwh`, branch `no-way-home`, 1,135 lines, commits `e2ba97a`→`12a49cc`, Aug 6–7 2026) rather than taken on the two design docs' say-so. Function and file references below are real, not illustrative.*

---

## 1. The concepts worth stealing, and from where

| Concept | Source | Serves which part of your framing | Status |
|---|---|---|---|
| **Manipulability continuum** (cue / index / signal as one β dial, not three boxes) | Maynard Smith & Harper, *Animal Signals* (2003); sharpened by Biernaskie, Perry & Grafen, *Evol. Letters* 2:201-209 (2018) | "spectrum, not positive/negative" — applied to *how fakeable* an emitted value is, which is the axis you actually need for REPORT vs. FORWARD vs. a new corroboration-style channel | Established |
| **Autogenic / allogenic emitter split** | Jones, Lawton & Shachak, *Oikos* 69:373-386 (1994) | "redefine the emitters" — a two-line classification test ("does this emit by existing, or by processing something else?") for every current and future emitter | Established |
| **Event Calculus fluents** (`Initiates` vs. `Happens`-only) | Kowalski & Sergot, *New Generation Computing* (1986) | Turns "ten forwards of one report are one evidence lineage" from a code-review convention into a schema-level axiom: `Initiates(FORWARD, Lineage(_), t)` is false by construction | Established |
| **Zero-intelligence as a formal limit, not a special case** | Gode & Sunder, *JPE* 101(3):119-137 (1993) — **already cited in your own `policies.py::zero_intelligence_constrained` docstring as I-9** | "emitters/subscribers/motives" — lets ZI sit on the same continuum as every other policy instead of being a bolted-on random baseline | Established |
| **Deneubourg choice function** as the deterministic-measurement/probabilistic-outcome primitive | Deneubourg, Aron, Goss & Pasteels, *J. Insect Behavior* 3:159-168 (1990) | "super deterministic in MEASURING, probabilistic in OUTCOMES" — nearly verbatim: `p = (k+C_i)^α/[(k+C_i)^α+(k+C_j)^α]` on exactly-measured trace values | Established |
| **Bounded trace weight (MAX-MIN Ant System)** | Stützle & Hoos, ACO stagnation-guard literature | Independent confirmation that your forward-storm finding is a *generic* stigmergic-medium failure (unbounded positive feedback), not a one-off bug — and gives the fix (floor/ceiling + decay) | Established |
| **Vector reward / Pareto dominance, never scalarized** | Roijers, Vamplew, Whiteson & Dazeley, *JAIR* 48:67-113 (2013) | The formal version of "radar chart, not positive/negative" for *reporting*, and the mechanism that keeps "metrics never silently become rewards" true structurally | Established |
| **Multinomial-logit discrete choice under attention cost** | Sims (2003); Matejka & McKay, *AER* 105(1):272-298 (2015) | A closed form for "a subscriber chooses what to read" — **scope-limited to one discrete choice per tick**, not a fractional portfolio (see §5 for why we don't use the portfolio version) | Established, narrowly scoped |
| **BHW informational cascades** | Bikhchandani, Hirshleifer & Welch, *JPE* 100(5):992-1026 (1992) | The Bayesian *reason* your forward-storm finding exists: once a cascade starts, each subsequent same-direction action carries ~zero incremental bits — theoretical backing for lineage-dedup being correct, not just empirically lucky | Established, **but the specific closed-form cascade formula circulating in both design docs (`1-(p-p²)^(n/2)`) was not verified against BHW's own paper by either critique pass — treat as a derived textbook approximation until traced to source, don't cite it as BHW's own equation** |
| Reward machines as a **pure observational interpreter** (never wired to reward) | Toro Icarte, Klassen, Valenzano & McIlraith, *JAIR* 73:173-208 (2022) | A citable, replayable way to state "what happened" as a small automaton over the event log — useful later, not now (see §4) | Established |

**What I deliberately did not carry forward from either design:** the Sims/Matejka-McKay *portfolio* attention-allocation machinery (a simplex over N channels spent from one shared bit-budget), the Bayesian-belief-manipulation shadow-oracle instrumentation, and Statistical-Model-Checking-with-an-uncomputed-N. All three are real, citable ideas — none of them are needed for the one experiment in §3, and both adversarial reviews independently caught specific problems in each (portfolio choice isn't actually what Matejka-McKay's closed form covers; BBM requires real new engineering surface; the SMC citation used in Design A (arXiv:2505.04322) doesn't say what it was cited to support). Better to earn them later than cite them now and be wrong.

---

## 2. The redesigned environment

**Formal skeleton — trimmed to what's load-bearing, not the full 8-tuple either design proposed:**

```
Channel = ⟨ id, emitter_class ∈ {autogenic, allogenic}, manipulability ∈ {cue, index, signal},
            lineage_role ∈ {Initiates, Happens-only}, decay_fn ⟩
```

Two fields do essentially all the work:

- **`lineage_role`** generalizes your existing `origin_id`/`is_forward` fields in `messages.py::Message` into an explicit axiom rather than an implicit convention: an emission either **Initiates** a fresh evidential fluent or is **Happens-only** relative to one that already exists. This is the single cheapest, highest-value idea in either design (both adversarial reviews independently flagged it as their #1 keeper) — it costs a field and an assertion, not a rewrite.
- **`manipulability`** is a *classification you audit*, not a continuous cost-function apparatus (both critiques found the full β-continuum-with-single-crossing-`cost_fn` machinery under-specified and not worth the engineering surface yet). The audit is one static invariant: *zero `channel_emit` events on an `index`-tagged channel from an `allogenic` emitter.* That's the entire enforcement mechanism, and it's a five-line test.

**Zero-intelligence, formalized once:** ZI is the `κ→0` / `λ→∞` limit of whatever choice rule a subscriber uses — uniform-random regardless of channel state. Your codebase already has this half-built and better than either design doc credited: `zero_intelligence` (unconstrained) and `zero_intelligence_constrained` (Gode-Sunder's own minimal-rationality, no-loss constraint, cited as I-9 in the docstring) already exist in `policies.py`. What's missing is wiring a ZI instrument into the *institution's* candidate set (`institutions.py::CANDIDATES`) — see Increment 2.

**Channel table for the food-medicine world.** I picked Design A's tighter table over Design B's 8–10-channel version — Design B's own critique showed two of its channels (#1 Locality Scarcity Index, #6 Executor Action Broadcast) are pure relabeling with zero new measurement, and the review of both flagged that going past ~5–6 channels outruns what a 1,135-line, two-day-old codebase actually has (per-locality economics isn't built — `world.py`'s own docstring: *"one shared latent blight severity and one shared food/medicine/wealth pool — the physical economy is still the single-locality one from v2"* — so a per-locality Scarcity Index channel would be describing infrastructure that doesn't exist yet, not relabeling infrastructure that does).

| # | Channel | Emitter class | Manipulability | Cost | Decay | `lineage_role` | Maps to existing code |
|---|---|---|---|---|---|---|---|
| 1 | **Blight Exposure** | autogenic | index (β=1, zero agent write) | 0 | — | Happens-only | `WorldState`'s hidden regime, `shift_tick` |
| 2 | **Resource Ledger** (shared pool — not yet per-locality) | autogenic | cue | 0 | — | Happens-only | `state.wealth`, food/medicine stocks |
| 3 | **REPORT** | allogenic | signal | fixed | exponential | **Initiates** | `messages.py::Message(is_forward=False)` |
| 4 | **FORWARD** | allogenic, content-frozen | index-of-a-claim | flat, small | inherits parent's clock — **does not reset it** | Happens-only | `messages.py::Message(is_forward=True)` |
| 5 | **CORROBORATION** *(new)* | allogenic | signal, high-β (costly to fake — requires actually observing the locality) | cost ≫ REPORT's | non-decaying | **Initiates** a *separate* `IndependentEvidence(origin_id)` fluent | doesn't exist yet — see Increment 4 |
| 6 | **Mandate/Ballot Instrument** | autogenic | index | 0 | dial: reset-on-ballot (term-committed) vs. continuous-recompute | Happens-only | `institutions.py::CANDIDATES`, your existing 4.4%-cost ablation as one parameter |

Channel 5 is the one genuinely new mechanism here (both adversarial reviews separately converged on it as the best new idea in either design) — everything else is a relabeling of what's already running, done carefully enough that the relabeling is actually true this time.

---

## 3. The precise problem

**Question:** *Does partial lineage-weighting recover most of the forward-storm resistance of full lineage-awareness, or is deduplication all-or-nothing?*

This is not "is aware better than naive" — you already proved that (0/30 vs. 24/30, `provenance_test_v1.md`). It's the *shape* of the curve between those two known endpoints, which is genuinely unknown and genuinely actionable: if a cheap 50%-discount-on-forwards already gets you most of the protection, you don't need full origin-deduplication machinery everywhere you'd otherwise want it.

**Instrument, built directly on existing code** (`institutions.py` / `policies.py` — no new channels, no new infrastructure):

```
score_β(state) = (1-β)·raw_message_rate(...) + β·unique_origin_rate(...)   # both already exist in messages.py
mitigate  iff  score_β > 0.30                                              # same threshold you already use
```

β=0 is exactly `instrument_responsive_naive` / `lineage_naive_heuristic`. β=1 is exactly `instrument_responsive` / `lineage_aware_heuristic`. Pre-registered grid: β ∈ {0, 0.25, 0.5, 0.75, 1.0}.

**Two required control arms, not optional extras** — this is where I overruled both design docs, because both adversarial reviews independently found the same hole: a monotone-looking F(β) curve is consistent with *either* "lineage information helps" *or* "any bounded/decayed counting helps, lineage or not" (the generic ACO-stagnation story), and consistent with *either* "the instrument matters" *or* "the ballot's 24-voter aggregation already absorbs the noise regardless of instrument" (the Gode-Sunder-shaped confound). Both must be ruled out, not assumed away:

- **Bounded-decay-EMA control, zero lineage information:** a `raw_message_rate` variant with MAX-MIN-Ant-System-style floor/ceiling clipping and exponential decay, *no origin dedup at all*. If this alone recovers resistance, the protection is coming from trace-bounding, not from knowing about lineage — a real, distinct, useful finding either way.
- **ZI-constrained-as-ballot-candidate control:** add a 5th entry to `institutions.py::CANDIDATES` using the existing `zero_intelligence_constrained` policy as an instrument. Structurally, this instrument has no β-knob and no rate-threshold at all — it cannot exhibit a dose-response curve. This is what makes the question "unanswerable by ZI alone": ZI gives you a floor, never a curve.

**Pre-registered numeric criterion:**

- Primary test: **Jonckheere-Terpstra** ordered-alternative trend test (not a per-point SPRT relabeled as a trend test — that was a real error in Design A, correctly caught by review) for a monotone non-increasing F(β) across the 5 ordered groups, α=0.05.
- Falsifiable shape claim, stated *before* running: a **knee** exists (`min{β : F(β) ≤ 0.10} < 1.0`) — i.e., partial weighting suffices — versus **no partial credit** (F(β) stays near 24/30 until β=1). Both outcomes are real, useful, and currently unknown.
- N: provisionally 30 seeds/arm to match existing convention, but **this must be confirmed by an actual power calculation (permutation-simulated JT null) before locking, not assumed by continuity with prior runs** — neither design doc did this, and both reviews caught it.
- Bin edges / thresholds are the existing `MESSAGE_WINDOW=30`, `LINEAGE_*_THRESHOLD=0.30` — reused verbatim, not re-tuned, so this experiment isn't quietly also a hyperparameter search.

Checks against the three criteria: **(a) measurable exactly** — `raw_message_rate`/`unique_origin_rate` are pure deterministic folds over the existing message log, no new estimator, no MI, no Bayes filter. **(b) unanswerable by ZI alone** — ZI-constrained has no β-knob, by construction. **(c) genuine generalization, not restatement** — the endpoints are already known; the knee location and the bounded-decay-vs-lineage separation are not.

---

## 4. The migration path

Every increment below runs on the *existing* 1,135-line codebase. Nothing requires the Channel abstraction, an event-sourced log rewrite, or attention economics — those are explicitly parked (see §5).

**Increment 1 (1–2 days) — schema invariant + manipulability audit.**
Add a `lineage_role` field to `messages.py::Message` (`Initiates` if `origin_id == event's own id` and `is_forward=False`, `Happens-only` otherwise — this is already fully determined by the existing `is_forward` flag, so it's a derived property, not new state). Tag the four live emitters (blight state, resource ledger, REPORT, FORWARD) with a `manipulability` class. Add one test: zero events where an allogenic emitter writes to an index-tagged channel. Net effect: the 24/30-vs-0/30 finding is now backed by an `assert`-checkable schema invariant instead of a code-review convention, with zero behavior change.

**Increment 2 (2–3 days) — the β-instrument, the bounded-decay control, and the ZI ballot candidate.**
Add `instrument_mixed(β)` to `institutions.py` per §3's formula. Add a bounded-decay-EMA rate function to `messages.py` (new counting method, zero lineage info). Add `E_zero_intelligence_constrained` to `CANDIDATES` using the policy that already exists. This is *all* the infrastructure §3 needs — no channels, no event log, no attention budgets.

**Increment 3 (3–5 days) — run the pre-registered experiment.**
Lock the β grid, the JT test, and the power-calculated N in a committed pre-registration note *before* running (this is the single biggest practical risk in the whole plan — see §5). Run the sweep plus both control arms. Produce the dose-response curve. This is the actual deliverable of this redesign — a real result, not more scaffolding.

**Increment 4 (1–2 weeks, v2, contingent) — the CORROBORATION channel.**
The one genuinely new mechanism (§2, channel 5): a second, expensive, independently-`Initiates`-ing evidence type, separate from REPORT/FORWARD lineage, testable with plain scripted policies — no attention budgets, no MI, no SMC harness. Build this *only if* Increment 3's curve is interesting enough to justify a new channel rather than more analysis of the existing one.

**Explicitly parked, not rejected:** the full Channel 8-tuple abstraction, a real event-sourced/lamport-versioned log, Sims/Matejka-McKay portfolio attention economics, Bayesian belief-manipulation instrumentation, reward-machine aspect automata, and SMC/MORL reporting infrastructure. Revisit only if Increment 3 shows genuine continuous structure worth the investment — both adversarial reviews converged on this exact gating condition independently, which is a stronger signal than either design doc's own confidence.

---

## 5. Honest risks

**Scope was oversold in both original designs, and the gap is not cosmetic.** The codebase is 1,135 lines, built in two days, with a *shared* (not per-locality) resource pool and events stored as a plain `list[dict]` — `messages.py`'s own docstring: *"Not a full I-11 event-sourced log — this is a smoke test, not the confirmatory kernel."* Both designs' framing of the redesign as "a relabeling with one new degree of freedom" describes infrastructure roughly 5–10x the size of what exists. The migration path above is scoped to reality; the full 8–10-channel apparatus in either original design is realistically 4–8+ weeks, not the days either doc's tone implies.

**This project already tried a comparably heavy formalization pass and it collapsed.** `.memory/no-way-home-phase0-v2/` — a from-scratch contract/schema/verifier layer for this same project — was rejected by independent review twice ("validation-plan operations were labels rather than a fail-closed dispatcher," "did not derive complete field/event/payload/projection/consumer routes") and is marked PARKED as of Aug 3, 2026, three days before this project's own founding document, with `implementation_authorized: false` and the stop reason logged as credit-limit exhaustion. This is direct local evidence, not a generic caution: heavy upfront schema formalization is a demonstrated failure mode *on this exact codebase*, which is the main reason I trimmed both designs' formal apparatus rather than merging them additively.

**The tautology risk is reduced, not eliminated.** Making the knee-location (not the endpoints) the falsifiable content genuinely avoids "F(0)≈0.8, F(1)≈0.0, therefore interesting" — but this framing has to be restated explicitly every time results get written up, or the same objection will correctly recur.

**The Gode-Sunder confound is addressed by the two control arms in §3, but they don't exist yet** — they're Increment 2, not already-running code. Until they're built, any β-sweep result is exactly as confounded as the original designs' were.

**`shift_tick=800` is a fixed constant across every seed** (`world.py`, confirmed directly). Neither design's mutual-information/informativeness-of-*timing* diagnostics would mean anything against a non-random shift point — which is one more reason those diagnostics are correctly dropped from this proposal rather than salvaged.

**Citation hygiene: verify before reuse, not before.** Both adversarial reviews independently caught real mismatches under primary-source checking — a Statistical-Model-Checking paper cited for a claim it doesn't make, an unverifiable exact equation (Lee, Flack & Krakauer 2024) behind a paywall, and a cascade-probability formula attributed to BHW that doesn't match what a direct search returns. None of these are load-bearing in the trimmed §3 design. If Increment 4 or later revives any of this machinery, re-verify those three specifically against primary text before citing exact numbers — a real paper with a wrong attached claim is exactly the failure mode "citable and real" is supposed to catch, and it survives a superficial check.

**Small honest finding, not from either design doc:** `world.py`'s module docstring currently says *"the full ballot/mandate/executor machinery from PREREGISTRATION.md §3 still isn't built"* — but `institutions.py` (committed one commit later) does implement it, fully (`CANDIDATES`, mandate authorization, executor election). The docstring is just stale. Small thing, but it's the same species of risk as the "schema drift" caution in the research digest, at the scale of one comment — worth a two-minute fix during Increment 1 so nobody re-derives the institution from scratch believing it doesn't exist.