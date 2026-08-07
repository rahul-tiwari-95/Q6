# No Way Home — A1 Pre-registration

**Status:** Working pre-registration, trimmed from `.memory/no-way-home-project-design-review.md`
(main branch history, commit `68a42c6`) §0–15. That source remains the fuller record if a
detail here is ambiguous. The seven-contract specification-verification system built around it
(`.memory/no-way-home-phase0-v2/`) is **parked, not deleted** — it produced no simulator code
in a full day and failed its own internal audit twice; nothing in it is load-bearing for what
follows. This document keeps the science, drops the verification bureaucracy.

**Branch:** `no-way-home`. **Worktree:** `/Users/rahul/Q6-nwh`.

---

## 1. Thesis and claim boundary

No Way Home is a generative-mechanism laboratory: does adaptive agent learning add explanatory
or functional behavior beyond what's explicitly supplied by the physical, informational, and
institutional rules of a persistent multi-agent world?

The claim form is narrow on purpose:

> In a fully specified artificial world, randomizing the learning condition while holding
> declared opportunities and exogenous streams fixed changed a preregistered outcome by a
> run-level amount, over the tested world/seed distribution.

A positive result establishes synthetic, world-relative sufficiency. It does not establish that
the mechanism is uniquely necessary, psychologically human, normatively legitimate, or
transportable to a real polity, market, epidemic, or economy. It is not a model of democracy,
capitalism, expertise, or morality. It is not a universal `BaseWorld` whose "resources" or
"risk" secretly define every later result. It does not call a dashboard vector a system-level
Q-function — that phrase gets used loosely elsewhere in this project's own brainstorming and
should be read as an aspiration to check, not a settled claim.

## 2. Invariants (I-1 through I-12)

These are the actual design discipline worth keeping. Most of them independently converged on
things the literature review for `Q6 No Way Home.md` found too — worth noting, since it means
two separate passes at this problem agree.

- **I-1 Laboratory, not universal ontology.** The reusable core is an experimental protocol and
  event contract — identity, time, observation, decision, lifecycle, provenance. Resources,
  risk, welfare, and authority stay world-specific.
- **I-2 Continuous history.** The environment can recover; the simulation never resets. Episodes
  are analysis windows, not restorations of world state.
- **I-3 Absence-driven recovery, not automatic reset.** The kernel must support agent
  activation/termination, full population absence, recovery, and successor spawning as
  first-class, not as an episode boundary.
- **I-4 Identity, lineage, lifecycle are first-class.** Every agent has stable identity and
  explicit predecessor links. A successor is never inferred from a reused name.
- **I-5 Equal voice is narrow and nontransferable.** Where a collective choice exists, one
  ballot per eligible agent, equal weight, no purchase. This says nothing about agenda access,
  reach, or informal influence — those are separately observed.
- **I-6 Competence ≠ civic standing.** Task qualification restricts *execution*, never grants
  extra votes or immunity from audit.
- **I-7 Information received is not ground truth.** Policies act only on logged deliveries, never
  on latent evaluator state.
- **I-8 Communication is a treatment, not decoration.** No-message, fixed-report, and structured
  message conditions must be separately expressible and switchable.
- **I-9 Permanent nonlearning controls.** ZI, ZI-C, scripted heuristics, and frozen checkpoints
  run through the *same* observation/action/event pipeline as the learner. This is the direct
  operational form of Gode & Sunder (1993) — market rules plus a no-loss constraint can do most
  of the work regardless of agent intelligence, and the only way to know if that's true here is
  to run the controls through the identical pipeline and check.
- **I-10 No universal welfare scalar.** Reward, private utility, physical outcome, and
  acceptance criteria stay separate objects. Nothing gets silently promoted to "the" welfare
  function.
- **I-11 Append-only truth.** Every metric must be reproducible from a versioned raw event
  stream — not from a binary checkpoint or an aggregate CSV alone.
- **I-12 Predeclare failure before scaling.** Name the estimand, controls, holdouts, and stop
  conditions before the confirmatory run, not after looking at results.

## 3. The A1 world — "the copied outbreak report"

A continuous food-medicine world. **24 agents across 4 localities** in the eventual confirmatory
design (the smoke test in §6 runs far smaller). Agents differ mechanically — production rate,
sensing quality, logistics reach, technical treatment skill, communication reach — never in
named personality. A latent blight/pathogen regime governs crop and medicine outcomes; at an
unannounced tick drawn from a held-out family, the signal-to-state or treatment-response mapping
shifts. Sensors are noisy, delayed, and correlated by calibration lineage — ten forwards of one
original report are one piece of evidence, not ten.

Every active agent gets one equal, nontransferable ballot on a bounded mitigation mandate, and a
separate equal ballot selecting one of four qualified, frozen candidate **executors** — only the
selected executor may choose the technical instrument (dosing/routing) within the mandate voters
set. Executors don't learn. Citizens may.

**Supplied, not claimed as emergent:** physical dynamics, production functions, signal lineage
rules, message vocabulary, ballot equality, the four-candidate roster, election/removal rules,
each candidate's frozen instrument policy, action legality, private utilities, lifecycle rules.

**Potentially emergent, worth actually measuring:** attention allocation, provenance-checking
behavior, strategic forwarding, trust, the realized mandate vote and executor vote, mandate
revision, specialization, whether behavior changes after the regime shift.

**The project must never claim the supplied ballot, election, or market mechanism itself
"emerged."** That distinction — supplied rule vs. observed behavior under the rule — is the one
most agent-based-economics work blurs, per Windrum, Fagiolo & Moneta's critique, and it's the
one this design is explicit about.

## 4. Minimal experiment specification (target — full confirmatory scale)

This is the eventual, fully-powered design. It is **not** what we run first (see §6).

- **Estimand:** paired contrast between one locked nonlearning comparator and the online citizen
  policy, on need-shortfall agent-ticks per 10,000 world-ticks, lower is better.
- **Window:** 25,000 ticks per run — 5,000 burn-in, shift drawn in [8,000, 10,000), 10,000-tick
  post-shift primary window, remainder for persistence diagnostics.
- **Core factorial:** 2×2×2 — citizen update on/off, provenance visible/hidden, regime
  stationary/one of three held-out shift families.
- **Permanent controls:** ZI, ZI-C, fixed need-first / fair-share / lineage-counting heuristics,
  a fixed Bayesian estimator, a frozen pre-shift checkpoint, a scripted voter/selector, and one
  treatment-independent physical-feasibility oracle used only as a regret diagnostic, never as
  the primary outcome.
- **Scale:** 12 calibration blocks + 36 confirmatory blocks per arm, stratified across shift
  families and candidate-roster difficulty.
- **Acceptance (H1 supported only if all hold, on untouched holdouts):** ≥15% primary loss
  reduction vs. the locked comparator; 95% paired CI excludes zero with a ≥10% lower bound; beats
  the frozen twin (isolating online updating from inherited competence); positive in ≥2 of 3
  shift families; worst-locality shortfall no more than 5% worse than the comparator; survives
  identity permutation and isn't carried by one seed or roster.
- **Stop conditions:** evaluator-only information reaching a policy; the confirmatory CI landing
  entirely inside a ±5% equivalence margin; the effect reversing under high-scarcity or
  clustered-lineage conditions; provenance visibility having no causal effect at all (making it
  decorative); denominators requiring exclusion of dead/absent agents to produce the result.

These thresholds (15%/10%/5%) were Rahul's own selection on 2026-08-02 and shouldn't be relaxed
after seeing results. They're preserved here as the target for whenever this reaches
confirmatory scale — which is explicitly **not** this week.

## 5. Risks, named in advance

- **Tautology:** if a supplied mechanism (the election, the mandate) is later described as
  "emergent," the claim is void by construction. §3's supplied/emergent split exists to prevent
  this, and every write-up must check against it before claiming anything emerged.
- **The Gode-Sunder problem:** if ZI-C already reproduces the online citizen's need-shortfall
  performance, the institution — not learning — is doing the work, and that's the actual
  finding, not a bug to iterate past.
- **Over-formalization of secondary diagnostics.** The regime-classification ("J8") diagnostic
  spent an entire day being formalized to the point of requiring injectivity proofs over a
  generator's mathematical image — before it's even the primary metric. It's demoted here to
  "compute it eventually, simply, once the primary metric already shows something interesting."
  Calibration/Brier scoring for the latent regime is useful, but it is not gating anything.
- **Benchmark formalism eating the runway.** The source document's workload-gate section is a
  single paragraph specifying exact disk-byte and timing formulas for benchmarking code that
  doesn't exist yet. Replaced here with: build it, run it, measure real throughput, decide
  scale from that — the same way every other Q6 benchmark has been done.

---

## 6. What we're actually running first — the nontriviality smoke test

This is new content, not in the source document, and it's the entire point of trimming: answer
one fast, cheap question before investing further.

**Question:** does this world produce *any* separable signal between good and bad policy on the
primary metric — or does zero-intelligence (or a trivial fixed heuristic) already explain
everything, the way it did for Sugarscape and for real double auctions?

**Scale, deliberately far below §4:**
- 8 agents, 1 locality (not 24 agents / 4 localities)
- 2,000 ticks per run (not 25,000), regime shift at a fixed tick around 800 (not drawn from a
  held-out family — determinism is fine for a smoke test, randomize once this shows signal)
- 5 seeds per policy (not 36 blocks)
- Policies compared: **C1 (zero-intelligence)**, **C2 (zero-intelligence, budget-constrained)**,
  **C3 (fixed need-first heuristic)**, **always-mitigate**, **never-mitigate**, and a
  **physical-feasibility oracle** (upper bound, not a real policy)
- No learning agent yet. No election, no message lineage, no provenance treatment. Just the
  physical world (food/medicine/blight + one regime shift) and the permanent nonlearning
  controls, because those have to be run through the pipeline and checked *before* anything
  else is worth building on top.

**What would make us stop and redesign before building anything more:** ZI-C's need-shortfall is
statistically indistinguishable from always-mitigate's, or the oracle's regret relative to ZI-C
is near zero. Either means the physical world itself isn't discriminating, and no amount of
citizen learning built on top of it will produce a meaningful result — exactly Gode & Sunder's
lesson, checked directly instead of assumed away.

Results are in `no_way_home/results/smoke_test_v1.md` once the run below completes.
