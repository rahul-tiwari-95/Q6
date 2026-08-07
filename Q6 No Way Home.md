# Q6: No Way Home

**Status:** Founding document for the next universe of Q6. Not yet started. Written 2026-08-01.
**Maintainer:** Rahul Tiwari. Research and drafting: Claude.
**Companion to:** [`Q6.md`](Q6.md) (mission, research questions, roadmap for the Hunter/Kṛṣṇa line of work — still active, not superseded).

---

Every version of this project so far has lived inside one 25×25 grid, with exactly two agents, one pellet-collector and one pursuer, and one question: does self-play converge to the strategy that completes the objective, or the strategy that merely survives. That question is still open, and it's not being abandoned. But on 2026-08-01, Rahul described a much bigger idea in one long, unfiltered message — and reading it back, it wasn't a departure from Q6. It was Q6's own instincts asking for more room than a 25×25 grid could give them. This document exists to make that continuity explicit, to lay out what a stranger would need to know to pick this up, and to commit — in writing, before any of it is built — to what we think, what we don't know, and what would tell us we're wrong.

The title is not decorative. Every prior version of Q6 shows up again here, recognizable, in a bigger world.

---

## 1. How This Started — In Rahul's Own Words

This section exists so nobody, including future-Rahul, mistakes this for a idea that arrived from nowhere. It's lightly organized from one message, typed in one sitting, unfiltered on purpose — the phrasing is preserved because the phrasing is the specification.

**The provocation — beyond the gridworld:**

> "I want to create/work on a broader multi agent system. Like lets go beyond a hunter gridworld — lets think of a big enough matrix — a smaller version I can run on mac for faster tests but the true version would maybe required more compute."

**The inspiration — how organisms actually learn:**

> "Taking inspiration from real world and especially how toddlers and kids learn continuously by taking feedback from the environment."

**The feedback philosophy — rich, not scalar:**

> "I want a very well thought off reward and punishment signals — different and minute representations."

**The population structure — heterogeneity as a first-class object:**

> "There are multi agent, and we can think of a radar spider graph — what 3-5 qualities we can have — which we can tune across different agents — and then these multi agent interact with each other."

**The reframing of success — away from a scoreboard:**

> "There is no defined winning or loss. Its more nuanced and broader, and they take feedback from environment and from interacting with each other all the time."

**The central metaphor — everything is a market:**

> "Because in reality, our world in which I live as a human is multi faceted. There is a market for everything — market meaning: you come with what you were given (at birth or instantiation), you come with what you acquired, and then you compare yourself on a market — supply and demand. So if we think like this, there is money market, job market, dating market — everything is a market. If we make/design a world where we first spend time understanding the foundations, and then you compete in this market..."

**The unit-of-analysis shift — this is the one that has no obvious prior art:**

> "We dont follow one agent like we did with Krishna, but we see the whole system as one. And have like a Q-table for the whole sim... we see what the sim is doing in general."

And then, explicitly, an instruction that shapes this whole document:

> "Do web and internet research extensively and share your thoughts. What do you think? Lets shape the clay pot."

That's the brief. Everything after this section is either "here is what Q6 already did that points here" (§2), "here is what the world already knows about each piece of this" (§3), or "here is what I'd actually build and why" (§4).

---

## 2. The Natural Progression — What Q6 Already Was Building Toward

This is the part Rahul asked for explicitly: not a clean-room pivot, but a demonstration that the market-world idea is where Q6's own unresolved instincts were already pointing. Eight threads, each traced from actual commits and actual files in this repo, not narrated after the fact.

### 2.1 Multi-agent was never really the destination — it was Q6's starting assumption

Q6 has had two learning agents since Phase 2. Kṛṣṇa and Hunter aren't a single-agent RL problem with a scripted adversary bolted on — from `train_phase2.py` onward, both networks update simultaneously, and Fictitious Self-Play exists specifically because two co-training agents *without* it cycle endlessly (Krishna beats Hunter, Hunter adapts, Krishna adapts back, no net progress). The opponent pool (`agent/opponent_pool.py`, then `utils/hierarchical_pool.py`) is, structurally, a small population — snapshots of past Hunter policies that Krishna has to remain competent against, not just the single current one. "No Way Home" doesn't introduce multi-agent dynamics to Q6. It removes the cap of two.

### 2.2 The opponent pool already IS a primitive market — we just hadn't named it that

This is the single most direct throughline, and it was sitting in the code before either of us called it a market. Ablation 1 (`utils/hierarchical_pool.py`, branch `v7-ablations`) reweights which Hunter snapshot Krishna trains against using:

```
weight_i = max(score_i - baseline, 0) + floor
baseline = mean(score) across the current hard tier, recomputed at sample time
```

Read that again with matching theory in mind: this is preference-weighted sampling toward counterparts you currently have a favorable outcome against, with a floor that guarantees the disfavored counterparts are never fully excluded from the market. It is, mechanically, a crude one-sided matching mechanism — Krishna doing something close to revealed-preference selection over a pool of potential "counterparties," rectified toward matches it currently wins. Ablation 1b then asked the natural next question of any market mechanism: what happens when you change the floor and the warmup schedule that governs how quickly the market's weighting takes effect? That is literally a market-design parameter sweep, run and pre-registered as one, five days before Rahul described "everything is a market" in the abstract. The market metaphor isn't new to Q6. It's been running training jobs for a week.

### 2.3 Reward design has been the site of every hard-won lesson so far — "minute representations" is a request for what Q6 already learned it needed the hard way

The single most expensive lesson in this project's history is a reward-scale story: a `-5` wall penalty completely drowned the `+50` pellet signal for hundreds of episodes; reducing it to `-1` helped, and `0` fixed it outright. CHER (Counterfactual Hindsight Experience Replay) exists because a single terminal reward wasn't a rich enough signal to teach collection under threat — it injects synthetic transitions with a bonus specifically at moments judged safe. Then v7's own confound — a *weighted* CHER bonus (`bonus = 30 × min(2.0, hunter_dist/12)`) that quietly undershot the flat bonus it replaced in the most common distance range, landing in the same commit as a harsher timeout penalty — is a first-hand, already-paid-for lesson in exactly the failure mode the reward-pathology research below formalizes: multiple reward channels changing at once, at different effective magnitudes, produce unattributable results. "A very well thought off reward and punishment signal, different and minute representations" isn't a new ambition. It's Rahul, having been burned by an under-specified reward twice, asking to do deliberately what Q6 has so far only done by trial, error, and a 36-hour wasted run.

### 2.4 The system was already drifting away from per-agent scores, toward population-level diagnostics

`cher_dependency` (later `cher_dep_idx`) is not Krishna's reward. It's a ratio describing whether the *population's* synthetic-teaching mechanism is still necessary — a property of the training system, not of any one episode's outcome. `mean_hard_tier_score` is an EMA over the whole hard tier's outcomes, not any single match. And the diagnostic that actually cracked open the ablation-1 vulnerability window wasn't a per-agent metric at all — it was splitting Hunter's win rate by `mode` (joint vs. fsp) and discovering the damage was concentrated in a mode the opponent-pool mechanism doesn't even touch. That is, in miniature, exactly the discipline Perolat et al.'s Utilitarian/Equality/Sustainability/Peace vector formalizes for full multi-agent systems (§3.5): describe the *system*, not the agent, and split it by structural category before trusting an aggregate. Q6's dashboard already had to grow algorithm-aware, presence-guarded charts to keep up with this instinct. "A Q-table for the whole sim" is a name for a habit this project already had.

### 2.5 "No defined winning or loss" is the same sentence as Q6's own diagnosis of Krishna

Q6's mission statement doesn't call Krishna's evasion a bug. It calls it *correct behavior under the wrong incentives* — the risk-dominant equilibrium, not a broken agent. That is already most of the way to open-endedness's actual stance on win conditions: replace "did the agent win" with "is the system's behavior novel and learnable to an observer" (Hughes et al., §3.1). Q6 arrived at "there's no single correct behavior, only equilibria under incentive structures" from a completely different direction — chasing why an agent kept losing — and No Way Home asks the same question one level up, about a whole population instead of one pair.

### 2.6 The continual-learning scoping decision already pointed at "toddlers"

When Rahul's original brief — "figure out continual learning" — got scoped down to plasticity loss and primacy bias (Dohare et al. 2024; Nikishin et al. 2022), that scoping decision was itself implicitly developmental: it's about whether a network *keeps the capacity to learn* under prolonged non-stationary experience, which is precisely the question developmental AI asks about children versus the field's usual "train once, deploy frozen" assumption. §3.4 below is the literature that scoping decision was always adjacent to, now engaged directly instead of by proxy.

### 2.7 Pre-registration and honest negative results are already Q6's house style — and they turn out to be exactly what a market-world needs to stay credible

Ablation 1b is a completed, code-audited, negative result: raising the rectification floor and ramping it in made the vulnerability window *worse*, not better, and reversed the one metric decline that made ablation 1 worth building on — written up in full, criteria stated before the run, not after. That discipline is not incidental to No Way Home; it is the field's own prescribed cure for agent-based economics' worst disease. Windrum, Fagiolo & Moneta's standard critique of thirty years of agent-based economics is that "the model can generate any result" once it has enough free parameters — and their remedy is exactly Rahul's existing habit, stated as a formal recommendation: pre-register numeric macro targets before the run, and require the model to hit several simultaneously with one parameter set. Q6 doesn't need to learn this discipline for the new project. It needs to keep doing what it already does.

### 2.8 The infrastructure already scales past one run

The orchestrator (`orchestrator/orchestrator.py`) — crash-restart, checkpoint-aware resume, config hot-reload, now with fixed shutdown semantics and collision-safe status paths — was built to run `v8` and `v7-ablations` unattended in parallel on one laptop. It doesn't know or care that it's running two-agent gridworlds; it supervises `{id, cwd, command}` tuples. A market-world with 200+ agents run as one more job in that same daemon is not new infrastructure. It's the same daemon doing what it was already asked to do.

---

## 3. The Research — What the World Already Knows About Each Piece of This

Seven parallel research threads, each mapped to one thing Rahul said. This was real search — real papers, real repos, real numbers — not synthesis from memory. Every claim below has a citable source.

### 3.1 "There is no defined winning or loss" → Open-Endedness and Quality-Diversity

The field has a real, current, formal definition: a system is **open-ended** relative to an observer if its output stream is both **novel** (the observer's prediction loss on future outputs keeps rising) and **learnable** (loss falls as the observer sees more history) — Hughes, Dennis, Parker-Holder et al., *Open-Endedness is Essential for Artificial Superhuman Intelligence*, ICML 2024 (arXiv:2406.04268). Before that, Soros & Stanley (ALIFE 2014) proposed four necessary conditions for open-ended evolution and showed each one's removal causes stagnation: a minimal criterion for reproduction; evolution creating novel ways to meet it; agents choosing their own interactions; and phenotype complexity not capped by the representation.

Your "radar spider graph of 3-5 qualities" is a **behavioral descriptor space**, and the practical machinery already exists: **MAP-Elites** (Mouret & Clune, 2015) discretizes a low-dimensional descriptor space and keeps the best solution per cell — the archive itself is a system-level object, literally a table over trait-space, i.e. a real candidate for "the Q-table for the whole sim." The 3-5 dimension instinct is *correct*, and for two separable reasons: naive grids cost n^d cells (fixed by CVT-MAP-Elites, which decouples archive size from dimension count), and — more fundamentally, per a 2026 ICLR Oral (Tjanaka et al.) — high-dimensional descriptors cause *distortion*, where distinct solutions collapse onto near-identical measures and exploration silently stalls even when the memory problem is solved.

**Concrete, stealable mechanisms found:**
- **Bounded-window minimal criterion** (POET, Wang et al. 2019): admit a new niche only if at least one agent can survive it but none dominates it — a two-sided viability test replacing a win condition.
- **PATA-EC** (Enhanced POET, Wang et al. 2020): characterize a niche by the *rank ordering it induces over the whole population*, not by hand-designed features — evaluate everyone, rank-normalize, and that vector is a domain-general novelty descriptor. This is exactly what a market's price/rank vector already is.
- **Lexicase selection** (Boldi, Ding & Spector, NeurIPS 2023): select on many fine-grained signals in random order rather than summing them into one scalar — the sharpest existing match for "very well thought off, minute representations" — and it beat MAP-Elites outright on deceptive domains *without optimizing for diversity at all*, which is also the field's strongest warning that a good diversity score doesn't prove your diversity mechanism did anything.
- **α-Rank** (Omidshafiei et al., *Scientific Reports* 2019): builds the empirical payoff table for a whole population and returns the stationary distribution of an evolutionary Markov chain over strategy profiles — a genuine, off-the-shelf "table for the whole sim," with sink strongly-connected components that would have named Q6's own evasion equilibrium as a system-level object, not an agent-level bug.
- **Minimal Criterion Coevolution** (Brant & Stanley, 2017/2019): two populations coevolving with *no fitness function, no novelty archive, no descriptor at all* — just a viability constraint and a fixed-size rotating queue — produces unbounded, non-plateauing complexity growth. The cheapest possible engine for "no defined winning or loss," small enough to build in a day.

**Honest failure modes:** an *unaligned* descriptor axis doesn't overcome deception and can perform significantly worse than plain fitness optimization (Pugh et al., tested directly); novelty search degenerates toward random search in large behavior spaces by the original authors' own admission; MAP-Elites is, by its creators' own words, "by definition" not open-ended, because its descriptor axes are fixed at design time; and every open-ended system in the literature — POET, hide-and-seek's six phases, Enhanced POET itself — eventually plateaus once its encoding is exhausted. Most relevant to Q6's own history: minimal-criterion coevolution's two populations can **collude** — mazes discovering the cheapest possible way to look complex without being hard — which coevolution researchers named decades ago as disengagement, cycling, and mediocre stable states (Ficici & Pollack, 1998). Krishna's evasion is a mediocre stable state. This literature has a fix, tested since 2004: reduce the stronger population's *virulence* — reward it for a moderate margin of victory, not maximal defeat — because an agent that wins completely destroys its own training gradient.

### 3.2 "There is a market for everything" → Agent-Based Computational Economics

Your vision was substantially built in 1996, as **Sugarscape** (Epstein & Axtell, *Growing Artificial Societies*). A toroidal grid, a renewable resource, and agents carrying a small integer trait vector — vision, metabolism, lifespan, starting wealth, each drawn from *uniform* priors — already produces a strongly skewed wealth distribution from perfectly symmetric inputs, with no win condition, ever. It has no neural networks, and it is the existence proof that the vision works at all. Rules are explicitly modular (growback, move-harvest-metabolize, death-and-replace, trade, inheritance, credit, culture, disease, combat) and independently switchable — the ablation design is *built into* the model rather than bolted on, exactly Q6's existing rule-flag discipline.

The methodologically richer precedent is the **Santa Fe Artificial Stock Market** (Arthur, Holland, LeBaron, Palmer & Tayler, 1996/97): 25 agents, each holding 100 condition/forecast "predictor" rules, two of whose bits are hardwired **controls carrying zero information by construction**. The claim "technical trading emerged" is then a statistical test — are the real information bits set significantly more than the control bits — not an anecdote. A single scalar (how often the population's rule-discovery process runs) flips the *entire system* between two regimes: a rational-expectations equilibrium (low volume, no bubbles) and a self-organizing "complex" regime (permanent technical trading, bubbles and crashes, volatility statistics matching real IBM data) — a genuine phase transition, measured on system-level observables, which is close to the cleanest existing demonstration of "we see what the sim is doing in general" as a real research result rather than an aspiration.

The modern deep-RL instantiation is **The AI Economist** (Zheng et al., Salesforce, *Science Advances* 2022): a 25×25 Gather-Trade-Build economy, heterogeneous Pareto-distributed skills, a discretized continuous double auction, and a co-adapting social-planner agent setting tax policy — trained with a three-phase curriculum specifically because, in their own words, "as one actor changes its policy, the shape of the entire reward function for other actors may change significantly." That instability-management pattern is structurally identical to rectified opponent sampling in Q6.

**The single most load-bearing empirical fact in this entire research effort:** DeepMind's **Emergent Bartering** work (Johanson, Hughes, Timbers & Leibo, 2022) shows agents *do not learn to trade* when the trading action is generic (drop-an-item, give-an-item) — a unilateral give only pays off if a stranger reciprocates later, which sits outside any usable credit-assignment horizon. Trade only emerges when exchange is an **atomic**, environment-enforced offer/accept swap. This is not a detail. It's the difference between a market existing and not existing.

**Honest failure modes, from the field's own internal critique:** Windrum, Fagiolo & Moneta (2007) name *over-parameterization* ("the model can generate any result") and *equifinality* (many different micro-rule sets produce identical output traces) as the field's central methodological threats. Axtell & Farmer's 2025 *Journal of Economic Literature* review — co-authored by Sugarscape's own creator, reviewing thirty years including their own work — states plainly: **"Today we lack an understanding of which rules of agent behavior are sufficient to produce realistic-looking multi-agent institutions."** That is the field's standing open scientific question, not a solved prerequisite. On the encouraging side, they name a genuine counterexample to "ABM produces pretty pictures and no science": NASDAQ's own agent-based model of its Small Order Execution System correctly predicted the effects of decimalization before it happened — a real, falsifiable, out-of-sample result, achieved with a year of work and proprietary micro-data, which is a useful calibration for what a solo project can honestly claim.

### 3.3 "Money market, job market, dating market" → Matching and Market Design Theory

These are not the same math wearing different clothes, and the choice of formalism determines what the simulation can possibly show. **Two-sided matching without money** (Gale & Shapley, 1962; Alvin Roth's Nobel-winning market-design program) replaces "winning" with **stability**: a matching is stable if no pair mutually prefers each other over their current assignment. That's a computable, system-level benchmark, not a per-agent score — the direct formalization of "no defined winning or loss, more nuanced." **Search-and-matching theory** (Diamond-Mortensen-Pissarides, Nobel 2010) is what a gridworld with local interaction actually *is*: an aggregate matching function with congestion externalities, generically inefficient — and Adachi (2003) proves the crucial bridge, that as search frictions vanish, decentralized search-equilibrium outcomes converge exactly onto the set of Gale-Shapley stable matchings. The gridworld and the theoretical clearinghouse are the same object at two different friction settings, and friction is a dial you can turn and measure.

**The single most dangerous result for this entire project**, found independently in this thread and the previous: **Gode & Sunder (1993)** showed *zero-intelligence* traders — bidding uniformly at random, constrained only to never accept a loss — hit 97–99.9% allocative efficiency in a real double-auction market, statistically indistinguishable from human traders. The market *structure*, not agent intelligence, does almost all the work. A badly-chosen mechanism makes the whole simulation unfalsifiable: it will look successful no matter what, or fail to, the agents learn. Cliff & Bruten (1997) supply the escape: the Gode-Sunder result is an artifact of *symmetric* supply and demand gradients and provably fails once you make them asymmetric — a concrete, testable design lever, not a dead end.

**The hardest open technical question for the radar graph specifically:** Chiappori, McCann & Pass prove a multidimensional trait vector collapses to a single scalar "market value" only under a narrow condition called index-separability — and it is *checkable directly*, as a derivative test, on whatever surplus function you choose. Generically, without that condition, the matching map is discontinuous in trait space and isn't determined by the trait distributions alone. This is real, hard, and worth knowing before assuming the radar graph reduces to anything simple.

**Other concrete, stealable mechanisms:** blocking-pair count as the primary system-level observable (sweep every non-matched pair, count how many would rather have each other — a scalar description of the whole market's unresolved tension, belonging to no single agent); a Gale-Shapley reference matching computed offline every N episodes, with the rank-gap between it and the realized outcome as a non-saturating efficiency metric; Roth & Xing's four-stage market-health classifier (unraveling → deadline congestion → clearinghouse adoption → possible re-unraveling), a whole-system pathology invisible from any individual's reward; and Spence's job-market signaling model, which turns "what you were given at birth, what you acquired" into a mechanism with a known equilibrium — an unobservable innate ability, an observable costly signal, cheaper to produce for the able.

**Other honest failure modes:** the Diamond paradox — with homogeneous agents and any positive search cost, the unique equilibrium price is the full monopoly price, and prices can collapse to a degenerate point with no meaningful search left to model; markets can **unravel**, with transaction timing creeping earlier and earlier until the market no longer functions as designed (documented across dozens of real historical markets by Roth & Xing); the Boston school-choice mechanism teaches agents to misreport their true preferences, meaning the sim would measure strategy rather than preference; and — provably, not just empirically — Cen & Shah (AISTATS 2022) show that in a matching market where agents learn their preferences via bandit feedback, **stability and low regret for all agents cannot be simultaneously guaranteed**, because exploration itself necessarily creates blocking pairs. Adding costs and transfers (making the market transferable-utility) restores simultaneous achievability — a real, provable reason money and non-money markets are not the same math.

### 3.4 "How toddlers and kids learn continuously" → Developmental AI and Intrinsic Motivation

There is a real, twenty-year research program here, and its central technical claim is sharper than the metaphor: raw prediction-error curiosity is the *wrong* signal, because it is maximized by irreducible randomness — Burda et al.'s literal noisy-TV experiment showed a TV the agent could switch channels on permanently captured a curiosity-driven agent's attention, because the TV's content is genuinely unpredictable and that unpredictability never resolves. The field's fix, and the single most transferable idea in this whole research effort: reward the **derivative of competence** — learning progress — not its level (Oudeyer, Kaplan & Hafner, IAC, 2007). A purely stochastic goal yields near-zero learning progress and gets correctly deprioritized, because nothing is actually being learned, only observed.

**Why this matters specifically for a multi-agent market world:** in a population of learners, every *other* agent is a noisy TV, and a moving one. Knowledge-based intrinsic motivation (prediction error, novelty counts, Random Network Distillation) will be dominated by other agents' non-stationarity. Competence-based learning progress is the right family precisely because it measures change in *your own* success at a self-set goal, regardless of what caused the environment to shift.

**Concrete mechanisms:** Kanitscheider et al. (OpenAI, 2021) estimate learning progress as the gap between a fast and a slow exponential moving average of success rate, derive the optimal averaging window from an explicit bias-variance tradeoff, and — critically — make it **bidirectional**, so *drops* in competence also trigger renewed attention, which is exactly the signal you want when a rival displaces you in a market. IMGEP (Forestier, Mollard & Oudeyer, 2022) gives a *modular* goal-space architecture — one goal module per environment object, selected by a bandit weighted toward whichever module currently shows the most progress — which maps one-to-one onto "there is a market for everything": one module per market, with attention shifting to whichever market an agent is currently making progress in. Colas, Karch, Sigaud & Oudeyer's survey (JAIR 2022) formalizes goals as *weight vectors over objectives*, z_g = (β₁,...,β_N) — meaning your radar graph can be something an agent **selects**, not just a chart the experimenter draws after the fact: heterogeneity across agents becomes different β-distributions, "what you were given at birth" becomes a fixed β-prior, "what you acquired" becomes a learned β-policy.

**On the social side**, Ndousse et al. (ICML 2021) found something important and counterintuitive: model-free RL agents **do not spontaneously socially learn** — if a novice observes a useful expert state but receives no reward for it, the policy gradient is exactly zero, so the demonstration carries no signal at all, no matter how informative it looked. Social learning required two additions: a reward-independent auxiliary loss (predicting the next state, forcing implicit modeling of others) and a visible "prestige" cue — literally recoloring agents by their accumulated competence, so there's something to condition attention on. In a market world, wealth or reputation is that cue for free.

**Honest failure modes:** Taïga et al. (ICLR 2020) reassessed popular curiosity bonuses inside one common framework and found they don't reliably beat simple ε-greedy exploration, and gain nothing from 5× more data — intrinsic motivation is not a free win, and the honest posture is to pre-register a uniform-sampling baseline and commit to reporting it regardless of outcome. Learning progress itself is zero at *both* ends — competence saturated, or competence flat at zero — which means it structurally **abandons** hard-exploration problems rather than solving them, a failure mode directly relevant to Q6's own evasion collapse (if pursuit never produces any measurable progress at all, learning-progress-driven exploration will correctly, and unhelpfully, stop trying it). Melting Pot's own headline empirical finding is worth sitting with directly: **maximizing collective reward produces policies that are less robust to novel social partners than maximizing individual reward** — meaning the correct design is to *measure* at the system level while continuing to *optimize* at the individual level, never the reverse. And plasticity loss connects here directly: Dohare et al.'s dead-unit accumulation, declining effective rank, and monotonically growing weight norms are all driven by non-stationary inputs and bootstrapped target shift — exactly the conditions a self-play market world creates by construction — while the field's own honest finding is that general-purpose regularization (LayerNorm, weight decay) currently beats every domain-specific plasticity intervention tried so far.

### 3.5 "We see the whole system as one... a Q-table for the whole sim" → System-Level Observables

This is Rahul's least-precedented idea, and the literature that actually answers it is not complexity science — most of complexity science's glamorous measures (integrated information, Hoel-style causal emergence, assembly theory) are either uncomputable at any real scale or actively disputed as unusable. What *does* work comes from three more grounded places.

First, the sequential-social-dilemma line converged on a small, fixed vector of aggregate metrics that belong to no agent: **Utilitarian efficiency, Equality (1 − Gini), Sustainability (how early reward is collected), and Peace** (Perolat, Leibo et al., 2017) — reused almost verbatim by later DeepMind work and generalized into Melting Pot's evaluation protocol. This is the most directly buildable version of "a Q-table for the whole sim": four numbers, logged every episode, describing the population's regime, not any individual's score.

Second, done-correctly descriptive statistics from economics and ecology: the **Gini coefficient** and full Lorenz curve for inequality; **Hill numbers** (the mathematically correct generalization of "diversity," resolving the fact that Shannon entropy and the Gini-Simpson index are not even the same *kind* of quantity and shouldn't be compared to each other); **Shorrocks mobility**, measuring how much rank churns between wealth quintiles over time; and — only when you actually have the tail data for it — the **Clauset-Shalizi-Newman procedure** for correctly testing a power-law claim, which matters because their own re-analysis debunked 17 of 24 canonical "power law" claims across other fields when properly tested.

Third, from artificial life: **Bedau-Packard evolutionary activity statistics** and **ANNECS** (Accumulated Number of Novel Environments Created and Solved, Enhanced POET) — a monotone counter whose *slope*, not its value, is the meaningful anti-stagnation signal.

**The single most important design fact in this thread:** the two most sophisticated existing attempts reached *opposite* conclusions about whether a scalar welfare number is even meaningful. The AI Economist optimizes equality × productivity as its central scalar. Melting Pot 2.0 explicitly *refuses* total collective return as a primary metric, because it's identically zero by construction in zero-sum substrates. Whether a single welfare number means anything depends entirely on whether your world is constant-sum — and a market world with real competition for scarce resources will often be closer to zero-sum than either of those two systems assumed.

**Other concrete mechanisms worth having on the dashboard from day one:** System Neural Diversity (Bettini, Shankar & Prorok, 2025) — a proper, theoretically-characterized Gini-flavored scalar for how behaviorally different the population's policies actually are, computed via Wasserstein distance between action distributions rather than KL (which diverges as policies sharpen and isn't even a proper metric); a **paired neutral-shadow run** — an identical simulation with the learning signal deliberately severed (rewards shuffled, or policies resampled rather than gradient-updated) — reporting every headline metric as (real − shadow), which is the deep-RL analogue of the Santa Fe market's zero-information control bits; and a frozen-policy re-injection probe, straight from the Santa Fe team's own playbook — snapshot good agents, keep training the population, reinsert the frozen snapshot much later, and see whether it now performs *below* average. It did, in the original experiment, which was direct proof the system had genuinely moved rather than converged. That specific probe is cheap, requires no new theory, and is reusable on Q6's *existing* Hunter/Krishna setup today, before any of the rest of this is built.

**Honest failure modes, several of them sharp:** a naive Gini coefficient breaks the moment agents can hold debt, since the standard formula is only bounded by 1 for non-negative wealth; role-diversity entropy can be maximal while nothing is actually specializing, because entropy says nothing about whether individual role *assignments* persist over time (Project Sid hit this directly and had to add a persistence metric alongside it); and — the one that most directly threatens the "Q-table" framing itself — a macro-state built by binning aggregates together is only a valid Markov chain if the binning doesn't destroy information relevant to the future, and **nothing guarantees that is true**. It has to be tested (compare the squared one-step transition matrix against the empirical two-step one), never assumed.

### 3.6 "A big enough matrix... a smaller version on Mac" → Large-Scale MARL Engineering

The practical, decision-relevant finding here is blunt and well-quantified. **JAX is the wrong bet for the Mac-scale version, specifically.** JaxMARL's own published benchmark table shows JAX running a *single* environment instance is **15× slower** than the plain numpy original (5,480 vs. 83,400 steps/second on MPE Simple Spread) — the entire advertised 478×–32,600× speedup only appears at 10,000 parallel environments on a datacenter A100. And there is no working JAX GPU path on Apple Silicon: `jax-metal`'s last release was October 2024, it fails the JAX test suite, and JAX's own maintainers closed the Metal issues citing no active development.

The right foundation is **PufferLib** (Suarez, RLC 2025) — pure-C environments (their "Ocean" suite: 60+ reference environments, ~20k lines of C) hitting **over 1 million agent-steps per second on a single CPU core**, with no accelerator required at all. That number is the whole ballgame: the same code that's fast on a MacBook with no GPU is, unmodified, fast on a rented RTX 6000. Their `shared_pool.h` — a complete 8-agent commons dilemma in roughly 600 lines of C — is a ready-made template to fork.

**Melting Pot** (Leibo et al., DeepMind, 2021/2023) contributes the single best *structural* idea in this whole literature review: split every world into a **substrate** (pure physics — map, resources, movement, exchange rules, no opponent policy at all), a **background population** (a named, versioned, frozen set of co-players), and a **scenario** (the pairing of the two). Train on substrates, evaluate *only* against background populations the focal agents never trained with. This directly operationalizes "no defined winning or loss": you stop scoring an agent against a win condition and start scoring a population against its ability to generalize to social partners it has never met — and every frozen opponent-pool checkpoint Q6 already produces is, immediately, a candidate background-population member.

**Neural MMO** (Suarez et al., NeurIPS 2023) is the closest thing that already exists to the full vision — a procedurally generated map, ~128 agents, resource gathering, skills, and an environment-wide market — but it's also the sharpest cautionary tale: its own competition report states plainly that participants "rarely utilized market mechanics effectively or learned to buy/sell strategically," and the current maintained C port ships with `reward_market = 0.0` — the market's reward is turned off by default, in the actively maintained implementation, because agents mostly ignore it. That is a direct, published warning that a rich world's optional systems get ignored unless the reward structure makes them load-bearing, not a shortcut. Note too: the Python package itself is a dependency dead end (`numpy==1.23.3` hard-pinned, no push since August 2024) — read the paper for the predicate/task design, don't build on the repo.

**The reference cost anchor, so Mac-scale expectations are calibrated correctly:** Melting Pot's own baselines train for roughly one billion environment steps per run. At PufferLib's C-environment throughput (~150k steps/second achievable on a Mac, conservatively), that's under two hours. At pure-Python throughput (~5k steps/second), the same run is **fifty-five days**. That gap — a hundred-to-a-thousand-fold — is the entire reason the engineering choice here is existential rather than aesthetic.

**Other useful, concrete engineering findings:** parameter sharing across all N agents (one policy, N agents) silently makes every agent identical unless you concatenate an agent-identifying signal to the observation — proven to still recover per-agent-optimal distinct policies (Terry et al., 2020) — and your trait vector *is* that signal, for free; a declarative reward description language (MAgent, Zheng et al. 2018) — boolean event expressions that fire named reward rules — is a working precedent for structured, minute reward representations that ships today; naive Python multiprocessing caps out around 10,000 steps/second regardless of how fast the underlying environment is, purely from IPC overhead, so a 1M-sps C environment wrapped naively can lose two full orders of magnitude to plumbing alone.

### 3.7 "Very well thought off reward and punishment signals" → Reward Design Pathology

This thread is unusually decisive, and mostly against the plan as literally stated. **Skalse et al. (NeurIPS 2022)** prove, formally, that over the full space of stationary stochastic policies, any two reward functions that are jointly "unhackable" and non-trivial must be equivalent — meaning **hackability is the default property of a rich reward, not an engineering failure to be designed around.** There is, provably, no such thing as an elaborate, safe-to-optimize proxy reward.

Two results should directly reshape the plan. First, **potential-based reward shaping** (Ng, Harada & Russell, 1999) is the *only* shaping form proven to preserve the optimal policy — but Wiewiora (2003) proves it is mathematically equivalent to Q-value initialization, meaning it can only buy faster exploration, never richer expressiveness, and Devlin & Kudenko (2011) show that even this provably "safe" form still changes *which* Nash equilibrium a multi-agent system converges to. That is Q6's own risk-dominance-versus-payoff-dominance thread reappearing, unprompted, inside a paper about a completely different game: even a mathematically safe reward tweak is an equilibrium-selection intervention at population scale.

Second, and this is the one that should most directly reshape the reward design: **linear scalarization of a multi-objective reward can only ever recover the *convex* region of the Pareto front** (Vamplew et al. 2008; Roijers et al. 2013) — policies that live in a *concave* region are Pareto-optimal and genuinely desirable, and are provably unreachable by *any* weighting of the individual channels. A weighted sum over five trait-linked reward components doesn't just risk being badly tuned. It **forecloses a real region of possible agent behavior**, permanently, no matter how carefully the weights are searched.

**What actually replaces "elaborate hand-designed reward" in the literature, and what I'd recommend instead:** **homeostatic drive-reduction** (Keramati & Gutkin, *eLife* 2014, as a model of biological reward) — each agent carries an internal need-state vector (energy, safety, social contact, information), reward is the negative deviation from a setpoint, and successful market transactions replenish it. This is one mechanism, not five independent bonus terms, so it doesn't hit the linear-scalarization problem at all — reward isn't a *sum* over separable objectives, it's error against a need vector, a genuinely different mathematical object. If a true multi-objective vector reward is wanted anyway, **Chebyshev scalarization** against a reference point (never a weighted sum) at least recovers the full Pareto front rather than only its convex slice.

**Concrete, stealable diagnostic mechanisms:** **proxy/true reward pairs**, declared before any run — for every channel the agents actually optimize, define a separate welfare metric that is logged every episode and *never* enters any loss function, so reward hacking becomes visible as proxy-rising-while-true-metric-falling rather than invisible (Pan, Bhatia & Steinhardt, ICLR 2022 — who also found reward hacking arrives as a genuine **phase transition** in model capacity, meaning a clean Mac-scale run is not evidence about how the RTX 6000 version will behave); **difference rewards** (Wolpert & Tumer's COIN framework) — re-run the market clearing with one agent counterfactually removed, and credit that agent only with the difference in system-level outcome, which directly solves free-riding under any aggregate objective and is unusually cheap in a market specifically, because clearing is already a pure function of the participant list; a periodic **exploitability probe** — freeze the population, train one fresh adversary whose only job is to extract value from it, and report how much it wins by (Gleave et al., ICLR 2020, found a fresh adversary can beat a self-play-trained population using under 3% of the victim's original training compute); and a **universalization test**, straight from Melting Pot's scenario design — clone the current focal policy into every seat in the world and check what happens, a cheap, one-line diagnostic for the common failure where a policy looks excellent specifically *because* it's surrounded by agents unlike itself.

**Other empirically documented, honest failure modes:** **Calvano et al. (*American Economic Review*, 2020)** show independent Q-learners in a repeated pricing game reliably learn **supracompetitive collusion**, sustained by real punishment-and-forgiveness strategies — in exactly the repeated-market setting this project wants — and the correct response is to pre-register collusion as a *predicted* possible outcome, not something to be surprised by or to engineer away entirely; two-level co-adaptive designs (any planner, tax authority, or adaptive market-maker learning alongside the population) are provably unstable without a staged curriculum, because each level's policy update reshapes the other's entire reward landscape; and, most relevant to the "everyone plays their own equilibrium" framing — Devlin & Kudenko's finding above has a blunt practical consequence: **average outcomes across seeds are actively misleading** when a system has multiple stable equilibria, because you'd be averaging over runs that converged to structurally different worlds. Report the *distribution* over which equilibrium each seed reached, never the mean.

---

## 4. My Own Touch — What I'd Actually Build, and What I'd Watch For

Everything above is a map of the territory. This section is a position, and it's opinionated on purpose — Rahul asked what I think, not what a neutral summary would say.

### 4.1 The three problems that will actually decide whether this works

**The tautology trap.** If a trait axis is named "attractiveness" and the system then produces assortative matching on attractiveness, nothing was learned — the conclusion was built into the premise. This is the oldest and most-repeated failure in agent-based economics (Windrum, Fagiolo & Moneta's "the model can generate any result"), and Sugarscape's own discipline against it is precise: draw every trait from **uniform**, not shaped, priors, so that whatever skew or clustering emerges in the *outcome* cannot have been inherited from the *input*. Traits should be mechanical and low-level (a metabolism rate, a perception radius, a discount factor) — never named after the social outcome they're meant to explain.

**The Gode-Sunder problem — does agent learning even matter?** Zero-intelligence traders hit 97-99% allocative efficiency in a real double auction. If this world's market mechanism has that property, agent learning is decoration on top of a mechanism that was already doing all the work, and the whole project is unfalsifiable — it will look successful no matter what the agents actually do. The escape hatch is real and testable: Cliff & Bruten proved the effect is an artifact of *symmetric* supply-and-demand gradients. Build the endowment distributions deliberately asymmetric, and run a zero-intelligence baseline through the exact same pipeline as a permanent column on every headline number, from the very first run. If ZI-C already matches the learned agents, that's the finding, not a bug to iterate past.

**Is "a Q-table for the whole sim" even a coherent idea, and what does it concretely mean?** Yes, and here is the buildable version: log, every N steps, the Perolat-style Utilitarian/Equality/Sustainability/Peace vector, System Neural Diversity as the heterogeneity scalar, blocking-pair count as the matching-theory stability measure, and a MAP-Elites archive's coverage and QD-score over the trait axes — four numbers plus one archive, all belonging to the system rather than any agent. But it breaks in specific, documented ways that have to be respected: aggregate welfare goes identically flat in genuinely zero-sum settings; Gini blows past its bounds the moment agents can hold debt; and a "macro-state" built from binned aggregates is only a valid Markov chain if the binning preserves the information that actually predicts the future — which has to be *tested*, every time, never assumed.

### 4.2 What I'd actually build

Not the full vision on day one — a version specifically designed to fail fast and honestly if the core premise is wrong.

**Engine:** C, against PufferLib's `ocean/target` template, forking `shared_pool.h` as the starting skeleton. Not JAX — JaxMARL's own numbers show JAX is *slower* than plain numpy at the single-environment scale a laptop actually runs, and there is no working JAX GPU path on Apple Silicon. The payoff of C is that the exact same environment code is fast on the M5 Pro today and fast on the rented RTX 6000 later — only the trainer changes.

**World:** Sugarscape's three-rule core as the floor — resource growback, move-harvest-pay-metabolism, death-and-replace-with-a-fresh-random-trait-draw. That alone delivers "no defined winning or loss" almost for free: there's no terminal condition, only a viability constraint an agent can fail, at which point the population regenerates it with a new random draw rather than the simulation ending.

**Agents:** One shared-parameter policy, conditioned at every step on each agent's continuous 3-5 dimensional trait vector (Terry et al.'s agent-indication trick, proven to still recover per-agent-distinct optimal behavior despite full parameter sharing). Traits sampled from uniform priors, split explicitly into an endowed component (fixed for the agent's life) and an acquired component (updated by experience) — directly implementing "what you were given at birth, what you acquired" as two separately-observable channels, the way Sugarscape's inheritance and credit rules do.

**Markets:** One contract-matching engine (Hatfield-Milgrom's unification of college admissions, wage matching, and package auctions into a single model), governed by two structural switches rather than four hand-built market types: transferable-utility on/off, and substitutes/complements on/off. Money market and job market are TU-on with different contract terms; dating market is TU-off. Whatever the switch settings, exchange must be **atomic** — an offer/accept action that swaps both sides in a single transition — because the single most repeated empirical finding across two independent research threads is that generic drop/give actions never produce trade at all.

**Feedback:** homeostatic drive-reduction as the one designed mechanism — an internal need vector per agent (energy, safety, social contact), reward as deviation from setpoint, replenished by successful transactions — rather than five independently-tuned bonus terms, because linear scalarization of a channel vector *provably* forecloses whole regions of possible behavior, and this sidesteps that by not being a weighted sum in the first place.

**The falsifiable question for version zero, stated before any of it runs:** does replacing zero-intelligence traders with learned agents change the system-level vector (Utility/Equality/Sustainability/Peace, System Neural Diversity, blocking-pair count) by more than seed-to-seed noise, under deliberately asymmetric supply and demand? A numeric threshold, stated in advance — something like "learned agents' Equality metric differs from ZI-C's by more than 2σ across 10 seeds" — and a "no" is a complete, legitimate, Q6-style negative result, not a failed project.

### 4.3 The honest risk assessment

Axtell & Farmer's 2025 review — written by one of Sugarscape's own creators, looking back on thirty years of the field including his own contributions — states the scope of the actual unsolved problem without hedging: **"we lack an understanding of which rules of agent behavior are sufficient to produce realistic-looking multi-agent institutions."** That is not a gap waiting for the right architecture to close it. It is the field's standing open question, after three decades and a great deal of serious work. This project should not aim at a credible artificial economy. It should aim at one falsifiable sufficiency claim, at Mac scale — exactly the discipline Q6 already has, applied to a bigger world.

On sequencing against the rest of Q6, since a real opinion was asked for and not a hedge: **don't drop what's already running, and don't let it block this either.** `v8` is parked mid-run at episode 2015 of 6000. Two isolation runs for ablation-1b were recommended and never started. Neither needs active attention — only machine cycles, and both machines are currently idle. Resume `v8` and start the isolation runs in the background this week, and build the market-world's version zero as the active-attention project in parallel. They compete for compute, not for Rahul's time, and there's enough of the former to go around. The one thread genuinely worth carrying forward on purpose: the risk-dominance question becomes, at population scale, the real-world **market participation puzzle** — why households systematically under-hold equities despite the long-run premium — which is Krishna's evasion, scaled up, with actual economic literature already asking the same question about actual humans. And the plasticity-loss thread has a flagged-open, genuinely unclaimed question sitting inside it: nobody has checked whether capacity collapse behaves differently when the non-stationarity comes from *other learning agents changing* rather than from the task itself changing. That's a clean, cheap bridge from Q6's existing narrow scope directly into this one.

### 4.4 The first two weeks, and what would make me stop

Days 1-2: the pre-registration document — trait axes, the falsifiable question, the numeric success criterion — written before a line of environment code, exactly the existing `versions/` habit applied one level up. Days 2-4: fork `shared_pool.h`, implement the three-rule Sugarscape core plus one atomic offer/accept trade action. Days 4-6: implement zero-intelligence-uniform and zero-intelligence-constrained as first-class policies running through the identical pipeline, and reproduce Gode-Sunder's published efficiency numbers as a correctness check before trusting anything built on top of it. Days 6-8: the shared-parameter policy with trait conditioning, reusing the existing `v8` IPPO implementation behind a PettingZoo emulation wrapper. Days 8-10: the system-level dashboard — the U/E/S/P analog, System Neural Diversity, blocking-pair count, the trait archive — built with the same discipline as Q6's existing per-run dashboard. Days 10-14: run it, compare learned agents against both zero-intelligence baselines, and write up whatever actually happened.

**What would make me stop and redesign, rather than push through:** if the zero-intelligence-constrained baseline already saturates every system-level metric that matters, *and* the deliberately-asymmetric supply/demand configuration doesn't break that saturation. That would mean the Gode-Sunder problem is real and unescapable in this specific mechanism — a signal to fix the market design itself before spending another cycle on agent learning, not a reason to add more agents and hope the problem resolves on its own.

---

## Appendix: Full Reference List

**Open-endedness and quality-diversity**
- Hughes, Dennis, Parker-Holder et al. — *Open-Endedness is Essential for Artificial Superhuman Intelligence*, ICML 2024 (arXiv:2406.04268)
- Mouret & Clune — *Illuminating Search Spaces by Mapping Elites*, 2015 (arXiv:1504.04909)
- Soros & Stanley — *Identifying Necessary Conditions for Open-Ended Evolution*, ALIFE 14, 2014
- Brant & Stanley — *Minimal Criterion Coevolution*, GECCO 2017; *Benchmarking Open-Endedness in MCC*, GECCO 2019
- Wang, Lehman, Clune, Stanley — *POET*, 2019 (arXiv:1901.01753); Enhanced POET, ICML 2020
- Pugh, Soros, Szerlip, Stanley — *Confronting the Challenge of Quality Diversity*, GECCO 2015
- Vassiliades, Chatzilygeroudis, Mouret — *CVT-MAP-Elites*, IEEE TEVC 2017 (arXiv:1610.05729)
- Tjanaka, Chen, Fontaine, Nikolaidis — *Discount Model Search for QD in High-Dimensional Measure Spaces*, ICLR 2026 Oral
- Fontaine & Nikolaidis — *CMA-MAE*, GECCO 2023 / ACM TELO 2024
- Ingvarsson, Samvelyan, Lim et al. — *Mix-ME: Quality-Diversity for Multi-Agent Learning*, 2023 (arXiv:2311.01829)
- Bettini, Shankar, Prorok — *System Neural Diversity*, JMLR 26, 2025 (arXiv:2305.02128)
- Omidshafiei, Papadimitriou, Piliouras et al. — *α-Rank*, *Scientific Reports* 2019 (arXiv:1903.01373)
- Leibo et al. — *Melting Pot*, ICML 2021 (arXiv:2107.06857)
- Boldi, Ding, Spector — *Objectives Are All You Need*, NeurIPS 2023 ALOE workshop (arXiv:2311.02283)
- Ficici & Pollack — coevolutionary pathologies, ALIFE VI, 1998; Cartlidge & Bullock, 2004
- Baker et al. — *Emergent Tool Use From Multi-Agent Autocurricula*, ICLR 2020 (arXiv:1909.07528)
- Parker-Holder, Jiang et al. — *ACCEL*, ICML 2022 (arXiv:2203.01302)

**Agent-based computational economics**
- Epstein & Axtell — *Growing Artificial Societies*, MIT Press/Brookings, 1996
- Kehoe — *The Specification of Sugarscape*, arXiv:1505.06012
- Arthur, Holland, LeBaron, Palmer, Tayler — *Asset Pricing Under Endogenous Expectations in an Artificial Stock Market*, SFI, 1996/97
- Zheng, Trott, Srinivasa, Parkes, Socher — *The AI Economist*, *Science Advances* 8(18), 2022
- Johanson, Hughes, Timbers, Leibo — *Emergent Bartering Behaviour in MARL*, 2022 (arXiv:2205.06760)
- Kiyotaki & Wright — *On Money as a Medium of Exchange*, *JPE* 97(4), 1989
- Marimon, McGrattan, Sargent — money emergence with classifier-system agents, *JEDC* 14, 1990
- Windrum, Fagiolo, Moneta — *Empirical Validation of Agent-Based Models*, JASSS 10(2)8, 2007
- Axtell & Farmer — *Agent-Based Modeling in Economics and Finance: Past, Present, and Future*, *JEL* 63(1), 2025
- Epstein — *Inverse Generative Social Science*, JASSS 26(2)9, 2023
- Park, O'Brien, Cai et al. — *Generative Agents*, UIST 2023 (arXiv:2304.03442)

**Matching and market design**
- Gale & Shapley — *College Admissions and the Stability of Marriage*, 1962
- Roth — market design survey work, IJGT 2008
- Diamond, Mortensen, Pissarides — search-and-matching theory
- Adachi — search-friction convergence to stable matchings, 2003
- Gode & Sunder — *Allocative Efficiency of Markets with Zero-Intelligence Traders*, *JPE* 101(1), 1993
- Cliff & Bruten — ZI-C limits under asymmetric supply/demand, 1997
- Chiappori, McCann, Pass — multidimensional matching and index reducibility
- Shimer & Smith — assortative matching under search frictions, 2000
- Roth & Xing — market unraveling, *AER* 1994
- Spence — job market signaling, *QJE* 1973
- Cen & Shah — stability/regret tradeoff in learned matching markets, AISTATS 2022

**Developmental AI and intrinsic motivation**
- Oudeyer, Kaplan, Hafner — *Intrinsic Motivation Systems for Autonomous Mental Development (IAC)*, 2007
- Burda et al. — *Random Network Distillation*, and the noisy-TV problem, 2019 (arXiv:1810.12894)
- Kanitscheider, Huizinga, Farhi et al. (OpenAI) — bidirectional learning-progress estimation, 2021
- Forestier, Mollard, Portelas, Oudeyer — *IMGEP with Automatic Curriculum Learning*, JMLR 23, 2022
- Colas, Karch, Sigaud, Oudeyer — *Autotelic Agents*, JAIR 2022 (arXiv:2012.09830)
- Portelas, Colas, Hofmann, Oudeyer — *ALP-GMM*, CoRL 2019
- Ndousse, Eck, Levine, Jaques — social learning in MARL, ICML 2021
- Taïga, Fedus, Machado, Courville, Bellemare — reassessing exploration bonuses, ICLR 2020
- Dohare et al. — *Loss of Plasticity in Deep Continual Learning*, *Nature* 2024
- Nikishin et al. — *The Primacy Bias in Deep RL*, ICML 2022 (arXiv:2205.07802)
- Achille et al. — critical learning periods in deep networks

**System-level observables**
- Perolat, Leibo, Zambaldi, Beattie, Tuyls, Graepel — *Commons Game*, U/E/S/P metrics, NIPS 2017 (arXiv:1707.06600)
- Hughes, Leibo et al. — *Inequity Aversion Improves Cooperation*, NeurIPS 2018 (arXiv:1803.08884)
- Clauset, Shalizi, Newman — *Power-Law Distributions in Empirical Data*, correct fitting procedure
- Dakos et al. — early-warning indicators for critical transitions
- Jost — Hill numbers and the correct measurement of diversity
- Bedau, Snyder, Packard — evolutionary activity statistics

**Large-scale MARL engineering**
- Suarez — *PufferLib 2.0*, RLC 2025
- Suarez, Isola et al. — *Neural MMO 2.0*, NeurIPS 2023 Datasets & Benchmarks (arXiv:2311.03736)
- Agapiou, Vezhnevets et al. — *Melting Pot 2.0*, DeepMind tech report 2022/2023 (arXiv:2211.13746)
- Rutherford, Ellis et al. — *JaxMARL*, AAMAS 2024 (arXiv:2311.10090)
- Matthews, Beukman et al. — *Craftax*, ICML 2024 (arXiv:2402.16801)
- Zheng, Yang, Cai et al. — *MAgent*, AAAI 2018 (arXiv:1712.00600)
- Terry, Grammel, Son, Black et al. — parameter sharing / agent indication, arXiv:2005.13625
- Terry, Black, Grammel et al. — *PettingZoo*, NeurIPS 2021 (arXiv:2009.14471)

**Reward design pathology**
- Skalse, Farrugia-Roberts, Russell, Abate, Gleave — reward hacking is generic, NeurIPS 2022
- Ng, Harada, Russell — *Policy Invariance Under Reward Transformations*, 1999
- Wiewiora — PBRS equivalence to Q-value initialization, 2003
- Devlin & Kudenko — multi-agent PBRS and equilibrium selection, 2011/2012
- Vamplew, Dazeley, Berry, Issabekov, Dekker — limits of linear scalarization, 2008
- Roijers, Vamplew, Whiteson, Dazeley — multi-objective decision-theoretic planning survey, 2013
- Pan, Bhatia, Steinhardt — *The Effects of Reward Misspecification*, ICLR 2022
- Calvano, Calzolari, Denicolò, Pastorello — algorithmic collusion, *AER* 2020
- Amodei, Olah, Steinhardt, Christiano, Schulman, Mané — *Concrete Problems in AI Safety*, 2016
- Manheim & Garrabrant — Goodhart's Law taxonomy
- Gleave, Dennis, Wild, Kant, Levine, Russell — *Adversarial Policies*, ICLR 2020
- Keramati & Gutkin — homeostatic reinforcement learning, *eLife* 2014
- Lehman et al. — *The Surprising Creativity of Digital Evolution*, *Artificial Life* 26(2), 2020

---

*This document was researched and drafted by Claude on 2026-08-01, at Rahul's direction, following extensive parallel literature research across the seven domains above. It is a starting position for the market-world thread, not a commitment — the honest risk assessment in §4.3 and the stop condition in §4.4 are meant to be taken as seriously as the enthusiasm everywhere else in this document.*
