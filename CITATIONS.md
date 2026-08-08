# Citations & Working Bibliography

Every URL used in this project's research, with one line each: why it's here and the key
learning taken from it. Two parts: the **August 2026 field surveys** (five parallel research
sweeps on what's new), then the **foundational references** that shaped the project's design.
Maintained by hand; add to it whenever research informs a design decision.

---

## Part 1 — Field state as of August 2026

### Open-endedness, quality-diversity, autocurricula

- **Darwin Gödel Machine — Zhang, Hu, Lu, Lange, Clune (ICLR 2026)** — https://arxiv.org/abs/2505.22954 — Where open-endedness energy actually went: self-improving coding agents sampled from a POET-flavored archive of diverse variants; the most-cited living open-endedness result right now.
- **Red Queen Gödel Machine (Jun 2026)** — https://arxiv.org/abs/2606.26294 — Co-evolves the *evaluator* alongside the agent so benchmarks can't go stale; directly relevant to any automated "is this world still interesting" check we might want.
- **Digital Red Queen — Sakana AI + MIT (Jan 2026)** — https://arxiv.org/abs/2601.03335 — LLM self-play arms race in Core War; the closest modern successor to hide-and-seek-style autocurricula, reframed around LLM-authored code.
- **Picbreeder with VLM curators (May 2026, GECCO best-paper nominee)** — https://arxiv.org/abs/2605.23908 — Honest negative: VLM judges notice novelty as readily as humans but get trapped in local basins, missing the big conceptual leaps — a caution against LLM-as-curator shortcuts.
- **PACE (ICML 2026)** — https://arxiv.org/abs/2605.01358 — UED signal from actual induced policy-parameter change instead of noisy regret proxies; the current best descendant of the ACCEL/PLR lineage.
- **VLM-judged multi-agent UED (Jul 2026)** — https://arxiv.org/abs/2607.08193 — Marker of the "LLM-as-judge enters environment design" trend; a video-LLM watches rollouts to score curriculum difficulty.
- **DéjàQ — Rocktäschel et al. (Jan 2026)** — https://arxiv.org/abs/2601.01931 — MAP-Elites archive of synthetic math problems co-evolved with the model's training — the clearest QD→LLM-training crossover.
- **SQUAD (ICLR 2026)** — https://openreview.net/forum?id=LCKixyFShR — QD without a discretized behavior archive; relevant if our trait/channel descriptors ever outgrow a grid.

### LLM-agent societies & computational economics

- **AgentSociety 2 — Tsinghua FIB Lab** — https://arxiv.org/abs/2502.08691 · https://github.com/tsinghua-fib-lab/agentsociety/ — The scale reference for LLM societies (10k+ agents), matured into a platform; the thing we deliberately are *not* building, and the contrast that defines our niche.
- **Diagon — "When Agent Markets Arrive" (Apr 2026)** — https://arxiv.org/abs/2604.06688 · https://github.com/assassin808/diagon — Agent labor-market testbed; key finding: naive "more transparency/competition" interventions can *degrade* market performance — institutional design details matter more than intuition says.
- **Institutional AI: governance graphs vs LLM collusion (Jan 2026)** — https://arxiv.org/abs/2601.11369 — First mechanism-design-level (not agent-level) anti-collusion intervention for LLM oligopolies; institution-level alignment as a real technique.
- **Moltbook Observatory Archive (May 2026)** — https://arxiv.org/abs/2605.13860 — 2.73M interactions among ~90,700 autonomous agents on an agent-only social network — first field-scale *observational* data on spontaneous LLM-population dynamics.
- **Moltbook emergent-coordination benchmark (Mar 2026)** — https://arxiv.org/abs/2603.03555 — Companion benchmarking paper to the archive above.
- **EconSimulacra (Jun 2026)** — https://arxiv.org/html/2606.26883v2 — Digital-twin platform coupling consumer economy, mobility, and social networks via shared agent states.
- **TRAILS — "Stop Drawing Scientific Claims from LLM Social Simulations Without Robustness Audits" (May 2026)** — https://arxiv.org/abs/2605.18890 — The paper that validates this project's entire methodological bet: small implementation perturbations swing LLM-sim macro outcomes by up to 76 percentage points; the field's formal complaint that nobody verifies what we verify by hand at every increment.
- **Agentopia (Jun 2026)** — https://arxiv.org/abs/2606.07513 — Project-Sid-lineage persistent learning agent societies.
- **LASS Workshop @ CIKM 2026** — https://lassworkshop26.github.io/ — New academic venue for LLM social simulation with an economics/policy track; a possible publication target.

### MARL engineering & environments

- **JaxMARL** — https://github.com/FLAIROx/JaxMARL — The most active hub in the MARL-environments ecosystem (commits through Aug 2026); also the source of the decisive benchmark table showing JAX@1-env is ~15x *slower* than numpy — why we build in plain Python/C at laptop scale.
- **PufferLib** — https://github.com/PufferAI/PufferLib — The C-environments-at-1M-steps/s framework; 4.0 rewrite in progress as of Aug 2026, Apple Silicon support still an open PR — the designated scale-up path when we outgrow the Mac.
- **jax-metal status discussion** — https://github.com/jax-ml/jax/discussions/34648 — Confirms JAX-on-Apple-Silicon is effectively dead (maintainers closed Metal issues, community replacement only ~3x over CPU); closed the case against a JAX-based design for local iteration.
- **Melting Pot CHANGELOG** — https://github.com/google-deepmind/meltingpot/blob/main/CHANGELOG.md — Maintenance-mode evidence; we take its substrate/scenario/background-population *design* (see Part 2), not its codebase.
- **TABX (ICML 2026)** — https://arxiv.org/abs/2602.01665 — GPU-parallel battle sim built on JaxMARL interfaces; scale reference for what vectorized MARL looks like in 2026.
- **Craftax-MA — Foerster lab (Nov 2025)** — https://arxiv.org/abs/2511.04904 — 250M interactions/hour heterogeneous-agent benchmark with trading; the "hyperscale" contrast to our small-and-interpretable bet.
- **JaxMARL-HFT (Nov 2025)** — https://arxiv.org/abs/2511.02136 — Same stack repurposed for high-frequency-trading MARL; evidence the JAX ecosystem is spreading beyond games.
- **XLand-MiniGrid** — https://github.com/dunnolab/xland-minigrid — Procedurally-generated *rule trees* (compositional task generation); stalled since Dec 2025 but the rule-generation idea is worth remembering.

### Developmental AI, plasticity, continual learning

- **Plasticity-loss survey, v3 (Apr 2026)** — https://arxiv.org/abs/2411.04832 — 50+ mitigation methods in one taxonomy; headline: general-purpose regularization beats domain-specific fixes — the current reference map for the Dohare lineage Q6's continual-learning scope is built on.
- **Spectral Collapse Drives Loss of Plasticity (ICML 2026)** — https://arxiv.org/pdf/2509.22335 — New mechanistic story: plasticity loss as spectral collapse of representations, not just dead units — the post-Dohare causal account.
- **Plasticine benchmark (Feb 2026 rev.)** — https://arxiv.org/abs/2504.17490 — Standardized plasticity evaluation; fixes the inconsistent-benchmarking complaint.
- **Survey of Continual RL (Jun 2026)** — https://arxiv.org/pdf/2506.21872 — Fresh broad synthesis; companion venue: CATS workshop @ ICML 2026 — https://cats-icml.github.io/
- **MEAL — first Continual-MARL benchmark (ICML 2026)** — https://arxiv.org/abs/2506.14990 — Naive CL+MARL combinations fail on coordination-heavy tasks; the direct citable bridge for Q6's open question about plasticity under *multi-agent* non-stationarity.
- **MAGELLAN (ICML 2025)** — https://arxiv.org/abs/2502.07709 — Oudeyer-lineage learning-progress intrinsic motivation ported to LLM agents in open-ended goal spaces.
- **HERAKLES (Aug 2025)** — https://arxiv.org/pdf/2508.14751 — MAGELLAN follow-up adding hierarchical skill compilation.
- **In-context curiosity impossibility (Jun 2026)** — https://arxiv.org/abs/2606.19476 — Proof that unbiased learning-progress estimation from in-context prediction error is impossible in general MDPs — an important negative result before anyone builds LP-driven curiosity here.
- **Information-theoretic open-endedness — Van Roy group (Jun 2026)** — https://arxiv.org/abs/2606.08369 — First rigorous "bit-equivalent" formal definition of open-endedness; mathematical teeth for a previously informal concept.
- **Evolved developmental reward schedules (Jun 2026)** — https://arxiv.org/abs/2606.20858 — Rare direct developmental-psych→RL design paper: evolving time-varying agency/novelty weightings.
- **OMAR (Feb 2026)** — https://arxiv.org/html/2602.03109 — Where the social-learning energy went: LLM multi-agent social self-play, rather than the original Ndousse imitation-auxiliary-loss framing.

### Matching markets, algorithmic collusion, reward hacking

- **On the Fragility of AI Agent Collusion (Mar 2026)** — https://arxiv.org/html/2603.20281 — LLM pricing collusion collapses under agent heterogeneity — an important corrective to collusion alarmism.
- **Fish, Gonczarowski & Shorrer — LLM pricing collusion (2024, rev. Mar 2026)** — https://arxiv.org/abs/2404.00806 — The origin result the 2026 collusion literature extends: symmetric LLM agents reliably converge to supracompetitive prices.
- **Prompt Optimization Enables Stable Collusion (Apr 2026)** — https://arxiv.org/html/2604.17774v1 — Meta-learned prompt guidance *restores* the collusion that heterogeneity breaks — deployment practice, not model capability, drives the risk.
- **Breaking the Secret: economic anti-collusion interventions (Apr 2026)** — https://arxiv.org/pdf/2604.23511 — Complementary economic-mechanism countermeasures for embodied multi-agent collusion.
- **Do Matching Mechanisms Work with LLM Agents? (Jun 2026)** — https://arxiv.org/abs/2606.03030 — Yes: deferred-acceptance-style mechanisms substantially beat free-form LLM negotiation on stability/efficiency — a vote of confidence for building real market structure rather than letting agents freelance.
- **Learn2Match (Jun 2026)** — https://arxiv.org/abs/2606.06744 — Two-sided matching as a partially-observable Markov game with a decentralized MARL benchmark — fills the gap between bandit-feedback matching theory and real RL interaction.
- **Matching Markets meet Cumulative Prospect Theory (Jun 2026)** — https://arxiv.org/abs/2606.19883 — Behaviorally-realistic (CPT) preferences in competitive bandit matching, with adversarial-robustness guarantees.
- **MAC-Bench (Jun 2026)** — https://arxiv.org/abs/2606.07805 — Dynamic benchmark for Machiavellian rule-gaming in *multi-agent* deployments — the reward-hacking benchmark to point at our world once a stronger learner exists.
- *Regulatory context (no single URL; from news reporting): a March 11, 2026 flash crash (23 autonomous trading agents, ~$500M in 47 seconds) drew SEC scrutiny; California's algorithmic-pricing collusion law effective Jan 1, 2026; the federal Preventing Algorithmic Collusion Act (S.232) pending; EU AI Act high-risk obligations deadline Aug 2026.*

---

## Part 2 — Foundational references that shaped the design

### Falsifiability & controls (the project's spine)

- **Gode & Sunder 1993 — Allocative Efficiency of Markets with Zero-Intelligence Traders (JPE)** — https://doi.org/10.1086/261868 — The single most load-bearing citation in the project: random-but-budget-constrained traders hit 97–99.9% allocative efficiency, so market *structure* can do all the work — why ZI/ZI-C baselines run permanently through our identical pipeline, and why a world where they match the learner is unfalsifiable.
- **Windrum, Fagiolo & Moneta 2007 — Empirical Validation of Agent-Based Models (JASSS)** — https://www.jasss.org/10/2/8.html — Names the field's two diseases: over-parameterization ("the model can generate any result") and equifinality (many rule-sets, same output) — the reason our pre-registration and rule-modularity discipline exists.
- **Axtell & Farmer 2025 — ABM in Economics and Finance (JEL)** — https://ora.ox.ac.uk/objects/uuid:8af3b96e-a088-4e29-ba1e-0760222277b7 — Thirty-year retrospective by Sugarscape's co-creator: "we lack an understanding of which rules of agent behavior are sufficient to produce realistic-looking multi-agent institutions" — the field's standing open question, and this project's actual target.
- **Arthur, Holland, LeBaron, Palmer, Tayler — Santa Fe Artificial Stock Market (1996/97)** — https://sites.santafe.edu/~wbarthur/Papers/Arthur-HollandStockMarket.pdf — The methodological gold standard: hardwired zero-information *control bits* as a within-model placebo, mechanism-kill ablations, and a whole-system phase transition governed by one exploration-rate knob.

### Artificial societies & markets

- **Epstein & Axtell 1996 — Growing Artificial Societies (Sugarscape)** — spec: https://arxiv.org/abs/1505.06012 (Kehoe's formal Z-specification) — The existence proof for the whole genre: a handful of modular rules and uniform trait priors produce skewed wealth distributions with no win condition; also the warning that under-specified update semantics silently change results.
- **Zheng et al. 2022 — The AI Economist (Science Advances)** — https://www.science.org/doi/10.1126/sciadv.abk2607 — Two-level RL (agents + tax planner) with the documented curriculum needed to keep co-adaptive learning stable; the modern deep-RL sibling of our institution design.
- **Johanson, Hughes, Timbers & Leibo 2022 — Emergent Bartering (DeepMind)** — https://arxiv.org/abs/2205.06760 — The most load-bearing empirical detail we adopted: trade *never emerges* from generic drop/give actions; exchange must be an atomic offer/accept swap or credit assignment can't reach it.
- **Kiyotaki & Wright 1989 — On Money as a Medium of Exchange (JPE)** — https://econpapers.repec.org/RePEc:ucp:jpolec:v:97:y:1989:i:4:p:927-54 — Money as a self-fulfilling equilibrium (fundamental vs speculative), and the warning that a learning system landing on one equilibrium is a coordination accident, not a discovery.
- **Park et al. 2023 — Generative Agents (UIST)** — https://arxiv.org/abs/2304.03442 — The reference architecture for LLM societies *and* the reason we don't use one: RLHF-tuned agents are documented as excessively cooperative — wrong material for markets that need defection and holdout.
- **Vezhnevets et al. 2023 — Concordia (DeepMind)** — https://arxiv.org/abs/2312.03664 — The Game-Master pattern for adjudicating open-ended agent actions; the clean world/agent separation worth remembering if action spaces ever get rich.
- **Tesfatsion — Agent-Based Computational Economics portal** — https://faculty.sites.iastate.edu/tesfatsi/archive/tesfatsi/ace.htm — The "modeler as pure observer" principle and the phase-portrait framing of what a system-level result even is.
- **Epstein 2023 — Inverse Generative Social Science (JASSS)** — https://www.jasss.org/26/2/9.html — The inverted workflow (macro target as objective, micro-rules as search space) and its honest cost: multiple distinct generators hit the same target.

### MARL evaluation & system-level metrics

- **Leibo et al. 2021 — Melting Pot (ICML)** — https://arxiv.org/abs/2107.06857 · 2.0: https://arxiv.org/abs/2211.13746 — The substrate / background-population / scenario split: never grade a multi-agent system on its own training reward; grade a population against held-out social partners.
- **Perolat et al. 2017 — Commons Game (NIPS)** — https://arxiv.org/abs/1707.06600 — The Utilitarian/Equality/Sustainability/Peace metric vector — the most directly buildable "measure the whole system, not the agent" precedent; also the warning that learning can make collective outcomes *worse* than random, invisibly, if you only watch per-agent reward.
- **Leibo et al. 2017 — Sequential Social Dilemmas (AAMAS)** — https://arxiv.org/abs/1702.03037 — Cooperation and defection as properties of the environment's incentive geometry, not of the agents — Q6's own thesis, stated at N agents.
- **Hughes et al. 2018 — Inequity Aversion in Intertemporal Social Dilemmas (NeurIPS)** — https://arxiv.org/abs/1803.08884 — Distribution-relative reward as a temporal-credit-assignment device, not a moral preference.
- **Bettini, Shankar & Prorok 2025 — System Neural Diversity (JMLR)** — https://arxiv.org/abs/2305.02128 — A theoretically-characterized scalar for population behavioral heterogeneity (Wasserstein-based, Gini-inspired) with per-agent decomposition.
- **Omidshafiei et al. 2019 — α-Rank (Scientific Reports)** — https://arxiv.org/abs/1903.01373 — Population-level evaluation via the stationary distribution of an evolutionary Markov chain — the closest existing thing to "a Q-table for the whole sim."
- **Gorsane et al. 2022 — Standardised MARL Evaluation (NeurIPS)** — https://arxiv.org/abs/2209.10485 — The reporting standard we pre-register against: 10 seeds, bootstrap CIs, IQM; a third of published MARL papers have no uncertainty quantification at all.

### Open-endedness & quality-diversity foundations

- **Hughes et al. 2024 — Open-Endedness is Essential for ASI (ICML)** — https://arxiv.org/abs/2406.04268 — The formal definition we use: open-ended = novel (observer's loss on the future keeps rising) *and* learnable (loss falls with history), observer-relative.
- **Mouret & Clune 2015 — MAP-Elites** — https://arxiv.org/abs/1504.04909 — The archive-over-descriptor-space machinery; by its authors' own statement not itself open-ended (fixed axes) — a measurement instrument, not an engine.
- **Wang et al. 2019/2020 — POET / Enhanced POET** — https://arxiv.org/abs/1901.01753 · http://proceedings.mlr.press/v119/wang20l/wang20l.pdf — The bounded-window minimal criterion (admit a niche only if learnable-but-unsolved) and PATA-EC (characterize a niche by the rank-ordering it induces over the population).
- **Pugh, Soros, Szerlip & Stanley 2015/16 — Quality Diversity frontier** — https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2016.00040/full — The empirical answer to "what if you pick descriptor axes badly": unaligned descriptors can be *worse* than plain fitness search.
- **Vassiliades et al. 2017 — CVT-MAP-Elites (IEEE TEVC)** — https://arxiv.org/abs/1610.05729 — Archive size decoupled from descriptor dimensionality.
- **Fontaine & Nikolaidis 2023 — CMA-MAE** — https://arxiv.org/abs/2205.10752 — The explicit dial between optimization and illumination.
- **Ingvarsson et al. 2023 — Mix-ME** — https://arxiv.org/pdf/2311.01829 — Archive cells holding whole *teams* — the system-level QD construction.
- **Boldi, Ding & Spector 2023 — Objectives Are All You Need** — https://arxiv.org/abs/2311.02283 — Lexicase selection over many fine-grained signals beats MAP-Elites on deceptive domains *without* optimizing diversity — both a warning about QD-score evidence and the best existing match for many-minute-reward-signals selection.
- **Parker-Holder et al. 2022 — ACCEL (ICML)** — https://arxiv.org/pdf/2203.01302 — Regret-based curricula at single-GPU compute (~3B steps vs POET's ~500B) via positive value loss — the compute-realistic open-endedness path.
- **Baker et al. 2020 — Emergent Tool Use / Hide-and-Seek (ICLR)** — https://arxiv.org/abs/1909.07528 — Six emergent phases then stop; two of the six were physics exploits — autocurricula find your simulator's bugs before your intended strategies.

### Learning, intrinsic motivation & plasticity

- **Colas, Karch, Sigaud & Oudeyer 2022 — Autotelic Agents survey (JAIR)** — https://arxiv.org/abs/2012.09830 — Goals as weight-vectors over objectives (the radar chart as a *goal space*), and the complete taxonomy of drivers-without-a-win-condition.
- **Forestier et al. 2022 — IMGEP (JMLR)** — https://www.jmlr.org/papers/volume23/21-0808/21-0808.pdf — Modular goal spaces with a learning-progress bandit — one module per market/domain, attention flowing to wherever progress is happening.
- **Portelas et al. 2019 — ALP-GMM (CoRL)** — http://proceedings.mlr.press/v100/portelas20a/portelas20a.pdf — The most implementable learning-progress teacher; *absolute* LP (fires on competence drops too) is the key detail for competitive worlds.
- **Burda et al. 2019 — RND and the noisy-TV problem** — https://arxiv.org/abs/1810.12894 — Why prediction-error curiosity fails (irreducible noise is maximally "interesting"), and thus why competence-progress signals are the right family in multi-agent worlds where every other learner is a noisy TV.
- **Nikishin et al. 2022 — The Primacy Bias in Deep RL (ICML)** — https://arxiv.org/abs/2205.07802 — Early-experience overfitting fixable by periodic resets; half of Q6's continual-learning scope.
- **Dohare et al. 2024 — Loss of Plasticity in Deep Continual Learning (Nature)** — *(cite by name)* — The other half: standard backprop loses the ability to keep learning under non-stationarity — exactly the condition a self-play market world creates.
- **Ndousse et al. 2021 — Social Learning in MARL (ICML)** — *(cite by name)* — Model-free RL agents do *not* spontaneously learn from observing experts (zero policy gradient from unrewarded observation); social learning needs an auxiliary prediction loss plus visible prestige cues.

### Matching theory & reward design

- **Terry et al. 2020 — Parameter Sharing / Agent Indication** — https://arxiv.org/abs/2005.13625 — One shared policy + a per-agent trait vector in the observation provably recovers distinct per-agent optimal policies — the mechanism that makes a continuous trait radar-chart trainable at all.
- **PettingZoo — Terry et al. 2021 (NeurIPS)** — https://arxiv.org/abs/2009.14471 — The multi-agent API standard worth exposing regardless of internal architecture.
- **Suarez 2025 — PufferLib 2.0 (RLC)** — https://rlj.cs.umass.edu/2025/papers/Paper151.html — 1M+ steps/s pure-C environments; the engineering existence proof that laptop-scale and GPU-scale can share one world implementation.
- **Suarez et al. 2023 — Neural MMO 2.0 (NeurIPS D&B)** — https://arxiv.org/abs/2311.03736 — The predicate/task reward system worth stealing, and the cautionary tale: its own maintained config ships with the market reward disabled because agents ignore rich systems that aren't load-bearing.
- **Cen & Shah 2022 — Regret, stability & fairness in matching markets (AISTATS)** — *(cite by name)* — Provable: bandit learners in matching markets cannot have stability and low regret simultaneously — unless transfers (money) exist; the sharpest reason "dating market" and "job market" are genuinely different math.
- **Chiappori, McCann & Pass — multidimensional matching** — *(cite by name)* — A trait *vector* collapses to a scalar "market value" only under a restrictive, testable index-separability condition; generically the matching map is discontinuous in trait space.
- **Roth & Xing 1994 — Market unraveling (AER)** — *(cite by name)* — Markets can destroy themselves by transaction-timing creep; a whole-system pathology invisible from any agent's reward.
- **Skalse et al. 2022 — Defining and Characterizing Reward Hacking (NeurIPS)** — *(cite by name)* — Provably, no non-trivial reward pair is unhackable: hackability is the default property of a rich reward, not an engineering failure.
- **Ng, Harada & Russell 1999 — Potential-based shaping** — *(cite by name)* — The only shaping form that provably preserves optimal policies; Devlin & Kudenko 2011 showed even it still changes *which equilibrium* a multi-agent system selects.
- **Vamplew et al. 2008 / Roijers et al. 2013 — Multi-objective RL limits** — *(cite by name)* — Linear scalarization can only reach the convex part of the Pareto front: a weighted sum over reward channels permanently forecloses real behaviors no weight tuning recovers.
- **Calvano et al. 2020 — Algorithmic collusion (AER)** — *(cite by name)* — Independent Q-learners in repeated pricing games reliably learn supracompetitive collusion with punishment strategies — pre-register collusion as a *predicted* outcome in any repeated-market world.
- **Keramati & Gutkin 2014 — Homeostatic RL (eLife)** — *(cite by name)* — Reward as deviation-from-setpoint on an internal need vector — the design we adopted to avoid hand-tuned bonus-term stacks entirely.

### Q6-lineage references (from the original Hunter/Kṛṣṇa thread)

- **Harsanyi & Selten 1988 — risk dominance vs payoff dominance** — *(cite by name)* — The Stag Hunt frame under all of Q6: the safe incomplete strategy beating the risky correct one is rational equilibrium selection, not a bug.
- **La Malfa et al. — "The Attacker in the Mirror"** — https://arxiv.org/abs/2605.08427 — Self-play Nash equilibria that are trivially-safe and useless, with the same game value as competence — the sharpest modern mirror of Kṛṣṇa's evasion.
- **Self-play survey (2024)** — https://arxiv.org/abs/2408.01072 — Uses "non-stationarity" throughout and never says "continual learning" — evidence FSP pools already are continual-learning machinery under another vocabulary.
- **Kaplanis, Shanahan & Clopath 2019 (ICML)** — https://arxiv.org/abs/1902.00255 — The one paper explicitly framing competitive self-play as continual learning.
- **de Witt et al. — IPPO** — https://arxiv.org/abs/2011.09533 · **Yu et al. — MAPPO** — https://arxiv.org/abs/2103.01955 — Why independent PPO (not centralized critics) matches zero-sum structure; the algorithmic basis of the v8 comparison.
- **Bansal et al. 2017 — Emergent Complexity via Multi-Agent Competition** — https://arxiv.org/abs/1710.03748 — Precedent for competition alone generating curricula.
- **Balduzzi et al. 2019 — PSRO rectified Nash** — https://arxiv.org/abs/1901.08106 — The basis of Q6's rectified opponent sampling (ablations 1/1b).
- **Lauffer et al. — Rational Policy Gradient** — https://arxiv.org/abs/2511.09535 · **risk-preference self-play** — https://arxiv.org/abs/2305.11476 · **equilibrium bias** — https://arxiv.org/abs/2405.02724 · **opponent-aware basin entry** — https://arxiv.org/abs/2605.18078 — The 2023–2026 cluster actively working equilibrium selection in self-play; the citation graph Q6's flagship result plugs into.
