# Q6 research review — 5 September 2026

Q6 contains worthwhile applied research and engineering. Its strongest current assets are an inspectable experimental history, a working RL stack, replay/monitoring tools, and a small synthetic decision laboratory. Its broad scientific explanations are less reliable than its code and recorded observations. The next step should make one experiment trustworthy and finishable.

My recommendation: release a corrected, bounded Q6 research lab, then pursue **adaptation across changing small worlds** as the next research question. Make efficient simulation serve that experiment. Introduce nested learning only after ordinary memory and learning baselines establish a measurable limitation.

This review does not change training or simulation source. It preserves findings and executable diagnostic artifacts. Three independent sub-agents audited RL, No Way Home, and engineering/reproducibility; the coordinating review traced GitHub history, compared branches, checked primary literature, and tested the PPO agent.

## Scope and evidence

- Inspected all six GitHub branch tips: `main`, `conitnous`, `v7`, `v7-ablations`, `v8`, `no-way-home`. Their retrieved histories contain 53 unique commits, from October 2025 through August 2026.
- The 157 published files in `no-way-home` at `379bd252792736956d6525f3f24dd2976e830d44` match this checkout byte for byte. The local `.git` points to an absent `/Users/rahul/Q6` worktree parent, so GitHub supplied history. This is a local checkout limitation, not an algorithm defect.
- Inspected separate source archives for `v7-ablations` at `1811d88a955252b44b2d698d9f16cb8fd572e77a` and `v8` at `d5ba3dd68ed8c4990d6bff02cb495bb34429038b`.
- The dashboard preserves 40 run summaries representing 71,272 recorded episode entries, including interrupted and smoke runs. These are not 40 independent scientific replications. All referenced raw run directories are absent here; no historical CSV, JSONL, or PTH files were found in the checkout. Historical headline numbers below are stored summaries, not newly rerun training results.
- Validation: **all 254 current-checkout tests passed across two executions**: 253 completed before the unfinished slow training smoke was interrupted; that one remaining test passed separately with one Torch CPU thread. The separate v8 PPO-agent suite passed **11/11**. An independent No Way Home subset also passed **32/32**. These are software checks, not proof of the scientific claims. Logs are preserved in this folder and the exact method is recorded in `engineering-audit.md`.
- Additional probes reproduced invalid CHER transitions, hidden pellets, hidden invulnerability, unreachable objectives, the beta curve, paired analysis, dropped learner terminal updates, and utility edge cases. Exploratory controls are explicitly distinguished from confirmation.

Detailed evidence: [RL audit](rl-audit.md), [No Way Home audit](nwh-audit.md), [engineering audit](engineering-audit.md). [GitHub progression](history.json) records the retrieved commit titles and revisions. Reproduction commands and data are listed at the end.

## What you built and explored

| Period | Evidence of work | What it represents |
|---|---|---|
| Oct 2025–Jan 2026 | Original “Sui Generis” launcher; archived DQN versions; curriculum, reward/epsilon fixes, logs and success scoring | Building a learning system and discovering how easily its feedback can mislead |
| May–early June 2026 | Cleanup/config/tests; CNN + Double DQN + dueling network; scripted-to-learning Hunter transition; historical opponent pool | Substantive applied RL implementation and self-play infrastructure |
| June 2026 | Two-head gated network, synthetic CHER, persistent easy opponents, reward redesign and warmup | Several hypotheses about failure to collect under adversarial pressure; outcomes strongly confounded |
| Late July–early August | Independent PPO branch, rectified opponent sampling, frozen-policy-anchor code, floor/warmup ablations, orchestrator and replay dashboards | Better instrumentation and attempts to isolate explanations; anchor exists as code but the visible main ablation runs did not enable it |
| Aug 7–9 | No Way Home resource model, message lineage, noisy elections, commitment ablation, tabular learner, pilot/power/pre-registration/beta sweep | A separate small decision-and-information laboratory, with much cheaper experiments |

The contribution is the integrated experimental program: constructing environments, running and observing failures, building tools to inspect them, and retaining negative results. These components mostly build on established algorithms. That is compatible with a valuable open-source artifact; it does not require claiming a new learning algorithm.

Several documents explicitly acknowledge AI assistance. Commit records establish project progression, not which individual wrote each line. The review credits the work visible in the project rather than attempting unverifiable line-level attribution.

## What survives validation, and what needs revision

**1. There is real learning infrastructure.** The base Double DQN target, dueling combination, gradient update, and compact replay representation are substantive implementations. The PPO branch implements GAE and clipped-surrogate updates, with hand-computed advantage tests. Historical summaries support substantial training activity. They do not establish held-out generalization or convergence to a game-theoretic equilibrium.

The often-cited ~85% Phase 1 success is a training-phase statistic within an 18,000-episode run. It is not evidence of an 85% held-out win rate after 6,000 episodes. Phase 1 and Phase 2 also change pellet density and other enemies, so their difference does not isolate the effect of a learned opponent. The v4 write-up says roughly 94% timeouts; its dashboard summary records 2,847/6,000, or **47.45%**. Reconcile these sources before republishing the claims.

**2. The central CHER mechanism is not valid counterfactual experience.** It changes an action and reward while retaining the successor produced by the original action (`utils/cher.py:174-190`). In a controlled reproduction, the synthetic reward was 29.699, versus 0.299 for executing the proposed action, and the successor was different. This problem remains on `v7-ablations`. The trainer additionally attaches post-step information to a pre-step state.

Ordinary [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495) reinterprets goals on experienced trajectories. A different action needs a corresponding valid successor. Q6 could use an explicit heuristic imitation loss, or clone and step the true simulator to generate real alternatives. Those would be new, clearly defined experiments.

**3. The v5→v7 interpretation overlooks a major historical confound.** [Commit cb5153bd48](https://github.com/rahul-tiwari-95/Q6/commit/cb5153bd48) fixes Hunter replay storing Krishna's action instead of Hunter's. It simultaneously changes approach shaping, timeout rewards, CHER weighting, gate regularization, and easy-opponent warmup. Repairing Hunter learning alone could change adversarial pressure. The record cannot attribute the outcome to weighted CHER versus reward asymmetry. The wrong-action bug remains in this branch's older Phase 3 trainer, but is fixed on v7/ablations; do not describe it as unresolved across all branches.

**4. Some apparent strategy problems have environment and measurement confounds.** Hunter movement permanently hides uncollected pellets from the grid observation. The feedforward observation omits invulnerability/time; identical observed grids can produce different catch outcomes. A flood-fill check found mandatory unreachable pellets in 22/2,000 seeded maps. That 1.1% cannot explain near-total failure, but is inappropriate unexplained noise in a benchmark.

The gate's two heads have labels, not demonstrated evade/collect specialization. Both receive the same mixed objective. On the ablation branch, the initial hard-tier score of 0.5 is an unobserved prior; it later becomes measured pellet fraction, not win rate. The “CHER dependency” ratio divides capped interventions by uncapped opportunities. A fall in that ratio is not proof the teacher became unnecessary. Teacher-disabled fixed-policy evaluation would test that claim directly.

Consequently, “task-progress collapse under this implementation” is supported much better than “rational risk-dominant Nash equilibrium.” There is no payoff analysis, exploitability measurement, or best-response evidence establishing the latter. The literature cited by Q6 addresses related but distinct mechanisms: [risk-sensitive objectives](https://arxiv.org/abs/2405.02724), [coordination-equilibrium selection](https://arxiv.org/abs/2605.18078), and [shared-parameter red-team collapse](https://arxiv.org/abs/2605.08427). Related papers do not prove the diagnosis in this game. Failure under both DQN and PPO would also retain shared environment/observation confounds.

**5. No Way Home's reported beta counts reproduce, but the interpretation should narrow.** We reconstructed **41, 41, 34, 0, 0** fooled seeds from the never-mitigate pre-shift message histories alone. Fifty direct policy checks matched. Increasing beta lowers the mixed score by construction because raw count is at least unique count. Pre-shift mitigation cannot change the sickness/message process. The curve locates a threshold crossing within this generator; it does not discover a general emergent law.

The original permutation shuffles observations across paired seeds. A within-seed reanalysis also gives p=0.0001 at 9,999 permutations: correcting the design does not erase the large contrast. The stronger issue is whether the comparison identifies the value of lineage rather than a more conservative action threshold.

On **exploratory** calibration seeds 15000–15009, a raw counter with threshold 1.0 had zero false alarms and mean shortfall 38,156, versus zero and 43,474 for the default lineage-aware counter. This is not a confirmed new winner: thresholds were explored here, with only ten seeds. It demonstrates why independently calibrated raw/decayed controls and full-task outcomes must come before stronger provenance claims. These calibration seeds have now been examined and must not be repurposed as untouched confirmation data.

**6. No Way Home has narrower useful results.** Separating world RNG from policy RNG is good experimental engineering. The reactive-mandate ablation isolates the cost of fixed-term commitment within the implemented setup. Keeping the surplus, threshold, and learning failures in the reports is valuable.

The kernel currently has shared resource pools, locality sickness counts, and one Boolean mitigation action. It is not yet spatial or a population of independently learning citizens. The election rule supplies a 70% probability of choosing the best calibration-ranked candidate. The learner receives unique-origin count as a feature; it learns a decision using a deduplicator, rather than discovering deduplication. Its stored performance is about 40% worse than the aware heuristic. The learning-rate floor becomes constant after 20 visits, and final transitions are dropped. These findings warrant corrected terminology and a learner lifecycle fix, not dismissal of the entire lab.

## Why the project became hard to steer

The observable pattern is **scope expansion plus unreliable feedback**. Success moved from learning a game, to preventing self-play collapse, to demonstrating rational equilibrium selection, to continual learning, to a society/market-world. Mechanisms, environments and success metrics changed together. Long runs then produced outcomes that could not decisively choose between explanations.

Some AI-assisted narrative also promotes analogies into conclusions: an opponent sampler becomes a market; evasion becomes an equilibrium; a gate is assumed to represent two specialized skills; a literature search becomes a confirmed unoccupied niche. These are hypotheses or metaphors. Treating them as established makes disappointing runs feel more conclusive than they are.

The practical repair is a trustworthy small experiment with a stopping rule. Ambition can remain broad while each deliverable stays narrow.

## What is worth releasing

| Artifact | Plausible audience and reason to care | Required preparation |
|---|---|---|
| Corrected Q6 failure-analysis lab | RL students and experimenters who want to reproduce, inspect, and diagnose concrete failure modes | Fix core semantics; attach errata and real data; one short runnable experiment with baselines |
| No Way Home provenance/control lab | People teaching or testing correlated evidence and sequential decisions | Fairly calibrated controls, welfare/response metrics, raw per-seed outputs, narrow claims |
| Replay recorder + static viewer | Small-gridworld researchers who need to see what policies actually do | A tiny schema, working example traces, relative artifact paths, missing-value fixes |
| Checkpoint-aware local supervisor | Individuals running laptop experiments | Exit-status and stale-log fixes; portable examples; explicit resume limits |

Audience fit is an assessment, not demonstrated demand. No external user interviews or adoption study were performed. Ask prospective users to reproduce one plot or adapt one scenario; successful independent use is a stronger signal than praise for a broad roadmap.

The repository is already public. It currently has no license file or formal GitHub release, and historical raw runs are not bundled. A practical first release is a licensed, tested snapshot containing one command, exact configuration/software versions, seed-level data, a few replays, and a short findings/limitations note. Preserve original experiments as “as run,” with errata; do not silently rewrite history. A large framework extraction is unnecessary.

## Paths forward, ranked

**1. Recommended research direction: adaptation across small changing worlds.** Ask: *How quickly can an agent adapt to a changed environment, and how much previous competence does it retain?* Use an A→B→A schedule: establish competent behavior in A, change one rule or opponent distribution in B, then revisit A. Hold maps/evaluation policies fixed when measuring retention; separate policy memory reset from weight updates. Include a frozen policy, ordinary learner, and recurrent/memory baseline. Measure B adaptation steps, A retention, held-out world performance, elapsed training time, and peak memory across repeated training seeds.

Start from one corrected kernel. A small Krishna/Hunter variant gives the shortest route to existing neural-RL code; NWH can remain a separate information-control example. An operational deadline is to finish a replicated baseline study before making a second environment or architecture. If baseline competence cannot be established in the chosen budget, reduce the task and diagnose it.

Efficient multi-world execution means batching many independent environments. Diverse worlds means varying their rules. Multiple interacting agents is another axis. Choose these independently. Profile first; implement batching only where it improves experiment turnaround. Compare scalar and batched transitions under matched random draws and test per-world resets. Report both simulator throughput and end-to-end learning throughput. Q6's existing DQN has 10.33 million parameters, with 99.16% in one flattened dense layer, so model/inference overhead deserves profiling too.

Existing [JaxMARL](https://github.com/bold-lab-ai/JaxMARL), [XLand-MiniGrid](https://github.com/dunnolab/xland-minigrid), and [Madrona](https://madrona-engine.github.io/) already address accelerated MARL, diverse tasks, and many-world simulation respectively. Q6's plausible distinction is a small, auditable experiment with clear failure analysis. Use existing backends/baselines where they fit; a new general engine is a much larger commitment.

**2. Fastest finished publication: close the provenance-control question.** Independently calibrate raw, unique, and decayed counters on training/calibration worlds; freeze them; evaluate on new seeds and changed forwarding/reporting/shift settings. Report welfare, unnecessary actions, missed crises and response delay. If the lineage benefit disappears under fair controls, publish that result. The experiment can still teach something useful about how a seemingly persuasive benchmark encoded its answer.

**3. Architecture direction, after baseline evidence: fast memory plus slow learning.** If “nested networks” means components adapting at different timescales, test one small fast-updating memory on a slow backbone against a GRU and matched ordinary learner. Distinguish episode-local state updates from online parameter learning. Freeze the world family and training budget during the comparison; measure adaptation, retention, compute and memory cost. Google's [Nested Learning work](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/) specifically concerns nested optimization processes and update frequencies. Q6's current gate is not evidence of that mechanism. A controlled result on a modest architecture is a credible outcome; reproducing an entire new paradigm is not needed to begin.

## A bounded restart

1. **Close the record:** publish this audit internally, attach errata, recover any historical data still stored elsewhere, and choose one release artifact. Do not retrain every old version.
2. **Make one reference trustworthy:** fix observation/transition semantics and provenance; run baseline competence and fixed-policy evaluation. Archive CHER/gating for later causal tests.
3. **Finish one study:** three to five training seeds for the first A→B→A baseline, with a declared compute cap and held-out evaluation set. Save all seed-level outcomes. Broader replication follows only if the question merits it.
4. **Package the result:** one reproducible command, one explanatory figure, a few working replays, and an honest account of what the result establishes. Then let user feedback and measured bottlenecks determine the next increment.

## Reproductions

Run from the Q6 repository root. Scripts are diagnostic artifacts and do not modify simulation/training source.

```sh
python3 research_review/2026-09-05/reproduce_rl.py .
python3 research_review/2026-09-05/reproduce_nwh.py
```

The NWH script writes generated JSON to `/tmp`. Its threshold search is exploratory. Saved outputs: [RL checks](rl-reproduction.json), [NWH checks](nwh-reproduction.txt), [paired beta data](nwh-beta-paired.json), [exploratory controls](nwh-exploratory-controls.json).

Validation covers existing tests and bounded probes. Historical multi-day training runs, full PPO training, new nested architectures, and multi-world speedups were not rerun or demonstrated during this audit.
