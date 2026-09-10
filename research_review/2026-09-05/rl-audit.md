# Original Q6 RL audit (2026-09-05)

Scope: read-only audit of current `/Users/rahul/Q6-nwh` original RL stack, training loops, tests, versions, articles, and embedded dashboard artifacts. No source changed. The worktree `.git` points to missing `/Users/rahul/Q6/.git/worktrees/Q6-nwh`; historical provenance must therefore be checked separately through GitHub. Claims below are current-code facts unless marked artifact or interpretation.

## Overall assessment

This is substantial applied RL engineering and a research notebook, with an actual working Double DQN/dueling CNN implementation, multiple environments, self-play snapshot infrastructure, trajectory recording, dashboard, checkpoint/resume plumbing, and a sequence of documented experiments. The original work is worth releasing as a reproducible learning/failure-analysis sandbox after correctness repair. It does **not yet establish** that collection collapse is a rational equilibrium, that GOP fixes conflicting gradients, that CHER is valid hindsight replay, or that a sampling change causes the reported causal mechanism. Some strongest retrospective explanations contradict code or stored run summaries.

The most promising contribution is the combination of transparent unsuccessful experiments and inspectable multi-agent behavior. The architecture naming is less novel or defensible than the experimental tooling and question. More algorithmic complexity is currently unlikely to provide interpretable progress without a trusted evaluation substrate.

## Critical correctness findings

### 1. Hunter learns the wrong action in Phase 3

`train_phase3.py:288-305` gets `a_k` and `a_h`, executes both, but at line 304 calls:

    hunter.step(state, a_k, rewards["hunter"], next_state, done or trunc)

The Hunter replay action must be `a_h`. As written, its Q function learns the reward and successor associated with its own executed action under Krishna's action label. Phase 2 correctly stores `a_h` at `train_phase2.py:167`. This is a serious confound for all Phase 3 learning claims if present in the run commit. Parent is checking history; do not automatically attribute every historical run to this current-code bug without that check.

### 2. CHER fabricates transitions that do not obey environment dynamics

`utils/cher.py:174-190` replaces action with a heuristic toward-pellet action and adds 30 to reward, while retaining the original `next_state`, `next_context`, and done. In an action-dependent gridworld, these are generally not the outcome of the replacement action. This is not valid goal relabeling and not a simulated counterfactual rollout. The documentation explicitly asserts the invalidity as a key invariant (`ARCHITECTURE.md`, Counterfactual HER section), and `tests/test_cher.py:220-223` asserts preservation of the wrong successor rather than checking dynamics.

Additionally:
- `train_phase3.py:292` overwrites `info` with next-state info, then `308-316` stores that with the previous state/action. Thus CHER computes the replacement action and safety condition from the **post-action** position for a **pre-action** state.
- `utils/cher.py:78-92` only compares Manhattan deltas; it never checks walls or reachability. Calling its action “optimal” or a pellet within four Manhattan units “reachable” overstates the computation.
- The heuristic teacher is not learned and there is no branching simulator or world model.
- Changing injected counts does not by itself show independence from teaching: state distribution, episode length, opportunity definitions, and cap all affect counts.

A valid direction would clone the complete simulator state, execute the alternate joint action, recompute reward/termination/context, and log the resulting true transition. Or explicitly use the heuristic as an auxiliary imitation objective and measure it as such. These are different experiments and should be named honestly.

### 3. Hunter passage permanently hides pellets from observations

`environment/selfplay_env.py:170-173` clears the old Hunter cell to EMPTY; `:197` paints the new cell HUNTER, even if it contains a pellet. Leaving the cell does not restore its pellet, although `pellet_positions` retains it. Collection consults the set (`:151`), while the agents see only the flattened grid (`:300-301`). Required objectives therefore become invisible without being completed. This particularly undermines the diagnosis “does not collect even when safe.”

This is better fixed by separate static map, pellet mask, and agent occupancy layers, constructing observations rather than using one mutable rendering grid as both world state and view.

### 4. Observations are not Markov, despite “full vision” framing

The flat grid omits lives, invulnerability timer, and remaining episode time. Lives affect future loss, hidden invulnerability affects catch immediately, and hidden time affects terminal status. The gated policy adds lives and pellet count through `info_to_context` (`agent/gated_dqn_agent.py:50-74`), but does not add invulnerability/time. Base DQN agents receive none of these. Hunter overlaps can also hide the Hunter marker. A feedforward Q-network cannot distinguish physically identical grids with different immediate catch outcomes.

This could be intentionally framed as a POMDP with recurrence; currently it is an accidental observation problem, and adding “nested networks” would not isolate it.

### 5. A subset of maps is unwinnable

`selfplay_env.py:259-282` places walls and four mandatory pellets without any flood-fill/connectivity check. A deterministic audit of seeds 0..1999 found **22/2000 (1.1%)** with at least one required pellet unreachable from Krishna. Seeds include 68,115,401,548,583,623,640,719,1097,1133. This is not enough to explain near-total training failure, but should be removed or labeled for a benchmark.

## Reproduced checks

Direct controlled-state checks in current code:

- Hunter crossed `(1,3)` then left it: pellet still exists in `pellet_positions`, observed cell is EMPTY=6 rather than PELLET=1.
- For the same initial state and Hunter action, CHER changed Krishna action from UP to RIGHT; its retained successor did **not** equal the true RIGHT successor. Assigned reward was 29.699; actual RIGHT reward was 0.299.
- CHER recommended RIGHT where that target cell was a wall and the action was a no-op.
- Two states with exactly the same grid but invulnerability timers 0 versus 10 yielded lives 2 versus 3 after the same joint action.
- Flood fill: 22/2000 unreachable-pellet maps as above.

Existing suite invocation:

    python3 -m pytest -q tests/test_cher.py tests/test_selfplay_env.py tests/test_state_encoder.py tests/test_opponent_pool.py tests/test_frozen_agent.py tests/test_train_phase2.py

**66 tests passed** before the five-episode training smoke was intentionally interrupted to keep the audit bounded. Final pytest output: `66 passed in 86.14s`, KeyboardInterrupt in torch Adam. This was not a test failure. It illustrates that passing shape/unit tests do not check the scientific semantics above. No long training/retraining was done.

## What is actually implemented

- `agent/dqn_v2_agent.py:155-186` correctly uses local argmax, target-network evaluation, detached Double DQN target, Huber loss, gradient clipping, and Polyak target update. `model/cnn_q_network.py:77-85` implements the dueling mean-centered advantage formula correctly.
- Raw uint8 replay (`agent/dqn_v2_agent.py:38-63`) is a useful memory-saving design; encode only when sampled.
- `model/gated_option_network.py:137-148` is a two-expert mixture of advantage functions with a context MLP. Both experts receive the same blended TD loss (`agent/gated_dqn_agent.py:304-318`). There is no independent evade/collect supervision, reward decomposition, option termination, or temporal abstraction. Head labels do not establish behavioral specialization, and swapping heads plus complementing the gate preserves the function. Gate values staying between 0 and 1 cannot prove collapse is eliminated (`versions/v5_phase3_gop_cher.md:145-147`). “Prevents gradient interference” is untested, especially because the trunk is shared.
- `train_phase2.py:125-167` and `train_phase3.py:258-305` implement asymmetric independent Q-learning with historical Hunter opponents. Only Krishna sees an opponent pool; Hunter sees current Krishna in joint mode. Historical snapshots are selected with recency bias and FIFO eviction, not a full empirical average strategy. It is useful FSP-inspired population training, not evidence of Nash convergence. README's “FSP prevents Nash cycling” (`:209-210`) exceeds what the code/experiments establish.
- “Easy” and “hard” pool tiers mean early versus late checkpoints (`utils/hierarchical_pool.py:75-92`), not measured difficulty. Later is not necessarily stronger, particularly with wrong-action training.
- Frozen opponents are greedy (`agent/frozen_agent.py:42-47`), while live opponents explore. A skill comparison by epoch confounds learned policy and exploration mode.

## Reproducibility gaps

1. Python `random` drives exploration and replay sampling (`agent/dqn_v2_agent.py:56,127-128`) but Phase 2/3 seed only NumPy generator and Torch (`train_phase2.py:81-82`, `train_phase3.py:158-159`). Thus the declared seed does not reproduce actions/replay. Phase 1 also initializes Torch networks without setting a seed (`train_v2.py:110,128`).
2. Agent checkpoints save network weights, epsilon, learning_step only (`agent/dqn_v2_agent.py:203-216`; `agent/gated_dqn_agent.py:354-367`). Adam moments, replay buffers, RNG states, update cadence are not restored. Phase 3 resume starts a new seed based on episode index. A resumed run is not scientifically equivalent to continuous training.
3. Phase 2 metadata hard-codes epsilon_decay=0.9999 (`train_phase2.py:260`) while config actually uses 0.9994 (`config.py:54`). A run manifest should serialize actual configuration rather than duplicate constants.
4. Training win rates are measured during exploration against evolving opponents (`train_v2.py:167-169,227-233`; `train_phase2.py:159-167`). No fixed held-out evaluation matrix is present in these loops. A declining loss/nonzero loss is not proof of improved behavior.
5. Raw training_runs/checkpoints are absent in this 3.7MB checkout and ignored by `.gitignore:41`; `dashboard/data/index.json` does retain useful embedded metadata and downsampled training summaries. It is evidence of running experiments, but not full reproducibility on its own.

## Artifact/narrative contradictions

The following are based on stored `dashboard/data/index.json` run objects; raw CSVs were not available to independently recompute all aggregates.

| Run | Embedded artifact | Documentation issue |
|---|---|---|
| Phase1 `20260530_030934_phase1_baseline` | 18,000 total training episodes, final phase 9,000 episodes, phase4 win rate 0.8506667; initial phase 91% | README:67 says 85% at 6,000 episodes. Correctly identify it as training performance and total budget; no held-out evidence. |
| Phase2 v4 `20260606_175631_phase2_v4_nowall` | 6 K wins, **3,147 Hunter wins**, **2,847 timeout (47.45%)** | `versions/v4_phase2_bimodal.md:47-48` says ~380 Hunter wins/~94% timeout. These are materially inconsistent. |
| Phase3 v5 `20260612_184130_phase3_v2_resume_enabled` | 2 K wins, 265 H wins, 11,733 timeout (97.775%), best_avg100 27.50108@789 | `versions/v5...:114` says best 41.22@5. Retrospective causal narrative and reported summary need reconstruction. |
| Ablation1 `20260729_194141_v7_ablations_run` | 6,000 eps, 2 K wins, 2,104 H wins (35.07%), 64.9% timeout, 45.5h | Enough to show task remains almost never completed; single training seed cannot establish the mechanism. |
| Ablation1b `20260731_222534_v7_ablations_v2_floor_tuning` | 6,000 eps, 2 K wins, 2,873 H wins (47.88%), 52.08% timeout, 12.9h | Negative result honestly logged, but changed floor and schedule simultaneously and no replication. |

Specific explanatory problems:
- `versions/v4...:66,82-86` attributes reward to +0.5 survival/step and computes expected survival reward ~495. Actual self-play code has **K_STEP=-0.001**, no survival bonus. Positive non-collection returns can arise from net approach shaping; they are not proof of positive survival reward. The probability assumptions 0.4/0.99 are not measured. There is no basis here to call risk aversion the mathematically correct equilibrium.
- Original Phase1 has ~50 pellets (10% of empty cells; `hunter_gridworld.py:323-327`) and a four-pellet target plus two greedy bots, two patrollers, one scripted Hunter (`train_v2.py:46`, `hunter_gridworld.py:145-176`). Phase2 has exactly **four** pellets and removes those other enemies. Its difficulty increase cannot be attributed solely to replacing a scripted Hunter by a learned one.
- Original Phase1's `pellet_positions` set is initialized once (`hunter_gridworld.py:107`) and `_place_pellets` adds without clearing (`:336-340`); reset does not clear it. This is a further suspected persistent-state bug worth testing if recovering Phase1 runs.
- `articles/ablation1b-floor-tuning-failed.md:39` initially says joint mode “structurally cannot be affected” by the pool, then describes spillover. Both modes train the same Krishna parameters and replay buffer, so training in one obviously can change the other. Modes are not separate control groups.
- `articles/ablation1b...:47` claims independently audited, “isn't a bug or metric artifact.” This is stronger than feasible from current evidence. Even an audit should not categorically rule out all implementation/measurement artifacts.
- “Bimodal collapse eliminated because gate never locked” and “3 pellets proves planning” are interpretations, not measurements of those mechanisms.

## Efficiency and possible direction

Measured current parameter counts:
- CNNDuelingQNetwork: **10,326,949 parameters**, 10,240,256 (99.16%) in 40,000→256 flattened dense layer.
- GatedOptionNetwork: **10,360,442 parameters**, 10,240,256 (98.84%) in the same projection.

Both preserve 25×25 spatial resolution through two conv layers and flatten (`model/cnn_q_network.py:47-51`). The version document depicts MaxPool layers that do not exist. Single-state actions and two separate learning agents create many small calls; environment stepping is Python/NumPy, not batched. This is a credible place to investigate efficient many-world simulation, but speedup must be measured against identical transition semantics and learning performance. No evidence of an efficient multi-world engine or nested neural architecture already exists in this stack.

Suggested bounded restart:
1. Freeze an archived “as-run” snapshot and write errata rather than deleting the negative history.
2. Build a tiny trusted 7×7/9×9 environment with independent entity layers, Markov observations, reachability, explicit collision/timeout semantics; test one-step transitions against a reference.
3. Establish no-Hunter/random/greedy/A* held-out baselines; random actions should not accidentally score as learning. Then use a fixed opponent evaluation matrix across training checkpoints.
4. Simplify to a compact CNN or spatial pooling + ordinary DDQN/PPO, no CHER/no gate initially. Three to five seeds, identical map seeds, report actual task win rate plus pellets/catches/timeouts and confidence intervals.
5. Only once that works, ask one question: does a batched functional simulator deliver more **validated learning per second/memory budget** across 1/16/128/1024 worlds? This is a concrete small open-source project. Keep original Q6 as benchmark/task and notebook.
6. If branching is the desired research angle, implement exact simulator state clone/step as a first-class API and compare real counterfactual rollouts against extra ordinary experience under matched compute. This connects user interest in multiple worlds to a fixable weakness, without claiming the existing CHER is valid.

Do not add nested networks, new reward terms, opponent curricula, and replay modifications together. The current obstacle is lack of trustworthy causal attribution, not shortage of mechanisms.

## Branch-qualified follow-up: v7-ablations and historical fix

Checked GitHub tarball `v7-ablations` at `1811d88`, extracted to `/tmp/q6-audit-ablations/rahul-tiwari-95-Q6-1811d88`. Also read the GitHub patch for `cb5153bd48` (June 15, “v7: correctness fixes + reward redesign + improved training”).

### Corrected versus still present

- **Hunter wrong-action bug was fixed in cb5153bd48**. Patch explicitly changes `a_k` to `a_h`; latest ablation `train_phase3.py:377` is correct. `tests/test_phase3_regression.py` includes a regression for it. Do not present this as an unresolved bug on v7/v8. It remains in the original Phase3 stack embedded in the no-way-home checkout.
- The same commit simultaneously introduced conditional approach shaping, zero-pellet timeout penalty, gate entropy regularization, weighted CHER, and FSP easy warmup. Therefore the retrospective v5→v7 explanation that focuses on reward vs weighted CHER misses a larger confound: **repairing Hunter's learning algorithm**, plus other changes.
- **Invalid CHER persists** in ablation `utils/cher.py:183-197`, including unchanged successor/action mismatch. Weighted default true changes only bonus magnitude.
- **Post-state info attached to pre-state persists** at ablation `train_phase3.py:365-389`.
- **Pellet disappearance persists** at ablation `environment/selfplay_env.py:175-176,201`.
- **Observation hidden-state and disconnected-map issues persist**, same observation/reset structure at `:265-307`.
- **Unseeded Python random and nonexact resumption persist**. Added bookkeeping safeguards improve interruption handling but do not save optimizer/replay/RNG state.
- **Gate regularization is now present**, but it maximizes mixture-gate entropy. It still does not establish semantic collect/evade specialization or temporal options.

### Ablation diagnostics need narrower interpretation

1. `mean_hard_tier_score` is **not opponent win rate**. `train_phase3.py:425-429` scores nonwins by `pellets_collected / 4`, meaning a zero-pellet timeout and a zero-pellet death both score zero. A low score demonstrates low collection under that sampling scheme, not necessarily being repeatedly caught.
2. Unplayed checkpoints default to **0.5** (`utils/hierarchical_pool.py:133-135,265-270`). During FSP easy warmup the hard pool is not played, so the reported 0.50 before episode 1000 is an initialization prior. After first outcome it jumps directly to the observed score (`:194-198`), not a smoothed decrease from that prior. Consequently the famous **0.5→0.05 “collapse” cannot be interpreted as a measured deterioration from previously adequate hard-tier skill**. It substantially reflects the transition from unobserved priors to measured low collection. There may still be a real Hunter-win vulnerability window; this metric does not establish the claimed cause.
3. `cher_dep_idx = len(capped_injections) / uncapped_opportunities` (`train_phase3.py:440-443`, CHER cap at `utils/cher.py:155-157`). Example: 500 wrong safe actions all qualify but cap=50, index=0.1; this could misleadingly meet “low dependency” with zero correct choices. Need uncapped errors/opportunities, per-step exposure normalization, and held-out teacher-disabled performance. Mean raw counts below 50 do not show the cap has no effect on episode-level ratios. It is an intervention-frequency proxy, not a causal measure of dependence.
4. Easy warmup (`train_phase3.py:323-340`) affects only the ~30% FSP subset. ~70% joint play still faces the learning Hunter. Statements that the agent has no practice with strong Hunters before episode 1000 are therefore too broad.
5. Floor/warmup are a plausible experiment, and declaring the predicted result failed is good scientific practice. But one seed per setting, changing floor and schedule together, cap/score artifacts, shared-policy training across modes, and invalid CHER mean the causal explanation is still provisional. It can be published as an exploratory negative result after corrected labeling and errata; it is not yet a validated general claim about rectified PSRO/self-play.

### Specific research takeaway

What should be preserved is the transparent sequence “hypothesis → implemented intervention → observed failure → revised hypothesis,” and the ability to inspect trajectories. What should be discarded is confidence in narratives that were inferred from proxy metrics without fixed-policy controlled evaluation. This is a recoverable research-method problem, not evidence that the user's entire effort lacks value.

Reproduction artifacts preserved: `/tmp/q6-rl-repro.py` and `/tmp/q6-rl-repro-results.json`. The script accepts an optional repository-root argument and does no training or source writes. It reruns the environment/CHER checks and connectivity audit.

For a next project focused on A→B→A adaptation, use one world-rule/distribution shift, fixed evaluation maps/opponents, a competent A baseline, and matched interaction/compute budgets. Report B adaptation cost and A retention, testing simple recurrent memory and replay before nested optimizers. This is coherent and more finite than building a general simulation framework. Profiling/batching should support that experiment rather than become a second unbounded project.
