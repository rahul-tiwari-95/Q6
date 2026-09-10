# Fixed data, exact versus bootstrapped targets — protocol v1

Status: locally declared before main execution, not externally registered.
The [supervised result](supervised_results_v1.md) informed this comparison.
Its training bank is deliberately reused; its observed fresh panel is excluded
from the new primary evaluation. Stop after this comparison, without tuning.

## Question and scope

With state/action coverage, initial weights, architecture, sampling, loss and
update budget held fixed, does one-step Double DQN target learning achieve
useful fresh-layout behavior compared with exact Q* supervision?

This isolates the specified target-learning procedure under complete offline
action coverage. It does not compare online exploration, replay collection,
interaction efficiency or memory. Enumerated hypothetical states and all four
transitions are privileged coverage unavailable to an ordinary trajectory-only
learner. A failure applies to this target schedule and optimization budget,
not every bootstrapped RL algorithm.

## Paired conditions

World A, observation, reward shaping and discount remain unchanged: a fully
visible 5×5 collection world, four walls, one pellet, 32-step horizon, ordinary
controls visible initially, 92 observation values and discount 0.97.

Both conditions use the existing **20,420-parameter DQN.online**, 92 → 128 →
64 → 4 with ReLU; Adam 0.001; batch size 64; gradient norm cap 5; Smooth L1
beta 1 averaged across the batch and **all four action outputs**. Learner seeds
are **0, 1, 2**, each initialized identically in its two paired conditions.
No previous trained weights, auxiliary loss, masks or architecture changes.

1. **exact_q:** fixed exact shaped finite-horizon Q* targets, float64 in the
   archive and converted to float32 for optimization.
2. **double_dqn:** for each sampled state and each action, use its saved reward,
   successor and terminal flag. The detached target is
   `r + 0.97 * Q_target(s_next, argmax_a Q_online(s_next,a))` for nonterminal
   transitions and exactly `r` at success or timeout. Online ties choose the
   lowest action label. Initialize the target network as a copy of the online
   network. **After every optimizer update**, update each target parameter by
   `target = 0.99 * target + 0.01 * online`, matching the existing DQN schedule.
   Do not backpropagate through successor predictions or target parameters.

The exact condition's target network is unused. Target-network computation is
part of the target procedure being compared; equal update counts do not mean
equal floating-point work or wall time. Report each condition's costs.

Run **30,000 updates per condition and seed** at checkpoints **0, 1,000, 3,000,
10,000, 30,000**. Reuse the existing `BatchSampler`: owned NumPy RNG initialized
by `SeedSequence([seed,66301])`; 64 distinct uniformly sampled rows per batch,
with repetition allowed across batches. Both conditions for a seed must have
identical ordered batch digests and per-row counts. Each sampled row supplies
the same four action labels, so neither condition receives additional action
coverage or a different loss reduction.

Totals: **180,000 optimizer updates**, **11,520,000 state presentations** and
**46,080,000 action targets consumed**. These are repeated offline samples,
not environment interactions. Evaluation never advances training RNGs or
changes parameters/optimizer state. Run conditions sequentially on one CPU
thread; no simultaneous training jobs or accelerator allocation.

## Dataset, transitions and fresh selection

Reuse exactly the preceding supervised training bank: seeds **300000–300255**,
256 wall-and-goal layouts, every row-major non-wall/non-pellet position and
remaining time 1–32 innermost. There are **163,840 states**. Retain impossible
and original-start-unreachable hypothetical Markov states. Match every training
array hash to the supervised archive before optimization. Preserve archive
source, protocol, dataset and model provenance.

For every training row preserve all four deterministic transitions: shaped
reward, terminal flag and compact successor-row index. Terminal index is −1;
nonterminal successors refer to the same training layout and remaining time
minus one. No fresh observation may enter this table. Store compact indices
instead of four duplicated 92-value successor observations per row. Verify
transition dynamics against the actual environment and Bellman consistency
against the archived exact targets before admitting the main result.

Select **64 fresh layouts** by scanning upward from seed **940000**, rejecting
training-layout matches, matches to the preceding supervised fresh panel, and
duplicates already accepted in this panel. Identity is walls + pellet + rules,
ignoring original start, clock and task ID. Preserve all rejected candidates
and reasons. Selection precedes predictions and uses no performance criterion.
Enumerate their same 640 states per layout, **40,960 fresh states**, for
evaluation only. Verify zero layout and complete-observation train/fresh
overlap, and zero layout overlap with the preceding supervised fresh panel.

All conditions and initializations share this fixed bank and fresh panel.
The training bank has informed earlier research; the new panel is the untouched
primary test for this comparison. This is not an independent task-bank
replication of the earlier supervised study.

## Evaluation and measurements

At every scheduled checkpoint evaluate all original starts in both panels:
one greedy and two epsilon-0.1 episodes per layout. Use existing isolated
evaluation draws keyed by learner seed, map seed, repetition and 55219, shared
across conditions/checkpoints. Save raw rows and first-map, first-repetition
recordings for every condition, seed, checkpoint, panel and mode, selected
independently of outcomes.

Evaluate every enumerated state with batched inference. Reuse the supervised
metrics and remaining-time buckets 1–8, 9–16, 17–24, 25–32, plus the separate
all-state summary: four-action MAE, RMSE, signed bias, 95th percentile of
per-state MAE, exact action regret and optimal-action agreement. Agreement
conditions on still-winnable states, with exact-Q absolute tolerance 1e−6 and
relative tolerance zero. Impossible states stay in value-error denominators.
Preserve additive components and recompute pooled quantiles from predictions.

Log pre-update minibatch loss every 100 updates, separately by condition.
Exact targets are stationary; DDQN targets move. Similar training loss values
do not imply comparable error to Q*, which is measured separately for both.

Uniform random actions use seeds 0/1/2 and two episodes per map, and the
shortest-path controller uses one deterministic episode per map, on both
panels. Share these references without counting their reuse as new evidence.
Historical RL policies are not another arm of this controlled comparison.

Preserve final per-layout and per-seed paired **DDQN minus exact** differences
in greedy success, episode length and blocked-action rate. Report pooled
outcomes and seed ranges descriptively, not as confidence intervals; three
initializations share one task bank and panel. Report episode length alongside
success because failures consume the full horizon.

## Declared gates and interpretation

The primary outcome is **final fresh-layout greedy success per seed**, and
its paired difference between conditions. A condition passes useful offline
fresh behavior if **every seed reaches at least 70% and exceeds pooled random
fresh success**. Do not select an intermediate checkpoint after seeing scores.

Keep two separate diagnostics for each condition:

- **Training fit:** every seed reaches at least 90% original-start greedy
  training success **and** 90% optimal-action agreement over winnable training
  states. This is not proof of exact regression or optimal navigation.
- **Efficient fresh success:** in every seed, at least **80% of all fresh
  greedy episodes** both succeed and use at most **twice that layout's planner
  step count**. Failures count against this denominator. This secondary
  diagnostic was chosen to expose the previous study's wasted moves and is
  independent of the primary success gate. Also report successful-episode
  lengths and blocked moves with their denominators.

The exact arm should reproduce the earlier run's final online weight hashes
and sampler digests in the pinned runtime because its training computation
is unchanged. Check and preserve that identity; a mismatch is an execution
consistency failure requiring investigation, not a fresh scientific result.
This consistency check is not an independent training replication.

Interpret the final fixed-budget result without adding an extra run:

- Both conditions pass fresh success: bootstrapping can work with this fixed
  coverage; investigate online collection/replay next, retaining efficiency
  and fit limitations rather than calling RL solved.
- Exact passes, DDQN fails: focus on the specified target/optimization
  procedure under fixed coverage before online collection or memory. This
  does not establish that all bootstrapping fails.
- DDQN passes, exact fails: inspect paired errors, optimization and panel
  variation; do not assume exact labels guarantee easier policy learning.
- Neither passes: investigate fitting/coverage and reproducibility before
  attributing the result solely to online exploration or network capacity.

A partial, smoke, deviating, inconsistent or resource-interrupted run is
ineligible for research gates. Report its status and preserved progress.

## Resources, artifacts and stopping rule

Use Python **3.12**, Torch **2.8.0**, NumPy **2.0.2**, deterministic CPU execution
and one Torch thread. Total admission cap is **1,200 seconds**, including
dataset/transition preparation, training and evaluation. Periodically check
process peak resident memory with platform-correct `resource.getrusage` units
and stop admitting work if it exceeds **4 GiB**. This is a sampled stop guard,
not an OS-enforced allocation limit or a measurement of all system memory.
Record peak RSS, resource checks and separate condition/preparation/evaluation
timings. Keep arrays compact and only one learner active during optimization
to respect other applications on the owner's 64 GB MacBook.

Before main training, commit this protocol and freeze/capture source revision,
dirty state, source/protocol hashes, complete command and runtime. Save dataset
arrays/metadata, transitions, counts/digests, raw episode/state/loss tables,
all 30 scheduled inference snapshots including online and target weights,
final dense predictions, paired results, gates and replays. Inference snapshots
do not promise optimizer resume. Include a SHA-256 manifest and export the
dashboard from the saved result. Separate smoke/tests use small alternate
training/fresh seeds and cannot supply research evidence.

One main comparison only; no automatic extra updates, seeds, tuning or memory
extension. Preserve negative results and document deviations. Historical
artifacts and the owner's undecided license remain unchanged.
