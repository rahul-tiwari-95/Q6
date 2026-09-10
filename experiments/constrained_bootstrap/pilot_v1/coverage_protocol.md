# Exhaustive versus collected state coverage — protocol v1

Status: locally declared before main execution, not externally registered.
The [fixed-target result](fixed_targets_results_v1.md) informed this study:
Double DQN learned useful fresh-layout behavior under exhaustive offline
coverage. This comparison changes the current-state sampling support while
retaining that target/update procedure. Stop at the declared budget.

## Question and scope

Does the same offline DDQN procedure learn useful fresh-layout behavior when
current training states are restricted to those visited by an exploratory
trajectory policy, compared with exhaustive state coverage?

Both arms receive all four transitions at each sampled current state. This
counterfactual action access is privileged, and training occurs after collection
has stopped. The study does not establish online-RL competence, trajectory-only
action learning, optimal replay weighting, or a need for memory. Uniform replay
over distinct visited states changes both support and its induced map/clock
distribution; this does not isolate support size from distribution weighting.

## Fixed world, learner and paired conditions

World A remains the same fully visible 5×5 collection task: four walls, one
pellet, 32-step horizon, ordinary controls shown initially, 92-value observation,
discount 0.97, existing rewards and potential shaping. Use the same
**20,420-parameter DQN.online**, 92 → 128 → 64 → 4 with ReLU. Initialization
seeds **0, 1, 2** are paired across conditions. Start untrained in each arm.

Reuse the existing `q6.fixed_targets.fixed_update(..., condition="double_dqn")`:
Adam 0.001; batch size 64; Smooth L1 beta 1 averaged across all four actions
and states; gradient norm cap 5. Online chooses the successor argmax, the
lagged target evaluates it, and the target is detached. Success and timeout
targets equal the reward. After each optimizer update, apply target tau 0.01.
No new loss, masks, architecture, memory, collector learning or target schedule.

- **exhaustive:** sample uniformly from all **163,840** current states in the
  preceding fixed-target training bank, seeds **300000–300255**. Preserve
  map order, row-major non-wall/non-pellet position, then remaining time 1–32
  innermost. This includes hypothetical valid states unreachable from the
  original start at that elapsed time, and deadline-impossible states.
- **collected_unique:** sample uniformly from the sorted unique global row
  IDs visited as **pre-action current states** by the collector below.
  Duplicate visits increase recorded exposure counts but never replay weight.
  Maps with more distinct visited rows consequently receive more sampling
  mass; each map has equal 640-row mass only in the exhaustive condition.

Train for **30,000 updates per arm and initialization**, evaluating at
**0, 1,000, 3,000, 10,000, 30,000**. Each arm uses the unchanged `BatchSampler`
with `SeedSequence([seed,66301])` over its local support positions: 64 distinct
positions per batch, repetition allowed between batches. Map local positions
to global training-bank rows. The supports differ, so **actual batches and
per-row counts are intentionally not paired across arms**. Preserve local
and global ordered index digests, support indices, and global exposure vectors.

Totals: **180,000 updates**, **11,520,000 current-state presentations** and
**46,080,000 action targets consumed**, independently of collector interaction
counts. These repeated offline presentations are not world interactions.
All initializations share one collected bank and one evaluation panel; they
are not independent collection or task-bank replications.

## Collector and immutable replay support

Before any learner updates or policy evaluation, collect **16 complete episodes
on each of the 256 training maps**, in ascending map-seed order followed by
repetition 0–15: **4,096 episodes**, at most **131,072 actual world steps**.
Reset each episode to that map's original sampled start and full horizon.
Stop each episode at collection success or timeout; do not extend it to fill
a transition quota or stop early based on coverage or outcomes.

The collector chooses uniformly among the four action labels by a direct
`rng.integers(4)` draw at every step, with one owned RNG per episode initialized
by **SeedSequence([map_seed, repetition, 77301])**. It has no trained network,
oracle-selected action, action mask or learner-seed dependence. Its action
draws are isolated from learner sampling and evaluation randomness.

Save every actual pre-action row ID, remaining clock, action, shaped reward,
next row ID and terminated/truncated flag, keyed by map, repetition and step.
Terminal next-row index is −1; terminal observations are never current-state
training rows. Save per-episode steps/success and the complete visited-count
vector over the full bank. Deduplicate only after collection, sort global IDs,
and freeze this support before optimization. Preserve the complete collector
configuration, actions and ordering so the audit can reconstruct collection.

Reusing a layout or returning to a position at a different remaining time is
not the same Markov state. Use the original global bank row identity, including
position and clock. Do not silently balance maps, reweight frequencies, add
unvisited rows, augment starts, or change collector policy after seeing scores.

## Counterfactual transitions and successor queries

Reuse the preceding full training-bank observation/target arrays and all
**655,360** compact action transitions. Check every array against the archived
fixed-target study before optimization. Exact labels are evaluation and
integrity references only; neither learner optimizes against them.

For either arm, each sampled current state receives all four saved rewards,
terminal flags and successor observations. Nonterminal successor indices
always address the **full training bank**, including rows outside collected
current-state support. Retain those observations for detached DDQN target
queries. Do not drop the transition, terminate it artificially, substitute a
zero value, or add its successor to the collected current-state pool.

This is one-step counterfactual access, not recursively expanded replay support.
Parameters still change only through the loss on sampled current states.
Verify that every collected-arm global sample index belongs to the frozen
collected support and that unsampled rows have zero direct exposure counts.
No fresh state, label, transition or outcome may enter optimizer updates.

## Fresh panel and coverage diagnostics

Select **64 fresh layouts** by scanning upward from **950000**, rejecting
training-layout matches, layouts in either prior supervised/fixed-target fresh
panel (930000 and 940000 banks), and duplicates already accepted. Layout
identity is walls + pellet + rules, ignoring start, clock and seed. Preserve
every candidate rejection and reason. Selection precedes predictions and uses
no outcome criterion. Enumerate the same 640 states per accepted layout,
**40,960 fresh states**, for evaluation only.

Check zero train/fresh layout and full-observation intersections and zero
layout overlap with both previously observed fresh panels. The training bank
has informed earlier research; this is a new primary panel, not an independent
replication of the task generator or the collected bank.

Before training, report current-state support and visit counts globally,
per map, and by remaining-time buckets **1–8, 9–16, 17–24, 25–32**, plus the
separate all-state total. Include the support fractions among winnable states
and **goal-near states**, defined by physical shortest-path distance at most
two moves **regardless of remaining clock**. Compute these diagnostic labels
after random collection; they must not guide collection or sampling.

Report nonterminal edges from unique collected current states whose successors
are outside that support, with the total nonterminal-edge denominator. Also
report distinct outside-support successor IDs and distinct successor IDs.
If reporting exposure-weighted target-query counts, distinguish those repeated
queries from unique edges/states and derive them from recorded training counts.

## Evaluations, outcomes and gates

At every checkpoint, evaluate all 256 original training starts and all 64
fresh starts: one greedy and two epsilon-0.1 episodes per map and learner seed.
Use the existing isolated draws keyed by seed, map, repetition and 55219,
shared across conditions/checkpoints. Greedy ties use the lowest action label.
Evaluation does not alter weights, optimizer, target state or training RNG.

Evaluate every full training and fresh state with batched inference using the
existing MAE, RMSE, signed bias, 95th percentile of per-state four-action MAE,
exact action regret and winnable optimal-action agreement. Use absolute Q*
tie tolerance 1e−6, relative tolerance zero. Keep impossible states in error
denominators but exclude them from action agreement. Preserve map/time buckets,
additive components and pooled quantiles. Additional supported/unvisited state
breakdowns, if reported, are diagnostic and do not change the declared gates.

Shared random references use seeds 0/1/2 and two episodes per map; shortest
path uses one deterministic episode per map on both panels. They are evaluated
separately from collector episodes. Reusing a reference does not create new
independent evidence. Preserve first-map, first-repetition learner recordings
for every arm, seed, checkpoint, panel and action mode, selected before outcomes.

The primary outcome is **final fresh greedy success per seed**, plus paired
**collected_unique minus exhaustive** differences. A condition passes useful
offline fresh behavior if **every seed reaches at least 70% and exceeds pooled
random fresh success**. Do not select an intermediate checkpoint.

Retain independent diagnostics:

- **Training fit:** at least 90% original-start greedy training success and
  90% optimal-action agreement on the **full bank's** winnable states in every
  seed. This common full-bank diagnostic is deliberately stricter than fitting
  just the collected subset; a failure alone is not evidence that subset
  optimization failed.
- **Efficient fresh success:** at least 80% of all fresh greedy episodes in
  every seed both succeed and take at most twice their own layout's planner
  length. Failures count against the denominator. Also report episode lengths,
  blocked actions and successful-only metrics with explicit denominators.

Save per-seed and per-layout paired greedy success, length, blocked-rate and
efficient-success differences. Report seed ranges descriptively, not as
confidence intervals. The exhaustive arm must reproduce the previous DDQN
arm's final online weight hashes, initial hashes and sampling digests under
the pinned runtime. An identity mismatch is an execution-consistency failure,
not research evidence. This check is not an independent training replication.

Interpret the final comparison without an extra tuning run:

- Both pass: this collector's deduplicated state support is sufficient for
  useful offline behavior under all-action access. Investigate replay weighting,
  observed-action access and online collection/update feedback in separately
  controlled follow-ups; do not assume online learning is solved.
- Exhaustive passes, collected fails: the collected support and induced sampling
  distribution matter under this collector/budget. Diagnose coverage, clocks
  and weighting before changing targets, architecture or memory.
- Collected passes, exhaustive fails: inspect support weighting and new-panel
  variation without assuming more states must always help.
- Neither passes: investigate reproducibility, fitting and panel variation;
  do not attribute the result uniquely to collection or capacity.

Smoke, incomplete, inconsistent, resource-interrupted or deviating runs cannot
pass research gates. Preserve stop reasons and any valid partial measurements.

## Resources, artifacts and stopping rule

Use Python **3.12**, Torch **2.8.0**, NumPy **2.0.2**, deterministic CPU execution
and one Torch thread. Run collection and all six learner fits sequentially,
at lower scheduler priority. The admission cap is **1,200 seconds** including
preparation, collection, training, evaluation and aggregation. Check process
peak RSS periodically using platform-correct `resource.getrusage` units; stop
admitting work above **4 GiB**. This sampled guard is not an OS allocation limit
or a measure of other applications' memory. Record costs, checks and peak RSS.

Commit this protocol and capture clean frozen source revision, source/protocol
hashes, command, runtime and all prior input identities before main work.
Preserve collection step/episode logs, visited/support arrays, coverage and
successor-access diagnostics, datasets/transitions, local/global sampling
digests, exposure counts, raw loss/state/episode tables, paired outcomes,
all 30 online/target inference snapshots, final dense predictions and replays.
Snapshots do not promise optimizer resume. Include a SHA-256 manifest and
export the dashboard directly from saved results.

Separate smoke/tests use alternate small banks (CLI smoke starts 720000 and
990000), one learner seed and 24 updates per arm. A replay support smaller
than the fixed 64-state batch must produce an explicit configuration failure;
do not silently change sampling to replacement. No main run may be tuned from
smoke performance.

One main comparison only. No automatic extra collection, training, seeds,
architecture change or memory extension. Preserve negative results. Historical
artifacts and the owner's undecided license remain unchanged.
