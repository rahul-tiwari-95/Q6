# Independent support-bank replications — protocol v1

Status: locally declared before main execution, not externally registered.
The frozen-policy panel study found uniform-minus-collected efficient success
positive on all eight panel means. Those evaluations still shared one
collector history and one uniform-subset draw. This study replicates bank
construction before choosing a collection/replay intervention.

## Question and unit of replication

Does the **uniform-minus-collected greedy efficient-success advantage** recur
across **three new independently randomized bank pairs**, with equal unique
state count within each pair and the same DDQN learning procedure?

Bank IDs are **1, 2, 3**, all new; the earlier bank is historical context only.
Each pair shares three learner initializations and all evaluation panels.
Independent RNG realizations on the same training layouts are collector/support
replications, not new world-family or training-layout samples. Three banks
remain a small descriptive replication. No significance, equivalence or
population-wide sampler ranking is declared.

## Fixed task and learning procedure

Use fully visible World A, unchanged: 5×5, four walls, one pellet, horizon 32,
ordinary action mapping visible initially, 92 observations, gamma 0.97 and
existing rewards/potential shaping. Reuse the archived **163,840 training
states** on maps **300000–300255**, original row order and all **655,360
four-action transitions** from `experiments/coverage/pilot_v1`. Check arrays,
metadata and transitions against their archived manifest before optimization.
Do not regenerate a different training bank.

Both conditions start untrained with the same **20,420-parameter**
92 → 128 → 64 → 4 ReLU network, learner seeds **0, 1, 2**. Preserve
`q6.fixed_targets.fixed_update(..., "double_dqn")`: Adam 0.001, batch size 64,
all-four-action Smooth L1 beta 1 averaged over states/actions, gradient norm
clip 5, detached Double DQN successor targets, terminal target equal to reward,
soft target tau 0.01 after each optimizer update. No architecture, memory,
loss, replay-frequency weighting or online policy-feedback change.

## Construct and freeze all bank pairs

Before **any optimizer update or learner-policy evaluation**, complete all
three collectors and all three paired uniform draws. Iterate bank ID, ascending
training map seed, repetition **0–15**. Reset each episode to that map's
original sampled start/full horizon and stop at success or timeout.

Each bank uses **4,096 complete collection episodes**, at most **131,072 actual
world steps**; all three total **12,288 episodes**, at most **393,216 steps**.
Do not extend collection to hit a desired unique-state count, stop early on
coverage, or recollect based on outcomes. Realized counts are measurements.

An owned RNG per episode is initialized with
**SeedSequence([map_seed, repetition, 77301, bank_id])** and draws
`rng.integers(4)` once per actual action. Appending bank ID defines new streams;
the old three-component stream remains unchanged for historical callers.
No oracle, trained policy, action mask or learner-seed dependence enters
collection. Record every pre-action global state ID, clock, action, reward,
next state ID and termination flag, and the original ordered episode history.
Terminal next index is −1. Preserve the full visited-count vector; its sorted
positive mask defines `collected_unique`. Repeated visits are provenance,
never replay weights. Position plus remaining clock defines state identity.

For each realized size **K_b**, create one owned generator with
**SeedSequence([88301, bank_id])**, call
`choice(163840, size=K_b, replace=False)` exactly once, sort ascending and save
int32 global IDs as `uniform_subset`. Draw independently of collection,
learner sampling, predictions and labels. No balancing/rejection/redraw.
If K_b cannot fit the 64-state batch, stop as a configuration failure; never
silently sample with replacement or add states.

Save all six supports and their hashes before learning. Match size within each
pair; do not require or imply identical sizes across banks. Record within-pair
overlap/Jaccard and between-bank support intersections. Describe every bank's
map/clock/winnable/goal-near coverage using the prior definitions, plus
nonterminal edges and distinct destinations outside its own support. Labels
describe already selected supports and never guide selection.

## Matched updates and privileged access

Execute sequentially by bank ID, then collected/uniform condition, then learner
seed. Train **30,000 updates per model**, **18 fits** total: **540,000 updates**,
**34,560,000 sampled current-state presentations** and **138,240,000 action
targets**. These repeated offline presentations are not collection steps.

Reuse `SupportSampler`/`BatchSampler` with **SeedSequence([learner_seed,66301])**,
64 distinct local support positions per update, allowing repetition between
updates. Within each bank/seed pair, ordered **local** index digests/count
vectors must match across conditions. Sorted supports map these local ranks
to intentionally different global rows. Different K_b changes sampling across
banks; do not claim cross-bank schedules match. Record local/global ordered
digests and counts and verify zero direct sampling outside each own support.

All four rewards, terminal flags and successors remain available per sampled
state. Detached nonterminal successor queries use the **full training bank**,
including observations outside current-state support. Do not truncate these
edges, substitute zero targets, expand support or introduce fresh states into
optimization. Exact labels are integrity/diagnostic references only.

Archive inference snapshots at **0, 1,000, 3,000, 10,000, 30,000** updates,
online and target hashes, initial identities and loss summaries every 100
updates. Preserve **90 snapshots**; no optimizer-resume guarantee. Initial
online/target identities must match the prior learner for each seed across all
bank/condition combinations. Do not select a checkpoint based on outcomes.

## Prospective evaluation, final checkpoints only

Select **eight new disjoint 64-map panels** before training or predictions,
panel_0…panel_7 in order, scanning upward from **1020000 + 1000 × panel index**.
Exclude the training bank, all four preceding fresh panels (supervised,
fixed-target, coverage, equal-size) and **all 512 layouts in the frozen-policy
panel study**, plus every already accepted layout. Identity is walls + pellet
and rules, ignoring spawn, clock and seed. Preserve original spawn and every
candidate rejection. No difficulty balancing or outcome-driven replacement.

After all 18 fits finish, evaluate only final 30,000-update policies, without
updates or replay access. Validate online/target/snapshot identities before,
after and around each model's panel evaluation. This changes evaluation timing
from earlier training studies, not update mathematics or training budget.

Each model/map receives one greedy and two epsilon-0.1 episodes, with existing
external draws **[learner_seed,map_seed,repetition,55219]** shared across banks
and conditions. Greedy ties choose the lowest action label. There are
**27,648 learner episodes** (9,216 greedy; 18,432 epsilon). Shared references
run once, not once per bank: three random seeds × two repetitions × 512 maps
= **3,072**, plus **512** planner episodes. **31,232 total evaluation episodes**.

No full-state training/fresh regression sweep or training-fit gate is added.
Retain existing rollout Q* diagnostics; exact values measure actions but do
not choose learned actions. Primary evidence is behavior of the final policies
across independent bank draws, not an updated claim about full-state fit.

## Outcomes and aggregation

Primary: **uniform minus collected greedy efficient-success rate**, separately
for each bank, pooled across its 512 maps and three learner seeds. Efficient
success means success within twice that map's planner length; **all episodes**,
including failures, enter the denominator. Report the equal-weight mean of
the three bank effects and their min/max/sign counts. Never hide a bank reversal
inside the pooled result or treat 18 models as 18 independent bank draws.

Also save per-bank/panel/seed and per-layout paired greedy success, efficient
success, steps and blocked-fraction differences. Preserve complete keys and
additive counts. Report pooled blocked steps/total steps separately from mean
per-seed or per-episode blocked fractions. Successful-only length is descriptive
because successful subsets differ. Keep success and route efficiency visible
separately, with exploratory-mode summaries distinguished from greedy results.

Each bank/condition has 1,536 greedy and 3,072 exploratory episodes. For
success, efficient success and mean steps, equal counts make equal-bank
aggregation numerically equal to complete episode pooling. Blocked-step ratios
retain their explicitly labeled denominators. Bank remains the replication
unit. Per-panel differences share
trained policies; per-learner differences share banks. Save **4,608 greedy
per-layout pairs**, **72 bank/panel/seed** pairs, **24 bank/panel** pairs,
**nine bank/all-panel/seed** pairs and **three bank/all-panel** effects, plus
clearly labeled across-bank summaries. Do not invent exploratory paired rows.

Show historical 70% success/above-panel-random and 80% efficiency reference
crossings descriptively, with no new competence gate or change to old gates.
Evidence eligibility requires complete expected work, paired-sampling/model
integrity, declared resource/provenance checks and no deviations. Eligibility
is an integrity condition, not a performance result.

If effects recur across banks, a subsequent bounded composition or
collection/replay intervention becomes better motivated. If effects reverse
or vary strongly, describe which bank properties co-vary without naming one
as the cause; do not draw extra banks or tune the collector in this run.
Keep memory deferred. Privileged offline action access still prevents a claim
of online-RL competence or readiness to interpret forgetting in A → B → A.

## Dashboard, resources and stopping

Show each bank pair and the explicitly equal-bank pooled result in the existing
dashboard, with bank size/collection cost, coverage, final panel comparisons
and recorded policies. Preserve earlier studies and active rules at frame zero.
Preselect the first accepted map per panel, repetition zero, every bank ×
condition × learner seed in both modes: **288 learner recordings**, plus
random seed 0 and planner per panel: **304 total recordings**. No outcome
selection or checkpoint selector implying intermediate policy evaluation.

Use Python **3.12**, Torch **2.8.0**, NumPy **2.0.2**, deterministic CPU, one
Torch thread and sequential work at `nice -n 10`. Main admission cap is
**1,200 seconds** including preparation, collection, learning, evaluation and
aggregation. Stop admitting work above sampled **4 GiB process peak RSS**;
this is not a hard OS limit or total-machine memory. Preserve partial records,
stop reasons and all completed work; incomplete/deviating runs are ineligible.

Commit/push and independently review this protocol, then freeze clean source
before the single main run. Capture source/protocol/git/environment/command,
input manifests, training data/transitions, selected panels/skips, collector
histories, all supports/coverage, local/global sampling, loss windows, snapshots,
raw episodes/pairs, summaries, replays, resource records and SHA-256 manifest.
Output `experiments/bank_replication/pilot_v1`; export saved results directly
to `dashboard/data/bank_replication.json`. Keep license undecided.

Smoke uses **three bank pairs**, two alternate training maps starting **740000**,
16 collector episodes/map, one learner seed, **24 updates per model**, and two
two-map panels starting **1100000** and **1101000**. It performs 96 collection
episodes, 144 optimizer updates, 72 learner evaluation episodes and 12 shared
references, with **28 recordings**. Smoke remains ineligible main evidence.
Tests may use smaller explicit fixtures without changing main parameters.
Audit saved evidence and preselected replay observations without repeating
the complete final-policy evaluation or running another training comparison.
