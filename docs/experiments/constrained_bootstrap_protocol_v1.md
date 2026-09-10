# Constrain recorded-successor bootstrap actions — protocol v1

Status: locally declared before main execution, not externally registered.
Recorded-only supervision reduced efficient success under fixed state replay.
This study changes the bootstrap action set while retaining recorded outcomes.

## Question and arms

Primary: **constrained_bootstrap minus recorded_actions greedy efficient-success
rate**, per bank, then the equal-bank mean. Does restricting training-time
successor action selection to logged actions improve behavior when data,
current-state masks, replay, loss and target count stay fixed?

Control `recorded_actions` reuses **nine final 30,000-update policies** from
`experiments/recorded_actions/pilot_v1`, banks **1, 2, 3**, seeds **0, 1, 2**.
It is not the older four-action control. Treatment `constrained_bootstrap`
trains nine new policies from matching seed-specific online/target initial
weights. No baseline retraining, checkpoint selection or new collection.

Original collected support sizes are **59,839 / 59,984 / 59,838**, on the same
256 maps **300000–300255**. Both arms use identical logged transition tables,
not synthetic within-map replacements. These banks share training maps,
initializations and prospective evaluation panels; crossed cells are not
independent replications. Report each bank before means/ranges/signs. No
significance, equivalence or population-wide method ranking is declared.

## Fixed outcomes, objective and replay

Verify original collector log bytes and manifests, reconstruct deterministic
deduplicated `(state,action)` outcomes, and compare them with the recorded-
action archive's tables. Preserve separate terminated/truncated identities,
rewards, logged successors, absent-entry sentinels, masks and counts. Every
supported state has a recorded action. Repeated visits have no extra loss
weight. Every queried nonterminal successor must have a nonempty logged-action
mask; reject missing masks rather than falling back to unrestricted selection.
There are **zero new collection episodes, steps or support draws**.

Preserve World A, observed movement rule initially, 5×5/four walls/one pellet,
horizon 32, 92 observations, existing rewards/potential shaping, gamma **0.97**,
the **20,420-parameter** 92 → 128 → 64 → 4 ReLU network, Adam **0.001**,
gradient clipping **5** and soft target **tau 0.01**. Per-state loss is the
mean SmoothL1 beta 1 over that state's distinct logged actions, then the mean
over the 64 sampled states. Current-action supervision masks do not change.

Use only logged rewards, ends and successor choices for optimization.
Exhaustive rows resolve recorded observations; full-world transitions/Q* may
be preserved as integrity/rollout diagnostics but never train or select a mask.
Terminal targets use logged rewards and make no successor query.

Replay is unchanged `SupportSampler` / `BatchSampler`, owned
**SeedSequence([learner_seed,66301])**, returning 64 distinct support-local
positions in order per update, with repeats allowed between updates. Reconstruct
the recorded-action control's full 30,000-update stream and verify archived
local/global/map digests and counts. Check treatment streams and counts match
exactly within each bank/seed, including ordered map identities and diversity.
No cross-bank schedule identity is assumed. Every available support state must
be sampled by the full budget, with zero direct off-support samples.

Nine fits × **30,000 updates** give **270,000 new updates / 17,280,000 state
presentations**. Each arm has exactly **24,273,701 supervised action targets**:
**1,062,190 terminal plus 23,211,511 nonterminal targets/queries**. Treatment
bank target totals must be **8,115,408 / 8,078,271 / 8,080,022**, matching the
archived recorded-action controls, with the same per-successor query vectors
and zero queries outside collected support. These are optimizer presentations,
not unique edges, probe inference queries or policy evaluation episodes.

## The sole target-selection change

For each observed nonterminal transition `(s,a,r,s')`, the unrestricted
control selects `argmax_a Q_online(s',a)` over all four action labels.
The treatment selects the online argmax **only over `A_log(s')`**, the
successor's recorded-action set. Evaluate that selected action with the lagged
target network and form `r + gamma * Q_target(s',selected)`, detached.
Ties choose the lowest label among allowed actions. Do not restrict the
target-network evaluator to its own argmax, change target update timing,
add a penalty or weight states/actions by collection frequency.

Current-state predictions still contain four outputs, with only logged
current actions receiving loss. Fresh-world rollout evaluation remains
**greedy over all four actions**, or the same epsilon-0.1 diagnostic. Fresh
states have no logged masks; using a mask during deployment would be a second
intervention and is not allowed.

## Prospective mechanism diagnostics

At every treatment optimizer update, use the same pre-update online and target
successor predictions to compute restricted and unrestricted action choices.
Track the number of nonterminal query presentations, how often unrestricted
argmax is outside `A_log(s')`, the nonnegative online-value gap
`Q_online(s',unrestricted) - Q_online(s',restricted)`, and the **signed** DDQN
target difference between the complete restricted and unrestricted float32
targets. Mathematically this is
`gamma * (Q_target(s',restricted) - Q_target(s',unrestricted))`; saved values
retain rounding from the actual full-target subtraction.
Record counts, sums, extrema and positive/negative/zero target-delta counts,
with a declared 1e-12 diagnostic sign tolerance. Use nonterminal query
presentations as denominator; retain activation-conditioned quantities
separately if included. Ties can change labels with zero online gap. Do not
assume target differences are always negative: selection and evaluation
networks may rank actions differently. Only the restricted target enters loss.

Save raw **100-update diagnostic windows**, **2,700** in the main run, plus
per-fit totals; arithmetic must reconcile counts with actual query exposure.
Do not invent historical diagnostics for frozen controls.

Also freeze one **64-state probe batch per bank/seed before fitting**: recreate
the original sampler's first batch using its seed and support, without
advancing training replay. Use that same probe at each saved treatment
checkpoint **0, 1,000, 3,000, 10,000, 30,000**: **45 probes**. Save source
state/action, logged successor, successor mask, full online/target Q vectors,
choices, online gap, signed target delta and snapshot identity for every
observed nonterminal probe edge. Probe inference is gradient-free, uses no
new outcomes/RNG draws, and does not alter parameters, replay or optimizer
counters. Count these extra inference queries separately. They are not policy
rollout evaluation, optimizer targets or checkpoint-selection evidence.

Full audits regenerate fixed probes from saved snapshots; portable audits
verify formulas from saved Q vectors. All-update window statistics are checked
for complete accounting and arithmetic, not independently reproduced by
retraining. This difference in audit scope must remain explicit.

## Work order, snapshots and evaluation

Verify logs, tables, support, frozen controls and initial identities; freeze
all training inputs/probe rows and select all evaluation panels before fits.
Train sequentially by bank then seed. Save **45 treatment inference snapshots
plus nine copied recorded-action control finals**, and **2,700 loss windows**.
All treatment fits precede any policy rollout evaluation. Only final policies
are evaluated, without optimizer/replay or gradients; preserve source/copy/
online/target identities around each model-panel evaluation. No new full-state
fit sweep, checkpoint tuning or second main run follows the fixed budget.

Select **eight disjoint 64-map panels**, scanning from **1100000 + 1000 × panel
index**. Exclude training, the named supervised/fixed-target/coverage/equal-
support fresh panels, and the full 512-map selections from frozen-panel,
bank-replication, map-replay, within-map and recorded-action studies. Exclude
already accepted layouts. Identity is walls + pellet + rules, ignoring
seed/spawn/clock; retain original spawns and every rejection. No difficulty
balancing or outcome-based selection. This does not claim exclusion of every
historical Q6 task.

Each model/map has one greedy and two epsilon-0.1 episodes, paired external
draws **[learner_seed,map_seed,repetition,55219]**, with lowest-label ties.
**27,648 learner episodes** comprise 9,216 greedy and 18,432 epsilon; shared
references add **3,072 random plus 512 planner episodes**, totaling **31,232**.
Retain rollout Q* diagnostics separately from training.

Efficient success means reaching the goal within twice planner path length,
including failures in the denominator. Each bank/condition has 1,536 greedy
and 3,072 epsilon episodes. Preserve **4,608 greedy layout pairs**, **81 seed
pairs** (72 panel plus nine all-panel), and **27 aggregate pairs** (24 panel
plus three all-panel); do not manufacture epsilon paired rows. Report success,
efficiency, all-episode steps and blocked steps separately. Equal-count bank
means equal episode pooling for success/efficiency/steps, while pooled blocked
ratios, mean bank ratios and mean learner ratios remain distinct. Successful-
only means compare different successful subsets.

Historical 70% success/above-random and 80% efficient-success reference
crossings remain descriptive, not new gates. Eligibility means integrity and
completion, not a performance pass. Preserve older fit/gate results. The old
four-action, synthetic-uniform and map-balanced policies are not arms here;
do not rank them using scores from other panels.

A repeatable benefit would support sensitivity to restricting the bootstrap
action set under matched recorded data and target count. This changes the
backup operator, can exclude genuinely useful actions and leaves unobserved
current-action outputs unconstrained. It does not establish overestimation
or unsupported-action extrapolation as the sole cause. A negative result
would not vindicate unrestricted backups generally. Memory and new online
collection remain deferred; these are offline deduplicated-state experiments,
not demonstrated online-RL competence or interpretable A → B → A retention.

## Dashboard, resources and reproducibility

Add **Constrained bootstrap** to the existing dashboard. Show the recorded-
action archived control, matched state/action/query exposure, exact replay,
training-only constraint, activation/value-gap/signed-target diagnostics,
fixed checkpoint probes and final-panel behavior. Historical controls have
no invented training-time diagnostic curves. Preserve all earlier studies
and initially visible rules. Preselect first accepted map per panel,
repetition zero, both modes/all bank-arm-seed combinations: **288 learner
plus 16 shared reference recordings = 304**, without favorable substitution.

Use Python **3.12**, Torch **2.8.0**, NumPy **2.0.2**, deterministic CPU,
one Torch thread and `nice -n 10`. The user's additional compute allowance
does not require changing the comparable setup. Admission cap **1,200
seconds** includes preparation, training/probes, evaluation and aggregation;
sampled peak-process-RSS guard **4 GiB** is not a hard OS or total-machine
limit. Guard/nonfinite/identity failures preserve partial evidence and
disqualify the run. Review/push this protocol, freeze clean pushed source,
then execute one main comparison.

Capture source/protocol/git/environment/command, manifests/logs/tables,
supports/probe rows, controls/initials/snapshots, state and action/query counts,
sampling/reconstruction, diagnostic windows/probes, panels/rejections,
losses/raw episodes/pairs/summaries/replays/resources and SHA-256 manifest.
Output `experiments/constrained_bootstrap/pilot_v1`; export
`dashboard/data/constrained_bootstrap.json`. License remains undecided.

Smoke uses all three archived supports/tables, only seed 0, **24 updates per
bank = 72**, six treatment snapshots/probes (0/final) plus three archived
30,000-update finals. Two two-map panels start **1180000 / 1181000**, producing
**72 learner + 12 reference episodes** and **28 recordings**. Match the short
state/target/query stream to the historical first-24-update prefix, retaining
the baseline's full 30,000-update labels. Unequal smoke budgets are ineligible
scientific evidence. Tests may use smaller fixtures. Audit saved evidence,
fixed probes and recordings without retraining or full policy reevaluation.
