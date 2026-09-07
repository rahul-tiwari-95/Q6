# Equal-map replay on fixed collected supports — protocol v1

Status: locally declared before main execution, not externally registered.
Three matched-bank replications found a uniform-support efficiency advantage.
Their collected banks assign very different unique-state counts to maps. This
study tests whether a practical replay change improves the use of that same
experience, without collecting or adding any states.

## Question, comparison and scope

Does **map-balanced minus collected-unique greedy efficient success** improve
on each of the **three archived collected banks**, with support, network,
DDQN procedure and update budget per policy held fixed?

Bank IDs **1, 2, 3** refer to the exact collected supports in
`experiments/bank_replication/pilot_v1`: **59,839 / 59,984 / 59,838** sorted
unique current-state IDs. All use training maps **300000–300255**. Do not
recollect, draw new subsets, add missing states, deduplicate differently or
weight by collector visit frequency. Hash-check input supports and provenance
against the completed archive before training and verify unchanged afterward.

The baseline **collected_unique** consists of its nine archived final
30,000-update policies, seeds **0, 1, 2** per bank. Reevaluate these frozen
policies on the newly selected panels; do not retrain, resume or select a
baseline checkpoint. The treatment **map_balanced** trains nine new policies
from the same seed-specific initial online and target weights. Compare those
initial identities with the baseline archive's update-zero snapshots.

These are three previously drawn support banks, with shared training layouts,
learner initializations and evaluation panels. They are not new independent
collector replications. Report each bank before the equal-bank mean; panel
and learner measurements do not increase the number of bank draws. No
significance, equivalence or population-wide replay ranking is declared.

## Fixed task, data and optimizer

Use unchanged fully visible World A: 5×5, four walls, one pellet, horizon 32,
ordinary action mapping visible initially, 92 observations, gamma 0.97 and
existing rewards/potential shaping. Reuse all **163,840 archived training
states**, original row order, metadata and **655,360 four-action transitions**.
Check their identities against the coverage and bank-replication manifests.
No replacement training dataset or fresh-state regression sweep.

Keep the **20,420-parameter** 92 → 128 → 64 → 4 ReLU network and unchanged
`fixed_targets.fixed_update(..., "double_dqn")`: Adam 0.001, batch 64,
all-four-action Smooth L1 beta 1 averaged over states/actions, gradient norm
clip 5, detached Double DQN targets, terminal reward targets, target tau 0.01.
No architecture, memory, loss, target-access or online-feedback change.

Nonterminal successor queries retain full-bank observations, including states
outside the current-state support. All four actions remain privileged offline
supervision. Exact labels are integrity/diagnostic references and never
select replay states or learned actions.

## The one replay intervention

Baseline training sampled 64 distinct positions uniformly over its full
unique-state support per update using `[learner_seed,66301]`. Its archived
counts and ordered local/global digests remain the control record.

Treatment constructs ascending training-map IDs and each map's ascending
supported global rows. All 256 maps must have at least one supported row;
otherwise stop as a configuration failure. At each optimizer update:

1. From an owned map generator **SeedSequence([learner_seed,99301])**, call
   `choice(256, size=64, replace=False)` once, retaining its returned order.
2. For each selected map in that order, use a separate owned state generator
   **SeedSequence([learner_seed,99302])** to draw one integer from zero through
   that map's supported-row count minus one, using one scalar `integers(K_map)`
   call. Map this within-map rank to the corresponding supported global row.
3. Pass those ordered 64 distinct global rows to the unchanged DDQN update.

Both generators reset per fit and exclude bank ID. Thus the ordered map
schedule matches across banks for a given learner seed. Separate generators
prevent different row-count bounds from changing later map selections.
Within-map ranks and global states need not match across banks. No matching
of baseline/treatment sampled rows is claimed; replay sampling is the change.

This design changes both map exposure and within-batch map diversity. It is
a practical replay-design comparison, not a pure causal estimate of marginal
map weighting. It does not balance clock or goal distance within maps. A map
with only a few visited states may repeat those states more often; retain and
report this consequence instead of adding rows or changing the sampler.

## Work order, sampling and resource evidence

Verify and freeze all three supports, archived controls and input identities;
select all evaluation panels before any update or learned-policy prediction.
Train sequentially by bank ID then learner seed, **30,000 updates per fit**:
**nine fits / 270,000 new updates / 17,280,000 state presentations /
69,120,000 action targets**. The nine baseline fits consumed the same budget
historically, with **zero new baseline updates** in this study.

Perform **zero new collection steps or support draws**. Collection sizes and
costs shown in the dashboard are inherited provenance, not this run's cost.
Record every treatment stream through ordered map-ID, within-map-rank,
support-local-index and global-row SHA-256 digests and their count summaries.
Save full global/local state counts, per-map selection counts and update count.
Independently reconstruct draws in the audit; assert no direct off-support
samples, 64 distinct maps/states per batch and identical map schedules across
banks for each seed. Sample counts must sum to updates × 64.

Compare actual baseline/treatment exposure per map, clock bucket (1–8, 9–16,
17–24, 25–32 and all), winnable/goal-near category and state-count distribution.
Report map exposure range/CV, unique rows sampled and query-weighted
nonterminal successor fractions outside support. Support membership is fixed;
training presentation frequency is a separate quantity. Baseline counts are
archived 30,000-update counts, clearly labeled even during smoke.

Save treatment online/target snapshots at **0, 1,000, 3,000, 10,000, 30,000**:
**45 new snapshots**, plus copies of the **nine archived baseline finals**.
Record source/copy file hashes and module identities; these are inference
snapshots, not optimizer-resume checkpoints. Log treatment loss every 100
updates (**2,700 raw windows**). Do not invent new baseline loss curves or
claim old snapshots were evaluated on these panels.

All nine treatment fits must finish before any learned-policy evaluation.
Only final treatment and archived final baseline policies are evaluated.
No checkpoint selection, training-fit gate, additional tuning or second main
run within this milestone. Nonfinite loss/weights, changed support/model hashes,
missing cells or provenance deviations disqualify evidence and preserve
partial work with the reason.

## Prospective final-policy evaluation

Select **eight new disjoint 64-map panels** in panel order, scanning upward
from **1040000 + 1000 × panel index**. Exclude training layouts, all four earlier
fresh panels (supervised, fixed-target, coverage, equal-size), all **512 frozen
panel-study layouts**, all **512 bank-replication layouts**, and every already
accepted new layout. Identity is walls + pellet + rules, ignoring seed/spawn/
clock. Preserve original spawns and every rejection; no difficulty balancing
or outcome-based replacement.

After all fits, use frozen inference without optimizer/replay/gradients.
Validate online/target and source/copied model hashes before, after and around
every model-panel evaluation, including the archived controls.

Each model/map has one greedy and two epsilon-0.1 episodes, paired external
draws **[learner_seed,map_seed,repetition,55219]** across conditions/banks.
Greedy ties choose the lowest action label. There are **27,648 learner episodes**
(9,216 greedy; 18,432 epsilon). Shared references run once: three random seeds
× two repetitions × 512 maps = **3,072**, plus **512** planner episodes.
Total **31,232 evaluation episodes**. Q* along rollouts remains diagnostic.

## Outcomes and decision

Primary: **map_balanced minus collected_unique greedy efficient-success rate**
for each bank across its 512 maps and three learner seeds; then equal-weight
mean, range and sign count of those three effects. Efficient success means
success within twice the map's planner length, with **all episodes including
failures** in the denominator. Historical uniform-subset performance is context,
not a newly evaluated third arm or a gap-closure estimate on these panels.

Keep success, efficient success and steps separate. Save per-bank/panel/seed
and per-layout greedy pairs: **4,608 layout pairs**, **72 panel/seed plus nine
all-panel/seed rows**, and **24 panel plus three all-panel aggregate effects**.
Do not invent epsilon paired rows. Preserve complete keys and additive counts.
Each bank/condition has 1,536 greedy and 3,072 epsilon episodes.

For success, efficient success and mean steps, equal counts make the equal-bank
mean numerically equal to complete episode pooling. Report blocked-step/total-
step pooled ratios separately from mean within-bank pooled ratios and mean
learner ratios. Successful-only lengths compare different successful subsets.
Show historical 70% success/above-panel-random and 80% efficient-success
reference crossings descriptively, without a new competence gate or rewriting
old gates. Eligibility means integrity/completeness, not a performance pass.

An improvement across banks supports better use of fixed experience via this
replay design; missing support would not be the whole explanation. Mixed or
negative effects leave missing states and within-map composition unresolved;
they do not prove either cause or justify tuning within this run. Keep memory
deferred. Privileged offline access still prevents an online-RL competence
claim or readiness to interpret A → B → A forgetting.

## Dashboard, execution and validation

Add the study to the existing dashboard, showing each bank and equal-bank
primary effect, unchanged support alongside changed exposure, final panels,
treatment loss and source-labeled controls. Preserve all earlier studies and
visible initial rules. Preselect the first accepted map per panel, repetition
zero, every bank × condition × learner in both modes: **288 learner recordings**
plus random seed 0 and planner per panel = **304 recordings**. No outcome-based
trace selection or intermediate-evaluation selector.

Use Python **3.12**, Torch **2.8.0**, NumPy **2.0.2**, deterministic CPU, one
Torch thread, sequential execution at `nice -n 10`. Admission cap **1,200
seconds** includes preparation, training, evaluation and aggregation; sampled
peak-process-RSS cap **4 GiB**. This is not a hard OS limit or total-machine
memory. Preserve partial artifacts and stop reasons when a guard fires.

Review and push this protocol before freezing clean implementation source and
running one main comparison. Capture source/protocol/git/environment/command,
input manifests/identities, data/transitions, fixed supports, archived controls,
selected panels/skips, exposure/counts/digests, losses/snapshots, raw episodes/
pairs, summaries/replays/resources and SHA-256 manifest. Output
`experiments/map_replay/pilot_v1`, export `dashboard/data/map_replay.json`.
Keep license undecided.

Smoke reuses the same three shipped supports and all 256 maps, preserving
64-distinct-map batch semantics. Train only seed 0, **24 updates per bank**:
**72 new updates**, using six treatment snapshots (0/final) and three archived
30,000-update baseline finals. This unequal smoke budget checks execution
only and is ineligible research evidence. Evaluate two two-map panels starting
**1120000 / 1121000**: **72 learner + 12 shared reference episodes**, **28
recordings**. No new collection or support generation. Tests may use explicit
smaller fixtures without changing main or CLI-smoke parameters.

Audit stored evidence and preselected recording observations without retraining
or repeating the complete learned-policy evaluation.
