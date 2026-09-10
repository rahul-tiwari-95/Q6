# Equal-size collected versus uniform state banks — protocol v1

Status: locally declared before main execution, not externally registered.
The [coverage result](coverage_results_v1.md) motivates this comparison:
collected support improved familiar-start success but weakened fresh transfer.
That intervention changed both unique-state count and state composition.
This study fixes the count while preserving the offline DDQN procedure.

## Question and limits

With exactly **59,626 current states** in each bank, does a uniformly selected
subset of the exhaustive bank transfer differently from the archived states
visited by random trajectories?

Equal size removes unique-row count and average presentations per row as
between-arm differences. It does not match map, clock, goal-near or transition-
graph composition. A uniformly scattered subset can require more successor
queries outside its direct training support. Both arms still receive all four
counterfactual transitions at each sampled state and detached successor queries
from the full bank. This is not trajectory-only or continuous online RL.
No new collection, architecture, memory, loss, replay-frequency weighting,
policy feedback or adaptive support selection enters the main comparison.

## Fixed world and learner

Keep fully visible World A: 5×5, four walls, one pellet, horizon 32, ordinary
controls visible initially, 92-value observation, discount 0.97 and unchanged
rewards/potential shaping. Use the same **20,420-parameter** feedforward network,
92 → 128 → 64 → 4 with ReLU, untrained initializations **0, 1, 2**, paired across
arms. Reuse `q6.fixed_targets.fixed_update(..., "double_dqn")` unchanged.

Adam learning rate 0.001, batch size 64, all-four-action Smooth L1 beta 1
averaged over states/actions, gradient norm clip 5. Online selects each
nonterminal successor's argmax; the lagged target network evaluates it.
Targets are detached; success/timeout labels equal reward. Soft target tau
0.01 follows each optimizer update. Exact Q* labels remain diagnostic only.

## Banks fixed before optimization

Reuse the preceding full training bank from
`experiments/coverage/pilot_v1`: **163,840 rows**, maps **300000–300255**, original
map/position/clock row ordering, and all **655,360** four-action transitions.
Validate training arrays and transitions against the archive before learning.
Hypothetical valid states and deadline-impossible states remain eligible for
the exhaustive reference and uniform subset, as in the preceding protocol.

Conditions, in fixed execution order:

1. **collected_unique:** use the archived sorted `support_indices` from
   `collection.npz` exactly, with **59,626** rows. Validate them against the
   archived positive visited-count mask. Repeated visits are provenance only,
   not replay weights. Do not recollect or alter the bank.
2. **uniform_subset:** create an owned generator with
   `np.random.default_rng(np.random.SeedSequence([88301]))`; make exactly one
   `choice(163840, size=59626, replace=False)` call, sort ascending and store
   global row IDs as int32. Use no prior/fresh predictions, Q* values, distances,
   outcomes, balancing, rejection sampling or additional draws in selection.

The support RNG is independent of learner initialization, batches and
evaluation. Persist both sorted supports and their array hashes before
optimization. Every learner initialization shares these same two fixed banks.
Three learner seeds are not three uniform-subset or collector replications.
Do not redraw based on overlap, composition, graph structure or performance.

## Matched sampling and budget

Train **30,000 updates per arm and initialization**, with checkpoints
**0, 1,000, 3,000, 10,000, 30,000**. Each arm uses the existing BatchSampler
with `SeedSequence([seed,66301])` over its 59,626 local support positions,
sampling 64 distinct positions per update, allowing repetition between updates.
Map local positions to the arm's sorted global row IDs.

Within each seed the **ordered local index digest and local count vector must
match exactly across arms**. Global indices and per-global-row exposures
intentionally differ because bank contents differ. A matched local rank is not
the same physical state or a semantic state pairing. Save both ordered local
and global digests, both count vectors and the supports used to map them.

Totals: **180,000 optimizer updates**, **11,520,000 current-state presentations**
and **46,080,000 action targets**, six sequential fits. Each model receives
1,920,000 state presentations; average presentations per unique bank row are
identical. These repeated offline examples are not environment interactions.
The archived collector's 4,096 episodes / 99,814 steps are a historical input
cost; **new collection episodes and steps are zero**.

The collected arm must reproduce the previous coverage study's final online
AND target weights, initial weights and global sampling digest/count vector
under the pinned runtime. Check all three seeds. A mismatch is an execution-
consistency failure and ineligible evidence. This control is not independent
training replication. Shared-runner refactoring must preserve the old CLI
and historical studies; never edit their captured sources or results.

## Successor access and support diagnostics

For each sampled current state retain all four rewards, terminal flags and
successor indices into the full training observation bank. Query nonterminal
successors even when outside that arm's support. Do not truncate them, replace
their values with zero, drop the edge or add them to current-state support.
Verify zero direct sampling outside each arm's own bank. No fresh observation,
label, transition or outcome may enter optimization.

Before training, describe each bank using membership, not invented visit
counts: included/full states globally and per map; buckets **1–8, 9–16, 17–24,
25–32**, plus all; included/full winnable states; included/full goal-near states,
where physical shortest-path distance is at most two regardless of clock.
These labels describe the already selected supports and never guide selection.
Report intersection count, union count, fraction of each bank in common and
Jaccard overlap. Keep archived collector visit provenance separately labeled.

For each bank report all-action edges, terminal/nonterminal edges, nonterminal
edges ending outside its own support and distinct total/outside destinations.
Use nonterminal edges as the outside-edge fraction denominator. Repeated
optimizer-weighted successor queries, if reported, are separate exposure
statistics computed from saved global sampling counts.

## Fresh panel and evaluations

Choose **64 fresh layouts** by scanning upward from **960000**, rejecting
training layouts, all layouts from preceding supervised/fixed-target/coverage
fresh panels (930000, 940000, 950000 banks), and previously accepted duplicates.
Layout identity is walls + pellet + rules, ignoring start, clock and seed.
Selection happens before any learner predictions and is independent of outcomes.
Preserve candidate skips/reasons and verify zero layout/full-observation
train/fresh intersection and zero overlap with all three previous fresh panels.
Enumerate all **40,960 fresh states** for diagnostic evaluation only.

At every checkpoint evaluate all 256 original training starts and 64 fresh
starts: one greedy plus two epsilon-0.1 episodes per map/seed, with existing
independent draws keyed by seed, map, repetition and 55219, shared across arms
and checkpoints. Greedy ties choose the lowest action label. Evaluation changes
no weights, optimizer state or learner sampling RNG.

Evaluate full training/fresh states using the existing MAE, RMSE, signed bias,
95th percentile of per-state four-action MAE, action regret and winnable
optimal-action agreement (absolute Q* tolerance 1e−6, relative tolerance zero).
Impossible states remain in regression errors and outside winnable agreement.
Preserve additive components and map/time buckets for exact aggregation.

Also report **final-only own-support and outside-own-support** training-state
MAE and winnable agreement from saved dense predictions, with per-seed/pooled
counts. Each condition uses its own support mask. This is a declared diagnostic,
not a new subset-fit gate; the common fit gate below still uses the full bank.

Shared random references use seeds 0/1/2, two episodes per map; shortest path
uses one deterministic episode per map on both panels. Archive first-map,
first-repetition learner replays for every arm, checkpoint, panel, mode and
seed, plus the shared first reference recordings. Selection precedes outcomes.
An individual replay need not reflect the full-panel comparison.

## Outcomes and interpretation

Primary outcome: **final fresh greedy success per seed**, and paired
**uniform_subset minus collected_unique** differences on the same seed/layout.
A condition passes if **every seed reaches 70% and exceeds pooled random fresh
success**. Do not select a different checkpoint or action mode afterward.

Retain the existing separate diagnostics:

- **Training fit:** at least 90% original-start greedy success AND 90% full-bank
  winnable optimal-action agreement in every seed. The full bank includes rows
  outside both subsets, so this alone is not a test of fitting either subset.
- **Efficient fresh success:** at least 80% of all fresh greedy episodes in
  every seed both succeed and use at most twice their own map's planner length.
  Include failures in the denominator. Preserve all-episode and successful-only
  lengths/blocked fractions with explicit denominators.

Save per-seed/per-layout success, length, no-op-rate and efficient-success
paired differences, including binary per-layout efficient-success deltas.
Report seed ranges descriptively; no significance or equivalence claim follows
from three initializations sharing one subset draw and panel.

A better uniform subset would show that unique-row count alone is insufficient
to explain this between-bank difference. Composition includes map/clock/goal
coverage and successor-graph structure; it does not isolate a single component.
If both sparse banks struggle, that does not prove row count caused the gap.
If collected matches or beats uniform, do not infer all collectors are adequate.
Primary gates, continuous paired differences and efficiency must be read
together. This new fresh panel is not interchangeable with prior panels;
collected control scores can change even with bit-identical weights.

Use the result to choose a subsequent bounded data/support control or an
independent bank replication. Do not add memory, online feedback, more training
or a retuned collector in response to these outcomes.

## Resources, provenance and stopping rule

Python **3.12**, Torch **2.8.0**, NumPy **2.0.2**, deterministic CPU execution,
one Torch thread, sequential fits at reduced scheduler priority. Admission cap
**1,200 seconds** includes preparation, support selection, optimization,
evaluation and aggregation. Sample platform-correct process peak RSS and stop
admitting work above **4 GiB**. This is not an OS allocation limit or total
machine memory. Preserve stop reasons and valid partial measurements.

Commit/push the protocol, independently review it, then freeze clean source
before the main run. Capture source/protocol hashes, git revision, environment,
command, archive/input identities, selected supports, composition/graph
metrics, data/transitions, local/global sampling counts/digests, all 30 online/
target inference snapshots, final dense predictions, raw loss/state/episode
logs, paired outcomes, own/outside diagnostics and 124 main replays. Snapshots
do not promise optimizer resume. Preserve a SHA-256 manifest and export the
dashboard directly from saved results.

Output `experiments/equal_support/pilot_v1`, dashboard
`dashboard/data/equal_support.json`; preserve the preceding coverage result
and all older dashboard tracks. Keep license undecided.

Separate smoke/tests use alternate banks: CLI smoke training scan **730000**,
fresh scan **991000**, two maps per panel, seed 0, 24 updates per arm.
Use the sorted every-third-row support as a clearly synthetic collected-bank
fixture, then a uniform subset of equal size; no collector runs. This fixture
checks plumbing only. Smoke/deviating/incomplete/inconsistent/resource-stopped
runs are ineligible for all research gates. Supports smaller than 64 are explicit
configuration failures, never a silent switch to replacement sampling.

**One main comparison only.** No extra subset draws, collection, seeds,
training extensions, checkpoint selection or architecture changes.
