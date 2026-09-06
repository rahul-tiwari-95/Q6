# Q6 / No Way Home implementation and research audit

Audit date: 2026-09-05. Scope: `no_way_home/`, its 32 tests, preregistrations, phase plan, and checked-in result reports. No project source was edited. Git provenance is being investigated separately by the parent agent. No AGENTS.md was found by the parent; this audit also read the NWH phase plan. Findings below distinguish numerical reproduction from interpretation.

## Overall assessment

This is a small, unusually readable synthetic decision laboratory with real executable experiments, sensible controls, explicit negative results, and a useful institutional ablation. It is worth releasing as a transparent research/teaching artifact. It is not yet a multi-world simulator, a spatial gridworld, or a society of independently learning agents. Nor do the existing experiments establish novel provenance discovery or successful online adaptation. The central opportunity is to make a narrow benchmark rigorous and reusable, rather than adding another conceptual layer.

The implemented simulation comprises aggregate shared food/medicine/wealth, four groups of Bernoulli sickness draws, one hidden global regime switch, and one centralized Boolean mitigation action each tick (`world.py:31-67`, `world.py:70-79`, `world.py:93-165`, `world.py:169-193`). There are no agent objects, spatial positions, individual observations, strategic message decisions, lifecycle/population replacement, or independently learning voters. "Multi-agent gridworld" in README/phase plan therefore describes aspiration more strongly than current machinery warrants. The base preregistration does explicitly mark many larger features as a target, which is good scope honesty.

## Strongest existing contributions

1. **A compact, inspectable experimental progression.** The narrative keeps the unbounded-surplus failure, threshold collapse, RNG-stream mistake, weak learner, and missing control visible (`results/README.md`). This can be a useful public case study of how small simulation assumptions create apparent intelligence. The high-value artifact is executable mechanisms plus their failure analysis, not the quantity of formal documents.
2. **A useful regression-tested mechanism ablation.** Re-evaluating mandate permission each tick recovers the hard-coded best-executor behavior in tested settings (`institutions.py:169-182`, `tests/test_no_way_home_smoke.py:414-435`). It supports the conditional claim that term commitment explains this implementation's measured gap. The result is interpretable and inexpensive.
3. **Real numerical beta result, reproducible.** The claimed `[41,41,34,0,0]` fooled counts across 50 seeds reproduce. This audit reconstructed them entirely from never-mitigate pre-shift message trajectories, then verified 50 direct instrument runs against that reconstruction. The data are saved at `/tmp/q6-nwh-beta-paired.json`.
4. **Clean basic interfaces.** Policies share a callable interface; world and policy RNG streams are separated; metrics are computed from logged per-tick events; functions are small. Approximately 1,691 Python lines in 15 NWH modules (2,272 including the two test files) make this realistic to package as a standalone example.

## Research-validity issues that materially change interpretation

### A. The beta monotonic trend is built into the instrument/outcome

`mixed_score=(1-beta)*raw+beta*unique` (`institutions.py:65-90`), while raw count is always at least unique count (`messages.py:81-90`). Therefore every fixed observation's score is nonincreasing in beta. In the pre-shift world severity is exactly zero; mitigating only spends wealth and cannot change sickness or messages (`world.py:38`, `world.py:102-125`). Before the first mitigation each arm has sufficient accumulated money, so the binary event "ever mitigated before the shift" (`run_beta_sweep_v1.py:30-31`) is precisely a threshold crossing on the same message history. Its monotonicity is structural, not an uncertain emergent phenomenon.

Executed check: for seeds 20000-20049, run the never-mitigate policy only until tick799; compute each arm's threshold crossings from those messages. Counts exactly equal `[41,41,34,0,0]`; zero paired rows are nonmonotonic. Ten seeds × five actual instrument policies all match reconstruction. Thus the curve is numerically correct but its major discovery is a particular generator's threshold-crossing location. A coarse beta grid identifies a successful tested point or interval, not a precisely located universal knee.

### B. Equal numeric thresholds do not isolate lineage information's value

The beta sweep changes score magnitude and action conservatism at the same time. The required decayed-count control is acknowledged as unfinished (`run_beta_sweep_v1.py:92-98`), but a more basic missing baseline is a raw counter with its own calibrated threshold. A lower false-positive rate alone cannot establish useful decision quality: never mitigating scores a perfect zero on "fooled" while performing terribly on welfare.

This audit ran a deliberately **exploratory**, small control check on the phase plan's disjoint calibration range 15000-15009 (10 seeds, default 2,000-tick worlds):

| Policy | Fooled / 10 | Mean shortfall/10k |
|---|---:|---:|
| Unique count threshold .30 | 0 | 43,474 |
| Raw count threshold .60 | 2 | 41,367 |
| Raw count threshold .80 | 0 | 39,272 |
| Raw count threshold 1.00 | 0 | 38,156 |
| Never mitigate | 0 | 113,176 |

This does not prove tuned raw counting universally dominates deduplication: thresholds were explored in this audit and no new confirmatory study was done. It does demonstrate that the existing result does not establish that provenance is needed for good performance in this generator. Future comparisons should calibrate all policies under an identical budget and compare welfare plus false positives/true-crisis response delay, preferably across varying forwarding/reporting distributions.

### C. The primary permutation test ignores the paired design

The same 50 seeds are reused for all arms (`run_beta_sweep_v1.py:27,43,47-50`), but `stats.py:60-76` concatenates and freely shuffles all observations. That is an independent-samples permutation, not a seed-blocked paired permutation. The power calculation similarly samples independent Bernoulli arms (`run_power_calculation.py:51-59`), and null-calibration tests simulate independent groups (`tests/test_stats.py:50-64`). They validate a different sampling design.

A paired reanalysis performed in this audit, permuting beta labels within each seed and using the equivalent binary JT linear contrast, also produces p=.0001 at 9,999 permutations. The enormous conditional contrast survives; the design mistake should be corrected rather than suggesting that it makes the observed counts disappear. The Monte Carlo floor is not an exact probability. The implementation's docstring also says returned J is always in natural order, but decreasing alternatives actually reverse groups before returning J (`stats.py:53-55,69-81`).

Primary reference: [SciPy permutation_test documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html) distinguishes independent pooling from within-pair sample-label permutations.

### D. The learner is given deduplication; it does not discover it

`feature_unique_rate` calls the implemented deduplicator and gives its result to the learner (`learning.py:40-45`). The learned policy can discover when to use that feature, which is a valid narrow observation. It cannot establish discovery of provenance checking from raw messages. The earlier provenance report explicitly identified avoiding a free dedup feature as the research question (`results/provenance_test_v1.md:61-65`); the later learner claim changes that question.

The both-feature learner scores60,864 versus43,526 for the aware heuristic—about40% worse—and57,295 for the naive heuristic (`results/learning_citizen_v1.md:5-13`). It slightly improves upon the raw-only learner, but does not demonstrate that learning is necessary or superior. The best need-first baseline from the smoke test (39,317) is omitted from the learner report (`run_learning_citizen.py:48`; `results/smoke_test_v1.md:9`).

Only one training seed sequence is used per learner config (`run_learning_citizen.py:22-31`). Comparing50 versus200 episodes uses nested prefixes of that sequence, not independent training replications (`tests/test_no_way_home_smoke.py:438-478`). Frozen evaluation cannot establish online regime adaptation. Wealth omission is a plausible limitation, but it has not been isolated: the features also omit stocks/history/regime, and gamma=.9 gives a short effective horizon for a budget problem (`learning.py:61-65`). Reporting wealth as *the* explanation overstates the evidence.

### E. "Floored decay converged" is not justified

`alpha=max(.05,1/visits)` is exactly constant .05 after20 updates to a state/action (`learning.py:101-103`). It is not an asymptotically diminishing learning rate and has no standard tabular Q-learning convergence guarantee. The state abstraction is also not Markov in the omitted world variables. Keeping responsiveness with a constant floor may be a practical choice, but two data amounts producing similar outcomes does not establish convergence. [Watkins and Dayan's original Q-learning paper](https://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf) specifies the setting and learning-rate conditions for its convergence result.

The per-episode last transition is also never learned: the update happens on the *next* policy call (`learning.py:94-103`), while `world.run()` stops after its final step (`world.py:190-193`). On the next run empty events suppress the update and overwrite the previous state/action. Executed minimal reproduction: two one-tick worlds, each with shortfall48, yield zero visit-count/Q updates. This is a small bias in a2,000-tick run, but a real lifecycle/API bug to fix before reusing the learner.

### F. Election success is supplied in the voting rule

Voters choose the best calibration-scored candidate with probability.7, otherwise uniformly (`institutions.py:208-215`). All see that same fixed ranking. The mandate is one Boolean evaluation, not24 actual mandate ballots (`institutions.py:194-204`). "Voting works" here means noisy copies of a supplied ranking usually preserve its winner; it does not demonstrate learning expertise, conflicting information aggregation, or emergent institutional competence. "Best" is best on calibration, not a proven oracle-optimal executor. The commitment ablation remains useful under these conditions.

## Engineering and reproducibility gaps

- **Not byte-identical message histories for the same seed.** IDs come from module-global `itertools.count()` (`messages.py:20-24,57-60`), rather than run-local state. Executed check: repeated seed0 worlds have identical event dicts but different message logs/first origin IDs191 versus210. Metrics ignore ID values, so reported numbers still reproduce. This matters for stable event artifacts, run ordering and process parallelism.
- **Observation boundaries are conventions.** Every policy receives the full mutable WorldState, including hidden blight, history, stocks and config (`world.py:191`). The channel registry labels conventions; it does not enforce policy write permissions (`channels.py:30-48`). Testing registered policies does not establish evaluator information can never leak to future policies.
- **Incomplete raw artifact provenance.** Runs return event/message lists in memory, but experiment drivers save aggregated Markdown, not per-seed outcomes, configs, versioned event streams, trained tables/checkpoints or environment hashes (`run_beta_sweep_v1.py:100-103`, `run_learning_citizen.py:133-136`). This falls short of the advertised event-stream invariant, despite metrics themselves being pure.
- **Scaling has not been implemented.** Both rate functions scan the entire message history every call (`messages.py:84,89`); fixed-window counting therefore becomes quadratic in episode duration. There is no batched world state or parallel simulation backend. Rolling deques/origin counters are an obvious first optimization, followed by measured batched stepping. Efficient multi-world simulation is a potential next project, not an existing achievement here.
- **ZI versus ZI-C has the same executed-action law.** The kernel suppresses unaffordable actions for every policy (`world.py:104-105`); ZI-C just declines to propose the action itself (`policies.py:27-38`). Current aggregate near-equality therefore follows from mechanics and independent RNG draws, not evidence that an independently varied no-loss institution had no effect.
- **Package friction.** Root requirements list torch, numpy, gymnasium, matplotlib and pytest but omit scipy required by stats tests (`tests/test_stats.py:14`). A small standalone NWH package could avoid most Q6 dependencies. No license was found in the scanned worktree; parent should verify repository-level licensing before recommending a public release.

## Recommended next step for this thread

Close this as "a reproducible synthetic provenance-and-control lab" before adding corroboration channels or complex institutions. First publish/fix the paired analysis, calibrated raw/EMA controls, whole-task outcomes and uncertainty, terminology, per-seed artifacts, run-local IDs and learner transition lifecycle. If the dedup advantage survives varied report/forward regimes and equally tuned baselines, test a learned estimator on actual message sequences with repeated training seeds and held-out world families. If it does not survive, publish that negative result: it is a better contribution than decorating an easy threshold problem with a neural network.

For the user's desired multi-world direction, this tiny reference kernel can be one correctness oracle in a separate batched simulation project. The deliverable should be a fast backend that exactly matches reference transitions under keyed randomness across heterogeneous world parameters, plus throughput/memory benchmarks. Do not couple that first systems milestone to new social mechanisms or nested neural architectures.

## Validation status

Executed: 50-seed analytic reconstruction of beta outcomes;50 direct instrument checks;9,999 within-seed permutations;50 exploratory threshold-control runs; deterministic-ID and terminal-learning micro reproductions. Full existing NWH pytest suite is running separately; final result to be appended when complete. No expensive new training experiment was initiated beyond the repository's existing tests.

Final existing-suite result: `python3 -m pytest tests/test_no_way_home_smoke.py tests/test_stats.py -q` — **32 passed in 121.23s**. The initial environment emits an urllib3/LibreSSL warning, but tests pass. Consolidated reproducible audit command: from repository root, `python3 /tmp/q6-nwh-reproduce.py`; captured output is `/tmp/q6-nwh-reproduction.txt`. Per-seed beta outcomes and exploratory-control metrics are retained as `/tmp/q6-nwh-beta-paired.json` and `/tmp/q6-nwh-exploratory-controls.json`. No new EMA control, multi-training-seed learner study, or out-of-distribution world experiment was run; those remain untested.
