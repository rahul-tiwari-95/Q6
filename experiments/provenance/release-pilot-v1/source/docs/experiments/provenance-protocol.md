# Provenance control release pilot: protocol

Written 2026-09-06 before running the release pilot. **Exploratory release pilot**, not a new confirmatory claim. Previous Q6 data and the audit informed this protocol; fresh evaluation seeds prevent direct seed reuse but do not erase prior model selection. The original beta result reports remain historical artifacts.

Question: after giving raw-count, unique-origin-count and bounded exponentially decayed-count controllers the same calibration budget, does lineage information improve whole-task shortfall in this synthetic world, and does a selected policy retain its behavior under changed forwarding or reporting/shift settings?

## Fixed design

- Kernel: current `no_way_home.world`, 4 localities ×6 agents, 1,000 ticks, calibration/default regime switch at tick400. No new institution or learning agent.
- Three tuned controllers: raw count/window30; unique-origin count/window30; bounded exponential raw count with half-life15 ticks, floor0 and ceiling2 rate units. The exponential count is normalized by its infinite steady-state weight sum so thresholds use messages/tick. It sees no origin IDs. All act iff score > threshold; physical affordability remains identical.
- Equal tuning budget: each controller evaluates exactly nine thresholds `[0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.00, 1.30, 1.70]` on the same eight calibration seeds31000–31007. Half-life/window/bounds are fixed, not additionally tuned. Select the lowest mean whole-run shortfall; ties use the lowest threshold. This is equal search effort, not a claim of optimal calibration of every possible model.
- Untuned baseline: never mitigate. It exposes the inadequacy of false-alarm rate as a sole objective.
- Evaluation:24 fresh seeds41000–41023 per policy per scenario, paired across policies and scenarios. No held-out result enters calibration.
- Three evaluation scenarios: `default` (same generator as calibration); `forwarding_shift` (forward probability.8 and hub boost10); `reporting_and_timing_shift` (report threshold4 and regime shift tick600). All other values retain defaults. The latter two are uncalibrated parameter shifts, not evidence of broad distributional generalization.
- Primary outcome: whole-run need-shortfall agent-ticks per10,000 world ticks; lower is better. It sums food and medicine shortfall, so one person may contribute twice in a tick. This is a declared physical loss, not a universal welfare measure.
- Secondary diagnostics: any executed pre-shift mitigation (false-alarm flag); total executed false mitigation ticks; first executed post-shift mitigation delay (zero means at the shift tick); missed crisis (no post-shift mitigation). Report response delays including misses as the length of the remaining post-shift window, and separately report the missed fraction so fast responders alone cannot mask misses. Also retain post-shift shortfall and mitigation counts per seed.
- Uncertainty: percentile bootstrap95% intervals,2,000 resamples, fixed bootstrap seed52026. Resample whole seed rows; paired loss contrasts compare each policy minus unique counting on identical seeds. Report exploratory intervals, no discovery claim from p-values and no multiple-comparison significance stars.
- Resource budget:216 calibration runs +288 evaluation runs =504,000 world ticks. No training. Fixed sizes are a bounded pilot budget, not a power calculation.

## Reproducibility and stopping

Portable command: `python3 -m no_way_home.run_provenance_release`. It refuses to replace an existing run directory unless a different `--out` is supplied. Artifacts include protocol hash, source hashes, complete configs and grids, selected thresholds, Python/NumPy versions, seed-level calibration/evaluation CSV+JSON, full evaluation event/message streams, an uncertainty summary and dashboard JSON. Original reports are never rewritten.

Stop and report if nonfinite loss, missing seed/policy/scenario cells, calibration/evaluation seed overlap, score differences from the specified counting formulas, or nondeterministic reproduction occur. Do not add thresholds, retune decay, choose new seeds or redefine diagnostics after viewing this pilot. The next scientific step, if warranted, is a separately declared experiment across a larger family of evidence-generating conditions. Negative or ambiguous results are releaseable outcomes.
