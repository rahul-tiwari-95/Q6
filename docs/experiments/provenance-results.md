# Calibrated provenance controls: release pilot

**No controller wins across all three tested settings.** A decayed raw counter has the lowest shortfall in the calibration distribution; unique-origin counting retains better performance under the selected forwarding/reporting shifts. These are exploratory synthetic results, not evidence that provenance is universally necessary or unnecessary.

The [protocol](provenance-protocol.md) was written before the pilot: each tuned controller tested nine thresholds on the same eight calibration seeds, then froze its choice. Evaluation used24 fresh, paired seeds per setting. Raw and decay selected threshold1.0; unique selected.8. No extra decay settings were tuned. Whole-run need-shortfall is primary; false alarms and crisis response are diagnostics.

| Setting | Raw count | Unique count | Bounded decay | Never mitigate |
|---|---:|---:|---:|---:|
| Calibration distribution | 38,304 | 37,877 | **35,389** | 112,524 |
| More forwarding | 40,405 | **40,018** | 40,656 | 112,540 |
| Rarer reports + later shift | 69,302 | **63,609** | 71,085 | 74,713 |

Values are mean need-shortfall agent-ticks per10,000 world ticks; lower is better. Compare policies within each row: the later-shift setting has a different amount of crisis exposure. The loss sums food and medicine shortfalls, rather than counting distinct people.

Seed-paired differences give a more useful assessment than the ranking alone. In the default setting, raw minus unique is427 [bootstrap95% CI −181, 1,109], while decay minus unique is−2,488 [−3,075, −1,843]. Under more forwarding, raw minus unique is387 [179, 610], and decay minus unique is638 [384, 892]. With rarer reports/later shift, the corresponding differences are5,692 [4,576, 6,798] and7,475 [6,344, 8,607]. These are exploratory percentile intervals from2,000 seed resamples, not multiplicity-adjusted discovery tests.

All tuned controllers had zero observed false alarms and zero missed crises in these24-seed samples. That does not establish zero underlying risk. Never-mitigate also had zero false alarms, but missed every crisis and had much larger shortfall. Mean response delay in the rarer-report setting was69 ticks for raw,115 for unique and118 for decay: fast first response alone did not predict the best total outcome. The full report includes every response and uncertainty measure.

The practical finding is narrower and more useful than the original equal-threshold beta story: **calibration and the evidence-generating conditions matter**. A non-lineage baseline can perform well, and distribution shifts can change its relative performance. This motivates comparing estimators over a declared world family before investing in a larger learning architecture. No neural network or new institution was added.

## Reproduce and inspect

```bash
python3 -m no_way_home.run_provenance_release --out experiments/provenance/my-reproduction
python3 -m pytest tests/test_nwh_release.py -q
```

The runner refuses to overwrite an existing run directory. The optional `--dashboard PATH` selects where to export the compact dashboard data. The fixed release command evaluates504 worlds (504,000 ticks), without training.

- [Protocol](provenance-protocol.md) and [historical errata](provenance-errata.md).
- [Generated report](../../experiments/provenance/release-pilot-v1/RESULTS.md).
- [Combined results, intervals and provenance](../../experiments/provenance/release-pilot-v1/results.json).
- [Calibration CSV](../../experiments/provenance/release-pilot-v1/calibration.csv) and [evaluation CSV](../../experiments/provenance/release-pilot-v1/evaluation.csv).
- [Raw evaluation histories](../../experiments/provenance/release-pilot-v1/evaluation-traces.jsonl.gz), [metadata](../../experiments/provenance/release-pilot-v1/metadata.json), and [artifact hash manifest](../../experiments/provenance/release-pilot-v1/manifest.json).

The source snapshot inside the run directory fixes exactly which code generated these artifacts. A packaging-only rerun added repository-relative artifact links and source snapshots; configs, calibration choices, seed outcomes and compressed traces were unchanged.
