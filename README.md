# Q6

**Small, inspectable experiments in learning across changing worlds.**

Q6 studies how a neural agent adapts when a task changes, and what it retains when an earlier task returns. The main experiment follows **A → B → A** in a compact collection world. A separate No Way Home experiment compares ways of counting repeated evidence in a synthetic resource-allocation task.

This is a public research preview; a license has not yet been selected. Its earlier Krishna–Hunter self-play work, failed experiments, and methodological corrections remain available as a research history.

**Latest finding:** equal-map replay produces **mixed results** on the same three collected banks: efficient-success changes of **−3.91, +7.16 and +4.56 percentage points**. The equal-bank average improves **42.17% → 44.77%**, but bank 1 worsens, so this is not a reliable replacement for the original replay. Exposure balancing worked; it also repeated sparse-map states much more often. Next: test within-map state composition while preserving exact map quotas and replay schedules. Read the [map-replay result](docs/experiments/map_replay_results_v1.md). This remains privileged offline evidence, not online-RL competence.

## Start on CPU

Use Python 3.10 or later; Python 3.12 is a practical starting point. Run commands from the repository root.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

# Short execution check; writes to a new temporary output directory.
python -m q6.map_replay --output /tmp/q6-map-replay-smoke \
  --protocol-file docs/experiments/map_replay_protocol_v1.md --smoke
```

Choose a fresh `--output` directory when repeating a run. This smoke reuses three shipped supports and archived 30,000-update controls, trains three treatments for 24 updates each, and evaluates four alternate maps. Its unequal budgets check execution only; it is ineligible research evidence.

## Experiments

| Track | Question | Start here |
| --- | --- | --- |
| **Competence → adaptation — primary** | Can equal-map replay improve use of the same collected states? | [Current protocol](docs/experiments/map_replay_protocol_v1.md), [results](docs/experiments/map_replay_results_v1.md), [`q6/`](q6/) |
| **Provenance — bounded companion** | Does deduplicating message origins improve decisions compared with independently calibrated raw and decayed counts? | [Protocol](docs/experiments/provenance-protocol.md), [errata](docs/experiments/provenance-errata.md), [`no_way_home/`](no_way_home/) |

Reproduce the current replay comparison in a fresh directory:

```bash
# Match the main run's core dependencies in Python 3.12.
python -m pip install 'torch==2.8.0' 'numpy==2.0.2'
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 python -m q6.map_replay \
  --output experiments/map_replay/my-reproduction \
  --protocol-file docs/experiments/map_replay_protocol_v1.md
```

The [captured environment](experiments/map_replay/pilot_v1/environment.txt) records remaining versions. The runner preserves archived supports and control models, completes all nine treatment fits before evaluation and saves 45 new snapshots plus nine copied baseline finals. It runs sequentially on one CPU thread with a sampled 4 GiB process-RSS guard and a 1,200-second admission cap. The main comparison took **2.97 minutes** and peaked at **0.449 GiB**. No new collection or baseline training occurs. To display a reproduction locally, add `--dashboard dashboard/data/map_replay.json`; this replaces the viewer's current data, so preserve the original artifact. The [bank-replication protocol](docs/experiments/bank_replication_protocol_v1.md) reproduces the preceding support comparison.

Run the fixed-budget provenance pilot:

```bash
python -m no_way_home.run_provenance_release \
  --out experiments/provenance/my-reproduction
```

Stored studies: [map replay v1](experiments/map_replay/pilot_v1/results.json), [bank replications v1](experiments/bank_replication/pilot_v1/results.json), [panel robustness v1](experiments/panel_evaluation/pilot_v1/results.json), [equal-size banks v1](experiments/equal_support/pilot_v1/results.json), [state coverage v1](experiments/coverage/pilot_v1/results.json), [paired fixed-data targets v1](experiments/fixed_targets/pilot_v1/results.json), [exact supervision v1](docs/experiments/supervised_results_v1.md), [A-only competence v1](docs/experiments/competence_results_v1.md), [adaptation v2](docs/experiments/adaptation_pilot_v2.md), [archived adaptation v1](docs/experiments/adaptation_pilot_v1.md), and [provenance](experiments/provenance/release-pilot-v1/RESULTS.md). Both adaptation pilots failed initial competence; their later A→B→A outcomes do not establish forgetting or recovery. The A-only RL diagnosis learned its selected fixed tasks but failed fresh-world competence. Subsequent studies establish useful offline behavior under broad coverage, first with exact targets and then bootstrapping. The coverage control found weaker transfer from collected states on its panel. At equal bank size, uniform support improves route efficiency while success is close; the repeated collected control exposes panel sensitivity. The prospective eight-panel evaluation found an efficiency advantage in every panel mean for the original policies. Three new bank pairs replicated that advantage. Equal-map replay on those fixed supports now gives mixed behavior improvements despite successful balancing; the next control holds map exposure fixed while replacing states within each map. Each milestone stops at its declared budget. Runners refuse to overwrite an existing study directory; read the protocol before changing its seeds or budget.

## Inspect results

```bash
python3 -m http.server 8080 --bind 127.0.0.1
```

Open **[the current lab](http://127.0.0.1:8080/dashboard/lab.html#coverage)** and select **Experience coverage → Map-balanced replay**. Compare each bank's effect before the equal-bank mean. Use the map/learner controls to see actual presentation shares, repeated-state counts and clock/category exposure alongside unchanged support membership. The primary contrast stays greedy when replay mode changes. Inspect **304 preselected recordings**, including reversals, with the active rule and learned/exact values visible initially.

The study selector preserves **Bank replications**, **Panel robustness**, **Equal-size banks** and **Exhaustive vs collected**. All other experiment tracks and **1,098 earlier recordings** remain accessible. The [historical registry](http://127.0.0.1:8080/dashboard/index.html) preserves older summaries. To index legacy training runs on your machine, run `python dashboard/scan.py` before starting the server.

Historical summaries are not a complete artifact archive: many referenced CSV files, checkpoints, and replays are absent from a fresh checkout. Claims from the earlier reports should be read with the [September 2026 review](research_review/2026-09-05/README.md), which documents confounds, bugs, and unsupported interpretations. The new pilots also need independent seeds and fair baselines before supporting general conclusions. No Way Home is a synthetic decision model, not empirical evidence about human institutions.

## Develop and contribute

- [Contribution guide](CONTRIBUTING.md): fast and full test commands, experiment template, and reproducibility requirements.
- [Documentation index](docs/README.md) and [roadmap](docs/roadmap.md): current scope and criteria for the next experiment.
- [Map-replay validation](docs/validation/map-replay-v1.md), [bank-replication validation](docs/validation/bank-replication-v1.md), [panel-evaluation validation](docs/validation/panel-evaluation-v1.md), [equal-size validation](docs/validation/equal-support-v1.md), [coverage validation](docs/validation/coverage-v1.md), [fixed-target validation](docs/validation/fixed-targets-v1.md), [supervised validation](docs/validation/supervised-v1.md), [competence validation](docs/validation/competence-v1.md) and [earlier preview validation](docs/validation/2026-09-06.md): tests, CI, artifact audits and browser coverage.
- [Training supervisor](orchestrator/README.md): local job logging, crash restart, and resume conventions.
- [Citation metadata](CITATION.cff): cite the repository and the exact revision/artifact used.

The package installs the current experiment modules and selected legacy support modules. Historical trainer scripts and the dashboard are documented as commands run from this checkout; their APIs are not a stable library contract.

## Research history

The [version index](versions/README.md), [articles](articles/), [original Q6 narrative](Q6.md), [No Way Home design](Q6%20No%20Way%20Home.md), and [archive](archive/README.md) preserve the project's progression. Older ablations and the PPO port live on the [`v7-ablations`](https://github.com/rahul-tiwari-95/Q6/tree/v7-ablations) and [`v8`](https://github.com/rahul-tiwari-95/Q6/tree/v8) branches. These are historical experiments, not interchangeable controlled comparisons. The [`longer_memory/`](longer_memory/README.md) folder contains old planning notes, not an implemented memory architecture.
