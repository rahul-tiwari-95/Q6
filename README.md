# Q6

**Small, inspectable experiments in learning across changing worlds.**

Q6 studies how a neural agent adapts when a task changes, and what it retains when an earlier task returns. The main experiment follows **A → B → A** in a compact collection world. A separate No Way Home experiment compares ways of counting repeated evidence in a synthetic resource-allocation task.

This is a public research preview; a license has not yet been selected. Its earlier Krishna–Hunter self-play work, failed experiments, and methodological corrections remain available as a research history.

**Latest finding:** with the same data, batches, network, loss and update budget, **exact targets reach 84.38% fresh-layout success and Double DQN targets reach 86.46%**. Every seed in both conditions passes the declared offline fresh-success gate. Double DQN produces more efficient successes (66.15% versus 39.06%), although both miss the stricter 80% efficiency criterion and every-seed training-fit diagnostic. Bootstrapping works under broad fixed coverage; the next diagnosis focuses on state coverage and online collection/replay before memory. Read the [controlled result](docs/experiments/fixed_targets_results_v1.md). This remains privileged offline evidence, not online-RL competence.

## Start on CPU

Use Python 3.10 or later; Python 3.12 is a practical starting point. Run commands from the repository root.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

# Short execution check; writes to a new temporary output directory.
python -m q6.fixed_targets --output /tmp/q6-fixed-targets-smoke \
  --protocol-file docs/experiments/fixed_targets_protocol_v1.md --smoke
```

Choose a fresh `--output` directory when repeating a run. A smoke run checks execution and artifacts; its small training budget does not establish adaptation or retention.

## Experiments

| Track | Question | Start here |
| --- | --- | --- |
| **Competence → adaptation — primary** | With coverage fixed, can bootstrapped targets learn useful fresh-layout behavior? | [Current protocol](docs/experiments/fixed_targets_protocol_v1.md), [results](docs/experiments/fixed_targets_results_v1.md), [`q6/`](q6/) |
| **Provenance — bounded companion** | Does deduplicating message origins improve decisions compared with independently calibrated raw and decayed counts? | [Protocol](docs/experiments/provenance-protocol.md), [errata](docs/experiments/provenance-errata.md), [`no_way_home/`](no_way_home/) |

Reproduce the current paired target comparison in a fresh directory:

```bash
# Match the main run's core dependencies in Python 3.12.
python -m pip install 'torch==2.8.0' 'numpy==2.0.2'
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 python -m q6.fixed_targets \
  --output experiments/fixed_targets/my-reproduction \
  --protocol-file docs/experiments/fixed_targets_protocol_v1.md
```

The [captured environment](experiments/fixed_targets/pilot_v1/environment.txt) records the remaining versions. The runner uses the shipped supervised archive to check training-array and exact-arm weight identity. It runs sequentially on one CPU thread with a sampled 4 GiB process-RSS stop guard; the main study took 146 seconds and peaked at 0.54 GiB. To display a reproduction locally, add `--dashboard dashboard/data/fixed_targets.json`; this replaces the viewer's current data, so preserve the original artifact.

Run the fixed-budget provenance pilot:

```bash
python -m no_way_home.run_provenance_release \
  --out experiments/provenance/my-reproduction
```

Stored studies: [paired fixed-data targets v1](experiments/fixed_targets/pilot_v1/results.json), [exact supervision v1](docs/experiments/supervised_results_v1.md), [A-only competence v1](docs/experiments/competence_results_v1.md), [adaptation v2](docs/experiments/adaptation_pilot_v2.md), [archived adaptation v1](docs/experiments/adaptation_pilot_v1.md), and [provenance](experiments/provenance/release-pilot-v1/RESULTS.md). Both adaptation pilots failed initial competence; their later A→B→A outcomes do not establish forgetting or recovery. The A-only RL diagnosis learned its selected fixed tasks but failed fresh-world competence. Subsequent studies establish useful offline behavior under broad coverage, first with exact targets and now with bootstrapping. Each milestone stops at its declared budget. Runners refuse to overwrite an existing study directory; read the protocol before changing its seeds or budget.

## Inspect results

```bash
python3 -m http.server 8080 --bind 127.0.0.1
```

Open **[the current lab](http://127.0.0.1:8080/dashboard/lab.html#fixed)** and select **Fixed-data targets**. Compare exact versus Double DQN success, efficient success, episode length, state-action agreement and value error. The paired table shows final differences within each seed; the replay switches target condition, checkpoint, panel and action mode while displaying learned and exact pre-action values. The first fresh map is a four-step success for both methods in all seeds, selected before outcomes. Use the full-panel curves and tables to assess their differences.

**Exact targets** retains the earlier supervised-versus-historical-RL illustration, on its own evaluation panel. **Learn one world** retains the fixed-task/stream diagnosis; **Adaptation** and **Evidence** retain the earlier tracks. The [historical registry](http://127.0.0.1:8080/dashboard/index.html) preserves older run summaries. To index legacy training runs available on your machine, run `python dashboard/scan.py` before starting the server.

Historical summaries are not a complete artifact archive: many referenced CSV files, checkpoints, and replays are absent from a fresh checkout. Claims from the earlier reports should be read with the [September 2026 review](research_review/2026-09-05/README.md), which documents confounds, bugs, and unsupported interpretations. The new pilots also need independent seeds and fair baselines before supporting general conclusions. No Way Home is a synthetic decision model, not empirical evidence about human institutions.

## Develop and contribute

- [Contribution guide](CONTRIBUTING.md): fast and full test commands, experiment template, and reproducibility requirements.
- [Documentation index](docs/README.md) and [roadmap](docs/roadmap.md): current scope and criteria for the next experiment.
- [Fixed-target validation](docs/validation/fixed-targets-v1.md), [supervised validation](docs/validation/supervised-v1.md), [competence validation](docs/validation/competence-v1.md) and [earlier preview validation](docs/validation/2026-09-06.md): tests, CI, artifact audits and browser coverage.
- [Training supervisor](orchestrator/README.md): local job logging, crash restart, and resume conventions.
- [Citation metadata](CITATION.cff): cite the repository and the exact revision/artifact used.

The package installs the current experiment modules and selected legacy support modules. Historical trainer scripts and the dashboard are documented as commands run from this checkout; their APIs are not a stable library contract.

## Research history

The [version index](versions/README.md), [articles](articles/), [original Q6 narrative](Q6.md), [No Way Home design](Q6%20No%20Way%20Home.md), and [archive](archive/README.md) preserve the project's progression. Older ablations and the PPO port live on the [`v7-ablations`](https://github.com/rahul-tiwari-95/Q6/tree/v7-ablations) and [`v8`](https://github.com/rahul-tiwari-95/Q6/tree/v8) branches. These are historical experiments, not interchangeable controlled comparisons. The [`longer_memory/`](longer_memory/README.md) folder contains old planning notes, not an implemented memory architecture.
