# Q6

**Small, inspectable experiments in learning across changing worlds.**

Q6 studies how a neural agent adapts when a task changes, and what it retains when an earlier task returns. The main experiment follows **A → B → A** in a compact collection world. A separate No Way Home experiment compares ways of counting repeated evidence in a synthetic resource-allocation task.

This is a public research preview; a license has not yet been selected. Its earlier Krishna–Hunter self-play work, failed experiments, and methodological corrections remain available as a research history.

**Latest finding:** [pilot v2](docs/experiments/adaptation_pilot_v2.md) completed **1,080,000 training transitions** across three seeds and A → B → A, but mean A success after initial training was **31.25%**, below the predeclared **70% competence gate**. [Post-hoc diagnostics](experiments/adaptation/pilot_v2/diagnostics.json) on the same A evaluation panel gave **41.67%** for seeded random actions and **100%** for a shortest-path controller. This is a failed competence check; it does not establish forgetting or recovery. The next milestone is reliable initial-task learning before adding memory or a new architecture.

## Start on CPU

Use Python 3.10 or later; Python 3.12 is a practical starting point. Run commands from the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

# Short execution check; writes to a new temporary output directory.
python -m q6.adaptation --output /tmp/q6-adaptation-smoke \
  --seeds 0 --phase-steps 64 --eval-episodes 2 --max-seconds 30
```

Choose a fresh `--output` directory when repeating a run. A smoke run checks execution and artifacts; its small training budget does not establish adaptation or retention.

## Experiments

| Track | Question | Start here |
| --- | --- | --- |
| **Adaptation — primary** | Can the compact learner first acquire A reliably enough to study B and recovery of A? | [Current protocol](docs/experiments/adaptation_protocol_v2.md), [`q6/`](q6/) |
| **Provenance — bounded companion** | Does deduplicating message origins improve decisions compared with independently calibrated raw and decayed counts? | [Protocol](docs/experiments/provenance-protocol.md), [errata](docs/experiments/provenance-errata.md), [`no_way_home/`](no_way_home/) |

Reproduce adaptation pilot v2 in a fresh directory and export its dashboard data:

```bash
python -m q6.adaptation --output experiments/adaptation/my-reproduction \
  --dashboard dashboard/data/adaptation.json --seeds 0,1,2 \
  --phase-steps 120000 --eval-episodes 32 --eval-seed-start 910000 \
  --max-seconds 900 --protocol-file docs/experiments/adaptation_protocol_v2.md

# Post-hoc planner/random references; no further training.
python -m q6.diagnostics --study experiments/adaptation/my-reproduction
```

Run the fixed-budget provenance pilot:

```bash
python -m no_way_home.run_provenance_release \
  --out experiments/provenance/my-reproduction
```

Stored pilots: [adaptation v2](experiments/adaptation/pilot_v2/results.json), [archived v1](docs/experiments/adaptation_pilot_v1.md), and [provenance results](experiments/provenance/release-pilot-v1/RESULTS.md). V2 kept the learner and task fixed, used ten times v1's training budget, and evaluated on a fresh panel; the larger budget also stretched the existing exploration schedule. Both failed the competence gate. The milestone stops here, with no third tuning run. The runners refuse to overwrite an existing study directory; read the protocol before changing its seeds or budget.

## Inspect results

```bash
python3 -m http.server 8080 --bind 127.0.0.1
```

Open **[the current lab](http://127.0.0.1:8080/dashboard/lab.html)** for learning curves, comparisons, and recorded behavior. The [historical registry](http://127.0.0.1:8080/dashboard/index.html) preserves earlier run summaries. To index legacy training runs available on your machine, run `python dashboard/scan.py` before starting the server.

Historical summaries are not a complete artifact archive: many referenced CSV files, checkpoints, and replays are absent from a fresh checkout. Claims from the earlier reports should be read with the [September 2026 review](research_review/2026-09-05/README.md), which documents confounds, bugs, and unsupported interpretations. The new pilots also need independent seeds and fair baselines before supporting general conclusions. No Way Home is a synthetic decision model, not empirical evidence about human institutions.

## Develop and contribute

- [Contribution guide](CONTRIBUTING.md): fast and full test commands, experiment template, and reproducibility requirements.
- [Documentation index](docs/README.md) and [roadmap](docs/roadmap.md): current scope and criteria for the next experiment.
- [Training supervisor](orchestrator/README.md): local job logging, crash restart, and resume conventions.
- [Citation metadata](CITATION.cff): cite the repository and the exact revision/artifact used.

The package installs the current experiment modules and selected legacy support modules. Historical trainer scripts and the dashboard are documented as commands run from this checkout; their APIs are not a stable library contract.

## Research history

The [version index](versions/README.md), [articles](articles/), [original Q6 narrative](Q6.md), [No Way Home design](Q6%20No%20Way%20Home.md), and [archive](archive/README.md) preserve the project's progression. Older ablations and the PPO port live on the [`v7-ablations`](https://github.com/rahul-tiwari-95/Q6/tree/v7-ablations) and [`v8`](https://github.com/rahul-tiwari-95/Q6/tree/v8) branches. These are historical experiments, not interchangeable controlled comparisons. The [`longer_memory/`](longer_memory/README.md) folder contains old planning notes, not an implemented memory architecture.
