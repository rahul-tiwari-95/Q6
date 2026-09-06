"""Bounded exploratory provenance-control pilot; see the pre-run protocol.

Run from any directory: python3 -m no_way_home.run_provenance_release
Historical reports are preserved. A new output directory is required.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import platform
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from no_way_home.metrics import need_shortfall_per_10k
from no_way_home.policies import never_mitigate
from no_way_home.world import WorldConfig, run

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'docs/experiments/provenance-protocol.md'
THRESHOLDS = [0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.00, 1.30, 1.70]
CALIBRATION_SEEDS = list(range(31000, 31008))
EVALUATION_SEEDS = list(range(41000, 41024))
POLICY_LABELS = {'raw': 'Calibrated raw count', 'unique': 'Calibrated unique count',
                 'decay': 'Calibrated bounded decay', 'never': 'Never mitigate'}
BASE_CONFIG = WorldConfig(n_ticks=1000, shift_tick=400)
SCENARIOS = {
    'default': ('Calibration distribution', BASE_CONFIG),
    'forwarding_shift': ('More forwarding', replace(BASE_CONFIG, forward_prob_per_tick=0.8, hub_forward_boost=10.0)),
    'reporting_and_timing_shift': ('Rarer reports, later shift', replace(BASE_CONFIG, report_sick_threshold=4, shift_tick=600)),
}


class CountingPolicy:
    """Same threshold search budget for three fixed evidence summaries.

    The incremental decayed sum equals MessageLog.bounded_decay_rate after
    rate normalization. It reads only message timestamps, never origin IDs.
    """
    half_life = 15.0
    floor = 0.0
    ceiling = 2.0
    window = 30

    def __init__(self, kind, threshold):
        if kind not in ('raw', 'unique', 'decay'):
            raise ValueError(f'unknown counting policy: {kind}')
        self.kind = kind
        self.threshold = float(threshold)
        self.start_episode()

    def start_episode(self):
        self._decayed_total = 0.0
        self._last_tick = 0
        self._seen = 0

    def score(self, state):
        now = state.tick + 1
        if self.kind == 'raw':
            return state.messages.raw_message_rate(now, self.window)
        if self.kind == 'unique':
            return state.messages.unique_origin_rate(now, self.window)
        self._decayed_total *= 2.0 ** (-(now - self._last_tick) / self.half_life)
        for message in state.messages.messages[self._seen:]:
            self._decayed_total += 2.0 ** (-(now - message.tick) / self.half_life)
        self._seen = len(state.messages.messages)
        self._last_tick = now
        decay = 2.0 ** (-1.0 / self.half_life)
        normalization = decay / (1.0 - decay)
        return max(self.floor, min(self.ceiling, self._decayed_total / normalization))

    def __call__(self, state, rng):
        return self.score(state) > self.threshold


def outcome(state):
    """Executed actions; response delays include nonresponses at window length."""
    events = state.events
    if not events:
        raise ValueError('an evaluation run must contain events')
    pre = [event for event in events if not event['blight_high']]
    post = [event for event in events if event['blight_high']]
    if not post:
        raise ValueError('a crisis must occur within the evaluation window')
    false_ticks = sum(bool(event['mitigate']) for event in pre)
    responses = [event['tick'] for event in post if event['mitigate']]
    loss = need_shortfall_per_10k(state)
    if not np.isfinite(loss):
        raise ValueError('nonfinite shortfall')
    return {
        'shortfall_per_10k': loss,
        'post_shift_shortfall_per_10k': need_shortfall_per_10k(state, (post[0]['tick'], state.tick + 1)),
        'false_alarm': bool(false_ticks),
        'false_mitigation_ticks': false_ticks,
        'missed_crisis': not bool(responses),
        'response_delay_ticks': responses[0] - post[0]['tick'] if responses else len(post),
        'mitigation_ticks': sum(bool(event['mitigate']) for event in events),
        'post_shift_ticks': len(post),
    }


def calibrate(cfg, seeds, thresholds):
    rows, selected = [], {}
    for kind in ('raw', 'unique', 'decay'):
        candidates = []
        for threshold in thresholds:
            losses = []
            for seed in seeds:
                state = run(cfg, CountingPolicy(kind, threshold), seed)
                row = {'policy_id': kind, 'threshold': threshold, 'seed': seed, **outcome(state)}
                rows.append(row)
                losses.append(row['shortfall_per_10k'])
            candidates.append((float(np.mean(losses)), threshold))
        selected[kind] = min(candidates)[1]
    return selected, rows


def bootstrap_interval(values, draws):
    values = np.asarray(values, dtype=float)
    return np.quantile(values[draws].mean(axis=1), [0.025, 0.975]).tolist()


def summarize(rows, scenarios, selected, evaluation_seeds, n_resamples=2000):
    results = []
    rng = np.random.default_rng(52026)
    for scenario_id, (label, cfg) in scenarios.items():
        # Identical bootstrap seed rows retain policy pairing in every contrast.
        draws = rng.integers(0, len(evaluation_seeds), (n_resamples, len(evaluation_seeds)))
        by_policy = {}
        for policy_id in POLICY_LABELS:
            cells = {r['seed']: r for r in rows if r['scenario_id'] == scenario_id and r['policy_id'] == policy_id}
            if set(cells) != set(evaluation_seeds):
                raise ValueError('incomplete policy/scenario/seed cells')
            by_policy[policy_id] = [cells[seed] for seed in evaluation_seeds]
        unique_loss = np.array([r['shortfall_per_10k'] for r in by_policy['unique']])
        policy_results = []
        for policy_id, cells in by_policy.items():
            losses = np.array([r['shortfall_per_10k'] for r in cells])
            differences = losses - unique_loss
            policy_results.append({
                'policy_id': policy_id, 'n': len(cells),
                'mean_shortfall': float(losses.mean()),
                'ci95_shortfall': bootstrap_interval(losses, draws),
                'false_alarm_rate': float(np.mean([r['false_alarm'] for r in cells])),
                'mean_false_mitigation_ticks': float(np.mean([r['false_mitigation_ticks'] for r in cells])),
                'missed_crisis_rate': float(np.mean([r['missed_crisis'] for r in cells])),
                'mean_response_delay_ticks': float(np.mean([r['response_delay_ticks'] for r in cells])),
                'paired_difference_vs_unique': {'mean': float(differences.mean()),
                                                 'ci95': bootstrap_interval(differences, draws)},
            })
        results.append({'id': scenario_id, 'label': label, 'config': asdict(cfg), 'policy_results': policy_results})
    return results


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_provenance():
    files = sorted((ROOT / 'no_way_home').glob('*.py')) + [PROTOCOL]
    return {
        'python': platform.python_version(), 'numpy': np.__version__,
        'source_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        'protocol_sha256': hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        'rng': 'numpy SeedSequence(seed).spawn(2): world and policy streams',
    }


def run_pilot(out, *, cfg=BASE_CONFIG, scenarios=None, calibration_seeds=None,
              evaluation_seeds=None, thresholds=None, n_resamples=2000):
    """Keyword overrides support the tiny integration test, never the release CLI."""
    scenarios = SCENARIOS if scenarios is None else scenarios
    calibration_seeds = CALIBRATION_SEEDS if calibration_seeds is None else calibration_seeds
    evaluation_seeds = EVALUATION_SEEDS if evaluation_seeds is None else evaluation_seeds
    thresholds = THRESHOLDS if thresholds is None else thresholds
    if set(calibration_seeds) & set(evaluation_seeds):
        raise ValueError('calibration and evaluation seeds must be disjoint')
    if not calibration_seeds or not evaluation_seeds or not thresholds:
        raise ValueError('seed sets and threshold grid must be nonempty')
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'refusing to overwrite existing run: {out}')
    out.mkdir(parents=True)
    provenance = source_provenance()
    for relative_path in provenance['source_sha256']:
        destination = out / 'source' / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative_path).read_bytes())
    selected, calibration_rows = calibrate(cfg, calibration_seeds, thresholds)
    write_csv(out / 'calibration.csv', calibration_rows)
    (out / 'calibration.json').write_text(json.dumps(calibration_rows, indent=2) + '\n')
    rows = []
    # mtime=0 and no embedded filename make trace compression deterministic.
    with (out / 'evaluation-traces.jsonl.gz').open('wb') as raw:
        with gzip.GzipFile(fileobj=raw, filename='', mode='wb', mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding='utf-8') as traces:
                for scenario_id, (_, scenario_cfg) in scenarios.items():
                    for policy_id in POLICY_LABELS:
                        for seed in evaluation_seeds:
                            policy = never_mitigate if policy_id == 'never' else CountingPolicy(policy_id, selected[policy_id])
                            state = run(scenario_cfg, policy, seed)
                            key = {'scenario_id': scenario_id, 'policy_id': policy_id, 'seed': seed}
                            rows.append({**key, **outcome(state)})
                            trace = {**key, 'config': asdict(scenario_cfg), 'events': state.events,
                                     'messages': [asdict(message) for message in state.messages.messages]}
                            traces.write(json.dumps(trace, sort_keys=True, separators=(',', ':')) + '\n')
    write_csv(out / 'evaluation.csv', rows)
    (out / 'evaluation.json').write_text(json.dumps(rows, indent=2) + '\n')
    metadata = {
        'status': 'exploratory', 'generated_at': datetime.now(timezone.utc).isoformat(),
        'calibration_config': asdict(cfg), 'calibration_seeds': calibration_seeds,
        'evaluation_seeds': evaluation_seeds, 'threshold_grid': thresholds,
        'selected_thresholds': selected, 'bootstrap_resamples': n_resamples,
        'bootstrap_seed': 52026, 'provenance': provenance,
        'calibration_runs_per_tuned_policy': len(thresholds) * len(calibration_seeds),
        'scenarios': {key: asdict(value[1]) for key, value in scenarios.items()},
        'fixed_controller_settings': {'window': 30, 'decay_half_life': 15, 'decay_floor_rate': 0, 'decay_ceiling_rate': 2},
    }
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    try:
        artifact_base = str(out.resolve().relative_to(ROOT))
    except ValueError:
        artifact_base = str(out.resolve())
    dashboard = {
        'schema_version': 1, 'status': 'exploratory', 'protocol': 'docs/experiments/provenance-protocol.md',
        'generated_at': metadata['generated_at'],
        'primary_metric': 'Need-shortfall agent-ticks per 10,000 world ticks (lower is better)',
        'policies': [{'id': key, 'label': value, 'selected_parameters':
                      ({'threshold': selected[key]} if key != 'never' else {})} for key, value in POLICY_LABELS.items()],
        'scenarios': summarize(rows, scenarios, selected, evaluation_seeds, n_resamples),
        'limitations': [
            'Exploratory synthetic pilot; equal tuning budget does not establish optimal policies or novelty.',
            '24 evaluation seeds per scenario; uncertainty intervals resample seeds, not individual ticks.',
            'Response delay includes missed crises at the full post-shift window length; missed rate is reported separately.',
            'Unique-origin counting is supplied; no learner discovers provenance in this experiment.',
            'A centralized binary controller over shared resources; no independently learning citizens or spatial grid.',
        ],
        'artifacts': {key: f'{artifact_base}/{filename}' for key, filename in {
            'metadata': 'metadata.json', 'calibration': 'calibration.csv',
            'evaluation': 'evaluation.csv', 'traces': 'evaluation-traces.jsonl.gz',
            'results': 'results.json', 'manifest': 'manifest.json', 'report': 'RESULTS.md',
        }.items()},
    }
    (out / 'summary.json').write_text(json.dumps(dashboard, indent=2) + '\n')
    (out / 'results.json').write_text(json.dumps({**dashboard, 'metadata': metadata}, indent=2) + '\n')
    lines = ['# Provenance controls: exploratory release pilot', '',
             'Equal calibration budgets; primary loss is whole-run shortfall. Historical beta reports are preserved.', '',
             'Selected thresholds: ' + ', '.join(f'{key}={value:g}' for key, value in selected.items()) + '.', '',
             '| Scenario | Policy | Mean shortfall [bootstrap 95% CI] | False alarms | Missed crisis | Mean delay |',
             '|---|---|---:|---:|---:|---:|']
    for scenario in dashboard['scenarios']:
        for result in scenario['policy_results']:
            low, high = result['ci95_shortfall']
            lines.append(f"| {scenario['id']} | {result['policy_id']} | {result['mean_shortfall']:.0f} [{low:.0f}, {high:.0f}] | {result['false_alarm_rate']:.1%} | {result['missed_crisis_rate']:.1%} | {result['mean_response_delay_ticks']:.1f} |")
    lines += ['', 'Intervals are exploratory, per comparison; they are not multiplicity-adjusted hypothesis tests.',
              'Never mitigate demonstrates why zero false alarms alone is insufficient. See paired differences in summary.json.', '',
              'Full configs, source hashes, grids and seeds: metadata.json. Raw evaluation histories: evaluation-traces.jsonl.gz.', '']
    (out / 'RESULTS.md').write_text('\n'.join(lines))
    manifest = {str(path.relative_to(out)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(out.rglob('*')) if path.is_file()}
    (out / 'manifest.json').write_text(json.dumps({'algorithm': 'sha256', 'files': manifest}, indent=2) + '\n')
    return dashboard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'experiments/provenance/release-pilot-v1')
    parser.add_argument('--dashboard', type=Path, default=ROOT / 'dashboard/data/provenance.json')
    args = parser.parse_args()
    result = run_pilot(args.out)
    args.dashboard.parent.mkdir(parents=True, exist_ok=True)
    args.dashboard.write_text(json.dumps(result, indent=2) + '\n')
    print((args.out / 'RESULTS.md').read_text())


if __name__ == '__main__':
    main()
