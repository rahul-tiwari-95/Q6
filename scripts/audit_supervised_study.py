"""Independently audit a complete supervised-v1 study without training.

Checks archived inputs, all exact-Q targets against an independent Bellman
recurrence, bounded clone transitions, saved model/dense-prediction consistency,
minibatch sample digests, raw metrics and research gates. It performs model
forward inference and isolated dynamics checks, never optimizer steps or policy
rollouts. Three compact JSON summaries describe successful checks.

Use the recorded Python/Torch/NumPy environment for exact float32 predictions:
    python scripts/audit_supervised_study.py
    python scripts/audit_supervised_study.py --study experiments/supervised/my-run

This intentionally accepts the complete declared v1 configuration only. Smoke,
deviating or incomplete studies use verify_pilot_artifacts.py instead. Paths in
study metadata resolve against this repository; model/source artifacts are read
only. This is artifact validation, not an independent training replication.
"""
import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict, deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

def main():
    ROOT = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--study', type=Path, default=Path('experiments/supervised/pilot_v1'),
                        help='Complete declared-v1 study directory, absolute or relative to the repository root.')
    args = parser.parse_args()
    if not __debug__:
        parser.error('Run the audit with assertions enabled; do not use python -O.')
    P = (args.study if args.study.is_absolute() else ROOT / args.study).resolve()
    if not (P / 'results.json').is_file() or not (P / 'manifest.json').is_file():
        parser.error('The study needs completed results.json and manifest.json artifacts.')
    # Use the archived world/network implementation. Later development in q6/ must
    # not silently change the model architecture or transition checks for this run.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(P / 'source'))
    from q6.world import CollectionWorld, WorldConfig
    from q6.learning import DQN

    started = time.monotonic()
    torch.set_num_threads(1)
    J = lambda path: json.loads(path.read_text())
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    array_sha = lambda value: hashlib.sha256(value.tobytes()).hexdigest()

    def close(actual, expected, label=''):
        assert math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-8), (label, actual, expected)

    def read_csv(name):
        return list(csv.DictReader((P / name).open()))

    def layout_hash(env):
        payload = {'walls': env.walls.astype(int).tolist(), 'pellets': env.pellets.astype(int).tolist(), 'config': asdict(env.config)}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    protocol, metadata = J(P / 'protocol.json'), J(P / 'dataset_metadata.json')
    config = WorldConfig(**{**protocol['world'], 'action_mapping': tuple(protocol['world']['action_mapping'])})
    assert asdict(config) == asdict(WorldConfig()), 'Expected the unchanged declared World A configuration.'
    assert protocol['seeds'] == [0, 1, 2]
    assert protocol['checkpoints'] == [0, 1000, 3000, 10000, 30000]
    assert protocol['deviations'] == [] and not protocol['smoke']
    assert protocol['runtime']['python'].split('.')[:2] == ['3', '12']
    assert protocol['runtime']['torch'].split('+')[0] == '2.8.0'
    assert protocol['runtime']['numpy'] == '2.0.2'
    assert protocol['git']['dirty'] is False
    assert sha(P / 'protocol.md') == protocol['protocol_sha256']
    for name, digest in protocol['source_sha256'].items():
        assert sha(P / 'source' / name) == digest
    prior_protocol = J(P / 'prior_protocol.json')
    for name in ['q6/world.py', 'q6/learning.py']:
        assert protocol['source_sha256'][name] == prior_protocol['source_sha256'][name]

    data = dict(np.load(P / 'dataset.npz', allow_pickle=False))
    for name, spec in metadata['arrays'].items():
        assert list(data[name].shape) == spec['shape'] and str(data[name].dtype) == spec['dtype']
        assert array_sha(data[name]) == spec['sha256']
    assert metadata['status'] == 'complete'
    assert [x['map_seed'] for x in metadata['train']] == list(range(300000, 300256))
    assert len(metadata['heldout']) == 64
    env = CollectionWorld(config)
    for panel in ['train', 'heldout']:
        for layout in metadata[panel]:
            env.reset(seed=layout['map_seed'])
            assert layout_hash(env) == layout['layout_hash']
            assert list(env.position) == layout['original_start']
    train_hashes = {x['layout_hash'] for x in metadata['train']}
    accepted, rejected, accepted_hashes = [], [], set()
    candidate = 930000
    while len(accepted) < 64:
        env.reset(seed=candidate)
        key = layout_hash(env)
        if key in train_hashes or key in accepted_hashes:
            rejected.append({'map_seed': candidate, 'layout_hash': key, 'reason': 'training_layout' if key in train_hashes else 'earlier_fresh_layout'})
        else:
            accepted.append({'map_seed': candidate, 'layout_hash': key, 'original_start': list(env.position)})
            accepted_hashes.add(key)
        candidate += 1
    assert accepted == metadata['heldout'] and rejected == metadata['collision_skips']
    assert not train_hashes & accepted_hashes
    duplicates = {key: count for key, count in Counter(x['layout_hash'] for x in metadata['train']).items() if count > 1}
    assert duplicates == metadata['train_duplicate_layouts']
    obs_hashes = {panel: {hashlib.sha256(row.tobytes()).digest() for row in data[f'{panel}_observations']} for panel in ['train', 'heldout']}
    assert not obs_hashes['train'] & obs_hashes['heldout']
    assert metadata['split_check'] == {'layout_intersections': 0, 'observation_intersections': 0,
        'unique_training_observations': len(obs_hashes['train']), 'unique_fresh_observations': len(obs_hashes['heldout'])}
    del obs_hashes

    # Independent finite-horizon Bellman check on every saved target. The recurrence
    # is unique from remaining=1 upward; it does not use VisibleOptimalQ.
    target_entries = 0
    max_bellman_error = 0.0
    clone_checks = 0
    for panel in ['train', 'heldout']:
        obs = data[f'{panel}_observations']
        targets = data[f'{panel}_targets']
        assert obs.shape == (len(metadata[panel]) * 640, 92)
        assert targets.shape == (len(obs), 4) and targets.dtype == np.float64
        for map_index, layout in enumerate(metadata[panel]):
            env.reset(seed=layout['map_seed'])
            positions = [tuple(map(int, p)) for p in np.argwhere(~(env.walls | env.pellets))]
            pos_index = {position: index for index, position in enumerate(positions)}
            goal = tuple(map(int, np.argwhere(env.pellets)[0]))
            distance = {goal: 0}
            queue = deque([goal])
            deltas = ((-1, 0), (1, 0), (0, -1), (0, 1))
            while queue:
                cell = queue.popleft()
                for dr, dc in deltas:
                    nxt = cell[0] + dr, cell[1] + dc
                    if 0 <= nxt[0] < 5 and 0 <= nxt[1] < 5 and not env.walls[nxt] and nxt not in distance:
                        distance[nxt] = distance[cell] + 1
                        queue.append(nxt)
            assert len(distance) == 21
            start = map_index * 640
            local = targets[start:start + 640].reshape(20, 32, 4)
            sl = slice(start, start + 640)
            assert np.array_equal(data[f'{panel}_map_seeds'][sl], np.full(640, layout['map_seed']))
            assert np.array_equal(data[f'{panel}_positions'][sl], np.repeat(positions, 32, axis=0))
            assert np.array_equal(data[f'{panel}_remaining'][sl], np.tile(np.arange(1, 33), 20))
            for pi, position in enumerate(positions):
                for remaining in range(1, 33):
                    index = start + pi * 32 + remaining - 1
                    env.position, env.elapsed = position, 32 - remaining
                    assert np.array_equal(obs[index], env.observe())
                    assert bool(data[f'{panel}_winnable'][index]) == (distance[position] <= remaining)
                    for action, physical in enumerate(config.action_mapping):
                        dr, dc = deltas[physical]
                        nxt = position[0] + dr, position[1] + dc
                        if not (0 <= nxt[0] < 5 and 0 <= nxt[1] < 5) or env.walls[nxt]:
                            nxt = position
                        collected = nxt == goal
                        ended = collected or remaining == 1
                        before = -distance[position] / 5
                        after = 0.0 if ended else -distance[nxt] / 5
                        reward = config.step_cost + config.pellet_reward * collected + config.shaping_weight * (config.gamma * after - before)
                        expected = reward + (0.0 if ended else config.gamma * local[pos_index[nxt], remaining - 2].max())
                        max_bellman_error = max(max_bellman_error, abs(expected - targets[index, action]))
                        target_entries += 1
                        if map_index < 2 and remaining in [1, 2, 32]:
                            clone = env.clone()
                            next_obs, actual_reward, terminated, truncated, info = clone.step(action)
                            close(actual_reward, reward, 'world reward')
                            assert clone.position == nxt and (terminated or truncated) == ended
                            clone_checks += 1
    assert max_bellman_error < 1e-12, max_bellman_error
    print(json.dumps({'dataset_states': {p: len(data[f'{p}_observations']) for p in ['train', 'heldout']},
        'target_entries_checked': target_entries, 'max_bellman_error': max_bellman_error,
        'world_clone_transition_checks': clone_checks, 'train_unique_layouts': len(train_hashes),
        'fresh_unique_layouts': len(accepted_hashes), 'collision_rejections': len(rejected)}), flush=True)

    result, manifest = J(P / 'results.json'), J(P / 'manifest.json')
    assert result['protocol'] == protocol
    assert result['run']['status'] == 'complete'
    assert result['run']['train_updates'] == 90000
    assert result['run']['training_examples'] == 5760000
    assert result['run']['model_parameter_count'] == 20420
    assert result['run']['wall_seconds'] < protocol['budget']['admission_seconds'] == 900
    for name, digest in manifest['files'].items():
        assert sha(P / name) == digest, name
    assert set(manifest['files']) == {x.relative_to(P).as_posix() for x in P.rglob('*') if x.is_file() and x.name != 'manifest.json'}
    assert len(result['run']['progress']) == 15 and all(x['evaluation_complete'] for x in result['run']['progress'])
    assert {(x['seed'], x['checkpoint']) for x in result['run']['progress']} == {(s,c) for s in [0,1,2] for c in protocol['checkpoints']}

    sampling = J(P / 'sampling.json')
    assert set(sampling) == {'0', '1', '2'}
    sample_counts = dict(np.load(P / 'sample_counts.npz', allow_pickle=False))
    for seed in [0, 1, 2]:
        row = sampling[str(seed)]
        assert row['updates'] == 30000 and row['examples_seen'] == 1920000
        rng = np.random.default_rng(np.random.SeedSequence([seed, 66301]))
        digest = hashlib.sha256()
        counts = np.zeros(163840, dtype=np.uint32)
        for _ in range(30000):
            indices = rng.choice(163840, 64, replace=False)
            digest.update(indices.astype('<i8').tobytes())
            counts[indices] += 1
        assert digest.hexdigest() == row['batch_index_sha256']
        assert np.array_equal(counts, sample_counts[f'seed{seed}'])
        assert np.count_nonzero(counts) == row['unique_states_sampled']

    # Model/dense-prediction consistency uses forward inference only, no actions.
    model_count = 0
    final_predictions = {}
    prior_metadata = J(P / 'prior_reference_metadata.json')
    assert sha(P / 'prior_protocol.json') == prior_metadata['source_protocol_sha256']
    for name, digest in prior_metadata['historical_source_sha256'].items():
        assert sha(ROOT / prior_metadata['historical_source_directory'] / name) == digest
    exposure_path = ROOT / prior_metadata['input']
    assert sha(exposure_path) == prior_metadata['input_sha256']
    layout_cache = {}
    checked_prior_overlap = []
    for exposure in J(exposure_path):
        if exposure['condition'] != 'stream' or exposure['seed'] not in protocol['seeds']:
            continue
        layouts = set()
        for task in exposure['tasks']:
            map_seed = task['map_seed']
            if map_seed not in layout_cache:
                env.reset(seed=map_seed)
                layout_cache[map_seed] = layout_hash(env)
            layouts.add(layout_cache[map_seed])
        overlaps = sorted(layouts & accepted_hashes)
        expected_overlap = {'seed': exposure['seed'], 'training_task_ids': len(exposure['tasks']),
            'unique_training_layouts': len(layouts), 'fresh_layout_overlap': overlaps,
            'fresh_layout_overlap_count': len(overlaps)}
        assert expected_overlap == next(row for row in prior_metadata['seeds'] if row['seed'] == exposure['seed'])
        checked_prior_overlap.append(len(overlaps))
    assert len(layout_cache) == prior_metadata['unique_task_ids_regenerated']
    for path in sorted((P / 'models').glob('*.pt')):
        state = torch.load(path, map_location='cpu', weights_only=True)
        model = DQN(state['observation_size'])
        model.online.load_state_dict(state['online'])
        assert model.parameter_hash() == state['parameter_hash']
        assert sum(x.numel() for x in model.online.parameters()) == 20420
        assert state['purpose'] == 'inference_only_not_resumable'
        model_count += 1
        if path.name.startswith('prior_stream'):
            seed = int(path.stem.split('seed')[1])
            assert sha(path) == prior_metadata['models'][str(seed)]['sha256']
            assert sha(ROOT / prior_metadata['models'][str(seed)]['source']) == sha(path)
        elif path.stem.endswith('update0'):
            seed = int(path.stem.split('_seed')[1].split('_')[0])
            assert model.parameter_hash() == result['run']['initial_policy_hashes'][str(seed)]
            assert model.parameter_hash() == DQN(92, seed=seed).parameter_hash()
        elif path.stem.endswith('update30000'):
            seed = int(path.stem.split('_seed')[1].split('_')[0])
            final_predictions[seed] = dict(np.load(P / f'predictions_seed{seed}.npz', allow_pickle=False))
            for panel in ['train', 'heldout']:
                observations = data[f'{panel}_observations']
                expected = np.empty((len(observations), 4), np.float32)
                with torch.no_grad():
                    for start in range(0, len(observations), 4096):
                        expected[start:start+4096] = model.online(torch.from_numpy(observations[start:start+4096])).numpy()
                assert np.array_equal(expected, final_predictions[seed][panel]), ('dense predictions',seed,panel)
    assert model_count == 18
    print(json.dumps({'manifest_files': len(manifest['files']), 'model_snapshots': model_count,
        'sampler_digests_reconstructed': 3, 'dense_prediction_arrays_exactly_reproduced': 6,
        'prior_fresh_layout_overlap_by_seed': checked_prior_overlap}), flush=True)

    # Raw episode cardinality, accounting, summaries and independent gates.
    episodes, references, states, losses = [read_csv(name) for name in ['evaluations.csv','references.csv','state_metrics.csv','losses.csv']]
    assert len(episodes) == 14400
    assert len(states) == 24000
    assert len(references) == 2816
    assert len(losses) == 900
    expected_cells = {(seed,checkpoint,panel,mode,layout['map_seed'],rep)
        for seed in [0,1,2] for checkpoint in protocol['checkpoints']
        for panel in ['train','heldout'] for mode in ['greedy','epsilon_0_1']
        for layout in metadata[panel] for rep in range(1 if mode=='greedy' else 2)}
    actual_cells = [(int(r['seed']),int(r['checkpoint']),r['panel'],r['mode'],int(r['map_seed']),int(r['repetition'])) for r in episodes]
    assert len(set(actual_cells)) == len(actual_cells) and set(actual_cells) == expected_cells
    for row in episodes + references:
        assert int(row['checkpoint_complete']) == 1
        assert 1 <= int(row['steps']) <= 32
        close(row['base_return'], int(row['success']) - .01*int(row['steps']))
        assert int(row['noop_steps']) + int(row['moving_steps']) == int(row['steps'])
        assert int(row['moved_revisits']) <= int(row['moving_steps'])
        assert int(row['optimal_winnable_actions']) <= int(row['winnable_steps']) <= int(row['steps'])
        assert int(row['avoidable_failure_actions']) == 1 - int(row['success'])
    reference_cells = [(r['policy'],int(r['seed']),r['panel'],r['mode'],int(r['map_seed']),int(r['repetition'])) for r in references]
    expected_reference_cells = set()
    for policy in ['random_actions','shortest_path','prior_stream']:
        for seed in ([0] if policy=='shortest_path' else [0,1,2]):
            for panel in (['heldout'] if policy=='prior_stream' else ['train','heldout']):
                for mode in (['greedy','epsilon_0_1'] if policy=='prior_stream' else ['reference']):
                    for layout in metadata[panel]:
                        for rep in range(1 if policy=='shortest_path' or mode=='greedy' else 2):
                            expected_reference_cells.add((policy,seed,panel,mode,layout['map_seed'],rep))
    assert len(set(reference_cells)) == len(reference_cells) and set(reference_cells) == expected_reference_cells
    assert len({r['reference_sample_id'] for r in references}) == len(references)
    for summary in result['aggregate'] + result['seed_results']:
        selected = [r for r in episodes if all(str(r[k])==str(summary[k]) for k in ['condition','checkpoint','panel','mode'])
            and ('seed' not in summary or int(r['seed'])==summary['seed'])]
        assert summary['episodes'] == len(selected)
        for raw,metric in [('success','success_rate'),('steps','mean_steps'),('base_return','mean_return'),('shaped_return','mean_shaped_return')]:
            close(np.mean([float(r[raw]) for r in selected]), summary[metric], metric)
        for num,den,metric in [('noop_steps','steps','noop_rate'),('moved_revisits','moving_steps','moved_revisit_rate'),
                             ('optimal_winnable_actions','winnable_steps','optimal_action_rate'),
                             ('action_regret_sum','steps','mean_action_regret'),('q_abs_error_sum','q_error_steps','mean_abs_q_error')]:
            denominator = sum(float(r[den]) for r in selected)
            if denominator:
                close(sum(float(r[num]) for r in selected)/denominator, summary[metric], metric)
            else:
                assert summary[metric] is None
        if 'seed' not in summary:
            rates = [np.mean([int(r['success']) for r in selected if int(r['seed'])==seed]) for seed in [0,1,2]]
            close(summary['seed_success_min'], min(rates));close(summary['seed_success_max'], max(rates))
    for summary in result['references']:
        selected = [r for r in references if all(r[k]==summary[k] for k in ['policy','panel','mode'])]
        assert summary['episodes'] == len(selected)
        close(summary['success_rate'], np.mean([int(r['success']) for r in selected]))
    for seed in [0,1,2]:
        selected = [r for r in losses if int(r['seed'])==seed]
        assert [int(r['checkpoint']) for r in selected] == list(range(100,30001,100))
        assert all(int(r['updates_in_window'])==100 for r in selected)
        assert all(int(r['training_examples'])==int(r['checkpoint'])*64 for r in selected)
        assert all(np.isfinite(float(r['mean_loss'])) and float(r['mean_loss'])>=0 for r in selected)

    expected_state_cells = {(seed,checkpoint,panel,layout['map_seed'],bucket) for seed in [0,1,2]
        for checkpoint in protocol['checkpoints'] for panel in ['train','heldout']
        for layout in metadata[panel] for bucket in ['all','1-8','9-16','17-24','25-32']}
    actual_state_cells = [(int(r['seed']),int(r['checkpoint']),r['panel'],int(r['map_seed']),r['time_bucket']) for r in states]
    assert len(set(actual_state_cells))==len(actual_state_cells) and set(actual_state_cells)==expected_state_cells
    final_agreement = {}
    for seed in [0,1,2]:
        for panel in ['train','heldout']:
            prediction = final_predictions[seed][panel].astype(np.float64)
            target = data[f'{panel}_targets']
            error = prediction - target
            regret = target.max(1)-target[np.arange(len(target)),prediction.argmax(1)]
            win = data[f'{panel}_winnable']
            final_agreement[seed,panel] = float((regret[win]<=1e-6).mean())
            for row in states:
                if int(row['seed'])!=seed or int(row['checkpoint'])!=30000 or row['panel']!=panel:
                    continue
                low,high = (1,32) if row['time_bucket']=='all' else map(int,row['time_bucket'].split('-'))
                selected = (data[f'{panel}_map_seeds']==int(row['map_seed'])) & (data[f'{panel}_remaining']>=low) & (data[f'{panel}_remaining']<=high)
                e,r,w = error[selected],regret[selected],win[selected]
                expected = {'states':len(e),'action_values':e.size,'winnable_states':int(w.sum()),'impossible_states':int((~w).sum()),
                    'optimal_winnable_actions':int(((r<=1e-6)&w).sum()),'abs_error_sum':np.abs(e).sum(),'squared_error_sum':np.square(e).sum(),
                    'signed_error_sum':e.sum(),'action_regret_sum':r.sum(),'winnable_regret_sum':r[w].sum(),'impossible_regret_sum':r[~w].sum(),
                    'mean_abs_q_error':np.abs(e).mean(),'rmse_q_error':np.sqrt(np.square(e).mean()),
                    'q95_abs_q_error':np.quantile(np.abs(e).mean(1),.95),'mean_signed_q_bias':e.mean(),'mean_action_regret':r.mean()}
                for name,value in expected.items(): close(row[name],value,name)
                if w.any():
                    close(row['optimal_action_rate'],(r[w]<=1e-6).mean())
                    close(row['winnable_mean_action_regret'],r[w].mean())
                if (~w).any(): close(row['impossible_action_regret'],r[~w].mean())

    for summary in result['state_aggregate']:
        selected=[r for r in states if int(r['checkpoint'])==summary['checkpoint'] and r['panel']==summary['panel'] and r['time_bucket']==summary['time_bucket']]
        for metric in ['states','action_values','winnable_states','impossible_states','optimal_winnable_actions','abs_error_sum','squared_error_sum','signed_error_sum','action_regret_sum','winnable_regret_sum','impossible_regret_sum']:
            close(summary[metric],sum(float(r[metric]) for r in selected),metric)
        close(summary['mean_abs_q_error'],summary['abs_error_sum']/summary['action_values'])
        close(summary['rmse_q_error'],np.sqrt(summary['squared_error_sum']/summary['action_values']))
        close(summary['mean_signed_q_bias'],summary['signed_error_sum']/summary['action_values'])
        close(summary['mean_action_regret'],summary['action_regret_sum']/summary['states'])
        close(summary['optimal_action_rate'],summary['optimal_winnable_actions']/summary['winnable_states'])
        close(summary['winnable_mean_action_regret'],summary['winnable_regret_sum']/summary['winnable_states'])
        if summary['checkpoint']==30000:
            panel=summary['panel']; low,high=(1,32) if summary['time_bucket']=='all' else map(int,summary['time_bucket'].split('-'))
            mask=(data[f'{panel}_remaining']>=low)&(data[f'{panel}_remaining']<=high)
            errors=np.concatenate([np.abs(final_predictions[s][panel].astype(np.float64)-data[f'{panel}_targets']).mean(1)[mask] for s in [0,1,2]])
            close(summary['q95_abs_q_error'],np.quantile(errors,.95))

    random_rows=[r for r in references if r['policy']=='random_actions' and r['panel']=='heldout']
    random_rate=float(np.mean([int(r['success']) for r in random_rows]))
    assert len(random_rows)==384
    assert result['gates']['eligible'] is True
    assert {r['seed'] for r in result['gates']['per_seed']}=={0,1,2}
    for gate in result['gates']['per_seed']:
        seed=gate['seed']
        scores={panel:np.mean([int(r['success']) for r in episodes if int(r['seed'])==seed and int(r['checkpoint'])==30000 and r['panel']==panel and r['mode']=='greedy']) for panel in ['train','heldout']}
        close(gate['train_success'],scores['train']);close(gate['fresh_success'],scores['heldout'])
        close(gate['train_state_optimal'],final_agreement[seed,'train'])
        assert gate['training_fit']==bool(scores['train']>=.9 and final_agreement[seed,'train']>=.9)
        assert gate['fresh']==bool(scores['heldout']>=.7 and scores['heldout']>random_rate)
    for key in ['training_fit','fresh']:
        assert result['gates'][key]==all(row[key] for row in result['gates']['per_seed'])
    close(result['gates']['pooled_random_fresh_success'],random_rate)
    expected_interpretation = (
        'fresh_pass_fit_unresolved' if result['gates']['fresh'] and not result['gates']['training_fit'] else
        'fresh_competence_gate_met' if result['gates']['fresh'] else
        'training_fit_only' if result['gates']['training_fit'] else 'training_fit_not_established')
    assert result['run']['interpretation'] == expected_interpretation
    print(json.dumps({'evaluation_rows':len(episodes),'reference_rows':len(references),'state_metric_rows':len(states),
        'loss_rows':len(losses),'training_updates':result['run']['train_updates'],'supervised_state_presentations':result['run']['training_examples'],
        'gates':result['gates'],'read_only_review_seconds':time.monotonic()-started}),flush=True)


if __name__ == '__main__':
    main()
