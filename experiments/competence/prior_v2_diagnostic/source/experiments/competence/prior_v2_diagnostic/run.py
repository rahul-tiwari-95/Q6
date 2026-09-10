"""Post-hoc exact-Q diagnosis of frozen v2 after-A policies; never trains.

From repository root:
python3 experiments/competence/prior_v2_diagnostic/run.py --out /tmp/q6-prior-v2-reproduction
"""
import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from q6.learning import DQN
from q6.optimal import VisibleOptimalQ
from q6.world import CollectionWorld, WorldConfig


def mean(rows, key):
    return float(np.mean([row[key] for row in rows])) if rows else None


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=ROOT / 'experiments/adaptation/pilot_v2')
    parser.add_argument('--out', type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    args.study = args.study.resolve()
    if (args.out / 'results.json').exists():
        raise FileExistsError('choose a new output directory; existing diagnostic is preserved')
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    source_paths = [ROOT / 'q6/optimal.py', ROOT / 'q6/world.py', ROOT / 'q6/learning.py', Path(__file__).resolve()]
    source_hashes = {}
    for source in source_paths:
        relative = source.relative_to(ROOT)
        destination = args.out / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        source_hashes[str(relative)] = hashlib.sha256(destination.read_bytes()).hexdigest()
    torch.set_num_threads(1)
    protocol = json.loads((args.study / 'protocol.json').read_text())
    config_dict = dict(protocol['world'])
    config_dict['action_mapping'] = tuple(config_dict['action_mapping'])
    config = WorldConfig(**config_dict)
    current_world_hash = hashlib.sha256((ROOT / 'q6/world.py').read_bytes()).hexdigest()
    assert current_world_hash == protocol['source_sha256']['q6/world.py'], 'historical world implementation changed'
    with (args.study / 'evaluations.csv').open() as stream:
        prior = {(int(row['seed']), int(row['map_seed'])): row for row in csv.DictReader(stream)
                 if row['condition'] == 'continued' and row['checkpoint'] == 'after_a' and row['world'] == 'A'}
    reference = VisibleOptimalQ(config)
    steps, episodes, model_hashes = [], [], {}
    for seed in protocol['seeds']:
        checkpoint = args.study / 'models' / f'seed_{seed}_after_a.pt'
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        agent = DQN(saved['observation_size'], seed=seed, gamma=config.gamma, capacity=256)
        agent.online.load_state_dict(saved['online'])
        agent.online.eval()
        assert agent.parameter_hash() == saved['parameter_hash']
        model_hashes[checkpoint.name] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        for map_seed in protocol['eval_panel']:
            env = CollectionWorld(config)
            observation, _ = env.reset(seed=map_seed)
            seen = set()
            episode_steps = []
            base_return = 0.0
            while True:
                before = env.position
                with torch.no_grad():
                    learned = agent.online(torch.from_numpy(observation).unsqueeze(0))[0].numpy().copy()
                optimal = reference.q_values(observation)
                best = optimal.max()
                optimal_actions = np.flatnonzero(np.isclose(optimal, best, atol=1e-9, rtol=0))
                action = int(learned.argmax())
                can_finish = reference.can_finish(observation)
                action_finish = reference.action_can_finish(observation)
                next_observation, reward, terminated, truncated, info = env.step(action)
                base_return += info['base_reward']
                row = {
                    'seed': seed, 'map_seed': map_seed, 'tick': env.elapsed,
                    'position_row': before[0], 'position_col': before[1], 'action': action,
                    'winnable_before': bool(can_finish),
                    'action_preserves_solvability': bool(action_finish[action]),
                    'avoidable_deadline_loss': bool(can_finish and not action_finish[action]),
                    'optimal_action': bool(action in optimal_actions),
                    'optimal_q_regret': float(best - optimal[action]),
                    'q_absolute_error_mean': float(np.abs(learned - optimal).mean()),
                    'q_signed_error_mean': float((learned - optimal).mean()),
                    'learned_top_two_margin': float(np.sort(learned)[-1] - np.sort(learned)[-2]),
                    'learned_selected_advantage_over_best_optimal': float(learned[action] - learned[optimal_actions].max()),
                    'blocked': bool(env.position == before), 'position_revisit': before in seen,
                    'base_reward': info['base_reward'], 'shaped_reward': reward,
                    'terminated': terminated, 'truncated': truncated,
                    **{f'learned_q_{a}': float(learned[a]) for a in range(4)},
                    **{f'optimal_q_{a}': float(optimal[a]) for a in range(4)},
                }
                steps.append(row)
                episode_steps.append(row)
                seen.add(before)
                observation = next_observation
                if terminated or truncated:
                    break
            historical = prior[(seed, map_seed)]
            assert int(historical['success']) == int(terminated)
            assert int(historical['steps']) == env.elapsed
            assert np.isclose(float(historical['base_return']), base_return)
            active = [row for row in episode_steps if row['winnable_before']]
            episodes.append({
                'seed': seed, 'map_seed': map_seed, 'success': bool(terminated),
                'steps': env.elapsed, 'base_return': base_return,
                'blocked_steps': sum(row['blocked'] for row in episode_steps),
                'position_revisit_steps': sum(row['position_revisit'] for row in episode_steps),
                'winnable_decisions': len(active), 'optimal_fraction_when_winnable': mean(active, 'optimal_action'),
                'mean_q_regret_when_winnable': mean(active, 'optimal_q_regret'),
                'mean_absolute_q_error': mean(episode_steps, 'q_absolute_error_mean'),
                'avoidable_deadline_loss_steps': sum(row['avoidable_deadline_loss'] for row in episode_steps),
            })
        assert agent.parameter_hash() == saved['parameter_hash'] and agent.steps == 0 and agent.updates == 0
    write_csv(args.out / 'steps.csv', steps)
    write_csv(args.out / 'episodes.csv', episodes)
    summaries = []
    for seed in protocol['seeds']:
        seed_steps = [row for row in steps if row['seed'] == seed]
        active = [row for row in seed_steps if row['winnable_before']]
        wrong = [row for row in active if not row['optimal_action']]
        seed_episodes = [row for row in episodes if row['seed'] == seed]
        summaries.append({
            'seed': seed, 'episodes': len(seed_episodes), 'success_rate': mean(seed_episodes, 'success'),
            'visited_steps': len(seed_steps), 'winnable_steps': len(active),
            'optimal_action_fraction_when_winnable': mean(active, 'optimal_action'),
            'mean_q_regret_when_winnable': mean(active, 'optimal_q_regret'),
            'mean_absolute_q_error_all_actions': mean(seed_steps, 'q_absolute_error_mean'),
            'mean_signed_q_error_all_actions': mean(seed_steps, 'q_signed_error_mean'),
            'blocked_step_fraction': mean(seed_steps, 'blocked'),
            'revisit_step_fraction': mean(seed_steps, 'position_revisit'),
            'suboptimal_winnable_decisions': len(wrong),
            'mean_wrong_selected_advantage_over_best_optimal': mean(wrong, 'learned_selected_advantage_over_best_optimal'),
            'mean_wrong_top_two_margin': mean(wrong, 'learned_top_two_margin'),
        })
    results = {
        'status': 'posthoc_diagnostic_no_training', 'source_study': str(args.study.relative_to(ROOT)),
        'checkpoint': 'after_a', 'world': 'A', 'eval_panel': protocol['eval_panel'],
        'reference': 'Exact finite-horizon optimal shaped discounted Q from visible state',
        'primary_outcome': 'descriptive action-ranking diagnostics; not an adaptation or causal test',
        'verification': {'historical_episodes_reproduced': len(episodes), 'no_training_updates': True,
                         'world_hash_matches_historical': True},
        'summary_by_training_seed': summaries,
        'limitations': [
            'This panel already informed v2 diagnosis; post-hoc results are not fresh confirmation.',
            'Q* compares optimal discounted shaped return; primary task success is also reported.',
            'State metrics are trajectory-weighted and correlated; no independence or significance claim.',
            'Ranking quality is reported while collection remains possible; after missed deadlines, actions can tie.',
            'Greedy Q ranking errors are observed, but their cause is not identified: coverage, optimization, representation and exploration require controlled experiments.',
            'The reference is engineered and computationally privileged; it is not learned or compute-matched.',
        ],
        'provenance': {'world_sha256': current_world_hash, 'model_sha256': model_hashes,
            'code_sha256': source_hashes,
            'protocol_sha256': hashlib.sha256((args.study / 'protocol.json').read_bytes()).hexdigest(),
            'python': sys.version.split()[0], 'numpy': np.__version__, 'torch': torch.__version__,
            'device': 'cpu', 'torch_threads': torch.get_num_threads(),
            'elapsed_seconds': time.perf_counter() - started},
    }
    (args.out / 'results.json').write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')
    lines = ['# Prior v2: exact-Q diagnosis of frozen after-A policies', '',
             'Post-hoc, read-only analysis of saved v2 weights. No training or exploration was performed.',
             'The same32 A evaluation maps and three saved training seeds reproduce all96 original episodes exactly.', '',
             '| Training seed | Success | Blocked steps | Optimal actions while winnable | Mean optimal-Q regret while winnable | Mean absolute Q error |',
             '|---:|---:|---:|---:|---:|---:|']
    for row in summaries:
        lines.append(f"| {row['seed']} | {row['success_rate']:.2%} | {row['blocked_step_fraction']:.2%} | {row['optimal_action_fraction_when_winnable']:.2%} | {row['mean_q_regret_when_winnable']:.4f} | {row['mean_absolute_q_error_all_actions']:.4f} |")
    lines += ['', 'The reference computes exact optimal discounted shaped Q from visible walls, pellet, position, remaining time and action mapping. Its independent finite-horizon dynamics are checked against actual world transitions. It is an engineered diagnostic, not a learned or compute-matched competitor.', '',
              'These policies exhibit incorrect greedy action rankings and many blocked/revisited positions. Learned values also underestimate optimal values on average. This identifies observed behavior, not why learning produced it: optimization, state coverage, representation and exploration have not been causally separated.', '',
              'Step metrics are trajectory-weighted: a32-step failure contributes more observations than a short success. Once the remaining deadline makes success impossible, optimal actions can tie; ranking fraction/regret are therefore also restricted to still-winnable states. Absolute Q error averages all four actions over visited states. No significance or independence claim is attached to these correlated steps.', '',
              'This historical panel already informed diagnosis and is not a fresh test of the new A-only experiment. The source snapshots, exact old model hashes and measured CPU runtime are in results.json/source/manifest.json. Original pilot files were not changed.', '',
              'Reproduce from repository root with a new output directory:', '',
              '```bash', 'python3 experiments/competence/prior_v2_diagnostic/run.py --out /tmp/q6-prior-v2-reproduction', '```', '']
    (args.out / 'RESULTS.md').write_text('\n'.join(lines))
    manifest = {str(path.relative_to(args.out)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(args.out.rglob('*')) if path.is_file() and path.name != 'manifest.json'}
    (args.out / 'manifest.json').write_text(json.dumps({'algorithm': 'sha256', 'files': manifest}, indent=2) + '\n')
    print(json.dumps(results['summary_by_training_seed'], indent=2))


if __name__ == '__main__':
    main()
