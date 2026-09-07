"""Logged outcomes, equal-state loss and preserved offline experiment controls."""
import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import q6.recorded_actions as study
from q6.fixed_targets import fixed_update
from q6.learning import DQN


def logged_fixture():
    data = {'observations': np.zeros((4, 92), np.float32),
            'map_seeds': np.array([300000, 300000, 300001, 300001]),
            'remaining': np.array([2, 1, 2, 1])}
    def row(current, action, reward, following, terminated=0, truncated=0):
        return dict(current_row=current, action=action, reward=reward, next_row=following,
                    terminated=terminated, truncated=truncated,
                    map_seed=int(data['map_seeds'][current]), remaining=int(data['remaining'][current]))
    records = [row(0, 0, .1, 1), row(0, 2, 1., -1, terminated=1),
               row(1, 1, -.1, -1, truncated=1), row(2, 3, .1, 3),
               row(3, 0, -.1, -1, truncated=1)]
    records.append(dict(records[0]))
    return records, data, np.arange(4, dtype=np.int32)


def test_recorded_table_deduplicates_outcomes_without_frequency_weighting():
    records, data, support = logged_fixture()
    arrays, summary = study.build_recorded_table(records, data, support)
    assert summary['recorded_edges'] == 5 and summary['collector_steps'] == 6
    assert summary['duplicate_records'] == 1
    assert summary['states_by_observed_actions'] == [
        {'actions': 1, 'states': 3}, {'actions': 2, 'states': 1},
        {'actions': 3, 'states': 0}, {'actions': 4, 'states': 0}]
    assert summary['nonterminal_successors_outside_support'] == 0
    assert arrays['occurrences'][0, 0] == 2
    assert np.isnan(arrays['rewards'][~arrays['observed']]).all()
    assert all(not a.flags.writeable for a in arrays.values())
    repeated, _ = study.build_recorded_table(records + [dict(records[0])] * 7, data, support)
    for key in arrays:
        if key != 'occurrences':
            np.testing.assert_array_equal(arrays[key], repeated[key])


@pytest.mark.parametrize('mutation', ['reward', 'end_flags', 'missing_state', 'wrong_map', 'wrong_clock', 'terminal_next'])
def test_invalid_or_incomplete_recorded_outcomes_are_rejected(mutation):
    records, data, support = logged_fixture()
    if mutation == 'reward':
        records[-1]['reward'] += .01
    elif mutation == 'end_flags':
        records.append({**records[2], 'terminated': 1, 'truncated': 0})
    elif mutation == 'missing_state':
        records = [r for r in records if r['current_row'] != 3]
    elif mutation == 'wrong_map':
        records[0]['next_row'] = 3
    elif mutation == 'wrong_clock':
        records[0]['remaining'] = 1
    else:
        records[1]['next_row'] = 1
    with pytest.raises(study.ConsistencyError):
        study.build_recorded_table(records, data, support)


def tensor_table(size):
    return {'observed': torch.zeros((size, 4), dtype=torch.bool),
            'rewards': torch.full((size, 4), float('nan')),
            'ends': torch.zeros((size, 4), dtype=torch.bool),
            'successor_indices': torch.full((size, 4), 999999, dtype=torch.int32)}


def constant_agent():
    agent = DQN(92, seed=0)
    with torch.no_grad():
        for p in list(agent.online.parameters()) + list(agent.target.parameters()):
            p.zero_()
        agent.online[-1].bias.copy_(torch.tensor([0., 4., 2., 1.]))
        agent.target[-1].bias.copy_(torch.tensor([7., 1., 9., 3.]))
    return agent


def test_loss_weights_states_equally_and_uses_double_dqn_at_logged_successors():
    agent = constant_agent()
    observations = torch.zeros((2, 92))
    table = tensor_table(2)
    table['observed'][0, 0] = True
    table['observed'][1] = True
    table['rewards'][table['observed']] = .2
    table['successor_indices'][0, 0] = 1
    table['ends'][1] = True
    table['successor_indices'][1] = -1
    before_target = copy.deepcopy(agent.target.state_dict())
    # Online selects action1 at the recorded successor; target values it at1,
    # although target's own maximum is9 at action2.
    first = torch.nn.functional.smooth_l1_loss(torch.tensor(0.), torch.tensor(1.17)).item()
    second = torch.nn.functional.smooth_l1_loss(torch.tensor([0., 4., 2., 1.]), torch.full((4,), .2)).item()
    loss = study.recorded_update(agent, observations, table, np.arange(2))
    assert loss == pytest.approx((first + second) / 2)
    assert loss != pytest.approx((first + 4 * second) / 5)
    assert all(p.grad is None for p in agent.target.parameters())
    for key, online in agent.online.state_dict().items():
        torch.testing.assert_close(agent.target.state_dict()[key], before_target[key].lerp(online, .01), rtol=0, atol=0)
    assert agent.count == agent.steps == agent.updates == 0


def test_unobserved_poison_never_enters_loss_and_terminal_edges_never_query_successors(monkeypatch):
    first, second = constant_agent(), constant_agent()
    table = tensor_table(1)
    table['observed'][0, 2] = True
    table['rewards'][0, 2] = -.3
    table['ends'][0, 2] = True
    table['successor_indices'][0, 2] = -1
    altered = {k: v.clone() for k, v in table.items()}
    absent = ~table['observed']
    altered['rewards'][absent] = 1e20
    altered['ends'][absent] = True
    altered['successor_indices'][absent] = -999999
    def forbidden(*args, **kwargs):
        pytest.fail('terminal-only batch must never invoke the target network')
    monkeypatch.setattr(first.target, 'forward', forbidden)
    monkeypatch.setattr(second.target, 'forward', forbidden)
    calls = []
    handle = first.online.register_forward_pre_hook(lambda _, args: calls.append(args[0].shape))
    observations = torch.zeros((1, 92))
    a = study.recorded_update(first, observations, table, [0])
    b = study.recorded_update(second, observations, altered, [0])
    handle.remove()
    assert a == b and np.isfinite(a)
    assert calls == [torch.Size([1, 92])]
    assert first.parameter_hash() == second.parameter_hash()
    with pytest.raises(study.ConsistencyError, match='no recorded action'):
        study.recorded_update(first, observations, tensor_table(1), [0])


def test_all_four_recorded_actions_reduce_to_existing_ddqn_objective():
    torch.set_num_threads(1)
    first, second = DQN(92, seed=3), DQN(92, seed=3)
    observations = torch.from_numpy(np.random.default_rng(17).normal(size=(64, 92)).astype(np.float32))
    table = tensor_table(64)
    table['observed'][:] = True
    table['rewards'][:] = .2
    table['ends'][:, 2:] = True
    table['successor_indices'][:] = 0
    table['successor_indices'][:, 2:] = -1
    old = fixed_update(first, observations, None, table, np.arange(64), 'double_dqn')
    new = study.recorded_update(second, observations, table, np.arange(64))
    assert new == pytest.approx(old, abs=1e-7)
    for a, b in zip(first.online.parameters(), second.online.parameters()):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)


def fixture(tmp_path, **kwargs):
    protocol = tmp_path / 'protocol.md'
    protocol.write_text('Recorded action execution test, not research evidence.\n')
    return study.run_study(tmp_path / 'study', protocol, seeds=[0], updates=2,
        panel_count=1, maps_per_panel=2, panel_seed_start=1160000, smoke=True, **kwargs)


def test_logged_tables_and_controls_precede_fits_and_all_fits_precede_evaluation(tmp_path, monkeypatch):
    original_parser = study.build_recorded_table
    original_update, original_rollout = study.recorded_update, study.shared.rollout
    counts = {'tables': 0, 'updates': 0, 'rollouts': 0, 'targets': 0}
    output = tmp_path / 'study'
    def checked_parser(records, *args, **kwargs):
        arrays, summary = original_parser(records, *args, **kwargs)
        # A path-based call delegates to the iterable parser; count the outer
        # completed call so each bank contributes once.
        if isinstance(records, (str, Path)):
            counts['tables'] += 1
        return arrays, summary
    def checked_update(*args, **kwargs):
        assert counts['tables'] == 3 and counts['rollouts'] == 0
        with np.load(output / 'supports.npz') as arrays:
            assert len(arrays.files) == 6
            for bank in (1, 2, 3):
                np.testing.assert_array_equal(arrays[f'collected_unique_bank{bank}'], arrays[f'recorded_actions_bank{bank}'])
        assert all((output / f'models/bank{bank}_collected_unique_seed0_update30000.pt').exists() for bank in (1, 2, 3))
        counts['updates'] += 1
        counts['targets'] += int(args[2]['observed'][args[3]].sum())
        return original_update(*args, **kwargs)
    def checked_rollout(*args, **kwargs):
        assert counts['updates'] == 6
        assert all((output / f'models/bank{b}_recorded_actions_seed0_update2.pt').exists() for b in (1, 2, 3))
        counts['rollouts'] += 1
        return original_rollout(*args, **kwargs)
    def forbidden(*args, **kwargs):
        pytest.fail('recorded-action study must not collect, learn online, or invoke all-action updates')
    monkeypatch.setattr(study, 'build_recorded_table', checked_parser)
    monkeypatch.setattr(study, 'recorded_update', checked_update)
    monkeypatch.setattr(study.shared, 'rollout', checked_rollout)
    monkeypatch.setattr(study.shared, 'fixed_update', forbidden)
    monkeypatch.setattr(study.shared.DQN, 'observe', forbidden)
    monkeypatch.setattr(study.shared.DQN, 'learn', forbidden)
    import q6.coverage
    monkeypatch.setattr(q6.coverage, 'collect_support', forbidden)
    result = fixture(tmp_path)
    run = result['run']
    assert run['status'] == 'complete'
    assert run['train_updates'] == 6 and run['training_examples'] == 384
    assert run['collection_steps'] == run['baseline_new_updates'] == run['support_draws'] == 0
    assert run['baseline_reconstructed_updates'] == 90000
    assert run['action_target_presentations'] == counts['targets'] < 4 * run['training_examples']
    assert run['learner_episodes'] == 36 and run['reference_episodes'] == 6
    assert len(result['trajectories']) == 14
    assert result['provenance']['snapshot_integrity']['expected'] == 9
    assert result['provenance']['snapshot_integrity']['complete']
    assert all(v['verified'] for v in result['provenance']['baseline_sampling_reconstruction'])
    for row in result['provenance']['paired_map_sampling_consistency']:
        assert row['complete'] and row['compared_updates'] == 2 and row['baseline_updates'] == 30000
        assert row['map_digest_identical'] and row['local_digest_identical']
        assert row['global_digest_identical'] and row['global_counts_identical']
    assert all(v['optimizer_tables_unchanged'] for v in result['provenance']['recorded_table_integrity'])
    for row in result['action_exposure']['per_seed']:
        if row['condition'] == 'recorded_actions':
            assert row['state_presentations'] == 128 and row['outside_support_target_queries'] == 0
            assert row['terminal_action_targets'] + row['nonterminal_action_targets'] == row['action_target_presentations']
        else:
            assert row['updates'] == 30000 and row['action_target_presentations'] == 7680000
    assert all(v['unchanged'] and v['read_only'] for v in result['provenance']['support_integrity'])
    assert not result['robustness']['eligible']
    for row in csv.DictReader((output / 'evaluations.csv').open()):
        assert int(row['checkpoint']) == (30000 if row['condition'] == 'collected_unique' else 2)
    for path, digest in json.loads((output / 'manifest.json').read_text())['files'].items():
        assert hashlib.sha256((output / path).read_bytes()).hexdigest() == digest


def test_recorded_table_preparation_cap_preserves_partial_evidence(tmp_path, monkeypatch):
    def stopped(*args, **kwargs):
        raise study.shared.MemoryReached('recorded-table preparation test cap')
    monkeypatch.setattr(study, 'build_recorded_table', stopped)
    result = fixture(tmp_path)
    assert result['run']['status'] == 'incomplete_memory_cap'
    assert result['run']['train_updates'] == result['run']['learner_episodes'] == 0
    assert not result['robustness']['eligible']
    assert (tmp_path / 'study/results.json').exists()


def test_global_state_stream_mismatch_disqualifies_completed_measurements(tmp_path, monkeypatch):
    original = study.RecordedActionsComparison.consistency
    def altered(self, *args, **kwargs):
        records = original(self, *args, **kwargs)
        records[0]['global_digest_identical'] = False
        records[0]['complete'] = False
        return records
    monkeypatch.setattr(study.RecordedActionsComparison, 'consistency', altered)
    result = fixture(tmp_path)
    assert result['run']['status'] == 'inconsistent_not_evidence'
    assert result['run']['learner_episodes'] == 36
    assert not result['robustness']['eligible']
