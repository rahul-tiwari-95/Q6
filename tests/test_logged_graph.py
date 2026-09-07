"""Fixed targets must solve only the logged graph and preserve state weighting."""
import copy
import csv
import hashlib
import json

import numpy as np
import pytest
import torch

import q6.logged_graph as study
from q6.learning import DQN


def graph_fixture():
    remaining = np.array([3, 1, 2, 2, 1], np.int32)
    maps = np.full(5, 7, np.int32)
    table = {'observed': np.zeros((5, 4), bool),
             'rewards': np.full((5, 4), np.nan),
             'ends': np.ones((5, 4), bool),
             'terminated': np.zeros((5, 4), bool),
             'truncated': np.zeros((5, 4), bool),
             'successor_indices': np.full((5, 4), -1, np.int32),
             'occurrences': np.zeros((5, 4), np.uint32)}
    edges = [(0, 0, .1, 3), (0, 2, -.4, -1), (1, 1, -.2, -1),
             (1, 3, -.1, -1), (2, 1, -.01, 1), (3, 0, -.05, 1),
             (3, 2, .1, 4), (4, 0, .3, -1)]
    for row, action, reward, following in edges:
        table['observed'][row, action] = True
        table['rewards'][row, action] = reward
        table['ends'][row, action] = following == -1
        table['terminated'][row, action] = following == -1 and remaining[row] > 1
        table['truncated'][row, action] = following == -1 and remaining[row] == 1
        table['successor_indices'][row, action] = following
        table['occurrences'][row, action] = 1
    return table, remaining, maps


def test_backward_values_follow_logged_branches_in_clock_order_without_mutation():
    table, remaining, maps = graph_fixture()
    before = {k: a.copy() for k, a in table.items()}
    visits = []
    q, summary = study.solve_logged_graph(table, remaining, maps, enforce=lambda: visits.append(1))
    assert q.dtype == np.float64 and not q.flags.writeable
    assert q[0, 0] == pytest.approx(.1 + .97 * (.1 + .97 * .3), abs=1e-14)
    assert q[2, 1] == pytest.approx(-.01 + .97 * -.1, abs=1e-14)
    assert q[3, 0] == pytest.approx(-.05 + .97 * -.1, abs=1e-14)
    assert q[0, 2] == -.4 and np.isnan(q[~table['observed']]).all()
    assert summary['supported_states'] == 5 and summary['observed_edges'] == 8
    assert summary['terminal_edges'] == summary['nonterminal_edges'] == 4
    assert summary['graph_preparation_successor_backups'] == 4
    assert summary['neural_successor_queries'] == 0 and len(visits) >= 3
    for key, value in before.items():
        np.testing.assert_array_equal(table[key], value)


def test_absent_outcomes_and_visit_frequency_cannot_change_graph_targets():
    table, remaining, maps = graph_fixture()
    expected, _ = study.solve_logged_graph(table, remaining, maps)
    absent = ~table['observed']
    table['rewards'][absent] = 1e30
    table['successor_indices'][absent] = 99999
    table['ends'][absent] = False
    table['occurrences'] *= 100
    actual, _ = study.solve_logged_graph(table, remaining, maps)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('fault', ['missing_successor', 'cross_map', 'clock_cycle', 'terminal_successor', 'end_flags', 'nonfinite_reward'])
def test_invalid_recorded_graph_has_no_oracle_or_unrestricted_fallback(fault):
    table, remaining, maps = graph_fixture()
    if fault == 'missing_successor':
        table['observed'][1] = False
    elif fault == 'cross_map':
        maps[1] += 1
    elif fault == 'clock_cycle':
        table['successor_indices'][0, 0] = 0
    elif fault == 'terminal_successor':
        table['successor_indices'][1, 1] = 0
    elif fault == 'end_flags':
        table['terminated'][1, 1] = True
    else:
        table['rewards'][0, 0] = np.nan
    with pytest.raises(study.ConsistencyError):
        study.solve_logged_graph(table, remaining, maps)


def zero_agent():
    a = DQN(92, seed=0)
    with torch.no_grad():
        for parameter in list(a.online.parameters()) + list(a.target.parameters()):
            parameter.zero_()
    return a


def test_fixed_update_uses_equal_state_loss_without_successor_queries(monkeypatch):
    a = zero_agent()
    observations = torch.zeros((2, 92))
    observed = torch.tensor([[True, False, False, False], [True, True, True, True]])
    labels = torch.tensor([[2., float('nan'), float('nan'), float('nan')], [1., 1., 1., 1.]], requires_grad=True)
    calls = []
    handle = a.online.register_forward_pre_hook(lambda _, args: calls.append(len(args[0])))
    def forbidden(*args, **kwargs):
        pytest.fail('fixed-target optimization must not query a successor target network')
    monkeypatch.setattr(a.target, 'forward', forbidden)
    loss = study.logged_graph_update(a, observations, observed, labels, [0, 1])
    handle.remove()
    assert loss == pytest.approx((1.5 + .5) / 2)
    assert calls == [2] and labels.grad is None
    assert a.online[-1].bias.grad.tolist() == pytest.approx([-.625, -.125, -.125, -.125])
    assert all(p.grad is None for p in a.target.parameters())
    for target, online in zip(a.target.parameters(), a.online.parameters()):
        torch.testing.assert_close(target, online * .01, rtol=1e-6, atol=1e-10)


def test_bad_sampled_labels_fail_before_any_optimizer_step():
    a = zero_agent()
    observed = torch.tensor([[True, False, False, False]])
    labels = torch.full((1, 4), float('nan'))
    before = a.parameter_hash()
    with pytest.raises(study.ConsistencyError, match='nonfinite'):
        study.logged_graph_update(a, torch.zeros((1, 92)), observed, labels, [0])
    assert a.parameter_hash() == before and not a.optimizer.state
    with pytest.raises(study.ConsistencyError, match='no logged target'):
        study.logged_graph_update(a, torch.zeros((1, 92)), observed & False, labels, [0])


def test_fit_metrics_distinguish_state_and_edge_weighting_and_missing_action_choices():
    observed = np.array([[1, 0, 0, 0], [1, 1, 1, 1]], bool)
    labels = np.array([[2., np.nan, np.nan, np.nan], [1., 1., 1., 1.]])
    predictions = np.array([[0., 9., 9., 9.], [0., 0., 0., 0.]], np.float32)
    result = study.logged_graph_metrics(predictions, labels, observed)
    assert result['states'] == 2 and result['observed_edges'] == 5
    assert result['state_mean_abs_error'] == 1.5
    assert result['state_mean_squared_error'] == 2.5
    assert result['edge_mean_abs_error'] == 1.2 and result['edge_max_abs_error'] == 2
    assert result['restricted_action_agreement'] == 1 and result['mean_graph_regret'] == 0
    assert result['unrestricted_argmax_outside_logged_fraction'] == .5


def test_fit_rank_uses_lowest_logged_tie_and_declared_float64_tolerance():
    observed = np.array([[0, 1, 1, 0], [1, 1, 0, 0]], bool)
    labels = np.array([[np.nan, 1. - 5e-7, 1., np.nan], [0., 1., np.nan, np.nan]])
    predictions = np.zeros((2, 4), np.float32)
    result = study.logged_graph_metrics(predictions, labels, observed)
    assert result['restricted_action_agreement'] == .5
    assert result['mean_graph_regret'] == pytest.approx((1 + 5e-7) / 2, abs=1e-15)
    assert result['max_graph_regret'] == 1
    assert result['restricted_ranking_atol'] == 1e-6 and result['restricted_ranking_rtol'] == 0


def fixture(tmp_path, **kwargs):
    protocol = tmp_path / 'protocol.md'
    protocol.write_text('Logged graph execution test, not research evidence.\n')
    return study.run_study(tmp_path / 'study', protocol, seeds=[0], updates=2,
        panel_count=1, maps_per_panel=2, panel_seed_start=1200000, smoke=True, **kwargs)


def test_exact_replay_frozen_controls_and_final_fit_ordering(tmp_path, monkeypatch):
    output = tmp_path / 'study'
    original_update = study.logged_graph_update
    original_fit = study.LoggedGraphComparison.final_fit_diagnostics
    original_rollout = study.shared.rollout
    counts = {'updates': 0, 'targets': 0, 'fit_calls': 0, 'rollouts': 0}
    def forbidden(*args, **kwargs):
        pytest.fail('no bootstrap queries, control training, collection or online learning')
    def checked_update(agent, observations, mask, labels, indices):
        assert counts['fit_calls'] == counts['rollouts'] == 0
        with np.load(output / 'logged_graph_targets.npz') as arrays:
            assert len(arrays.files) == 6
            for bank in (1, 2, 3):
                assert arrays[f'bank{bank}_targets_float32'].dtype == np.float32
                relative = f'models/bank{bank}_constrained_bootstrap_seed0_update30000.pt'
                assert (output / relative).read_bytes() == (study.shared.ROOT / 'experiments/constrained_bootstrap/pilot_v1' / relative).read_bytes()
        monkeypatch.setattr(agent.target, 'forward', forbidden)
        batches = []
        handle = agent.online.register_forward_pre_hook(lambda _, args: batches.append(len(args[0])))
        try:
            result = original_update(agent, observations, mask, labels, indices)
        finally:
            handle.remove()
        assert batches == [64]
        counts['updates'] += 1
        counts['targets'] += int(mask[indices].sum())
        return result
    def checked_fit(self, models, records, *args, **kwargs):
        assert counts['updates'] == 6 and counts['rollouts'] == 0
        assert all((output / f'models/bank{b}_logged_graph_seed0_update2.pt').exists() for b in (1, 2, 3))
        before = {key: (m.parameter_hash(), study.module_hash(m.target)) for key, m in models.items()}
        optimizer_counts = copy.deepcopy(self.optimizer_counts)
        rng = torch.get_rng_state().clone()
        result = original_fit(self, models, records, *args, **kwargs)
        assert before == {key: (m.parameter_hash(), study.module_hash(m.target)) for key, m in models.items()}
        assert optimizer_counts == self.optimizer_counts
        torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
        counts['fit_calls'] += 1
        return result
    def checked_rollout(*args, **kwargs):
        assert counts['updates'] == 6 and counts['fit_calls'] == 1
        counts['rollouts'] += 1
        return original_rollout(*args, **kwargs)
    monkeypatch.setattr(study, 'logged_graph_update', checked_update)
    monkeypatch.setattr(study.LoggedGraphComparison, 'final_fit_diagnostics', checked_fit)
    monkeypatch.setattr(study.shared, 'rollout', checked_rollout)
    monkeypatch.setattr(study.shared, 'fixed_update', forbidden)
    monkeypatch.setattr(study.shared.DQN, 'observe', forbidden)
    monkeypatch.setattr(study.shared.DQN, 'learn', forbidden)
    import q6.constrained_bootstrap, q6.recorded_actions, q6.coverage
    monkeypatch.setattr(q6.constrained_bootstrap, 'constrained_update', forbidden)
    monkeypatch.setattr(q6.recorded_actions, 'recorded_update', forbidden)
    monkeypatch.setattr(q6.coverage, 'collect_support', forbidden)
    result = fixture(tmp_path)
    run = result['run']
    assert run['status'] == 'complete' and run['interpretation'] == 'smoke_or_deviation_descriptive_only'
    assert run['train_updates'] == 6 and run['training_examples'] == 384
    assert run['action_target_presentations'] == counts['targets']
    assert run['nonterminal_target_queries'] == run['checkpoint_probes'] == run['probe_query_count'] == 0
    assert run['baseline_new_updates'] == run['collection_steps'] == run['support_draws'] == 0
    assert run['learner_episodes'] == 36 and run['reference_episodes'] == 6
    assert len(result['trajectories']) == 14 and not result['robustness']['eligible']
    assert result['provenance']['snapshot_integrity']['expected'] == 9
    assert all(r['global_digest_identical'] and r['complete'] for r in result['provenance']['paired_map_sampling_consistency'])
    fit = result['fit_diagnostics']
    assert fit['integrity']['complete'] and fit['integrity']['actual_models'] == 6
    assert fit['inference_state_rows'] == 359322 and fit['observed_edge_comparisons'] == 504658
    assert fit['neural_successor_queries'] == 0
    with np.load(output / 'fit_predictions.npz') as arrays:
        for bank in (1, 2, 3):
            assert len(arrays[f'bank{bank}_state_rows']) in (59839, 59984, 59838)
        assert len(arrays.files) == 9
    with np.load(output / 'recorded_query_counts.npz') as arrays:
        for key in arrays.files:
            assert bool(arrays[key].sum()) == ('constrained_bootstrap' in key)
    old = json.loads((study.shared.ROOT / 'experiments/constrained_bootstrap/pilot_v1/action_exposure.json').read_text())
    for row in result['action_exposure']['per_seed']:
        if row['condition'] == 'constrained_bootstrap':
            archived = next(v for v in old['per_seed'] if v['condition'] == 'constrained_bootstrap' and v['bank_id'] == row['bank_id'] and v['seed'] == row['seed'])
            for field in ('action_target_presentations', 'nonterminal_target_queries'):
                assert row[field] == archived[field]
        else:
            assert row['nonterminal_action_targets'] > 0 and row['nonterminal_target_queries'] == 0
    for row in csv.DictReader((output / 'evaluations.csv').open()):
        assert int(row['checkpoint']) == (30000 if row['condition'] == 'constrained_bootstrap' else 2)
    for path, digest in json.loads((output / 'manifest.json').read_text())['files'].items():
        assert hashlib.sha256((output / path).read_bytes()).hexdigest() == digest


def test_graph_preparation_cap_preserves_partial_evidence(tmp_path, monkeypatch):
    def stopped(*args, **kwargs):
        raise study.shared.MemoryReached('graph preparation test cap')
    monkeypatch.setattr(study, 'solve_logged_graph', stopped)
    result = fixture(tmp_path)
    assert result['run']['status'] == 'incomplete_memory_cap'
    assert result['run']['train_updates'] == result['run']['learner_episodes'] == 0
    assert not result['robustness']['eligible']
    assert (tmp_path / 'study/results.json').exists()


def test_graph_fit_cap_preserves_finished_training_without_policy_evaluation(tmp_path, monkeypatch):
    def stopped(*args, **kwargs):
        raise study.shared.MemoryReached('final graph fit test cap')
    monkeypatch.setattr(study.LoggedGraphComparison, 'final_fit_diagnostics', stopped)
    result = fixture(tmp_path)
    assert result['run']['status'] == 'incomplete_memory_cap'
    assert result['run']['train_updates'] == 6 and result['run']['learner_episodes'] == 0
    assert not result['robustness']['eligible']
    assert (tmp_path / 'study/models/bank3_logged_graph_seed0_update2.pt').exists()


def test_replay_mismatch_disqualifies_logged_graph_result(tmp_path, monkeypatch):
    original = study.LoggedGraphComparison.consistency
    def altered(self, *args, **kwargs):
        rows = original(self, *args, **kwargs)
        rows[0]['global_digest_identical'] = rows[0]['complete'] = False
        return rows
    monkeypatch.setattr(study.LoggedGraphComparison, 'consistency', altered)
    result = fixture(tmp_path)
    assert result['run']['status'] != 'complete' and not result['robustness']['eligible']
