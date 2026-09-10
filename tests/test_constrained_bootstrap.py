"""Constrain training targets without changing logged supervision or evaluation."""
import copy
import csv
import hashlib
import json

import numpy as np
import pytest
import torch

import q6.constrained_bootstrap as study
from q6.learning import DQN
from q6.recorded_actions import recorded_update
from q6.optimal import VisibleOptimalQ
from q6.world import WorldConfig


def constant_agent(online=(0., 4., 2., 1.), target=(7., 1., 9., 3.)):
    agent = DQN(92, seed=0)
    with torch.no_grad():
        for parameter in list(agent.online.parameters()) + list(agent.target.parameters()):
            parameter.zero_()
        agent.online[-1].bias.copy_(torch.tensor(online))
        agent.target[-1].bias.copy_(torch.tensor(target))
    return agent


def table_fixture():
    table = {'observed': torch.zeros((2, 4), dtype=torch.bool),
             'rewards': torch.full((2, 4), float('nan')),
             'ends': torch.zeros((2, 4), dtype=torch.bool),
             'successor_indices': torch.full((2, 4), 999999, dtype=torch.int32)}
    table['observed'][0, 0] = True
    table['observed'][1, 2] = True
    table['rewards'][0, 0] = .2
    table['rewards'][1, 2] = -.3
    table['successor_indices'][0, 0] = 1
    table['successor_indices'][1, 2] = -1
    table['ends'][1, 2] = True
    return torch.zeros((2, 92)), table


def test_logged_online_argmax_can_raise_target_despite_lower_online_value():
    agent = constant_agent()
    observations, table = table_fixture()
    before_online, before_target = agent.parameter_hash(), study.module_hash(agent.target)
    rng = copy.deepcopy(agent.rng.bit_generator.state)
    counts = []
    h1 = agent.online.register_forward_pre_hook(lambda _, args: counts.append(('online', len(args[0]))))
    h2 = agent.target.register_forward_pre_hook(lambda _, args: counts.append(('target', len(args[0]))))
    result = study.constrained_targets(agent, observations, table, [0], include_edges=True)
    h1.remove(); h2.remove()
    assert counts == [('online', 1), ('target', 1)]
    assert result['targets'].item() == pytest.approx(.2 + .97 * 9)
    assert not result['targets'].requires_grad
    diag, edge = result['diagnostics'], result['edges'][0]
    assert edge['unrestricted_action'] == 1 and edge['restricted_action'] == 2
    assert diag['query_count'] == diag['outside_count'] == diag['positive_target_deltas'] == 1
    assert diag['online_gap_sum'] == 2
    assert diag['target_delta_sum'] == pytest.approx(.97 * 8, abs=1e-6)
    assert edge['target_delta'] == pytest.approx(edge['restricted_target'] - edge['unrestricted_target'])
    assert before_online == agent.parameter_hash() and before_target == study.module_hash(agent.target)
    assert rng == agent.rng.bit_generator.state and not agent.optimizer.state


def test_outside_argmax_tie_can_activate_constraint_with_zero_online_gap():
    agent = constant_agent(online=(4., 4., 2., 1.))
    observations, table = table_fixture()
    table['observed'][1] = torch.tensor([False, True, True, False])
    result = study.constrained_targets(agent, observations, table, [0], include_edges=True)
    edge = result['edges'][0]
    assert edge['unrestricted_action'] == 0 and edge['restricted_action'] == 1
    assert edge['online_gap'] == 0 and edge['unrestricted_argmax_outside']
    assert result['diagnostics']['negative_target_deltas'] == 1


def test_empty_successor_mask_has_no_unrestricted_fallback():
    agent = constant_agent()
    observations, table = table_fixture()
    table['observed'][1] = False
    before = agent.parameter_hash()
    with pytest.raises(study.ConsistencyError, match='no logged action'):
        study.constrained_update(agent, observations, table, [0])
    assert before == agent.parameter_hash() and not agent.optimizer.state


def test_terminal_and_absent_outcomes_never_trigger_successor_queries(monkeypatch):
    agent = constant_agent()
    observations, table = table_fixture()
    observations[0] = float('nan')
    def forbidden(*args, **kwargs):
        pytest.fail('terminal-only targets must not query either successor network')
    monkeypatch.setattr(agent.target, 'forward', forbidden)
    monkeypatch.setattr(agent.online, 'forward', forbidden)
    result = study.constrained_targets(agent, observations, table, [1], include_edges=True)
    assert result['targets'].item() == pytest.approx(-.3)
    assert result['diagnostics']['query_count'] == 0 and result['edges'] == []
    assert result['diagnostics']['target_delta_mean'] is None
    assert result['diagnostics']['target_delta_min'] is None


def test_all_allowed_successor_actions_recover_recorded_update_and_equal_state_loss():
    torch.set_num_threads(1)
    a, b = constant_agent(), constant_agent()
    observations, table = table_fixture()
    # State0 has one current action; its successor has four. Equal-state
    # normalization must survive this unequal action coverage.
    table['observed'][1] = True
    table['rewards'][1] = .2
    table['ends'][1] = True
    table['successor_indices'][1] = -1
    old = recorded_update(a, observations, table, [0, 1])
    new, diagnostics = study.constrained_update(b, observations, table, [0, 1])
    assert new == pytest.approx(old, abs=1e-7)
    assert diagnostics['outside_count'] == 0 and diagnostics['target_delta_sum'] == 0
    for left, right in zip(a.online.parameters(), b.online.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for left, right in zip(a.target.parameters(), b.target.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert all(p.grad is None for p in b.target.parameters())


def test_diagnostic_aggregation_weights_query_presentations_and_preserves_signs():
    observations, table = table_fixture()
    positive = study.constrained_targets(constant_agent(), observations, table, [0])['diagnostics']
    negative = study.constrained_targets(constant_agent(target=(7., 9., 1., 3.)), observations, table, [0])['diagnostics']
    combined = study.combine_diagnostics([positive, negative, study.empty_diagnostics()])
    assert combined['query_count'] == 2 and combined['outside_fraction'] == 1
    assert combined['positive_target_deltas'] == combined['negative_target_deltas'] == 1
    assert combined['target_delta_min'] < 0 < combined['target_delta_max']
    assert combined['target_delta_mean'] == pytest.approx((positive['target_delta_sum'] + negative['target_delta_sum']) / 2)


def test_fresh_policy_rollout_still_chooses_among_all_four_actions():
    agent = constant_agent()
    observations, table = table_fixture()
    probe = study.constrained_targets(agent, observations, table, [0], include_edges=True)
    assert probe['edges'][0]['restricted_action'] == 2
    config = WorldConfig()
    _, trajectory = study.shared.rollout(agent, config, VisibleOptimalQ(config),
        condition='constrained_bootstrap', seed=0, checkpoint=0, mode='greedy',
        panel='test', map_seed=1180000)
    assert trajectory['steps']
    assert all(step['action'] == 1 for step in trajectory['steps'])


def fixture(tmp_path, **kwargs):
    protocol = tmp_path / 'protocol.md'
    protocol.write_text('Constrained bootstrap execution test, not research evidence.\n')
    return study.run_study(tmp_path / 'study', protocol, seeds=[0], updates=2,
        panel_count=1, maps_per_panel=2, panel_seed_start=1180000, smoke=True, **kwargs)


def test_recorded_controls_exact_replay_and_pure_probes_precede_final_rollouts(tmp_path, monkeypatch):
    original_update, original_rollout = study.constrained_update, study.shared.rollout
    original_probe = study.ConstrainedBootstrapComparison.checkpoint_probe
    counts = {'updates': 0, 'rollouts': 0, 'probes': 0, 'targets': 0, 'queries': 0}
    output = tmp_path / 'study'
    def checked_update(*args, **kwargs):
        assert counts['rollouts'] == 0
        with np.load(output / 'supports.npz') as arrays:
            assert len(arrays.files) == 6
            for bank in (1, 2, 3):
                np.testing.assert_array_equal(arrays[f'recorded_actions_bank{bank}'], arrays[f'constrained_bootstrap_bank{bank}'])
                relative = f'models/bank{bank}_recorded_actions_seed0_update30000.pt'
                assert (output / relative).read_bytes() == (study.shared.ROOT / 'experiments/recorded_actions/pilot_v1' / relative).read_bytes()
        counts['updates'] += 1
        observed = args[2]['observed'][args[3]]
        counts['targets'] += int(observed.sum())
        counts['queries'] += int((observed & ~args[2]['ends'][args[3]]).sum())
        return original_update(*args, **kwargs)
    def checked_probe(self, agent, *args, **kwargs):
        before = (agent.parameter_hash(), study.module_hash(agent.target), copy.deepcopy(self.optimizer_counts))
        steps = [float(v['step']) for v in agent.optimizer.state.values()]
        rng = torch.get_rng_state().clone()
        result = original_probe(self, agent, *args, **kwargs)
        assert before == (agent.parameter_hash(), study.module_hash(agent.target), self.optimizer_counts)
        assert steps == [float(v['step']) for v in agent.optimizer.state.values()]
        torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
        counts['probes'] += 1
        return result
    def checked_rollout(*args, **kwargs):
        assert counts['updates'] == 6 and counts['probes'] == 6
        assert all((output / f'models/bank{b}_constrained_bootstrap_seed0_update2.pt').exists() for b in (1, 2, 3))
        counts['rollouts'] += 1
        return original_rollout(*args, **kwargs)
    def forbidden(*args, **kwargs):
        pytest.fail('this study must not train controls, collect or learn online')
    monkeypatch.setattr(study, 'constrained_update', checked_update)
    monkeypatch.setattr(study.ConstrainedBootstrapComparison, 'checkpoint_probe', checked_probe)
    monkeypatch.setattr(study.shared, 'rollout', checked_rollout)
    monkeypatch.setattr(study.shared, 'fixed_update', forbidden)
    monkeypatch.setattr(study.shared.DQN, 'observe', forbidden)
    monkeypatch.setattr(study.shared.DQN, 'learn', forbidden)
    import q6.recorded_actions, q6.coverage
    monkeypatch.setattr(q6.recorded_actions, 'recorded_update', forbidden)
    monkeypatch.setattr(q6.coverage, 'collect_support', forbidden)
    result = fixture(tmp_path)
    run = result['run']
    assert run['status'] == 'complete' and run['train_updates'] == 6
    assert run['training_examples'] == 384
    assert run['action_target_presentations'] == counts['targets']
    assert run['nonterminal_target_queries'] == counts['queries']
    assert run['baseline_new_updates'] == run['collection_steps'] == run['support_draws'] == 0
    assert run['learner_episodes'] == 36 and run['reference_episodes'] == 6
    assert len(result['trajectories']) == 14 and not result['robustness']['eligible']
    assert result['provenance']['snapshot_integrity']['expected'] == 9
    assert all(v['global_digest_identical'] and v['complete'] for v in result['provenance']['paired_map_sampling_consistency'])
    old = json.loads((study.shared.ROOT / 'experiments/recorded_actions/pilot_v1/action_exposure.json').read_text())
    for row in result['action_exposure']['per_seed']:
        assert row['outside_support_target_queries'] == 0
        if row['condition'] == 'recorded_actions':
            archived = next(v for v in old['per_seed'] if v['condition'] == 'recorded_actions' and v['bank_id'] == row['bank_id'] and v['seed'] == row['seed'])
            assert row['action_target_presentations'] == archived['action_target_presentations']
    for row in csv.DictReader((output / 'evaluations.csv').open()):
        assert int(row['checkpoint']) == (30000 if row['condition'] == 'recorded_actions' else 2)
    for path, digest in json.loads((output / 'manifest.json').read_text())['files'].items():
        assert hashlib.sha256((output / path).read_bytes()).hexdigest() == digest


def test_input_preparation_cap_preserves_partial_evidence(tmp_path, monkeypatch):
    def stopped(*args, **kwargs):
        raise study.shared.MemoryReached('constrained input preparation test cap')
    monkeypatch.setattr(study.ConstrainedBootstrapComparison, 'prepare_inputs', stopped)
    result = fixture(tmp_path)
    assert result['run']['status'] == 'incomplete_memory_cap'
    assert result['run']['train_updates'] == result['run']['learner_episodes'] == 0
    assert not result['robustness']['eligible']
    assert (tmp_path / 'study/results.json').exists()


def test_replay_mismatch_disqualifies_completed_comparison(tmp_path, monkeypatch):
    original = study.ConstrainedBootstrapComparison.consistency
    def altered(self, *args, **kwargs):
        rows = original(self, *args, **kwargs)
        rows[0]['global_digest_identical'] = False
        rows[0]['complete'] = False
        return rows
    monkeypatch.setattr(study.ConstrainedBootstrapComparison, 'consistency', altered)
    result = fixture(tmp_path)
    assert result['run']['status'] == 'inconsistent_not_evidence'
    assert result['run']['learner_episodes'] == 36
    assert not result['robustness']['eligible']
