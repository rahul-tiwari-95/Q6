"""Regression and artifact checks for the bounded provenance release."""
import gzip
import json
from dataclasses import asdict

import numpy as np
import pytest

from no_way_home.learning import RATE_BINS, TabularQMandateLearner, feature_raw_rate
from no_way_home.policies import never_mitigate
from no_way_home.run_provenance_release import CountingPolicy, outcome, run_pilot
from no_way_home.stats import jonckheere_terpstra_test
from no_way_home.world import WorldConfig, WorldState, run, step


def test_seeded_message_ids_do_not_depend_on_prior_runs():
    """A second run must reproduce lineage IDs, not just aggregate metrics."""
    cfg = WorldConfig(n_ticks=20, report_sick_threshold=1)
    first = run(cfg, never_mitigate, 3)
    run(cfg, never_mitigate, 900)
    second = run(cfg, never_mitigate, 3)
    assert first.messages.messages
    assert first.events == second.events
    assert first.messages.messages == second.messages.messages


def test_terminal_reward_is_consumed_once_without_cross_episode_bootstrap():
    """One-step episodes previously never performed even one Q update."""
    cfg = WorldConfig(n_ticks=1, starting_food_stock=0, base_food_yield_per_agent=0,
                      medicine_production_per_tick=0, starting_medicine_stock=0,
                      base_sickness_prob=1)
    learner = TabularQMandateLearner(feature_fns=[feature_raw_rate], bin_edges=[RATE_BINS], epsilon=0)
    learner.q_table[((0,), True)] = 100.0  # future values must not enter terminal target
    state = run(cfg, learner, 0)
    assert learner._q((0,), True) == -state.events[-1]['shortfall_this_tick']
    assert sum(learner.visit_counts.values()) == 1
    learner.end_episode(state)
    assert sum(learner.visit_counts.values()) == 1
    run(cfg, learner, 1)
    assert sum(learner.visit_counts.values()) == 2
    assert learner._last_bin is None


def test_every_transition_is_learned_and_frozen_evaluation_learns_none():
    cfg = WorldConfig(n_ticks=8)
    learner = TabularQMandateLearner(feature_fns=[feature_raw_rate], bin_edges=[RATE_BINS])
    run(cfg, learner, 0)
    assert sum(learner.visit_counts.values()) == 8
    before = learner.q_table.copy()
    learner.freeze()
    run(cfg, learner, 1)
    assert learner.q_table == before
    assert sum(learner.visit_counts.values()) == 8


def test_paired_permutation_retains_seed_offsets():
    """Large seed offsets must not drown a consistent within-seed contrast."""
    baseline = np.arange(12) * 1000
    groups = [baseline + offset * 0.01 for offset in range(4)]
    _, paired_p = jonckheere_terpstra_test(groups, 'increasing', 999, np.random.default_rng(7), paired=True)
    _, independent_p = jonckheere_terpstra_test(groups, 'increasing', 999, np.random.default_rng(7))
    assert paired_p < 0.01
    assert independent_p > 0.1
    with pytest.raises(ValueError, match='same number'):
        jonckheere_terpstra_test([[1, 2], [3]], 'increasing', 99, np.random.default_rng(0), paired=True)


def test_incremental_decay_matches_full_timestamp_sum_and_ignores_lineage():
    """Optimizing the EMA must preserve the declared control exactly."""
    cfg = WorldConfig(n_ticks=100, shift_tick=30, report_sick_threshold=1)
    state = WorldState.initial(cfg)
    policy = CountingPolicy('decay', 0.3)
    rng = np.random.default_rng(3)
    decay = 2 ** (-1 / policy.half_life)
    normalization = decay / (1 - decay)
    for _ in range(cfg.n_ticks):
        step(state, False, rng)
        expected = state.messages.bounded_decay_rate(state.tick + 1, policy.half_life,
                                                     0, 2 * normalization) / normalization
        assert policy.score(state) == pytest.approx(expected)
        for message in state.messages.messages:
            message.origin_id = 0  # all lineages collapse; decay score must be unchanged
        assert policy.score(state) == pytest.approx(expected)
    policy.start_episode()
    assert policy.score(WorldState.initial(cfg)) == 0


def test_missed_response_is_counted_in_delay_metric():
    cfg = WorldConfig(n_ticks=8, shift_tick=3)
    state = run(cfg, never_mitigate, 0)
    metrics = outcome(state)
    assert not metrics['false_alarm']
    assert metrics['missed_crisis']
    assert metrics['response_delay_ticks'] == 6
    assert metrics['post_shift_ticks'] == 6


def test_tiny_release_keeps_equal_calibration_budget_and_recomputable_traces(tmp_path):
    cfg = WorldConfig(n_ticks=16, shift_tick=7, report_sick_threshold=1)
    scenarios = {'tiny': ('Tiny integration fixture', cfg)}
    out = tmp_path / 'run'
    dashboard = run_pilot(out, cfg=cfg, scenarios=scenarios,
                          calibration_seeds=[1, 2], evaluation_seeds=[3, 4],
                          thresholds=[0.3, 0.8], n_resamples=50)
    calibration = json.loads((out / 'calibration.json').read_text())
    for kind in ('raw', 'unique', 'decay'):
        assert len([row for row in calibration if row['policy_id'] == kind]) == 4
    evaluation = json.loads((out / 'evaluation.json').read_text())
    assert len(evaluation) == 8
    with gzip.open(out / 'evaluation-traces.jsonl.gz', 'rt') as stream:
        traces = [json.loads(line) for line in stream]
    assert len(traces) == 8
    for trace, record in zip(traces, evaluation):
        loss = sum(event['shortfall_this_tick'] for event in trace['events']) / cfg.n_ticks * 10000
        assert loss == record['shortfall_per_10k']
        assert trace['config'] == asdict(cfg)
    assert len(dashboard['scenarios'][0]['policy_results']) == 4
    assert json.loads((out / 'metadata.json').read_text())['provenance']['protocol_sha256']
    with pytest.raises(FileExistsError):
        run_pilot(out)
    with pytest.raises(ValueError, match='disjoint'):
        run_pilot(tmp_path / 'bad', calibration_seeds=[1], evaluation_seeds=[1])
