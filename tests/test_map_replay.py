"""Fixed support, map exposure, archived control identity and bounded execution."""
import csv
import hashlib
import json

import numpy as np
import pytest

import q6.map_replay as study


def test_map_schedule_survives_different_support_sizes_and_global_rng_use():
    maps = np.repeat(np.arange(300000, 300256), 4)
    scarce = np.arange(0, len(maps), 4, dtype=np.int32)
    broad = np.arange(len(maps), dtype=np.int32)
    a, b = study.MapSampler(scarce, maps, 2), study.MapSampler(broad, maps, 2)
    for _ in range(20):
        left = a.next_batch()
        np.random.random(17)  # Unrelated users of global RNG cannot steer sampling.
        right = b.next_batch()
        assert len(np.unique(maps[left])) == len(np.unique(maps[right])) == 64
        assert len(np.unique(left)) == len(np.unique(right)) == 64
        np.testing.assert_array_equal(maps[left], maps[right])
        assert np.isin(left, scarce).all() and np.isin(right, broad).all()
    assert a.map_digest.hexdigest() == b.map_digest.hexdigest()
    np.testing.assert_array_equal(a.map_counts, b.map_counts)
    assert a.counts.sum() == b.counts.sum() == 20 * 64
    assert a.digest.hexdigest() != b.digest.hexdigest()
    replay = study.MapSampler(scarce, maps, 2)
    for _ in range(20):
        replay.next_batch()
    assert replay.digest.hexdigest() == a.digest.hexdigest()
    np.testing.assert_array_equal(a.local_counts, a.counts[scarce])


def test_every_map_gets_equal_exposure_when_batch_contains_all_maps():
    maps = np.repeat(np.arange(64), np.arange(1, 65))
    support = np.arange(len(maps), dtype=np.int32)
    sampler = study.MapSampler(support, maps, 0)
    for _ in range(17):
        rows = sampler.next_batch()
        assert sorted(maps[rows].tolist()) == list(range(64))
    np.testing.assert_array_equal(sampler.map_counts, np.full(64, 17))
    assert sampler.counts[0] == 17  # Sparse maps repeat existing rows, never invent states.
    assert sampler.local_counts.sum() == 17 * 64
    with pytest.raises(study.ConsistencyError, match="every training map"):
        study.MapSampler(support[1:], maps, 0)
    with pytest.raises(study.ConsistencyError):
        study.MapSampler(np.array([0, 0]), maps, 0)
    with pytest.raises(study.ConsistencyError):
        study.MapSampler(np.arange(63), np.arange(63), 0)


def fixture(tmp_path, **kwargs):
    protocol = tmp_path / 'protocol.md'
    protocol.write_text('Execution-only test on shipped supports.\n')
    return study.run_study(tmp_path / 'study', protocol, seeds=[0], updates=2,
        panel_count=1, maps_per_panel=2, panel_seed_start=1120000, smoke=True, **kwargs)


def test_reuses_every_support_and_control_before_updates_then_evaluates_frozen(tmp_path, monkeypatch):
    update, rollout = study.fixed_update, study.rollout
    sampler_class = study.MapSampler
    state = {'updates': 0, 'rollouts': 0, 'frozen_supports': []}
    output = tmp_path / 'study'
    class ObservedSampler(sampler_class):
        def next_batch(self):
            assert not self.support.flags.writeable
            return super().next_batch()
    def checked_update(*args, **kwargs):
        assert state['rollouts'] == 0
        with np.load(output / 'supports.npz') as saved:
            assert len(saved.files) == 6
            for bank in (1, 2, 3):
                np.testing.assert_array_equal(saved[f'collected_unique_bank{bank}'], saved[f'map_balanced_bank{bank}'])
                assert (output / f'models/bank{bank}_collected_unique_seed0_update30000.pt').exists()
        state['updates'] += 1
        return update(*args, **kwargs)
    def checked_rollout(*args, **kwargs):
        assert state['updates'] == 6
        assert all((output / f'models/bank{b}_map_balanced_seed0_update2.pt').exists() for b in (1, 2, 3))
        state['rollouts'] += 1
        return rollout(*args, **kwargs)
    def forbidden(*args, **kwargs):
        pytest.fail('offline replay control must not collect or use online replay learning')
    monkeypatch.setattr(study, 'MapSampler', ObservedSampler)
    monkeypatch.setattr(study, 'fixed_update', checked_update)
    monkeypatch.setattr(study, 'rollout', checked_rollout)
    monkeypatch.setattr(study.DQN, 'observe', forbidden)
    monkeypatch.setattr(study.DQN, 'learn', forbidden)
    import q6.coverage
    monkeypatch.setattr(q6.coverage, 'collect_support', forbidden)
    result = fixture(tmp_path)
    run = result['run']
    assert run['status'] == 'complete'
    assert run['train_updates'] == 6 and run['training_examples'] == 384
    assert run['baseline_new_updates'] == run['collection_steps'] == run['support_draws'] == 0
    assert run['fits'] == run['frozen_baselines'] == 3
    assert run['learner_episodes'] == 36 and run['reference_episodes'] == 6
    assert len(result['trajectories']) == 14
    assert result['provenance']['snapshot_integrity']['expected'] == 9
    assert result['provenance']['snapshot_integrity']['complete']
    assert all(x['unchanged'] and x['read_only'] for x in result['provenance']['support_integrity'])
    assert all(x['unchanged_during_evaluation'] for x in result['provenance']['models'])
    assert not result['robustness']['eligible']
    rows = list(csv.DictReader((output / 'evaluations.csv').open()))
    for row in rows:
        assert int(row['checkpoint']) == (30000 if row['condition'] == 'collected_unique' else 2)
    for x in result['exposure']['summaries']:
        assert x['presentations'] == (1920000 if x['condition'] == 'collected_unique' else 128)
        assert x['outside_support_direct_samples'] == 0
        maps = [y for y in result['exposure']['per_map'] if all(x[k] == y[k] for k in ('bank_id','condition','seed'))]
        assert sum(y['presentation_fraction'] for y in maps) == pytest.approx(1)
        assert sum(y['presentations'] for y in maps) == x['presentations']
    for x in result['pooled']['aggregate']:
        assert x['noop_rate'] == pytest.approx(x['noop_steps'] / x['evaluation_steps'])
    manifest = json.loads((output / 'manifest.json').read_text())
    for path, digest in manifest['files'].items():
        assert hashlib.sha256((output / path).read_bytes()).hexdigest() == digest


def test_nonfinite_update_keeps_control_and_initial_snapshot_without_evaluation(tmp_path, monkeypatch):
    monkeypatch.setattr(study, 'fixed_update', lambda *a, **k: float('nan'))
    result = fixture(tmp_path)
    assert result['run']['status'] == 'inconsistent_not_evidence'
    assert 'nonfinite' in result['run']['stop_reason']
    assert result['run']['learner_episodes'] == 0
    assert not result['robustness']['eligible']
    assert (tmp_path / 'study/models/bank1_map_balanced_seed0_update0.pt').is_file()
    assert (tmp_path / 'study/models/bank3_collected_unique_seed0_update30000.pt').is_file()


def test_post_aggregation_resource_guard_preserves_but_disqualifies_results(tmp_path, monkeypatch):
    original = study.descriptive_summaries
    def after_summary(*args, **kwargs):
        result = original(*args, **kwargs)
        def cap(*args, **kwargs):
            raise study.MemoryReached('post-aggregation test cap')
        monkeypatch.setattr(study, 'guard', cap)
        return result
    monkeypatch.setattr(study, 'descriptive_summaries', after_summary)
    result = fixture(tmp_path)
    assert result['run']['status'] == 'incomplete_memory_cap'
    assert result['run']['learner_episodes'] == 36
    assert not result['robustness']['eligible'] and not result['descriptive_thresholds']['eligible']
    assert not any(x['success_reference_met'] for x in result['descriptive_thresholds']['per_seed'])


def test_preparation_guard_prevents_training(tmp_path, monkeypatch):
    def cap(*args, **kwargs):
        raise study.MemoryReached('preparation test cap')
    monkeypatch.setattr(study, 'guard', cap)
    result = fixture(tmp_path)
    assert result['run']['status'] == 'incomplete_memory_cap'
    assert result['run']['train_updates'] == result['run']['learner_episodes'] == 0
    assert not result['robustness']['eligible']
    assert (tmp_path / 'study/results.json').is_file()
