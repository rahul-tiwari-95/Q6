"""Quota-preserving support changes keep archived replay's ordered map schedule."""
import csv
import hashlib
import json

import numpy as np
import pytest

import q6.within_map as study


def test_quota_draw_keeps_counts_and_uses_owned_map_rng_without_outcome_selection():
    maps = np.repeat(np.arange(300000, 300004), 32)
    support = np.array([0, 1, 32, 34, 37, 65, 68, 70, 71, 98], np.int32)
    replacement = study.quota_support(support, maps, 1)
    np.random.random(29)
    np.testing.assert_array_equal(replacement, study.quota_support(support, maps, 1))
    expected = []
    for m in np.unique(maps):
        candidates = np.flatnonzero(maps == m)
        size = np.count_nonzero(maps[support] == m)
        rng = np.random.default_rng(np.random.SeedSequence([1, int(m), 105301]))
        expected.extend(candidates[rng.choice(len(candidates), size=size, replace=False)])
    np.testing.assert_array_equal(replacement, sorted(expected))
    np.testing.assert_array_equal(maps[replacement], maps[support])
    assert not replacement.flags.writeable
    assert not np.array_equal(replacement, support)
    assert not np.array_equal(replacement, study.quota_support(support, maps, 2))


def test_empty_quota_stays_empty_and_interleaved_map_blocks_are_rejected():
    maps = np.repeat(np.arange(4), 5)
    support = np.array([0, 2, 11, 15, 16], np.int32)
    result = study.quota_support(support, maps, 1)
    assert not np.any(maps[result] == 1)
    np.testing.assert_array_equal(maps[result], maps[support])
    with pytest.raises(study.ConsistencyError, match='contiguous'):
        study.quota_support(np.arange(4, dtype=np.int32), np.array([0, 1, 0, 1]), 1)
    with pytest.raises(study.ConsistencyError):
        study.quota_support(np.array([1, 1]), maps, 1)


def test_original_sampler_matches_map_order_and_multiplicity_not_only_totals():
    maps = np.repeat(np.arange(300000, 300128), 8)
    original = np.flatnonzero(np.arange(len(maps)) % 8 < 3).astype(np.int32)
    replacement = study.quota_support(original, maps, 2)
    a, b = (study.QuotaSampler(x, maps, 1) for x in (original, replacement))
    observed_counts = np.zeros(128, np.uint32)
    for _ in range(40):
        left, right = a.next_batch(), b.next_batch()
        assert len(np.unique(left)) == len(np.unique(right)) == 64
        np.testing.assert_array_equal(maps[left], maps[right])
        np.add.at(observed_counts, maps[left] - 300000, 1)
    np.testing.assert_array_equal(a.local_counts, b.local_counts)
    np.testing.assert_array_equal(a.map_counts, b.map_counts)
    np.testing.assert_array_equal(a.map_counts, observed_counts)
    assert a.local_digest.hexdigest() == b.local_digest.hexdigest()
    assert a.map_digest.hexdigest() == b.map_digest.hexdigest()
    assert a.digest.hexdigest() != b.digest.hexdigest()
    assert a.map_counts.sum() == 40 * 64


def fixture(tmp_path, **kwargs):
    protocol = tmp_path / 'protocol.md'
    protocol.write_text('Quota comparison test, not research evidence.\n')
    return study.run_study(tmp_path / 'study', protocol, seeds=[0], updates=2,
        panel_count=1, maps_per_panel=2, panel_seed_start=1140000, smoke=True, **kwargs)


def test_fixed_quota_supports_and_reconstructed_controls_precede_training(tmp_path, monkeypatch):
    original_update, original_rollout = study.shared.fixed_update, study.shared.rollout
    count = {'updates': 0, 'rollouts': 0}
    output = tmp_path / 'study'
    def checked_update(*args, **kwargs):
        assert count['rollouts'] == 0
        with np.load(output / 'supports.npz') as arrays:
            assert len(arrays.files) == 6
        banks = json.loads((output / 'banks.json').read_text())
        assert len(banks) == 3 and all(b['quotas']['exact'] for b in banks)
        for bank in (1, 2, 3):
            assert (output / f'models/bank{bank}_collected_unique_seed0_update30000.pt').is_file()
        count['updates'] += 1
        return original_update(*args, **kwargs)
    def checked_rollout(*args, **kwargs):
        assert count['updates'] == 6
        assert all((output / f'models/bank{b}_within_map_uniform_seed0_update2.pt').exists() for b in (1, 2, 3))
        count['rollouts'] += 1
        return original_rollout(*args, **kwargs)
    def forbidden(*args, **kwargs):
        pytest.fail('this support control must not collect or learn online')
    monkeypatch.setattr(study.shared, 'fixed_update', checked_update)
    monkeypatch.setattr(study.shared, 'rollout', checked_rollout)
    monkeypatch.setattr(study.shared.DQN, 'observe', forbidden)
    monkeypatch.setattr(study.shared.DQN, 'learn', forbidden)
    import q6.coverage
    monkeypatch.setattr(q6.coverage, 'collect_support', forbidden)
    result = fixture(tmp_path)
    run = result['run']
    assert run['status'] == 'complete'
    assert run['train_updates'] == 6 and run['training_examples'] == 384
    assert run['collection_steps'] == run['baseline_new_updates'] == 0
    assert run['per_map_subset_draws'] == 768 and run['replacement_supports'] == 3
    assert run['baseline_reconstructed_updates'] == 90000
    assert run['learner_episodes'] == 36 and run['reference_episodes'] == 6
    assert len(result['trajectories']) == 14
    assert result['provenance']['snapshot_integrity']['complete']
    assert result['provenance']['snapshot_integrity']['expected'] == 9
    assert all(x['verified'] for x in result['provenance']['baseline_sampling_reconstruction'])
    for x in result['provenance']['paired_map_sampling_consistency']:
        assert x['complete'] and x['compared_updates'] == 2 and x['baseline_updates'] == 30000
        assert all(x[k] for k in ['local_digest_identical', 'local_counts_identical', 'map_digest_identical', 'map_counts_identical'])
    assert all(x['unchanged'] and x['read_only'] for x in result['provenance']['support_integrity'])
    assert all(x['unchanged_during_evaluation'] for x in result['provenance']['models'])
    assert not result['robustness']['eligible']
    for row in csv.DictReader((output / 'evaluations.csv').open()):
        assert int(row['checkpoint']) == (30000 if row['condition'] == 'collected_unique' else 2)
    with np.load(output / 'supports.npz') as arrays:
        for bank in (1, 2, 3):
            a, b = arrays[f'collected_unique_bank{bank}'], arrays[f'within_map_uniform_bank{bank}']
            assert len(a) == len(b) and not np.array_equal(a, b)
    for path, digest in json.loads((output / 'manifest.json').read_text())['files'].items():
        assert hashlib.sha256((output / path).read_bytes()).hexdigest() == digest


def test_quota_preparation_stop_preserves_partial_evidence_and_prevents_updates(tmp_path, monkeypatch):
    def stopped(*args, **kwargs):
        raise study.shared.MemoryReached('quota preparation test cap')
    monkeypatch.setattr(study, 'quota_support', stopped)
    result = fixture(tmp_path)
    assert result['run']['status'] == 'incomplete_memory_cap'
    assert result['run']['train_updates'] == result['run']['learner_episodes'] == 0
    assert not result['robustness']['eligible']
    assert (tmp_path / 'study/results.json').exists()


def test_schedule_mismatch_disqualifies_completed_measurements(tmp_path, monkeypatch):
    original = study.WithinMapComparison.consistency
    def altered(self, *args, **kwargs):
        records = original(self, *args, **kwargs)
        records[0]['map_digest_identical'] = False
        records[0]['complete'] = False
        return records
    monkeypatch.setattr(study.WithinMapComparison, 'consistency', altered)
    result = fixture(tmp_path)
    assert result['run']['status'] == 'inconsistent_not_evidence'
    assert result['run']['learner_episodes'] == 36
    assert not result['robustness']['eligible']
