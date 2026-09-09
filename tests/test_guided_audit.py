"""Audit invariants for unequal support and independent sampler accounting."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_guided_collection import reconstruct_sampling


def test_same_local_population_does_not_mean_same_global_experience():
    maps = np.repeat(np.arange(4) + 300000, 64)
    left = np.arange(0, 256, 2, dtype=np.int32)
    right = left + 1
    a = reconstruct_sampling(left, maps, seed=2, updates=20)
    b = reconstruct_sampling(right, maps, seed=2, updates=20)
    assert a["digests"]["local"] == b["digests"]["local"]
    assert a["digests"]["map"] == b["digests"]["map"]
    assert a["digests"]["global"] != b["digests"]["global"]
    assert np.array_equal(a["local_counts"], b["local_counts"])
    assert not a["global_counts"][right].any()
    assert not b["global_counts"][left].any()
    assert int(a["global_counts"].sum()) == int(b["global_counts"].sum()) == 1280


def test_different_support_sizes_use_owned_rng_and_separate_exposure():
    maps = np.repeat(np.arange(4) + 300000, 64)
    original_state = np.random.get_state()
    a = reconstruct_sampling(np.arange(128, dtype=np.int32), maps, seed=0, updates=20)
    b = reconstruct_sampling(np.arange(192, dtype=np.int32), maps, seed=0, updates=20)
    current_state = np.random.get_state()
    assert original_state[0] == current_state[0]
    assert np.array_equal(original_state[1], current_state[1])
    assert original_state[2:] == current_state[2:]
    assert a["digests"]["local"] != b["digests"]["local"]
    for result, support_size in ((a, 128), (b, 192)):
        assert result["local_counts"].shape == (support_size,)
        assert not result["global_counts"][support_size:].any()
        assert int(result["map_counts"].sum()) == int(result["global_counts"].sum()) == 1280


@pytest.mark.parametrize("support", [np.arange(63, dtype=np.int32), np.zeros(64, np.int32), np.arange(64, dtype=np.int32)[::-1]])
def test_rejects_a_support_that_cannot_supply_a_distinct_valid_batch(support):
    with pytest.raises(AssertionError):
        reconstruct_sampling(support, np.full(256, 300000), seed=0, updates=1)
