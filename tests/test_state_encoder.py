"""
Tests for utils/state_encoder.py — channel encoding of grid states.

These tests are intentionally property-based and round-trip oriented so they
remain truthful as the encoder evolves: we test invariants (shapes, sums,
mutual exclusion) rather than memorized outputs.
"""

import numpy as np
import pytest

from config import GRID_SIZE, WALL, PELLET, KRISHNA, HUNTER, GREEDY, PATROL, EMPTY
from utils.state_encoder import NUM_CHANNELS, encode_state, encode_batch


# ----------------------------- helpers -----------------------------

def _empty_grid() -> np.ndarray:
    return np.full(GRID_SIZE * GRID_SIZE, EMPTY, dtype=np.int32)


def _set(flat: np.ndarray, r: int, c: int, value: int) -> None:
    flat[r * GRID_SIZE + c] = value


# ----------------------------- shape / dtype -----------------------------

class TestShapeAndDtype:
    def test_output_shape(self):
        out = encode_state(_empty_grid())
        assert out.shape == (NUM_CHANNELS, GRID_SIZE, GRID_SIZE)

    def test_output_dtype(self):
        out = encode_state(_empty_grid())
        assert out.dtype == np.float32

    def test_values_are_binary(self):
        flat = _empty_grid()
        _set(flat, 5, 5, KRISHNA)
        _set(flat, 10, 10, HUNTER)
        out = encode_state(flat)
        unique = np.unique(out)
        assert set(unique.tolist()).issubset({0.0, 1.0})


# ----------------------------- correctness -----------------------------

class TestEncodingCorrectness:
    def test_empty_grid_all_zeros(self):
        out = encode_state(_empty_grid())
        assert out.sum() == 0.0

    def test_krishna_lands_on_channel_2(self):
        flat = _empty_grid()
        _set(flat, 3, 7, KRISHNA)
        out = encode_state(flat)
        assert out[2, 3, 7] == 1.0
        # No other channel should fire at that cell
        for ch in range(NUM_CHANNELS):
            if ch != 2:
                assert out[ch, 3, 7] == 0.0

    def test_each_entity_maps_to_correct_channel(self):
        mapping = [(WALL, 0), (PELLET, 1), (KRISHNA, 2),
                   (HUNTER, 3), (GREEDY, 4), (PATROL, 5)]
        for entity, expected_channel in mapping:
            flat = _empty_grid()
            _set(flat, 12, 12, entity)
            out = encode_state(flat)
            assert out[expected_channel, 12, 12] == 1.0, \
                f"entity {entity} should fire channel {expected_channel}"
            # All other channels at that cell must be 0
            other_sum = out[:, 12, 12].sum() - out[expected_channel, 12, 12]
            assert other_sum == 0.0, f"entity {entity} also activated other channels"

    def test_channel_sums_match_entity_counts(self):
        flat = _empty_grid()
        _set(flat, 0, 0, WALL)
        _set(flat, 0, 1, WALL)
        _set(flat, 0, 2, WALL)
        _set(flat, 5, 5, PELLET)
        _set(flat, 5, 6, PELLET)
        _set(flat, 10, 10, KRISHNA)
        _set(flat, 15, 15, HUNTER)
        out = encode_state(flat)
        assert out[0].sum() == 3  # walls
        assert out[1].sum() == 2  # pellets
        assert out[2].sum() == 1  # krishna
        assert out[3].sum() == 1  # hunter
        assert out[4].sum() == 0  # no greedy
        assert out[5].sum() == 0  # no patrol

    def test_mutual_exclusion_at_every_cell(self):
        """A cell holds exactly one entity (or EMPTY). Sum across channels must be <=1 everywhere."""
        rng = np.random.default_rng(42)
        # Build a random valid grid: each cell is one of the 7 values.
        flat = rng.integers(0, 7, size=GRID_SIZE * GRID_SIZE, dtype=np.int32)
        out = encode_state(flat)
        per_cell_sum = out.sum(axis=0)  # (25, 25)
        assert per_cell_sum.max() <= 1.0
        # And EMPTY cells should be all-zero across channels
        empty_mask = (flat.reshape(GRID_SIZE, GRID_SIZE) == EMPTY)
        assert np.all(per_cell_sum[empty_mask] == 0.0)
        # And non-empty cells should sum to exactly 1
        assert np.all(per_cell_sum[~empty_mask] == 1.0)


# ----------------------------- batch -----------------------------

class TestBatchEncoder:
    def test_batch_shape(self):
        B = 8
        batch = np.stack([_empty_grid() for _ in range(B)])
        out = encode_batch(batch)
        assert out.shape == (B, NUM_CHANNELS, GRID_SIZE, GRID_SIZE)

    def test_batch_matches_single(self):
        """encode_batch must produce identical output to encode_state per row."""
        rng = np.random.default_rng(0)
        rows = [rng.integers(0, 7, size=GRID_SIZE * GRID_SIZE, dtype=np.int32) for _ in range(4)]
        batched = encode_batch(np.stack(rows))
        for i, row in enumerate(rows):
            single = encode_state(row)
            np.testing.assert_array_equal(batched[i], single)


# ----------------------------- validation -----------------------------

class TestInputValidation:
    def test_rejects_2d_input_to_single(self):
        with pytest.raises(ValueError):
            encode_state(np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32))

    def test_rejects_wrong_length(self):
        with pytest.raises(ValueError):
            encode_state(np.zeros(624, dtype=np.int32))

    def test_rejects_1d_input_to_batch(self):
        with pytest.raises(ValueError):
            encode_batch(_empty_grid())


# ----------------------------- integration with env -----------------------------

class TestIntegrationWithEnv:
    """Encoder must work on actual environment states."""

    def test_encodes_real_env_state(self):
        from environment.hunter_gridworld import HunterGridworld
        env = HunterGridworld(difficulty_level=4, curriculum_phase=4)
        state, _ = env.reset(seed=0)
        out = encode_state(state)
        assert out.shape == (NUM_CHANNELS, GRID_SIZE, GRID_SIZE)
        # Krishna and Hunter must each appear exactly once
        assert out[2].sum() == 1, "Krishna should appear exactly once"
        assert out[3].sum() == 1, "Hunter should appear exactly once at difficulty 4"
        # Greedy bots: should be 2
        assert out[4].sum() == 2, "Should be 2 greedy bots"

    def test_persistent_after_step(self):
        from environment.hunter_gridworld import HunterGridworld
        env = HunterGridworld(difficulty_level=4, curriculum_phase=4)
        state, _ = env.reset(seed=0)
        for action in [0, 1, 2, 3, 0]:
            state, _, done, trunc, _ = env.step(action)
            out = encode_state(state)
            # Krishna always present
            assert out[2].sum() == 1
            if done or trunc:
                break
