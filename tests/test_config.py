"""
Tests for config.py - Verify centralized constants match the codebase.
"""
import pytest


class TestConfigConstants:
    """Verify config.py has correct values that match the active codebase."""

    def test_grid_constants(self):
        from config import GRID_SIZE, STATE_SIZE, ACTION_SIZE
        assert GRID_SIZE == 25
        assert STATE_SIZE == 625  # 25 * 25
        assert ACTION_SIZE == 4

    def test_grid_cell_values(self):
        from config import WALL, PELLET, KRISHNA, HUNTER, GREEDY, PATROL, EMPTY
        assert WALL == 0
        assert PELLET == 1
        assert KRISHNA == 2
        assert HUNTER == 3
        assert GREEDY == 4
        assert PATROL == 5
        assert EMPTY == 6

    def test_reward_values(self):
        from config import (
            REWARD_PELLET, REWARD_WIN, REWARD_CAUGHT,
            REWARD_WALL_HIT, REWARD_STEP,
        )
        assert REWARD_PELLET == 50.0
        assert REWARD_WIN == 100.0
        assert REWARD_CAUGHT == -10.0
        assert REWARD_WALL_HIT == -5.0
        assert REWARD_STEP == -0.001  # v0.3 corrected from -0.01

    def test_dqn_hparams(self):
        from config import (
            HIDDEN_SIZES, LEARNING_RATE, GAMMA, TAU,
            BATCH_SIZE, BUFFER_SIZE, UPDATE_EVERY,
        )
        assert HIDDEN_SIZES == (256, 128)
        assert LEARNING_RATE == 1e-4
        assert GAMMA == 0.99
        assert TAU == 1e-3
        assert BATCH_SIZE == 64
        assert BUFFER_SIZE == 100_000
        assert UPDATE_EVERY == 4

    def test_epsilon_params(self):
        from config import EPSILON_START, EPSILON_MIN, EPSILON_DECAY, PHASE_EPSILON_RESET
        assert EPSILON_START == 1.0
        assert EPSILON_MIN == 0.05
        assert EPSILON_DECAY == 0.9994
        assert PHASE_EPSILON_RESET == {1: 1.0, 2: 0.5, 3: 0.3, 4: 0.2}

    def test_curriculum_phases(self):
        from config import PHASE_EPISODES, CURRICULUM_PHASES, DEFAULT_EPISODES
        assert PHASE_EPISODES == (3000, 3000, 3000, 9000)
        assert sum(PHASE_EPISODES) == 18_000  # Total episodes needed
        assert len(CURRICULUM_PHASES) == 4
        # Verify default episodes matches phase allocation
        assert DEFAULT_EPISODES == sum(PHASE_EPISODES)

    def test_success_scoring_weights(self):
        from config import (
            WEIGHT_OBJECTIVES, WEIGHT_SURVIVAL,
            WEIGHT_EFFICIENCY, WEIGHT_STRATEGY,
        )
        total = WEIGHT_OBJECTIVES + WEIGHT_SURVIVAL + WEIGHT_EFFICIENCY + WEIGHT_STRATEGY
        assert abs(total - 1.0) < 1e-9
        assert WEIGHT_OBJECTIVES == 0.40
        assert WEIGHT_SURVIVAL == 0.30
        assert WEIGHT_EFFICIENCY == 0.20
        assert WEIGHT_STRATEGY == 0.10
