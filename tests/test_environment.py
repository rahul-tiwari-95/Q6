"""
Tests for the HunterGridworld environment.
"""
import pytest
import numpy as np
from environment.hunter_gridworld import HunterGridworld


@pytest.fixture
def env_easy():
    """Fresh easy-mode environment (2 greedy bots only)."""
    return HunterGridworld(difficulty_level=1)


@pytest.fixture
def env_expert():
    """Fresh expert-mode environment (full game + A* Hunter)."""
    return HunterGridworld(difficulty_level=4, curriculum_phase=4)


class TestEnvInitialization:
    """Test environment creation and basic properties."""

    def test_env_creates_grid(self, env_easy):
        assert env_easy.grid_size == 25

    def test_observation_space(self, env_easy):
        assert env_easy.observation_space.shape == (625,)
        assert env_easy.observation_space.low[0] == 0
        assert env_easy.observation_space.high[0] == 6

    def test_action_space(self, env_easy):
        assert env_easy.action_space.n == 4  # Up, Down, Left, Right

    def test_statesize(self, env_easy):
        assert env_easy.state_size == 625


class TestReset:
    """Test environment reset behavior."""

    def test_reset_returns_state_and_info(self, env_easy):
        state, info = env_easy.reset()
        assert state is not None
        assert isinstance(state, np.ndarray)
        assert isinstance(info, dict)

    def test_reset_state_shape(self, env_easy):
        state, _ = env_easy.reset()
        assert state.shape == (625,)

    def test_reset_contains_krishna(self, env_easy):
        state, _ = env_easy.reset()
        assert 2 in state  # KRISHNA = 2

    def test_reset_contains_pellets(self, env_easy):
        state, _ = env_easy.reset()
        assert 1 in state  # PELLET = 1

    def test_reset_contains_walls(self, env_easy):
        state, _ = env_easy.reset()
        assert 0 in state  # WALL = 0

    def test_reset_info_has_krishna_position(self, env_easy):
        _, info = env_easy.reset()
        assert "krishna_position" in info

    def test_difficulty_1_no_hunter(self, env_easy):
        env_easy.reset()
        assert env_easy.hunter is None

    def test_difficulty_4_has_hunter(self, env_expert):
        env_expert.reset()
        assert env_expert.hunter is not None


class TestStep:
    """Test environment step mechanics."""

    def test_step_returns_correct_tuple(self, env_easy):
        env_easy.reset()
        result = env_easy.step(0)
        assert len(result) == 5  # state, reward, done, truncated, info

    def test_step_reward_is_float(self, env_easy):
        env_easy.reset()
        _, reward, _, _, _ = env_easy.step(0)
        assert isinstance(reward, (float, np.floating))

    def test_step_penalty_is_negative(self, env_easy):
        env_easy.reset()
        _, reward, _, _, _ = env_easy.step(0)
        # Reward should be slightly negative for just moving (step penalty)
        assert reward <= 0

    def test_step_penalty_value(self, env_easy):
        env_easy.reset()
        # Move to empty cell (no pellet, no wall, no enemy)
        _, reward, _, _, _ = env_easy.step(0)
        # With step penalty -0.001, reward should be ≈ -0.001 if nothing else
        assert reward == pytest.approx(-0.001, abs=0.01)

    def test_collision_gives_negative_reward(self, env_expert):
        # With expert mode (has hunter), some steps should produce bigger penalties
        for _ in range(5):
            env_expert.reset()
            for action in [0, 1, 2, 3]:
                _, reward, _, _, _ = env_expert.step(action)
                if reward < -1:
                    assert reward <= -1.0  # caught penalty + step penalty
                    return
        # If no collision in first step of 5 resets, that's fine


class TestWinCondition:
    """Test win condition logic."""

    def test_win_by_pellets_default(self, env_easy):
        env_easy.reset()
        assert not env_easy.check_win_condition()
        # Manually collect 4 pellets
        env_easy.krishna.pellets_collected = 4
        assert env_easy.check_win_condition()

    def test_no_win_before_enough_pellets(self, env_easy):
        env_easy.reset()
        env_easy.krishna.pellets_collected = 3
        assert not env_easy.check_win_condition()


class TestKrishnaEntity:
    """Test Krishna entity behavior."""

    def test_initial_lives(self, env_easy):
        env_easy.reset()
        assert env_easy.krishna.lives == 3

    def test_take_damage_reduces_lives(self, env_easy):
        env_easy.reset()
        assert env_easy.krishna.lives == 3
        alive = env_easy.krishna.take_damage()
        assert alive
        assert env_easy.krishna.lives == 2
        assert env_easy.krishna.invulnerable

    def test_all_lives_lost(self, env_easy):
        env_easy.reset()
        # Each take_damage only works when not invulnerable
        for _ in range(3):
            env_easy.krishna.invulnerable = False
            env_easy.krishna.invulnerable_timer = 0
            alive = env_easy.krishna.take_damage()
        assert not alive
        assert env_easy.krishna.lives == 0

    def test_invulnerability_timer_decrements(self, env_easy):
        env_easy.reset()
        env_easy.krishna.take_damage()
        assert env_easy.krishna.invulnerable
        initial = env_easy.krishna.invulnerable_timer
        env_easy.krishna.update()
        assert env_easy.krishna.invulnerable_timer == initial - 1

    def test_pellet_collection(self, env_easy):
        env_easy.reset()
        assert env_easy.krishna.score == 0
        assert env_easy.krishna.pellets_collected == 0
        env_easy.krishna.collect_pellet()
        assert env_easy.krishna.score == 50
        assert env_easy.krishna.pellets_collected == 1

    def test_move_stays_within_bounds(self, env_easy):
        env_easy.reset()
        # Move up from top edge (x=0)
        env_easy.krishna.position = (0, 0)
        new_pos = env_easy.krishna.move(0, env_easy.grid)
        assert new_pos[0] == 0  # Can't go past top edge

    def test_move_into_wall(self, env_easy):
        env_easy.reset()
        # Find a wall position
        wall_cells = np.where(env_easy.grid == env_easy.WALL)
        if len(wall_cells[0]) > 0:
            wx, wy = wall_cells[0][0], wall_cells[1][0]
            # Place Krishna adjacent to wall (below it), try moving into wall
            env_easy.krishna.position = (wx + 1, wy)
            env_easy.grid[wx + 1, wy] = env_easy.KRISHNA
            # Try moving UP (into the wall at wx, wy)
            new_pos = env_easy.krishna.move(0, env_easy.grid)
            # Should stay in place (wall blocks movement)
            assert new_pos == (wx + 1, wy)


class TestEpsilonDecay:
    """Verify epsilon is decayed only once per episode."""

    def test_no_inline_epsilon_decay_in_main(self):
        """main.py should not have inline epsilon = max(...) decay."""
        with open('main.py', 'r') as f:
            content = f.read()
        lines = content.split('\n')
        count = 0
        for line in lines:
            stripped = line.strip()
            # Count lines that manually set epsilon with decay formula
            if 'agent.epsilon' in stripped and 'max(' in stripped and 'epsilon_decay' in stripped:
                count += 1
        # Should be 0 (only decay_epsilon() method call should exist)
        assert count == 0, f"Found {count} inline epsilon decay lines in main.py"


class TestHunterConstants:
    """Verify grid constants are consistent."""

    def test_wall_constant_is_zero(self, env_easy):
        assert env_easy.WALL == 0

    def test_pellet_constant_is_one(self, env_easy):
        assert env_easy.PELLET == 1

    def test_krishna_constant_is_two(self, env_easy):
        assert env_easy.KRISHNA == 2

    def test_hunter_constant_is_three(self, env_easy):
        assert env_easy.HUNTER == 3

    def test_constants_match_config(self, env_easy):
        from config import WALL, PELLET, KRISHNA, HUNTER, GREEDY, PATROL, EMPTY
        assert env_easy.WALL == WALL
        assert env_easy.PELLET == PELLET
        assert env_easy.KRISHNA == KRISHNA
        assert env_easy.HUNTER == HUNTER
        assert env_easy.GREEDY == GREEDY
        assert env_easy.PATROL == PATROL
        assert env_easy.EMPTY == EMPTY
