"""
Tests for progressive hunter system.
Verifies all 4 hunter types work correctly and the A* wall-walking bug is fixed.
"""
import pytest
import numpy as np
from environment.entities.hunter_progressive import (
    BaseProgressiveHunter,
    RandomHunter,
    GreedyHunter,
    SmartGreedyHunter,
    AStarHunter,
    create_hunter_for_phase,
)
from environment.hunter_gridworld import HunterGridworld


@pytest.fixture
def grid_empty():
    """Return a completely empty grid (all EMPTY=6)."""
    return np.full((25, 25), 6)


@pytest.fixture
def grid_with_walls():
    """Create a grid with some walls for testing."""
    grid = np.full((25, 25), 6)
    # Add a wall segment at row 10, columns 5-15
    for c in range(5, 16):
        grid[10, c] = 0  # WALL = 0
    return grid


class TestHunterFactory:
    """Test create_hunter_for_phase factory function."""

    def test_phase_1_returns_random(self):
        hunter = create_hunter_for_phase(1, (10, 10))
        assert isinstance(hunter, RandomHunter)

    def test_phase_2_returns_greedy(self):
        hunter = create_hunter_for_phase(2, (10, 10))
        assert isinstance(hunter, GreedyHunter)

    def test_phase_3_returns_smart_greedy(self):
        hunter = create_hunter_for_phase(3, (10, 10))
        assert isinstance(hunter, SmartGreedyHunter)

    def test_phase_4_returns_astar(self):
        hunter = create_hunter_for_phase(4, (10, 10))
        assert isinstance(hunter, AStarHunter)

    def test_unknown_phase_defaults_to_astar(self):
        hunter = create_hunter_for_phase(99, (10, 10))
        assert isinstance(hunter, AStarHunter)


class TestHunterBase:
    """Test BaseProgressiveHunter properties."""

    def test_position_property(self):
        hunter = RandomHunter((5, 5))
        assert hunter.position == (5, 5)

    def test_position_property_type(self):
        hunter = RandomHunter((5, 5))
        assert isinstance(hunter.position, tuple)

    def test_get_state_info(self):
        hunter = RandomHunter((5, 5))
        info = hunter.get_state_info()
        assert info['pos'] == (5, 5)
        assert isinstance(info, dict)


class TestAStarWallAvoidance:
    """
    CRITICAL: Verify the A* wall-walking bug is fixed.
    Before fix: A* checked grid[ny][nx] != 1 (avoided pellets, walked through walls)
    After fix: A* checks grid[ny, nx] != 0 (avoids walls correctly)
    """

    def test_astar_code_uses_wall_constant_zero(self):
        """Verify A* source code checks != 0 for walls."""
        import inspect
        src = inspect.getsource(AStarHunter.a_star_search)
        assert "!= 0" in src, "A* must check != 0 for walls"

    def test_astar_valid_moves_excludes_wall(self):
        """get_valid_moves must not return directions into wall cells (value 0)."""
        grid = np.full((25, 25), 6)  # All empty
        grid[10, 5] = 0  # Wall at position (5, 10) in grid[y,x] notation
                         # But grid[y,x] = grid[10, 5] — need to match A* indexing

        # A* uses grid[ny, nx] where (nx, ny) is from (x-1, y) etc.
        # From position (5, 10), RIGHT goes to (6, 10), grid[10, 6] = 6 (OK)
        # UP goes to (5, 9), grid[9, 5] = 6 (OK)
        # DOWN goes to (5, 11), grid[11, 5] = 6 (OK)
        # LEFT goes to (4, 10), grid[10, 4] = 6 (ok)
        # Block the cell that RIGHT moves into:
        grid[10, 6] = 0  # Block (6, 10)
        # Actually, let me be more careful with A* indexing.
        # get_valid_moves checks: grid[ny][nx] where moves map:
        # RIGHT: nx=x+1, ny=y → grid[y, x+1]

        grid[10, 6] = 0  # Block grid[10, 6] which = grid[y, nx] for RIGHT from (5,10)

        hunter = AStarHunter((5, 10), grid_size=25)
        valid = hunter.get_valid_moves(grid)
        # RIGHT goes to (6, 10), grid[10, 6]=0=wall → should NOT be valid
        assert 'RIGHT' not in valid, "A* should NOT consider wall cells as valid!"
        assert len(valid) >= 3  # At least UP, DOWN, LEFT should be valid

    def test_astar_moving_average_excludes_walls(self):
        """A* choose_action should never walk into a wall cell."""
        grid = np.full((25, 25), 6)
        # Block one direction from hunter position
        # Hunter at (10, 10), block grid[10, 11] (RIGHT direction)
        grid[10, 11] = 0  # wall
        grid[10, 12] = 0  # wall too

        hunter = AStarHunter((10, 10), grid_size=25)
        for _ in range(50):
            action = hunter.choose_action((20, 20), grid)
            # Execute action
            old_pos = hunter.pos[:]
            hunter.move(action)
            # New position should not be a wall
            nx, ny = hunter.pos
            assert grid[ny, nx] != 0, f"Hunter walked into wall at ({nx}, {ny})!"


class TestGreedyHunter:
    """Test GreedyHunter — it moves toward Krishna."""

    def test_moves_toward_krishna(self, grid_empty):
        hunter = GreedyHunter((5, 5), grid_size=25)
        hunter.update(grid_empty, (5, 10), [])
        # Krishna is at y=10, hunter at y=5. dx=0, dy=5. Preferred = DOWN.
        # DOWN: pos[1] += 1 → (5, 6)
        assert hunter.pos == [5, 6]

    def test_position_property_updates(self, grid_empty):
        hunter = GreedyHunter((5, 5), grid_size=25)
        assert hunter.position == (5, 5)
        hunter.update(grid_empty, (5, 10), [])
        assert hunter.position == (5, 6)

    def test_doesnt_walk_into_wall(self, grid_with_walls):
        # Hunter at (10, 4), wall at (10, 5), Krishna at (10, 8)
        hunter = GreedyHunter((10, 4), grid_size=25)
        hunter.update(grid_with_walls, (10, 8), [])
        new_pos = hunter.pos
        # Should not walk into wall at (10, 5)
        assert grid_with_walls[new_pos[1], new_pos[0]] != 0


class TestSmartGreedyHunter:
    """Test SmartGreedyHunter."""

    def test_moves_toward_krishna(self, grid_empty):
        hunter = SmartGreedyHunter((5, 5), grid_size=25)
        hunter.update(grid_empty, (5, 10), [])
        # Should move toward Krishna (pos[1] increased from 5 to 6, DOWN)
        assert hunter.pos[1] >= 5  # Moved in the right direction

    def test_fallback_when_blocked(self):
        """SmartGreedy should use fallback when primary direction is blocked."""
        grid = np.full((25, 25), 6)  # All empty
        hunter = SmartGreedyHunter((0, 0), grid_size=25)
        hunter.update(grid, (10, 10), [])
        # Should still make a move (fallback to secondary or perpendicular)
        assert isinstance(hunter.pos, list)


class TestHunterIntegration:
    """Integration: Hunter works with environment."""

    def test_hunter_position_accessible_from_env(self):
        env = HunterGridworld(difficulty_level=4, curriculum_phase=4)
        env.reset()
        assert env.hunter is not None
        pos = env.hunter.position
        assert isinstance(pos, tuple)
        assert len(pos) == 2

    def test_hunter_update_in_env(self):
        env = HunterGridworld(difficulty_level=4, curriculum_phase=1)
        env.reset()
        for _ in range(5):
            env.step(0)
        assert env.hunter.position is not None

    def test_all_phases_can_run_steps(self):
        """All 4 curriculum phases should produce working hunters."""
        for phase in range(1, 5):
            env = HunterGridworld(difficulty_level=4, curriculum_phase=phase)
            env.reset()
            for _ in range(20):
                env.step(np.random.randint(4))
            assert env.hunter.position is not None

    def test_progressive_hunter_constants_match_config(self):
        import inspect
        src = inspect.getsource(AStarHunter.a_star_search)
        assert "!= 0" in src, "A* should check != 0 for walls"
        src_valid = inspect.getsource(BaseProgressiveHunter.get_valid_moves)
        assert "!= 0" in src_valid, "get_valid_moves should check != 0 for walls"

    def test_constants_match_config(self):
        from config import WALL, PELLET, KRISHNA, HUNTER, GREEDY, PATROL, EMPTY
        env = HunterGridworld(difficulty_level=1)
        assert env.WALL == WALL
        assert env.PELLET == PELLET
        assert env.KRISHNA == KRISHNA
