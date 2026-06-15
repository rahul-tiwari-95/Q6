"""
Tests for utils.cher.CounterfactualHER.

Validates:
  - No counterfactuals when hunter is close (unsafe)
  - No counterfactuals when pellet is far (unreachable)
  - No counterfactuals when Krishna already takes optimal action
  - Counterfactual injected when all conditions are met
  - Optimal action computation (cardinal direction toward pellet)
  - Reward is original + collect_bonus
  - State / next_state / context preserved unchanged
  - max_cf_per_ep cap is respected
  - count_opportunities() matches relabel() output size
  - Works on empty trajectory
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pytest

from utils.cher import CounterfactualHER, _optimal_action_toward

GRID = 25

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_context() -> np.ndarray:
    return np.array([0.8, 0.5, 1.0], dtype=np.float32)


def _make_state() -> np.ndarray:
    s = np.full(GRID * GRID, 6, dtype=np.uint8)
    return s


def _make_step(
    *,
    hunter_dist: int,
    pellet_dist: float,
    pellet_pos: Tuple[int, int] | None,
    krishna_pos: Tuple[int, int],
    pellets_remaining: int,
    action: int,
    reward: float = 0.0,
    done: bool = False,
) -> Dict:
    state       = _make_state()
    context     = _make_context()
    next_context = _make_context()
    return {
        "state":        state,
        "context":      context,
        "action":       action,
        "reward":       reward,
        "next_state":   state.copy(),
        "next_context": next_context,
        "done":         done,
        "info": {
            "hunter_manhattan_dist": hunter_dist,
            "nearest_pellet_dist":   pellet_dist,
            "nearest_pellet_pos":    pellet_pos,
            "krishna_position":      krishna_pos,
            "pellets_remaining":     pellets_remaining,
        },
    }


# ---------------------------------------------------------------------------
# Optimal action helper
# ---------------------------------------------------------------------------

class TestOptimalAction:
    def test_move_down(self):
        assert _optimal_action_toward((5, 5), (8, 5)) == 1   # DOWN

    def test_move_up(self):
        assert _optimal_action_toward((8, 5), (5, 5)) == 0   # UP

    def test_move_right(self):
        assert _optimal_action_toward((5, 5), (5, 8)) == 3   # RIGHT

    def test_move_left(self):
        assert _optimal_action_toward((5, 8), (5, 5)) == 2   # LEFT

    def test_prefer_row_on_tie(self):
        # dr == dc → prefer row (UP or DOWN)
        result = _optimal_action_toward((5, 5), (8, 8))      # dr=3, dc=3 → DOWN (row)
        assert result == 1

    def test_prefer_column_when_larger(self):
        result = _optimal_action_toward((5, 5), (6, 9))      # dr=1, dc=4 → RIGHT
        assert result == 3


# ---------------------------------------------------------------------------
# Core filter conditions
# ---------------------------------------------------------------------------

class TestRelabelFilters:
    def setup_method(self):
        self.cher = CounterfactualHER(hunter_safe_dist=6, pellet_reach=4,
                                     collect_bonus=30.0)

    def test_no_cf_when_hunter_close(self):
        step = _make_step(
            hunter_dist=3,       # <= safe_dist=6  → unsafe
            pellet_dist=2.0,
            pellet_pos=(7, 5),
            krishna_pos=(5, 5),
            pellets_remaining=2,
            action=2,            # not toward pellet
        )
        result = self.cher.relabel([step])
        assert result == []

    def test_no_cf_when_pellet_far(self):
        step = _make_step(
            hunter_dist=10,
            pellet_dist=6.0,     # > reach=4
            pellet_pos=(11, 5),
            krishna_pos=(5, 5),
            pellets_remaining=2,
            action=2,
        )
        result = self.cher.relabel([step])
        assert result == []

    def test_no_cf_when_no_pellets(self):
        step = _make_step(
            hunter_dist=10,
            pellet_dist=2.0,
            pellet_pos=(7, 5),
            krishna_pos=(5, 5),
            pellets_remaining=0,  # no pellets
            action=2,
        )
        result = self.cher.relabel([step])
        assert result == []

    def test_no_cf_when_optimal_action_taken(self):
        """Krishna already moving toward pellet — no counterfactual needed."""
        step = _make_step(
            hunter_dist=10,
            pellet_dist=2.0,
            pellet_pos=(7, 5),   # below krishna → optimal = DOWN (1)
            krishna_pos=(5, 5),
            pellets_remaining=2,
            action=1,            # DOWN = optimal, no CF
        )
        result = self.cher.relabel([step])
        assert result == []

    def test_cf_generated_when_conditions_met(self):
        step = _make_step(
            hunter_dist=10,
            pellet_dist=2.0,
            pellet_pos=(7, 5),   # below krishna → optimal DOWN (1)
            krishna_pos=(5, 5),
            pellets_remaining=2,
            action=2,            # LEFT ≠ optimal
            reward=-0.001,
        )
        result = self.cher.relabel([step])
        assert len(result) == 1

    def test_no_cf_when_pellet_pos_none(self):
        step = _make_step(
            hunter_dist=10,
            pellet_dist=0.0,
            pellet_pos=None,    # no nearest pellet position
            krishna_pos=(5, 5),
            pellets_remaining=2,
            action=2,
        )
        result = self.cher.relabel([step])
        assert result == []


# ---------------------------------------------------------------------------
# Counterfactual transition content
# ---------------------------------------------------------------------------

class TestCounterfactualContent:
    def setup_method(self):
        # weighted=False: tests in this class validate transition structure,
        # not the weighting formula (that is tested in test_phase3_regression.py)
        self.cher = CounterfactualHER(hunter_safe_dist=6, pellet_reach=4,
                                     collect_bonus=30.0, weighted=False)

    def _get_cf_step(self):
        step = _make_step(
            hunter_dist=10,
            pellet_dist=3.0,
            pellet_pos=(8, 5),   # below krishna → DOWN (1)
            krishna_pos=(5, 5),
            pellets_remaining=3,
            action=2,            # LEFT → wrong
            reward=-0.5,
        )
        return step, self.cher.relabel([step])[0]

    def test_action_is_optimal(self):
        _, cf = self._get_cf_step()
        state, context, action, reward, next_state, next_context, done = cf
        assert action == 1  # DOWN toward pellet at (8,5) from (5,5)

    def test_reward_is_original_plus_bonus(self):
        step, cf = self._get_cf_step()
        _, _, _, reward, *_ = cf
        assert abs(reward - (-0.5 + 30.0)) < 1e-5

    def test_state_preserved(self):
        step, cf = self._get_cf_step()
        state, *_ = cf
        np.testing.assert_array_equal(state, step["state"])

    def test_next_state_preserved(self):
        step, cf = self._get_cf_step()
        _, _, _, _, next_state, *_ = cf
        np.testing.assert_array_equal(next_state, step["next_state"])

    def test_context_preserved(self):
        step, cf = self._get_cf_step()
        _, context, *_ = cf
        np.testing.assert_array_almost_equal(context, step["context"])

    def test_next_context_preserved(self):
        step, cf = self._get_cf_step()
        _, _, _, _, _, next_context, _ = cf
        np.testing.assert_array_almost_equal(next_context, step["next_context"])

    def test_done_preserved(self):
        step, cf = self._get_cf_step()
        *_, done = cf
        assert done == step["done"]


# ---------------------------------------------------------------------------
# Multiple steps in trajectory
# ---------------------------------------------------------------------------

class TestTrajectory:
    def setup_method(self):
        self.cher = CounterfactualHER(hunter_safe_dist=6, pellet_reach=4,
                                     collect_bonus=25.0)

    def test_multiple_opportunities_all_returned(self):
        steps = [
            _make_step(
                hunter_dist=10, pellet_dist=2.0, pellet_pos=(7, 5),
                krishna_pos=(5, 5), pellets_remaining=2, action=2,
            )
            for _ in range(5)
        ]
        result = self.cher.relabel(steps)
        assert len(result) == 5

    def test_mixed_trajectory(self):
        """Only steps meeting all conditions should produce counterfactuals."""
        steps = [
            # Safe opportunity — should produce CF
            _make_step(hunter_dist=10, pellet_dist=2.0, pellet_pos=(7, 5),
                       krishna_pos=(5, 5), pellets_remaining=2, action=2),
            # Hunter close — no CF
            _make_step(hunter_dist=2, pellet_dist=2.0, pellet_pos=(7, 5),
                       krishna_pos=(5, 5), pellets_remaining=2, action=2),
            # Pellet far — no CF
            _make_step(hunter_dist=10, pellet_dist=8.0, pellet_pos=(13, 5),
                       krishna_pos=(5, 5), pellets_remaining=2, action=2),
            # Safe opportunity — should produce CF
            _make_step(hunter_dist=12, pellet_dist=3.0, pellet_pos=(5, 8),
                       krishna_pos=(5, 5), pellets_remaining=1, action=0),
        ]
        result = self.cher.relabel(steps)
        assert len(result) == 2

    def test_max_cf_per_ep_cap(self):
        cher_capped = CounterfactualHER(hunter_safe_dist=6, pellet_reach=4,
                                       collect_bonus=30.0, max_cf_per_ep=3)
        steps = [
            _make_step(hunter_dist=10, pellet_dist=2.0, pellet_pos=(7, 5),
                       krishna_pos=(5, 5), pellets_remaining=2, action=2)
            for _ in range(10)
        ]
        result = cher_capped.relabel(steps)
        assert len(result) == 3

    def test_empty_trajectory(self):
        result = self.cher.relabel([])
        assert result == []


# ---------------------------------------------------------------------------
# count_opportunities()
# ---------------------------------------------------------------------------

class TestCountOpportunities:
    def setup_method(self):
        self.cher = CounterfactualHER(hunter_safe_dist=6, pellet_reach=4,
                                     collect_bonus=30.0)

    def test_count_matches_eligible_steps(self):
        steps = [
            # Eligible
            _make_step(hunter_dist=10, pellet_dist=2.0, pellet_pos=(7, 5),
                       krishna_pos=(5, 5), pellets_remaining=2, action=2),
            # Not eligible (hunter close)
            _make_step(hunter_dist=3, pellet_dist=2.0, pellet_pos=(7, 5),
                       krishna_pos=(5, 5), pellets_remaining=2, action=2),
            # Eligible
            _make_step(hunter_dist=8, pellet_dist=3.0, pellet_pos=(5, 8),
                       krishna_pos=(5, 5), pellets_remaining=1, action=0),
        ]
        assert self.cher.count_opportunities(steps) == 2


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestConstructorValidation:
    def test_negative_hunter_dist_raises(self):
        with pytest.raises(ValueError):
            CounterfactualHER(hunter_safe_dist=-1)

    def test_negative_pellet_reach_raises(self):
        with pytest.raises(ValueError):
            CounterfactualHER(pellet_reach=-1)

    def test_negative_bonus_raises(self):
        with pytest.raises(ValueError):
            CounterfactualHER(collect_bonus=-1.0)
