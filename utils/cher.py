"""
Counterfactual Hindsight Experience Replay (CHER) for adversarial self-play.

Motivation
----------
In training against a strong Hunter, Krishna learns to evade effectively but
develops "collection paralysis" — it never approaches pellets even when the
Hunter is far away.  The replay buffer contains many evasion-only episodes
with zero safe-collect signal.

CHER solves this by scanning completed episodes for timesteps where:
  1. The Hunter was far enough away to be "safe" (hunter_dist > safe_dist).
  2. A pellet was close enough to be reachable (pellet_dist <= pellet_reach).
  3. Krishna chose an action that was NOT optimal toward the nearest pellet.

For each such timestep we inject a *counterfactual* transition into the
replay buffer:
  - action    → the optimal toward-pellet action
  - reward    → original reward + collect_bonus
  - state / next_state unchanged (counterfactual: "what if Krishna had moved
    toward the pellet instead?")

This creates a synthetic safe-collection training signal without any new
environment interactions.

Usage
-----
    cher = CounterfactualHER(hunter_safe_dist=6, pellet_reach=4,
                             collect_bonus=30.0, grid_size=25)

    # After an episode, pass the trajectory list:
    cf_exps = cher.relabel(trajectory)

    # Each element of cf_exps is a 7-tuple:
    #   (state, context, action, reward, next_state, next_context, done)
    for exp in cf_exps:
        agent.add_experience(*exp)

Trajectory format
-----------------
Each step in the trajectory must be a dict with keys:
    state        np.ndarray  flat (625,) uint8
    context      np.ndarray  (3,) float32
    action       int
    reward       float
    next_state   np.ndarray  flat (625,) uint8
    next_context np.ndarray  (3,) float32
    done         bool
    info         dict        env info containing:
                               hunter_manhattan_dist  int
                               nearest_pellet_dist    float
                               nearest_pellet_pos     tuple | None
                               krishna_position       tuple
                               pellets_remaining      int
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# Action indices (must match selfplay_env.py)
_UP    = 0
_DOWN  = 1
_LEFT  = 2
_RIGHT = 3

_DELTAS: Dict[int, Tuple[int, int]] = {
    _UP:    (-1, 0),
    _DOWN:  ( 1, 0),
    _LEFT:  ( 0,-1),
    _RIGHT: ( 0, 1),
}


def _optimal_action_toward(
    src: Tuple[int, int],
    dst: Tuple[int, int],
) -> int:
    """
    Return the cardinal action that reduces Manhattan distance to `dst` most.

    Tie-breaking: prefer row movement over column movement so the agent doesn't
    oscillate in corridors.
    """
    dr = dst[0] - src[0]
    dc = dst[1] - src[1]
    if abs(dr) >= abs(dc):
        return _UP if dr < 0 else _DOWN
    return _LEFT if dc < 0 else _RIGHT


# ---------------------------------------------------------------------------

class CounterfactualHER:
    """
    Scans completed episode trajectories and generates counterfactual
    transitions that teach Krishna to collect pellets when it is safe.

    Args:
        hunter_safe_dist: Hunter Manhattan distance above which a timestep is
                          considered "safe for collection" (default 6).
        pellet_reach:     Nearest-pellet Manhattan distance at or below which
                          there is a reachable pellet to collect (default 4).
        collect_bonus:    Additional reward added to the counterfactual
                          transition (default 30.0).
        grid_size:        Grid side length (default 25).
        max_cf_per_ep:    Maximum counterfactual transitions per episode.
                          Caps the injection rate so CHER doesn't swamp the
                          buffer in long timeout episodes (default 50).
    """

    def __init__(
        self,
        hunter_safe_dist: int = 6,
        pellet_reach: int = 4,
        collect_bonus: float = 30.0,
        grid_size: int = 25,
        max_cf_per_ep: int = 50,
    ) -> None:
        if hunter_safe_dist < 0:
            raise ValueError("hunter_safe_dist must be >= 0")
        if pellet_reach < 0:
            raise ValueError("pellet_reach must be >= 0")
        if collect_bonus < 0:
            raise ValueError("collect_bonus must be >= 0")
        self.hunter_safe_dist = int(hunter_safe_dist)
        self.pellet_reach = int(pellet_reach)
        self.collect_bonus = float(collect_bonus)
        self.grid_size = int(grid_size)
        self.max_cf_per_ep = int(max_cf_per_ep)

    # ------------------------------------------------------------------

    def relabel(
        self,
        trajectory: List[Dict],
    ) -> List[Tuple]:
        """
        Generate counterfactual transitions from one episode trajectory.

        Args:
            trajectory: List of step dicts (see module docstring).

        Returns:
            List of (state, context, action, reward, next_state, next_context, done)
            tuples — ready to be injected into the agent's replay buffer.
        """
        cf_experiences: List[Tuple] = []

        for step in trajectory:
            if len(cf_experiences) >= self.max_cf_per_ep:
                break

            info: Dict = step["info"]
            hunter_dist: int  = info.get("hunter_manhattan_dist", 0)
            pellet_dist: float = info.get("nearest_pellet_dist", 0.0)
            pellet_pos: Optional[Tuple[int, int]] = info.get("nearest_pellet_pos")
            krishna_pos: Optional[Tuple[int, int]] = info.get("krishna_position")
            pellets_remaining: int = info.get("pellets_remaining", 0)

            # Conditions for a "safe collect opportunity"
            if hunter_dist <= self.hunter_safe_dist:
                continue
            if pellet_dist > self.pellet_reach:
                continue
            if pellets_remaining == 0:
                continue
            if pellet_pos is None or krishna_pos is None:
                continue

            # Compute the optimal action toward the nearest pellet
            optimal_action = _optimal_action_toward(krishna_pos, pellet_pos)

            # Only inject if Krishna did NOT take the optimal action
            if step["action"] == optimal_action:
                continue

            # Build counterfactual transition
            cf_reward = float(step["reward"]) + self.collect_bonus
            cf_exp = (
                step["state"],
                step["context"],
                optimal_action,
                cf_reward,
                step["next_state"],
                step["next_context"],
                step["done"],
            )
            cf_experiences.append(cf_exp)

        return cf_experiences

    # ------------------------------------------------------------------
    # Convenience stats
    # ------------------------------------------------------------------

    def count_opportunities(self, trajectory: List[Dict]) -> int:
        """Count timesteps in a trajectory that qualify as safe-collect opportunities."""
        count = 0
        for step in trajectory:
            info = step["info"]
            if (
                info.get("hunter_manhattan_dist", 0) > self.hunter_safe_dist
                and info.get("nearest_pellet_dist", 0.0) <= self.pellet_reach
                and info.get("pellets_remaining", 0) > 0
                and info.get("nearest_pellet_pos") is not None
            ):
                count += 1
        return count
