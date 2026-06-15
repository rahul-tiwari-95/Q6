"""
Self-Play Gridworld — clean 1v1 Krishna vs Hunter.

This is a NEW env, not a modification of `HunterGridworld`. The original env
(curriculum + scripted hunter + greedy + patrol) is preserved for Phase 1
reproduction and A/B comparison.

Differences vs HunterGridworld:
- No greedy bots, no patrollers (1v1 clean dynamics for FSP).
- Hunter is controlled by an external action, not a scripted policy.
- `step` takes a dict of actions and returns a dict of rewards.
- Same flat-625 state encoding (both agents see the same grid → full vision).

Action layout: 0=Up, 1=Down, 2=Left, 3=Right.

Cell IDs match `HunterGridworld`:
    WALL=0, PELLET=1, KRISHNA=2, HUNTER=3, EMPTY=6
(GREEDY=4 and PATROL=5 are unused here but kept reserved so the state encoder
 can be shared.)
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SelfPlayGridworld(gym.Env):
    """Two-player gridworld: Krishna collects pellets, Hunter chases Krishna."""

    metadata = {"render_modes": []}

    # Cell IDs (must match HunterGridworld so state_encoder is shared)
    WALL = 0
    PELLET = 1
    KRISHNA = 2
    HUNTER = 3
    EMPTY = 6

    AGENTS = ("krishna", "hunter")

    # Reward constants (kept as class attrs so tests can introspect/modify)
    K_PELLET = 50.0
    K_WIN = 100.0
    K_CAUGHT = -10.0
    K_WALL = 0.0    # zero: blocked moves are already a no-op; penalty caused wall-hugging collapse
    K_STEP = -0.001
    K_APPROACH = 0.3  # shaping: reward per Manhattan-step closer to nearest pellet
    K_APPROACH_SAFE_DIST = 8  # v7: proximity shaping only fires when hunter is this far away
    K_TIMEOUT_ZERO_PELLETS = -20.0  # v7: penalty for timing out without collecting any pellets
    K_LOSE = -50.0

    H_CATCH = 30.0
    H_WIN = 50.0
    H_LOSE = -50.0
    H_STEP = -0.001
    H_APPROACH = 0.1  # shaping: reward per Manhattan-step closer to Krishna

    MAX_STEPS = 1000
    TARGET_PELLETS = 4
    KRISHNA_LIVES = 3
    INVULN_FRAMES = 30

    def __init__(self, grid_size: int = 25, wall_density: float = 0.20):
        super().__init__()
        self.grid_size = int(grid_size)
        self.wall_density = float(wall_density)
        self.state_size = self.grid_size * self.grid_size

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=6, shape=(self.state_size,), dtype=np.int32
        )

        # Runtime state
        self.grid: np.ndarray | None = None
        self.krishna_pos: Tuple[int, int] | None = None
        self.hunter_pos: Tuple[int, int] | None = None
        self.pellet_positions: set = set()
        self.krishna_lives = 0
        self.pellets_collected = 0
        self.times_caught = 0
        self.invuln_timer = 0
        self.steps = 0
        self.walls_hit = 0

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None, options: Dict | None = None
              ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.steps = 0
        self.walls_hit = 0
        self.pellets_collected = 0
        self.times_caught = 0
        self.invuln_timer = 0
        self.krishna_lives = self.KRISHNA_LIVES

        self.grid = np.full((self.grid_size, self.grid_size), self.EMPTY, dtype=np.int32)
        self._generate_walls()
        self._place_pellets(self.TARGET_PELLETS)

        # Spawn Krishna and Hunter in opposite halves of the board to avoid
        # the trivial "caught on step 1" episodes.
        self.krishna_pos = self._random_empty_in_quadrant(top=True)
        self.grid[self.krishna_pos] = self.KRISHNA
        self.hunter_pos = self._random_empty_in_quadrant(top=False)
        self.grid[self.hunter_pos] = self.HUNTER

        return self._get_state(), self._get_info()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, actions: Dict[str, int]
             ) -> Tuple[np.ndarray, Dict[str, float], bool, bool, Dict[str, Any]]:
        """Execute one dual-agent timestep.

        Args:
            actions: {'krishna': int, 'hunter': int} with values in {0,1,2,3}.
        Returns:
            state, rewards_dict, terminated, truncated, info
        """
        if self.grid is None:
            raise RuntimeError("Call reset() before step().")
        if "krishna" not in actions or "hunter" not in actions:
            raise ValueError("actions must contain both 'krishna' and 'hunter' keys.")

        self.steps += 1
        r_k = self.K_STEP
        r_h = self.H_STEP
        terminated = False
        winner: str | None = None

        # -- 1. Move Krishna --
        kx, ky = self.krishna_pos
        hunter_dist_pre = int(abs(kx - self.hunter_pos[0]) + abs(ky - self.hunter_pos[1]))
        # Pellet proximity shaping: measure distance before move
        d_before = self._nearest_pellet_dist((kx, ky))
        nkx, nky = self._apply_action((kx, ky), actions["krishna"])
        if (nkx, nky) == (kx, ky) and self._would_have_moved(actions["krishna"], (kx, ky)):
            # Tried to move but blocked by wall or boundary
            r_k += self.K_WALL
            self.walls_hit += 1
        else:
            # Successful move (or no-op chosen). Clear old, write new.
            self.grid[kx, ky] = self.EMPTY
            self.krishna_pos = (nkx, nky)
            # Pellet collection
            if (nkx, nky) in self.pellet_positions:
                r_k += self.K_PELLET
                self.pellets_collected += 1
                self.pellet_positions.discard((nkx, nky))
            else:
                # Proximity shaping: only when Hunter is far enough (safe to collect)
                d_after = self._nearest_pellet_dist((nkx, nky))
                if hunter_dist_pre > self.K_APPROACH_SAFE_DIST:
                    r_k += self.K_APPROACH * (d_before - d_after)
            self.grid[nkx, nky] = self.KRISHNA

        # -- 2. Move Hunter --
        hx, hy = self.hunter_pos
        # Chase shaping: measure distance to Krishna after Krishna moves.
        hd_before = float(abs(hx - self.krishna_pos[0]) + abs(hy - self.krishna_pos[1]))
        nhx, nhy = self._apply_action((hx, hy), actions["hunter"])
        # Hunter can't enter wall, but it CAN step onto Krishna's cell (that's a catch).
        if self.grid[nhx, nhy] == self.WALL:
            nhx, nhy = hx, hy  # blocked; no wall penalty for Hunter
        else:
            # Clear old hunter cell unless it was overwritten by Krishna step
            if self.grid[hx, hy] == self.HUNTER:
                self.grid[hx, hy] = self.EMPTY
            self.hunter_pos = (nhx, nhy)
            hd_after = float(abs(nhx - self.krishna_pos[0]) + abs(nhy - self.krishna_pos[1]))
            # Do not add shaping on immediate catch steps to keep catch reward clear.
            if (nhx, nhy) != self.krishna_pos:
                r_h += self.H_APPROACH * (hd_before - hd_after)

        # -- 3. Collision check --
        caught = (self.krishna_pos == self.hunter_pos) and self.invuln_timer == 0
        if caught:
            r_h += self.H_CATCH
            r_k += self.K_CAUGHT
            self.times_caught += 1
            self.krishna_lives -= 1
            self.invuln_timer = self.INVULN_FRAMES
            if self.krishna_lives <= 0:
                terminated = True
                winner = "hunter"
                r_h += self.H_WIN
                r_k += self.K_LOSE

        # Repaint hunter cell (so collision visual is correct on grid output)
        if not terminated:
            # Always ensure hunter shows; Krishna takes visual priority if on same cell
            if self.hunter_pos != self.krishna_pos:
                self.grid[self.hunter_pos] = self.HUNTER

        # Tick invulnerability
        if self.invuln_timer > 0:
            self.invuln_timer -= 1

        # -- 4. Win check for Krishna --
        if not terminated and self.pellets_collected >= self.TARGET_PELLETS:
            terminated = True
            winner = "krishna"
            r_k += self.K_WIN
            r_h += self.H_LOSE

        truncated = (not terminated) and (self.steps >= self.MAX_STEPS)
        if truncated:
            winner = "timeout"
            if self.pellets_collected == 0:
                r_k += self.K_TIMEOUT_ZERO_PELLETS

        info = self._get_info()
        info["winner"] = winner
        info["caught_this_step"] = caught

        rewards = {"krishna": float(r_k), "hunter": float(r_h)}
        return self._get_state(), rewards, terminated, truncated, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _apply_action(self, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        x, y = pos
        if action == 0:
            nx, ny = x - 1, y
        elif action == 1:
            nx, ny = x + 1, y
        elif action == 2:
            nx, ny = x, y - 1
        elif action == 3:
            nx, ny = x, y + 1
        else:
            raise ValueError(f"Invalid action {action}; expected 0..3")
        if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
            return pos
        # Krishna-style: walls block movement
        if self.grid[nx, ny] == self.WALL:
            return pos
        return (nx, ny)

    def _nearest_pellet_dist(self, pos: Tuple[int, int]) -> float:
        """Manhattan distance to nearest pellet; 0 if no pellets remain."""
        if not self.pellet_positions:
            return 0.0
        px, py = pos
        return float(min(abs(px - qx) + abs(py - qy) for qx, qy in self.pellet_positions))

    def _would_have_moved(self, action: int, pos: Tuple[int, int]) -> bool:
        """True iff the action would have changed position absent walls/edges."""
        x, y = pos
        target = {
            0: (x - 1, y), 1: (x + 1, y),
            2: (x, y - 1), 3: (x, y + 1),
        }.get(action, pos)
        return target != pos

    def _generate_walls(self) -> None:
        n = int(self.wall_density * self.state_size)
        placed = 0
        attempts = 0
        max_attempts = n * 20
        while placed < n and attempts < max_attempts:
            attempts += 1
            x = int(self.np_random.integers(1, self.grid_size - 1))
            y = int(self.np_random.integers(1, self.grid_size - 1))
            if self.grid[x, y] != self.EMPTY:
                continue
            self.grid[x, y] = self.WALL
            placed += 1

    def _place_pellets(self, count: int) -> None:
        self.pellet_positions = set()
        empties = np.argwhere(self.grid == self.EMPTY)
        if len(empties) < count:
            raise RuntimeError("Not enough empty cells to place pellets.")
        idxs = self.np_random.choice(len(empties), size=count, replace=False)
        for i in idxs:
            pos = (int(empties[i][0]), int(empties[i][1]))
            self.grid[pos] = self.PELLET
            self.pellet_positions.add(pos)

    def _random_empty_in_quadrant(self, top: bool) -> Tuple[int, int]:
        """Pick a random empty cell in the top or bottom half of the grid."""
        half = self.grid_size // 2
        if top:
            mask = np.zeros_like(self.grid, dtype=bool)
            mask[:half, :] = (self.grid[:half, :] == self.EMPTY)
        else:
            mask = np.zeros_like(self.grid, dtype=bool)
            mask[half:, :] = (self.grid[half:, :] == self.EMPTY)
        candidates = np.argwhere(mask)
        if len(candidates) == 0:
            # Fallback: any empty cell
            candidates = np.argwhere(self.grid == self.EMPTY)
        idx = int(self.np_random.integers(0, len(candidates)))
        return (int(candidates[idx][0]), int(candidates[idx][1]))

    def _get_state(self) -> np.ndarray:
        return self.grid.flatten().astype(np.int32)

    def _get_info(self) -> Dict[str, Any]:
        # Compute derived distances for gate context and CHER relabeling.
        hunter_dist = 0
        if self.krishna_pos is not None and self.hunter_pos is not None:
            hunter_dist = int(
                abs(self.krishna_pos[0] - self.hunter_pos[0])
                + abs(self.krishna_pos[1] - self.hunter_pos[1])
            )

        nearest_pellet_pos: Tuple[int, int] | None = None
        nearest_pellet_dist = 0.0
        if self.pellet_positions and self.krishna_pos is not None:
            nearest_pellet_pos = min(
                self.pellet_positions,
                key=lambda p: abs(p[0] - self.krishna_pos[0]) + abs(p[1] - self.krishna_pos[1]),
            )
            nearest_pellet_dist = float(
                abs(nearest_pellet_pos[0] - self.krishna_pos[0])
                + abs(nearest_pellet_pos[1] - self.krishna_pos[1])
            )

        return {
            "steps": self.steps,
            "pellets_collected": self.pellets_collected,
            "pellets_remaining": len(self.pellet_positions),
            "times_caught": self.times_caught,
            "lives": self.krishna_lives,
            "krishna_position": self.krishna_pos,
            "hunter_position": self.hunter_pos,
            "walls_hit": self.walls_hit,
            # ---- extended fields for GOP gate and CHER ----
            "hunter_manhattan_dist": hunter_dist,
            "nearest_pellet_dist": nearest_pellet_dist,
            "nearest_pellet_pos": nearest_pellet_pos,
        }
