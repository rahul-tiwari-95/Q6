"""
Hierarchical Opponent Pool for Fictitious Self-Play.

Motivation
----------
In v4 training the FSP pool quickly filled with strong late-game Hunters.
Once the easy early-game Hunters were evicted, Krishna had no "safe practice"
opponents left.  The bimodal strategy collapse followed shortly after.

Two-tier design
---------------
Easy tier   — first `easy_max` snapshots (early training, low episode number).
              Never evicted.  Gives Krishna a permanently available weak
              opponent to practice pellet collection against.
Hard tier   — rolling FIFO, `hard_max` slots.  Always contains the most
              recent and strongest opponents.

Sampling
--------
With probability `p_easy` (default 0.25), sample from the easy tier.
Otherwise sample from the hard tier.  Within each tier `p_latest` controls
recency bias (default 0.7 → 70% chance of sampling the latest snapshot).

Usage
-----
    pool = HierarchicalOpponentPool(run_dir / "pool",
                                    easy_max=5, hard_max=15)

    # Add a snapshot — automatically routed to the correct tier.
    pool.add_snapshot(agent, metadata={"episode": ep})

    # Sample an opponent path.
    path = pool.sample(rng, p_easy=0.25, p_latest=0.7)
    if path is not None:
        frozen = FrozenAgent.load(path, device=device)

    # Total pool size (both tiers).
    print(len(pool))
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from agent.opponent_pool import OpponentPool


class HierarchicalOpponentPool:
    INDEX_NAME = "hier_index.json"

    def __init__(
        self,
        pool_dir: str | Path,
        easy_max: int = 5,
        hard_max: int = 15,
    ) -> None:
        self.pool_dir = Path(pool_dir)
        self.pool_dir.mkdir(parents=True, exist_ok=True)

        self.easy_pool = OpponentPool(self.pool_dir / "easy", max_size=easy_max)
        self.hard_pool = OpponentPool(self.pool_dir / "hard", max_size=hard_max)

        # Load persisted state (tracks whether easy tier is full)
        idx = self._load_index()
        self._easy_full: bool = idx.get("easy_full", False)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_snapshot(
        self, agent: Any, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a snapshot to the appropriate tier and return the path written.

        The easy tier fills first (up to `easy_max` entries) and is never
        evicted.  All subsequent snapshots go into the FIFO hard tier.
        """
        if not self._easy_full:
            path = self.easy_pool.add_snapshot(agent, metadata)
            if len(self.easy_pool) >= self.easy_pool.max_size:
                self._easy_full = True
            self._save_index()
            return path

        path = self.hard_pool.add_snapshot(agent, metadata)
        return path

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        rng: np.random.Generator,
        p_easy: float = 0.25,
        p_latest: float = 0.7,
    ) -> Optional[str]:
        """
        Return a snapshot path, or None if both tiers are empty.

        Args:
            p_easy:   Probability of sampling from the easy tier when it has
                      entries.  Ensures Krishna always has weak-opponent
                      practice even as the hard tier grows stronger.
            p_latest: Within a tier, probability of returning the most recent
                      snapshot (recency bias).
        """
        has_easy = len(self.easy_pool) > 0
        has_hard = len(self.hard_pool) > 0

        if not has_easy and not has_hard:
            return None

        use_easy = has_easy and (not has_hard or rng.random() < p_easy)
        tier = self.easy_pool if use_easy else self.hard_pool
        return tier.sample(rng, p_latest=p_latest)

    def latest(self) -> Optional[str]:
        """Return the most recent snapshot across both tiers (hard tier first)."""
        return self.hard_pool.latest() or self.easy_pool.latest()

    def __len__(self) -> int:
        return len(self.easy_pool) + len(self.hard_pool)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _index_path(self) -> Path:
        return self.pool_dir / self.INDEX_NAME

    def _save_index(self) -> None:
        self._index_path().write_text(
            json.dumps({"easy_full": self._easy_full}, indent=2)
        )

    def _load_index(self) -> Dict:
        p = self._index_path()
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
