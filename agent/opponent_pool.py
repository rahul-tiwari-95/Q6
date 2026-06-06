"""
Fictitious Self-Play opponent pool.

Stores historical snapshots of an agent (state_dicts on disk) and samples from
them so that the training agent doesn't overfit to the latest opponent.

Usage:
    pool = OpponentPool(pool_dir, max_size=20)
    pool.add_snapshot(agent, metadata={"episode": 500})
    path = pool.sample(rng, p_latest=0.7)  # may be None if empty
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class OpponentPool:
    INDEX_NAME = "pool_index.json"

    def __init__(self, pool_dir: str | Path, max_size: int = 20):
        self.pool_dir = Path(pool_dir)
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = int(max_size)
        self._next_id = 0
        self.entries: List[Dict[str, Any]] = []  # each: {"path": str, "metadata": {...}}
        self._load_index()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def add_snapshot(self, agent: Any, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save the agent's state to a new snapshot file. Returns the path written."""
        snap_id = self._next_id
        self._next_id += 1
        fname = f"snapshot_{snap_id:06d}.pth"
        fpath = self.pool_dir / fname
        agent.save(str(fpath))
        entry = {"path": str(fpath), "metadata": dict(metadata or {})}
        self.entries.append(entry)
        # FIFO eviction
        while len(self.entries) > self.max_size:
            old = self.entries.pop(0)
            try:
                Path(old["path"]).unlink(missing_ok=True)
            except OSError:
                pass
        self._save_index()
        return str(fpath)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def sample(self, rng: np.random.Generator, p_latest: float = 0.7) -> Optional[str]:
        """Return a snapshot path or None if pool empty.

        With probability `p_latest` returns the most recent snapshot; otherwise
        a uniformly random historical snapshot (including the latest)."""
        if not self.entries:
            return None
        if rng.random() < p_latest:
            return self.entries[-1]["path"]
        idx = int(rng.integers(0, len(self.entries)))
        return self.entries[idx]["path"]

    def latest(self) -> Optional[str]:
        return self.entries[-1]["path"] if self.entries else None

    def __len__(self) -> int:
        return len(self.entries)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _index_path(self) -> Path:
        return self.pool_dir / self.INDEX_NAME

    def _save_index(self) -> None:
        payload = {"next_id": self._next_id, "entries": self.entries}
        self._index_path().write_text(json.dumps(payload, indent=2))

    def _load_index(self) -> None:
        p = self._index_path()
        if not p.exists():
            return
        data = json.loads(p.read_text())
        self._next_id = int(data.get("next_id", 0))
        # Keep only entries whose files still exist
        self.entries = [
            e for e in data.get("entries", [])
            if Path(e["path"]).exists()
        ]
