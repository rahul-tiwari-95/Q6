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
Otherwise sample from the hard tier.

Hard-tier sampling has two modes, selected by the `rectified` constructor
flag:

  rectified=False (legacy / v7 behaviour)
      Pure recency sampling within the tier: `p_latest` controls the chance
      of returning the most recent snapshot, otherwise uniform-random over
      the tier.  No awareness of how Krishna is currently performing against
      any given snapshot.

  rectified=True (default — ablation 1)
      "Rectified" sampling inspired by PSRO's rectified Nash response
      (Balduzzi et al., 2019): snapshots Krishna is currently beating or
      tying get *more* sampling weight, snapshots that are dominating
      Krishna get *less* — but never zero, via a uniform floor.  This is
      meant to concentrate training signal on opponents Krishna is close to
      solving, while the floor keeps truly hard opponents in rotation (the
      risk being that a floor of 0 would let Krishna retreat into an
      easy-opponent-only niche and never learn to beat harder snapshots).

      Weight for hard-tier snapshot i:
          weight_i = max(score_i - baseline, 0) + floor
      where `score_i` is Krishna's EMA outcome score against snapshot i
      (see `record_outcome`), `baseline` is the mean score across the hard
      tier (recomputed at sample time — "rolling" in the sense that it
      always reflects current EMA state, not a fixed target), and `floor`
      is `rectify_floor` (default 0.1).  Weights are normalized to a
      probability distribution.

      `rectify_floor` can be raised at construction time (see the
      `train_phase3.py` `--rectify-floor` flag) or ramped in gradually at
      runtime via `set_rectify_floor()` (see `train_phase3.py`
      `--rectify-warmup-eps`, which mirrors the `anchor_weight` decay
      pattern used for ablation 2) — this was added after ablation 1's
      first real run found a mid-training vulnerability window caused by
      full-strength rectification kicking in at the exact same episode the
      easy-warmup curriculum ends (see versions/v7_ablation1_rectified.md).

Fitness tracking
-----------------
Call `record_outcome(path, score)` after every FSP-mode episode, where
`score` is in [0, 1] (0 = full loss / no pellets collected, 1 = win).  This
updates an EMA (`ema_alpha`, default 0.1) of Krishna's performance against
that specific snapshot.  Scores are persisted alongside the pool index so
they survive process restarts / --resume.

Usage
-----
    pool = HierarchicalOpponentPool(run_dir / "pool", easy_max=5, hard_max=15)

    # Add a snapshot — automatically routed to the correct tier.
    pool.add_snapshot(agent, metadata={"episode": ep})

    # Sample an opponent path.
    path = pool.sample(rng, p_easy=0.25, p_latest=0.7)
    if path is not None:
        frozen = FrozenAgent.load(path, device=device)

    # After the FSP episode completes, report Krishna's outcome.
    pool.record_outcome(path, score=1.0 if krishna_won else 0.0)

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

    # Starting floor for a `rectify_warmup_eps` ramp (see train_phase3.py).
    # `record_outcome` clips scores to [0, 1], so `baseline` is also always
    # in [0, 1] and `max(score_i - baseline, 0)` never exceeds 1.0. A floor
    # of 1.0 therefore guarantees the fitness term contributes at most half
    # of any snapshot's weight — a bounded, "near-uniform" starting point
    # for the ramp (not perfectly uniform; that would need floor -> inf).
    UNIFORM_WARMUP_FLOOR = 1.0

    def __init__(
        self,
        pool_dir: str | Path,
        easy_max: int = 5,
        hard_max: int = 15,
        rectified: bool = True,
        ema_alpha: float = 0.1,
        rectify_floor: float = 0.1,
        default_score: float = 0.5,
    ) -> None:
        """
        Args:
            pool_dir:       Directory holding both tiers + persisted index.
            easy_max:       Max snapshots in the permanent easy tier.
            hard_max:       Max snapshots in the rolling-FIFO hard tier.
            rectified:      If True (default), hard-tier sampling uses
                             fitness-rectified weighting (ablation 1). If
                             False, falls back to the original p_latest
                             recency-only sampling — kept for a clean A/B
                             comparison against the new behaviour.
            ema_alpha:      EMA decay for `record_outcome` (default 0.1).
            rectify_floor:  Minimum weight floor for rectified sampling so
                             no hard-tier snapshot is ever fully excluded
                             (default 0.1).
            default_score:  Score assumed for a hard-tier snapshot that has
                             not yet had any outcome recorded (default 0.5,
                             i.e. treated as a "tie" until observed).
        """
        self.pool_dir = Path(pool_dir)
        self.pool_dir.mkdir(parents=True, exist_ok=True)

        self.easy_pool = OpponentPool(self.pool_dir / "easy", max_size=easy_max)
        self.hard_pool = OpponentPool(self.pool_dir / "hard", max_size=hard_max)

        self.rectified = bool(rectified)
        self.ema_alpha = float(ema_alpha)
        self.rectify_floor = float(rectify_floor)
        self.default_score = float(default_score)

        # Load persisted state (tracks whether easy tier is full, and the
        # per-snapshot EMA outcome scores keyed by snapshot path).
        idx = self._load_index()
        self._easy_full: bool = idx.get("easy_full", False)
        self._scores: Dict[str, float] = dict(idx.get("scores", {}))
        self._prune_stale_scores()

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
        # FIFO eviction may have dropped an old snapshot — drop its score too.
        self._prune_stale_scores()
        self._save_index()
        return path

    def record_outcome(self, path: str, score: float) -> None:
        """
        Update the EMA outcome score for a stored snapshot.

        Called by the training loop after an FSP-mode episode against the
        snapshot at `path`. `score` should be in [0, 1]: 0 = full loss / no
        pellets collected, 1 = win. Values outside [0, 1] are clipped.

        Paths that are no longer present in either tier (e.g. evicted
        between sampling and recording) are still recorded — they will be
        pruned on the next add_snapshot() / reload.
        """
        score = float(np.clip(float(score), 0.0, 1.0))
        prev = self._scores.get(path)
        if prev is None:
            self._scores[path] = score
        else:
            self._scores[path] = (1.0 - self.ema_alpha) * prev + self.ema_alpha * score
        self._save_index()

    def set_rectify_floor(self, floor: float) -> None:
        """
        Update the effective rectify floor at runtime.

        Mirrors `GatedDQNAgent.set_anchor_weight()` — intended to be called
        once per episode by the training loop when ramping the floor in via
        `rectify_warmup_eps` (see `train_phase3.py`). Takes effect on the
        very next `sample()` / `hard_tier_weights()` call, since both read
        `self.rectify_floor` fresh rather than caching it.
        """
        self.rectify_floor = float(floor)

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
            p_latest: Within the easy tier (always) and within the hard tier
                      when `rectified=False`, probability of returning the
                      most recent snapshot (recency bias).  Ignored for the
                      hard tier when `rectified=True`.
        """
        has_easy = len(self.easy_pool) > 0
        has_hard = len(self.hard_pool) > 0

        if not has_easy and not has_hard:
            return None

        use_easy = has_easy and (not has_hard or rng.random() < p_easy)
        if use_easy:
            return self.easy_pool.sample(rng, p_latest=p_latest)

        if self.rectified:
            return self._sample_hard_rectified(rng)
        return self.hard_pool.sample(rng, p_latest=p_latest)

    def _sample_hard_rectified(self, rng: np.random.Generator) -> str:
        """Fitness-rectified sampling within the hard tier (ablation 1)."""
        weights = self.hard_tier_weights()
        total = float(weights.sum())
        entries_paths = self.hard_pool.paths()
        if total <= 0.0:
            idx = int(rng.integers(0, len(entries_paths)))
        else:
            probs = weights / total
            idx = int(rng.choice(len(entries_paths), p=probs))
        return entries_paths[idx]

    # ------------------------------------------------------------------
    # Fitness / rectification helpers
    # ------------------------------------------------------------------

    def hard_tier_scores(self) -> np.ndarray:
        """EMA outcome score per current hard-tier snapshot (in tier order)."""
        paths = self.hard_pool.paths()
        return np.array(
            [self._scores.get(p, self.default_score) for p in paths],
            dtype=np.float64,
        )

    def hard_tier_weights(self) -> np.ndarray:
        """
        Rectified sampling weight per current hard-tier snapshot.

        weight_i = max(score_i - baseline, 0) + floor

        `baseline` is the mean score across the hard tier, recomputed from
        current EMA state every call. Returns an empty array if the hard
        tier is empty.
        """
        scores = self.hard_tier_scores()
        if scores.size == 0:
            return scores
        baseline = float(scores.mean())
        return np.maximum(scores - baseline, 0.0) + self.rectify_floor

    def mean_hard_tier_score(self) -> float:
        """Mean Krishna EMA score across current hard-tier snapshots (for logging)."""
        scores = self.hard_tier_scores()
        return float(scores.mean()) if scores.size else 0.0

    def _prune_stale_scores(self) -> None:
        """Drop EMA scores for snapshots no longer present in either tier."""
        live_paths = set(self.easy_pool.paths()) | set(self.hard_pool.paths())
        self._scores = {p: s for p, s in self._scores.items() if p in live_paths}

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
            json.dumps({"easy_full": self._easy_full, "scores": self._scores}, indent=2)
        )

    def _load_index(self) -> Dict:
        p = self._index_path()
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
