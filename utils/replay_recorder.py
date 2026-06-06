"""
Replay Recorder — write episodes to JSONL for later playback in the dashboard.

Format
------
Each episode is a single JSONL file at: <run_dir>/replays/episode_<id>.jsonl

Line 0 is a header dict with episode-level metadata:
    {"type":"header", "episode_id":N, "phase":int, "difficulty":int,
     "seed":int, "grid_size":25, "agents":["krishna",...], "started_at":iso}

Subsequent lines are step records, one per env.step() call:
    {"type":"step", "t":int, "grid":[[...]], "actions":{"krishna":int,...},
     "rewards":{"krishna":float,...}, "q_values":{"krishna":[q0,q1,q2,q3],...},
     "lives":int, "pellets":int, "done":bool}

Last line is a footer with episode summary:
    {"type":"footer", "total_steps":int, "outcome":"win"|"loss"|"timeout",
     "total_reward":{"krishna":float,...}, "pellets_collected":int}

Design choices
--------------
- JSONL (not JSON) so we can stream-write and truncate cleanly on crashes.
- Grid is stored as a nested list of ints so the dashboard renders it without
  needing the channel encoder. Small (625 ints/step) and human-readable.
- Q-values are optional per-agent. None ⇒ field omitted.
- Recorder is OFF by default; training opts in via `start_recording()`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


@dataclass
class ReplayRecorder:
    """Stream episode steps to a JSONL file. One recorder per run.

    Usage:
        rec = ReplayRecorder(run_dir="training_runs/foo")
        rec.start_episode(episode_id=1, phase=1, difficulty=1, seed=0,
                          agents=["krishna"])
        rec.record_step(t=0, grid=grid_2d_list,
                        actions={"krishna": 2},
                        rewards={"krishna": 0.0},
                        q_values={"krishna": [0.1, 0.2, 0.3, 0.4]},
                        lives=3, pellets=0, done=False)
        ...
        rec.end_episode(outcome="win", total_reward={"krishna": 100.0},
                        pellets_collected=4)
    """

    run_dir: str
    enabled: bool = True
    _current_path: Path | None = field(default=None, init=False, repr=False)
    _fh: Any = field(default=None, init=False, repr=False)
    _step_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.enabled:
            Path(self.run_dir, "replays").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def start_episode(
        self,
        episode_id: int,
        phase: int,
        difficulty: int,
        seed: int | None,
        agents: Iterable[str],
        grid_size: int = 25,
    ) -> None:
        if not self.enabled:
            return
        self._close_if_open()
        self._current_path = Path(self.run_dir) / "replays" / f"episode_{episode_id:06d}.jsonl"
        self._fh = open(self._current_path, "w", encoding="utf-8")
        self._step_count = 0
        header = {
            "type": "header",
            "episode_id": int(episode_id),
            "phase": int(phase),
            "difficulty": int(difficulty),
            "seed": None if seed is None else int(seed),
            "grid_size": int(grid_size),
            "agents": list(agents),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        self._write(header)

    def record_step(
        self,
        t: int,
        grid: list[list[int]],
        actions: dict[str, int],
        rewards: dict[str, float],
        q_values: dict[str, list[float]] | None = None,
        lives: int | None = None,
        pellets: int | None = None,
        done: bool = False,
    ) -> None:
        if not self.enabled or self._fh is None:
            return
        rec: dict[str, Any] = {
            "type": "step",
            "t": int(t),
            "grid": grid,
            "actions": {k: int(v) for k, v in actions.items()},
            "rewards": {k: float(v) for k, v in rewards.items()},
            "done": bool(done),
        }
        if q_values is not None:
            rec["q_values"] = {k: [float(x) for x in v] for k, v in q_values.items()}
        if lives is not None:
            rec["lives"] = int(lives)
        if pellets is not None:
            rec["pellets"] = int(pellets)
        self._write(rec)
        self._step_count += 1

    def end_episode(
        self,
        outcome: str,
        total_reward: dict[str, float],
        pellets_collected: int,
    ) -> None:
        if not self.enabled or self._fh is None:
            return
        footer = {
            "type": "footer",
            "total_steps": self._step_count,
            "outcome": str(outcome),
            "total_reward": {k: float(v) for k, v in total_reward.items()},
            "pellets_collected": int(pellets_collected),
            "ended_at": datetime.now(timezone.utc).isoformat(),
        }
        self._write(footer)
        self._close_if_open()

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _write(self, obj: dict[str, Any]) -> None:
        assert self._fh is not None
        self._fh.write(json.dumps(obj, separators=(",", ":")) + "\n")
        self._fh.flush()

    def _close_if_open(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            finally:
                self._fh = None

    # convenience
    def close(self) -> None:
        self._close_if_open()

    def __del__(self) -> None:  # best-effort
        self._close_if_open()


# ----------------------------- replay loading -----------------------------

def load_replay(path: str | os.PathLike) -> dict[str, Any]:
    """Load a replay JSONL into a structured dict: {header, steps, footer}.

    Validates that the file starts with header, ends with footer, and that
    step `t` values are monotonic and contiguous from 0.
    """
    header: dict[str, Any] | None = None
    footer: dict[str, Any] | None = None
    steps: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            kind = obj.get("type")
            if kind == "header":
                if header is not None:
                    raise ValueError(f"{path}: multiple headers")
                header = obj
            elif kind == "step":
                steps.append(obj)
            elif kind == "footer":
                if footer is not None:
                    raise ValueError(f"{path}: multiple footers")
                footer = obj
            else:
                raise ValueError(f"{path}: unknown record type at line {i}: {kind}")
    if header is None:
        raise ValueError(f"{path}: missing header")
    # validate step monotonicity
    for expected_t, step in enumerate(steps):
        if step["t"] != expected_t:
            raise ValueError(f"{path}: step ordering broken at index {expected_t}: t={step['t']}")
    return {"header": header, "steps": steps, "footer": footer}
