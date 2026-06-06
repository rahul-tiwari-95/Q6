"""Smoke test for train_phase2 (5 episodes, fast)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from train_phase2 import run_training


def test_smoke_run_writes_artifacts(tmp_path, monkeypatch):
    # Redirect REPO_ROOT to tmp by symlinking training_runs
    import train_phase2 as tp2
    monkeypatch.setattr(tp2, "REPO_ROOT", tmp_path)
    (tmp_path / "training_runs").mkdir()

    run_dir = run_training(
        episodes=5,
        device="cpu",
        seed=7,
        name="smoke",
        snapshot_every=2,
        p_latest=0.5,
        replay_every=2,
        warm_start_krishna=None,
    )

    assert run_dir.exists()
    exp = run_dir / "experiment.json"
    assert exp.exists()
    data = json.loads(exp.read_text())
    assert data["phase_label"] == "phase2_selfplay"
    assert data["training"]["total_episodes"] == 5
    assert "wins" in data["results"]
    assert (run_dir / "logs" / "episode_stats.csv").exists()
    assert (run_dir / "checkpoints" / "krishna_final.pth").exists()
    assert (run_dir / "checkpoints" / "hunter_final.pth").exists()
    # Pool should have at least the init snapshot
    assert any((run_dir / "pool").glob("snapshot_*.pth"))
    # At least one replay file
    assert any((run_dir / "replays").glob("episode_*.jsonl"))
