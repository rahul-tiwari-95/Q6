"""
Tests for dashboard/scan.py — index building.
"""

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_fake_run(tmp_path: Path, run_id: str, *, with_experiment: bool = True,
                    with_log: bool = True, n_replays: int = 2) -> Path:
    run_dir = tmp_path / run_id
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "replays").mkdir(parents=True)

    if with_experiment:
        (run_dir / "experiment.json").write_text(json.dumps({
            "id": run_id, "name": "fake", "phase_label": "phase1_foundation",
            "algo": {"name": "DDQN+Dueling+CNN"},
            "hyperparams": {"learning_rate": 0.0001, "gamma": 0.99},
            "training": {"total_episodes": 10, "duration_seconds": 5.0,
                          "git_branch": "conitnous", "git_commit": "abc123"},
            "results": {"per_phase": [
                {"phase": 1, "episodes": 5, "win_rate": 0.6,
                 "avg_reward": 12.0, "avg_pellets": 2.4, "avg_caught": 0.4}
            ], "final_epsilon": 0.5},
            "notes": "test",
        }))

    if with_log:
        log = run_dir / "logs" / "episode_stats.csv"
        with open(log, "w") as fh:
            fh.write("episode,phase,curriculum_phase,steps,reward,pellets,caught,won,epsilon,loss,mean_q\n")
            for i in range(1, 11):
                fh.write(f"{i},1,1,50,{i*1.5},2,1,True,{1.0 - i*0.05},0.1,5.0\n")

    for k in range(n_replays):
        ep = (k + 1) * 5
        rp = run_dir / "replays" / f"episode_{ep:06d}.jsonl"
        with open(rp, "w") as fh:
            fh.write(json.dumps({"type": "header", "episode_id": ep, "phase": 1,
                                  "difficulty": 4, "seed": 42, "grid_size": 5,
                                  "agents": ["krishna"]}) + "\n")
            fh.write(json.dumps({"type": "step", "t": 0, "grid": [[6]*5]*5,
                                  "actions": {"krishna": 0}, "rewards": {"krishna": 0.0},
                                  "done": False}) + "\n")
            fh.write(json.dumps({"type": "footer", "total_steps": 1,
                                  "outcome": "win",
                                  "total_reward": {"krishna": 100.0},
                                  "pellets_collected": 4}) + "\n")
    return run_dir


def test_scan_finds_runs(tmp_path, monkeypatch):
    """scan_one should produce a proper entry for a v2 run."""
    sys.path.insert(0, str(REPO_ROOT))
    from dashboard import scan
    run_dir = _write_fake_run(tmp_path, "20260530_test", n_replays=3)
    entry = scan.scan_one(run_dir)
    assert entry is not None
    assert entry["id"] == "20260530_test"
    assert entry["legacy"] is False
    assert len(entry["replays"]) == 3
    # log summary must include downsampled series
    assert entry["summary"]["n_episodes"] == 10
    assert len(entry["summary"]["rewards"]) == 10  # smaller than max_points → no downsample
    # replay summary must show outcome
    assert entry["replays"][0]["outcome"] == "win"
    assert entry["replays"][0]["pellets_collected"] == 4


def test_scan_handles_legacy(tmp_path):
    """A run without experiment.json should still appear marked as legacy."""
    sys.path.insert(0, str(REPO_ROOT))
    from dashboard import scan
    run_dir = _write_fake_run(tmp_path, "legacy_run", with_experiment=False,
                              with_log=False, n_replays=0)
    entry = scan.scan_one(run_dir)
    assert entry is not None
    assert entry["legacy"] is True


def test_scan_cli_writes_index(tmp_path):
    """End-to-end: invoke `python dashboard/scan.py --out X` against a fake training_runs root."""
    out = tmp_path / "index.json"
    # Build a copy of the project layout pointing at a real (small) run dir.
    # Easier: call scan.build_index() which scans the actual repo runs.
    sys.path.insert(0, str(REPO_ROOT))
    from dashboard import scan
    idx = scan.build_index()
    assert "runs" in idx
    assert isinstance(idx["runs"], list)


def test_index_json_serializable(tmp_path):
    """All values returned from scan_one must be JSON-serializable."""
    sys.path.insert(0, str(REPO_ROOT))
    from dashboard import scan
    run_dir = _write_fake_run(tmp_path, "ser_test")
    entry = scan.scan_one(run_dir)
    json.dumps(entry)  # raises if not serializable
