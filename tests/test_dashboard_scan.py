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


# ---------------------------------------------------------------------------
# Merge behavior — regression coverage for a near-miss found while adding
# v8/v7-ablations dashboard support: training_runs/ is gitignored, so no
# local checkout has every run that was ever indexed. A naive rebuild would
# silently delete the historical record for any run not present locally.
# ---------------------------------------------------------------------------

def test_build_index_preserves_historical_entry_not_present_locally(tmp_path, monkeypatch):
    sys.path.insert(0, str(REPO_ROOT))
    from dashboard import scan
    monkeypatch.setattr(scan, "RUNS_DIR", tmp_path / "training_runs_empty")
    # No local run dirs at all -- simulates a fresh checkout.
    existing = tmp_path / "index.json"
    existing.write_text(json.dumps({
        "generated_at": "then", "repo_root": "x",
        "runs": [{"id": "historical_only_run", "phase_label": "phase1_foundation"}],
    }))

    idx = scan.build_index(existing_path=existing)

    ids = [r["id"] for r in idx["runs"]]
    assert "historical_only_run" in ids, "regenerating the index must not delete runs missing locally"


def test_build_index_local_run_wins_over_stale_indexed_copy(tmp_path, monkeypatch):
    sys.path.insert(0, str(REPO_ROOT))
    from dashboard import scan
    runs_dir = tmp_path / "training_runs"
    runs_dir.mkdir()
    monkeypatch.setattr(scan, "RUNS_DIR", runs_dir)
    _write_fake_run(runs_dir, "same_id_run", n_replays=1)

    existing = tmp_path / "index.json"
    existing.write_text(json.dumps({
        "generated_at": "then", "repo_root": "x",
        "runs": [{"id": "same_id_run", "phase_label": "STALE_SHOULD_BE_REPLACED"}],
    }))

    idx = scan.build_index(existing_path=existing)

    entries = [r for r in idx["runs"] if r["id"] == "same_id_run"]
    assert len(entries) == 1, "must not duplicate an id present both locally and in the existing index"
    assert entries[0]["phase_label"] != "STALE_SHOULD_BE_REPLACED"


def test_build_index_no_merge_flag_ignores_existing(tmp_path, monkeypatch):
    sys.path.insert(0, str(REPO_ROOT))
    from dashboard import scan
    monkeypatch.setattr(scan, "RUNS_DIR", tmp_path / "training_runs_empty")
    existing = tmp_path / "index.json"
    existing.write_text(json.dumps({
        "generated_at": "then", "repo_root": "x",
        "runs": [{"id": "historical_only_run", "phase_label": "phase1_foundation"}],
    }))

    idx = scan.build_index(existing_path=None)  # what --no-merge passes

    assert idx["runs"] == []


# ---------------------------------------------------------------------------
# Optional per-algorithm series — a PPO-only column must not leak into a DQN
# run's summary (or vice versa); see dashboard/scan.py's _OPTIONAL_SERIES.
# ---------------------------------------------------------------------------

def _write_csv(run_dir: Path, header: str, rows: list[str]) -> None:
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    log = run_dir / "logs" / "episode_stats.csv"
    with open(log, "w") as fh:
        fh.write(header + "\n")
        for r in rows:
            fh.write(r + "\n")


def test_ppo_columns_absent_for_dqn_style_csv(tmp_path):
    sys.path.insert(0, str(REPO_ROOT))
    from dashboard import scan
    run_dir = tmp_path / "dqn_run"
    _write_csv(
        run_dir,
        "episode,reward,pellets,won,epsilon,loss",
        [f"{i},{i*1.5},2,True,{1.0 - i*0.05},0.1" for i in range(1, 6)],
    )
    summary = scan._read_log_summary(run_dir / "logs" / "episode_stats.csv")
    assert "epsilons" in summary
    assert "entropies" not in summary
    assert "approx_kls" not in summary
    assert "clipfracs" not in summary


def test_dqn_columns_absent_for_ppo_style_csv(tmp_path):
    sys.path.insert(0, str(REPO_ROOT))
    from dashboard import scan
    run_dir = tmp_path / "ppo_run"
    _write_csv(
        run_dir,
        "episode,krishna_reward,pellets,winner,krishna_loss,krishna_entropy,krishna_approx_kl,krishna_clipfrac,avg100",
        [f"{i},{i*1.5},2,timeout,0.1,1.3,0.01,0.05,10.0" for i in range(1, 6)],
    )
    summary = scan._read_log_summary(run_dir / "logs" / "episode_stats.csv")
    assert "entropies" in summary
    assert "approx_kls" in summary
    assert "clipfracs" in summary
    assert "epsilons" not in summary
    assert "gates" not in summary
