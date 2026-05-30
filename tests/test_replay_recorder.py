"""
Tests for utils/replay_recorder.py — JSONL episode recording.
"""

import json
from pathlib import Path

import pytest

from utils.replay_recorder import ReplayRecorder, load_replay


# ----------------------------- helpers -----------------------------

def _tiny_grid() -> list[list[int]]:
    g = [[6] * 5 for _ in range(5)]
    g[0][0] = 2  # krishna
    g[4][4] = 3  # hunter
    return g


def _write_simple_episode(run_dir: Path, episode_id: int = 1, n_steps: int = 3,
                          outcome: str = "win") -> Path:
    rec = ReplayRecorder(run_dir=str(run_dir))
    rec.start_episode(episode_id=episode_id, phase=1, difficulty=4,
                      seed=42, agents=["krishna", "hunter"], grid_size=5)
    for t in range(n_steps):
        rec.record_step(
            t=t, grid=_tiny_grid(),
            actions={"krishna": t % 4, "hunter": (t + 1) % 4},
            rewards={"krishna": -0.001, "hunter": 0.0},
            q_values={"krishna": [0.1, 0.2, 0.3, 0.4]},
            lives=3, pellets=t, done=(t == n_steps - 1),
        )
    rec.end_episode(outcome=outcome,
                    total_reward={"krishna": 50.0, "hunter": -10.0},
                    pellets_collected=n_steps)
    return Path(run_dir) / "replays" / f"episode_{episode_id:06d}.jsonl"


# ----------------------------- write/read round-trip -----------------------------

class TestRoundTrip:
    def test_file_is_created(self, tmp_path):
        path = _write_simple_episode(tmp_path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_header_step_footer_structure(self, tmp_path):
        path = _write_simple_episode(tmp_path, n_steps=3)
        replay = load_replay(path)
        assert replay["header"]["episode_id"] == 1
        assert replay["header"]["agents"] == ["krishna", "hunter"]
        assert len(replay["steps"]) == 3
        assert replay["footer"]["outcome"] == "win"
        assert replay["footer"]["total_steps"] == 3

    def test_step_ordering_preserved(self, tmp_path):
        path = _write_simple_episode(tmp_path, n_steps=5)
        replay = load_replay(path)
        ts = [s["t"] for s in replay["steps"]]
        assert ts == [0, 1, 2, 3, 4]

    def test_grid_preserved(self, tmp_path):
        path = _write_simple_episode(tmp_path)
        replay = load_replay(path)
        assert replay["steps"][0]["grid"] == _tiny_grid()

    def test_q_values_preserved(self, tmp_path):
        path = _write_simple_episode(tmp_path)
        replay = load_replay(path)
        assert replay["steps"][0]["q_values"]["krishna"] == [0.1, 0.2, 0.3, 0.4]

    def test_disabled_recorder_writes_nothing(self, tmp_path):
        rec = ReplayRecorder(run_dir=str(tmp_path), enabled=False)
        rec.start_episode(episode_id=1, phase=1, difficulty=1, seed=0, agents=["krishna"])
        rec.record_step(t=0, grid=_tiny_grid(), actions={"krishna": 0},
                        rewards={"krishna": 0.0})
        rec.end_episode("loss", {"krishna": -1.0}, 0)
        # No replays dir should be created, or if it exists, empty.
        replays = tmp_path / "replays"
        if replays.exists():
            assert list(replays.iterdir()) == []


# ----------------------------- multi-episode -----------------------------

class TestMultipleEpisodes:
    def test_two_episodes_same_recorder(self, tmp_path):
        rec = ReplayRecorder(run_dir=str(tmp_path))
        for ep in [1, 2]:
            rec.start_episode(episode_id=ep, phase=1, difficulty=1, seed=ep,
                              agents=["krishna"], grid_size=5)
            rec.record_step(t=0, grid=_tiny_grid(), actions={"krishna": 0},
                            rewards={"krishna": 0.0}, done=True)
            rec.end_episode("loss", {"krishna": 0.0}, 0)
        p1 = tmp_path / "replays" / "episode_000001.jsonl"
        p2 = tmp_path / "replays" / "episode_000002.jsonl"
        assert p1.exists() and p2.exists()
        # files must be independent
        r1 = load_replay(p1)
        r2 = load_replay(p2)
        assert r1["header"]["seed"] == 1
        assert r2["header"]["seed"] == 2

    def test_implicit_close_on_new_episode(self, tmp_path):
        """Calling start_episode again should cleanly close the previous file."""
        rec = ReplayRecorder(run_dir=str(tmp_path))
        rec.start_episode(episode_id=1, phase=1, difficulty=1, seed=0,
                          agents=["krishna"], grid_size=5)
        rec.record_step(t=0, grid=_tiny_grid(), actions={"krishna": 0},
                        rewards={"krishna": 0.0})
        # forget end_episode and start a new one
        rec.start_episode(episode_id=2, phase=1, difficulty=1, seed=1,
                          agents=["krishna"], grid_size=5)
        rec.record_step(t=0, grid=_tiny_grid(), actions={"krishna": 1},
                        rewards={"krishna": 0.0})
        rec.end_episode("loss", {"krishna": 0.0}, 0)
        # ep1 should still be readable (header + 1 step, no footer)
        p1 = tmp_path / "replays" / "episode_000001.jsonl"
        with open(p1) as fh:
            lines = [json.loads(l) for l in fh if l.strip()]
        assert lines[0]["type"] == "header"
        assert lines[-1]["type"] == "step"


# ----------------------------- validation -----------------------------

class TestLoadValidation:
    def test_rejects_missing_header(self, tmp_path):
        bad = tmp_path / "bad.jsonl"
        bad.write_text(json.dumps({"type": "step", "t": 0, "grid": [], "actions": {},
                                   "rewards": {}, "done": False}) + "\n")
        with pytest.raises(ValueError, match="missing header"):
            load_replay(bad)

    def test_rejects_out_of_order_steps(self, tmp_path):
        bad = tmp_path / "bad.jsonl"
        with open(bad, "w") as fh:
            fh.write(json.dumps({"type": "header", "episode_id": 1, "phase": 1,
                                 "difficulty": 1, "seed": 0, "grid_size": 5,
                                 "agents": ["k"]}) + "\n")
            fh.write(json.dumps({"type": "step", "t": 0, "grid": [], "actions": {},
                                 "rewards": {}, "done": False}) + "\n")
            fh.write(json.dumps({"type": "step", "t": 2, "grid": [], "actions": {},
                                 "rewards": {}, "done": False}) + "\n")
        with pytest.raises(ValueError, match="step ordering"):
            load_replay(bad)


# ----------------------------- jsonl-ness -----------------------------

class TestStreamability:
    def test_each_line_is_valid_json(self, tmp_path):
        path = _write_simple_episode(tmp_path, n_steps=4)
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                json.loads(line)  # raises if malformed

    def test_records_flush_immediately(self, tmp_path):
        """Mid-episode the file should be readable up to the last completed step."""
        rec = ReplayRecorder(run_dir=str(tmp_path))
        rec.start_episode(episode_id=1, phase=1, difficulty=1, seed=0,
                          agents=["krishna"], grid_size=5)
        rec.record_step(t=0, grid=_tiny_grid(), actions={"krishna": 0},
                        rewards={"krishna": 0.0})
        # Read it without closing
        path = tmp_path / "replays" / "episode_000001.jsonl"
        content = path.read_text()
        assert "header" in content
        assert '"t":0' in content
        rec.close()
