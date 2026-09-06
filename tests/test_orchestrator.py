"""
Tests for orchestrator/orchestrator.py.

Two levels: pure unit tests for the small stateless pieces (command
construction, log parsing), and integration tests that spawn real short-lived
subprocesses through the orchestrator's own scheduling/polling methods (not
the full blocking run() loop with signal handlers, so tests stay fast and
don't fight pytest's own signal handling) to verify crash detection and
resume-flag construction actually work end to end against a real process.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from orchestrator.orchestrator import (
    JobSpec, JobState, Orchestrator, build_orchestrator, default_paths, load_config,
)


# ---------------------------------------------------------------------------
# Unit tests — pure logic, no subprocesses
# ---------------------------------------------------------------------------

class TestBuildCommand:
    def test_fresh_job_has_no_resume_flag(self):
        orch = Orchestrator(jobs=[], log_dir=Path("/tmp/x"), status_path=Path("/tmp/x/s.json"))
        job = JobState(spec=JobSpec(id="a", cwd=".", command=["python3", "s.py", "--x", "1"]))
        assert orch._build_command(job) == ["python3", "s.py", "--x", "1"]

    def test_resuming_job_appends_resume_dir_after_original_flags(self):
        # This is the exact footgun found while testing v8/v7-ablations resume:
        # hyperparameters are NOT restored from the checkpoint, so the ORIGINAL
        # flags must be replayed, with --resume appended, not substituted.
        orch = Orchestrator(jobs=[], log_dir=Path("/tmp/x"), status_path=Path("/tmp/x/s.json"))
        job = JobState(
            spec=JobSpec(id="a", cwd=".", command=["python3", "s.py", "--rollout-len", "2048"]),
            run_dir="training_runs/20260101_000000_a",
        )
        cmd = orch._build_command(job)
        assert cmd == [
            "python3", "s.py", "--rollout-len", "2048",
            "--resume", "training_runs/20260101_000000_a",
        ]


class TestExtractRunDir:
    def test_finds_run_dir_line(self, tmp_path):
        log = tmp_path / "job.log"
        log.write_text("[start] 100 episodes  device=cpu  run_dir=/foo/bar/baz\nmore text\n")
        orch = Orchestrator(jobs=[], log_dir=tmp_path, status_path=tmp_path / "s.json")
        job = JobState(spec=JobSpec(id="a", cwd=".", command=[]), log_path=log)
        orch._extract_run_dir(job)
        assert job.run_dir == "/foo/bar/baz"

    def test_does_not_overwrite_existing_run_dir(self, tmp_path):
        log = tmp_path / "job.log"
        log.write_text("run_dir=/should/not/be/used\n")
        orch = Orchestrator(jobs=[], log_dir=tmp_path, status_path=tmp_path / "s.json")
        job = JobState(spec=JobSpec(id="a", cwd=".", command=[]), log_path=log, run_dir="/already/set")
        orch._extract_run_dir(job)
        assert job.run_dir == "/already/set"

    def test_missing_log_file_is_a_noop(self, tmp_path):
        orch = Orchestrator(jobs=[], log_dir=tmp_path, status_path=tmp_path / "s.json")
        job = JobState(spec=JobSpec(id="a", cwd=".", command=[]), log_path=tmp_path / "nope.log")
        orch._extract_run_dir(job)  # must not raise
        assert job.run_dir is None

    def test_uses_latest_run_dir_in_current_output(self, tmp_path):
        log = tmp_path / "job.log"
        log.write_text("run_dir=/earlier/path\nrun_dir=/current/path\n")
        orch = Orchestrator(jobs=[], log_dir=tmp_path, status_path=tmp_path / "s.json")
        job = JobState(spec=JobSpec(id="a", cwd=".", command=[]), log_path=log)
        orch._extract_run_dir(job)
        assert job.run_dir == "/current/path"


class TestExtractProgress:
    def test_finds_last_episode_line(self, tmp_path):
        log = tmp_path / "job.log"
        log.write_text(
            "[start] ...\n"
            "  ep    10/100  mode=joint  steps=1000  avg100=1.23\n"
            "  ep    20/100  mode=fsp    steps=1000  avg100=4.56\n"
        )
        orch = Orchestrator(jobs=[], log_dir=tmp_path, status_path=tmp_path / "s.json")
        job = JobState(spec=JobSpec(id="a", cwd=".", command=[]), log_path=log)
        orch._extract_progress(job)
        assert "ep    20/100" in job.last_episode_line


class TestLoadAverageThrottle:
    def test_none_means_always_ok(self):
        orch = Orchestrator(jobs=[], log_dir=Path("/tmp/x"), status_path=Path("/tmp/x/s.json"),
                             max_load_average=None)
        assert orch._load_average_ok() is True

    def test_very_high_threshold_is_ok(self):
        orch = Orchestrator(jobs=[], log_dir=Path("/tmp/x"), status_path=Path("/tmp/x/s.json"),
                             max_load_average=10_000.0)
        assert orch._load_average_ok() is True

    def test_zero_threshold_blocks(self):
        orch = Orchestrator(jobs=[], log_dir=Path("/tmp/x"), status_path=Path("/tmp/x/s.json"),
                             max_load_average=-1.0)
        assert orch._load_average_ok() is False


class TestDefaultPaths:
    def test_defaults_are_keyed_off_config_stem(self):
        log_dir, status_path = default_paths(Path("/a/b/q6_jobs.json"), None, None)
        assert log_dir == Path("/a/b/orchestrator_logs_q6_jobs")
        assert status_path == Path("/a/b/orchestrator_status_q6_jobs.json")

    def test_two_configs_in_same_directory_do_not_collide(self):
        log_dir_1, status_path_1 = default_paths(Path("/a/b/q6_jobs.json"), None, None)
        log_dir_2, status_path_2 = default_paths(Path("/a/b/q6_jobs_v2.json"), None, None)
        assert log_dir_1 != log_dir_2
        assert status_path_1 != status_path_2

    def test_explicit_log_dir_overrides_default(self):
        log_dir, _ = default_paths(Path("/a/b/q6_jobs.json"), Path("/custom/logs"), None)
        assert log_dir == Path("/custom/logs")

    def test_explicit_status_file_overrides_default(self):
        _, status_path = default_paths(Path("/a/b/q6_jobs.json"), None, Path("/custom/status.json"))
        assert status_path == Path("/custom/status.json")

    def test_explicit_args_take_precedence_even_for_colliding_stems(self):
        # If the caller passes explicit paths, two configs with the same stem
        # (different directories) still don't collide -- the explicit args
        # win outright, config_path is irrelevant to the result.
        log_dir, status_path = default_paths(
            Path("/other/dir/q6_jobs.json"), Path("/custom/logs"), Path("/custom/status.json")
        )
        assert log_dir == Path("/custom/logs")
        assert status_path == Path("/custom/status.json")


# ---------------------------------------------------------------------------
# Integration tests — real short-lived subprocesses
# ---------------------------------------------------------------------------

_FAKE_SUCCESS = """
import sys
print("[start] fake job  run_dir={}".format(sys.argv[1]), flush=True)
print("  ep 1/1  fake progress line", flush=True)
sys.exit(0)
"""

_FAKE_CRASH_THEN_RESUME_SUCCEEDS = """
import sys
run_dir = sys.argv[1]
print("[start] fake job  run_dir={}".format(run_dir), flush=True)
if "--resume" in sys.argv:
    print("  ep 2/2  resumed and finished", flush=True)
    sys.exit(0)
else:
    print("  ep 1/2  about to crash", flush=True)
    sys.exit(1)
"""


def _write_script(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.write_text(body)
    return p


def _poll_until(orch: Orchestrator, job: JobState, predicate, timeout: float = 10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        orch._schedule()
        orch._poll_job(job)
        if predicate(job):
            return
        time.sleep(0.05)
    pytest.fail(f"condition not met within {timeout}s; last status={job.status}")


class TestIntegrationRealSubprocess:
    def test_fresh_launch_does_not_recover_an_old_run_from_reused_log(self, tmp_path):
        gate = tmp_path / "print_new_run"
        new_run = str(tmp_path / "new_run")
        script = _write_script(tmp_path, "wait_then_report.py", """
import sys
import time
from pathlib import Path
for _ in range(500):
    if Path(sys.argv[1]).exists():
        break
    time.sleep(0.01)
else:
    sys.exit(2)
print('run_dir=' + sys.argv[2], flush=True)
""")
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        (log_dir / "reused.log").write_text("run_dir=/old/unrelated_run\n")
        spec = JobSpec(id="reused", cwd=str(tmp_path),
                       command=[sys.executable, str(script), str(gate), new_run])
        orch = Orchestrator(jobs=[spec], log_dir=log_dir, status_path=tmp_path / "status.json")
        job = orch.jobs[0]

        orch._start_job(job)
        orch._extract_run_dir(job)
        before_new_output = job.run_dir
        gate.touch()
        _poll_until(orch, job, lambda j: j.status == "completed")

        assert before_new_output is None
        assert job.run_dir == new_run

    def test_successful_job_reaches_completed(self, tmp_path):
        script = _write_script(tmp_path, "fake_success.py", _FAKE_SUCCESS)
        run_dir_arg = str(tmp_path / "training_runs" / "fake_run")
        spec = JobSpec(id="succeeds", cwd=str(tmp_path),
                        command=[sys.executable, str(script), run_dir_arg])
        orch = Orchestrator(jobs=[spec], log_dir=tmp_path / "logs", status_path=tmp_path / "status.json")
        job = orch.jobs[0]

        _poll_until(orch, job, lambda j: j.status == "completed")

        assert job.run_dir == run_dir_arg
        assert "fake progress line" in job.last_episode_line
        assert job.restart_count == 0

    def test_crash_triggers_restart_with_resume_flag_and_then_succeeds(self, tmp_path):
        script = _write_script(tmp_path, "fake_crash.py", _FAKE_CRASH_THEN_RESUME_SUCCEEDS)
        run_dir_arg = str(tmp_path / "training_runs" / "fake_run2")
        spec = JobSpec(id="crashes_once", cwd=str(tmp_path),
                        command=[sys.executable, str(script), run_dir_arg],
                        restart_backoff_seconds=0.1, max_restarts=3)
        orch = Orchestrator(jobs=[spec], log_dir=tmp_path / "logs", status_path=tmp_path / "status.json")
        job = orch.jobs[0]

        # First launch: crashes (exit 1). Orchestrator should mark it pending
        # for restart, having already captured run_dir from its stdout.
        _poll_until(orch, job, lambda j: j.status == "pending" and j.restart_count == 1)
        assert job.run_dir == run_dir_arg

        # Backoff elapses, orchestrator restarts it — this time WITH --resume,
        # which the fake script uses to decide to succeed instead of crash.
        _poll_until(orch, job, lambda j: j.status == "completed", timeout=10.0)
        assert job.restart_count == 1  # only needed one restart

    def test_exceeding_max_restarts_marks_failed(self, tmp_path):
        # A script that always crashes, regardless of --resume.
        always_crash = "import sys\nprint('run_dir=%s' % sys.argv[1], flush=True)\nsys.exit(1)\n"
        script = _write_script(tmp_path, "fake_always_crash.py", always_crash)
        run_dir_arg = str(tmp_path / "training_runs" / "fake_run3")
        spec = JobSpec(id="always_crashes", cwd=str(tmp_path),
                        command=[sys.executable, str(script), run_dir_arg],
                        restart_backoff_seconds=0.05, max_restarts=2)
        orch = Orchestrator(jobs=[spec], log_dir=tmp_path / "logs", status_path=tmp_path / "status.json")
        job = orch.jobs[0]

        _poll_until(orch, job, lambda j: j.status == "failed", timeout=10.0)
        assert job.restart_count == 3  # initial attempt + 2 retries, then gives up

    def test_status_file_is_written_and_json_valid(self, tmp_path):
        script = _write_script(tmp_path, "fake_success2.py", _FAKE_SUCCESS)
        run_dir_arg = str(tmp_path / "training_runs" / "fake_run4")
        spec = JobSpec(id="succeeds2", cwd=str(tmp_path),
                        command=[sys.executable, str(script), run_dir_arg])
        status_path = tmp_path / "status.json"
        orch = Orchestrator(jobs=[spec], log_dir=tmp_path / "logs", status_path=status_path)
        job = orch.jobs[0]

        _poll_until(orch, job, lambda j: j.status == "completed")
        orch._write_status()

        payload = json.loads(status_path.read_text())
        assert payload["jobs"][0]["id"] == "succeeds2"
        assert payload["jobs"][0]["status"] == "completed"

    @pytest.mark.parametrize("child_exit, expected_cli_exit, expected_status", [
        (0, 0, "completed"),
        (3, 1, "failed"),
    ])
    def test_cli_exit_status_reflects_child_failure(
        self, tmp_path, child_exit, expected_cli_exit, expected_status
    ):
        config = tmp_path / "jobs.json"
        config.write_text(json.dumps({
            "poll_interval_seconds": 0.01,
            "jobs": [{
                "id": "child", "cwd": str(tmp_path),
                "command": [sys.executable, "-c", f"raise SystemExit({child_exit})"],
                "max_restarts": 0,
            }],
        }))
        status = tmp_path / "status.json"
        cli = Path(__file__).resolve().parents[1] / "orchestrator" / "orchestrator.py"
        result = subprocess.run(
            [sys.executable, str(cli), "--config", str(config),
             "--status-file", str(status)],
            cwd=tmp_path, capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == expected_cli_exit, result.stdout + result.stderr
        assert json.loads(status.read_text())["jobs"][0]["status"] == expected_status

    def test_recovers_run_dir_from_prior_status_file(self, tmp_path):
        # Simulate a previous orchestrator session having written a status
        # file with a job mid-flight, then a fresh Orchestrator instance
        # (as if the whole process/machine restarted) picking it back up.
        status_path = tmp_path / "status.json"
        status_path.write_text(json.dumps({
            "updated_at": "x",
            "jobs": [{"id": "recovering", "status": "running", "run_dir": "/prior/run/dir",
                      "restart_count": 2, "pid": None, "last_progress": "", "log": None,
                      "command": []}],
        }))
        spec = JobSpec(id="recovering", cwd=".", command=["python3", "whatever.py"])
        orch = Orchestrator(jobs=[spec], log_dir=tmp_path / "logs", status_path=status_path)

        assert orch.jobs[0].run_dir == "/prior/run/dir"
        assert orch.jobs[0].restart_count == 2

    def test_does_not_recover_a_completed_job_as_needing_resume(self, tmp_path):
        status_path = tmp_path / "status.json"
        status_path.write_text(json.dumps({
            "updated_at": "x",
            "jobs": [{"id": "done_already", "status": "completed", "run_dir": "/prior/run/dir",
                      "restart_count": 0, "pid": None, "last_progress": "", "log": None,
                      "command": []}],
        }))
        spec = JobSpec(id="done_already", cwd=".", command=["python3", "whatever.py"])
        orch = Orchestrator(jobs=[spec], log_dir=tmp_path / "logs", status_path=status_path)

        assert orch.jobs[0].status == "completed"

    def test_shutdown_reconciles_a_job_that_crashed_in_the_signal_race_window(self, tmp_path):
        # Regression test for a real bug found via a live end-to-end run:
        # a job that crashes near-instantly (fast crash loop) can exit on its
        # own in the narrow window between _handle_shutdown's initial
        # "running" snapshot and the reconciliation pass -- without the fix,
        # it's left reported as "running" with an already-dead PID forever.
        instant_crash = "import sys\nprint('run_dir=%s' % sys.argv[1], flush=True)\nsys.exit(1)\n"
        script = _write_script(tmp_path, "fake_instant_crash.py", instant_crash)
        run_dir_arg = str(tmp_path / "training_runs" / "fake_run5")
        spec = JobSpec(id="crashes_instantly", cwd=str(tmp_path),
                        command=[sys.executable, str(script), run_dir_arg])
        orch = Orchestrator(jobs=[spec], log_dir=tmp_path / "logs", status_path=tmp_path / "status.json")
        job = orch.jobs[0]

        orch._start_job(job)
        assert job.status == "running"
        job.proc.wait(timeout=5)  # let the real process actually finish exiting on its own
        assert job.proc.poll() is not None  # confirm it's genuinely dead already

        # _handle_shutdown's own "running" snapshot will exclude this job
        # (poll() is no longer None), so its ONLY path to a correct final
        # status is the reconciliation pass.
        orch._handle_shutdown(signum=15, frame=None)

        assert job.status == "stopped"
        assert job.status != "running"

    def test_shutdown_does_not_mislabel_a_graceful_exit0_as_completed(self, tmp_path):
        # Regression test for a real bug found on a live run: train_v8.py and
        # train_phase3.py both catch SIGTERM, save a checkpoint, and exit 0
        # (a *successful* shutdown from the script's own point of view). The
        # orchestrator used to check `rc == 0` before checking `self._stopping`,
        # so a job Ctrl-C'd partway through -- at episode 2,015 of a planned
        # 6,000, in the incident that found this -- got permanently logged as
        # "completed successfully", identical to actually finishing all 6,000.
        # The interrupted run's own checkpoint (krishna_latest.pth / last_state.json,
        # not krishna_final.pth) was the only place the truth was visible.
        graceful_on_sigterm = """
import signal, sys, time
def handler(signum, frame):
    sys.exit(0)  # simulates a real checkpoint-and-exit -- exit CODE is 0
signal.signal(signal.SIGTERM, handler)
print("run_dir=%s" % sys.argv[1], flush=True)
time.sleep(30)  # would exit 1 (unhandled) if it ever got here uninterrupted
"""
        script = _write_script(tmp_path, "fake_graceful_sigterm.py", graceful_on_sigterm)
        run_dir_arg = str(tmp_path / "training_runs" / "fake_run6")
        spec = JobSpec(id="graceful_job", cwd=str(tmp_path),
                        command=[sys.executable, str(script), run_dir_arg])
        orch = Orchestrator(jobs=[spec], log_dir=tmp_path / "logs", status_path=tmp_path / "status.json")
        job = orch.jobs[0]

        orch._start_job(job)
        assert job.status == "running"
        # Give it a moment to install the signal handler and print run_dir.
        for _ in range(50):
            if job.log_path.exists() and "run_dir=" in job.log_path.read_text():
                break
            time.sleep(0.05)

        orch._handle_shutdown(signum=15, frame=None)  # sends SIGTERM, waits, reconciles

        assert job.proc.returncode == 0  # confirm this really was a graceful exit-0, not a crash
        assert job.status == "stopped"
        assert job.status != "completed"


# ---------------------------------------------------------------------------
# Config hot-reload on restart -- regression test for the v8_ippo incident
# ---------------------------------------------------------------------------

_FAKE_HOT_RELOAD_SCRIPT = """
import sys
print("[start] hot-reload fake job  run_dir={}".format(sys.argv[1]), flush=True)
print("  ep 1/1  reloaded config was used", flush=True)
sys.exit(0)
"""


class TestConfigHotReloadOnRestart:
    def test_restart_picks_up_fixed_cwd_from_edited_config_file(self, tmp_path, capsys):
        # Regression test for exactly tonight's incident: v8_ippo's `cwd` in
        # orchestrator/q6_jobs.json was wrong (pointed at a branch worktree
        # instead of the main repo). The orchestrator process was already
        # running, with the broken JobSpec loaded into memory, by the time
        # the config file was fixed on disk -- so every restart kept
        # replaying the same broken cwd. It burned through all max_restarts
        # in ~5 minutes and then sat silently "failed" for the rest of the
        # ~45 hour run, unnoticed, while a second job in the same process
        # ran fine to completion.
        #
        # Here: cwd starts broken (the fake script doesn't exist there), so
        # the first launch crashes almost immediately -- same failure mode
        # as python3 exiting nonzero because it can't find the script file.
        # We then edit the job's entry in the config file on disk, exactly
        # like fixing q6_jobs.json while the orchestrator keeps running, and
        # assert the NEXT restart actually launches with the corrected cwd
        # instead of replaying the stale one from construction time.
        bad_dir = tmp_path / "wrong_worktree"
        good_dir = tmp_path / "correct_repo"
        bad_dir.mkdir()
        good_dir.mkdir()
        # The script only exists in the correct directory -- mirrors
        # train_v8.py existing in the main repo but not the stale worktree.
        (good_dir / "fake_job.py").write_text(_FAKE_HOT_RELOAD_SCRIPT)

        run_dir_arg = str(tmp_path / "training_runs" / "hot_reload_run")
        config_path = tmp_path / "jobs.json"
        config = {
            "poll_interval_seconds": 10,
            "status_print_interval_seconds": 30,
            "max_concurrent_jobs": None,
            "max_load_average": None,
            "jobs": [
                {
                    "id": "hot_reload_job",
                    "cwd": str(bad_dir),
                    "command": [sys.executable, "fake_job.py", run_dir_arg],
                    "max_restarts": 3,
                    "restart_backoff_seconds": 0.1,
                }
            ],
        }
        config_path.write_text(json.dumps(config))

        orch = build_orchestrator(
            load_config(config_path),
            log_dir=tmp_path / "logs",
            status_path=tmp_path / "status.json",
            config_path=config_path,
        )
        job = orch.jobs[0]

        # First launch uses the broken cwd captured in the spec at
        # construction time -- python3 can't find fake_job.py there, so the
        # process exits nonzero almost immediately.
        _poll_until(orch, job, lambda j: j.status == "pending" and j.restart_count == 1)
        assert job.spec.cwd == str(bad_dir)

        # Fix the bug on disk WHILE the orchestrator process (and this `orch`
        # object) is still alive -- exactly what happened with v8_ippo.
        config["jobs"][0]["cwd"] = str(good_dir)
        config_path.write_text(json.dumps(config))

        capsys.readouterr()  # discard output so far before checking for the reload message

        # Backoff elapses; the restart should re-read the config file and
        # launch with the corrected cwd -- not the stale one from __init__.
        _poll_until(orch, job, lambda j: j.status == "completed", timeout=10.0)

        assert job.restart_count == 1  # fixed on the very next attempt, no more crashes
        assert job.spec.cwd == str(good_dir)

        # The whole point of this fix is that config drift is made visible,
        # not just handled -- assert the reload was actually logged.
        out = capsys.readouterr().out
        assert "config changed on reload" in out
        assert str(bad_dir) in out
        assert str(good_dir) in out

    def test_missing_config_file_falls_back_to_in_memory_spec(self, tmp_path, capsys):
        # If the config file has since been deleted/moved, a restart must
        # not crash the orchestrator or drop the job -- it should fall back
        # to whatever spec is already in memory, loudly.
        always_crash = "import sys\nprint('run_dir=%s' % sys.argv[1], flush=True)\nsys.exit(1)\n"
        script = _write_script(tmp_path, "fake_always_crash2.py", always_crash)
        run_dir_arg = str(tmp_path / "training_runs" / "fake_run_missing_cfg")
        config_path = tmp_path / "gone.json"  # deliberately never written

        spec = JobSpec(id="no_config_file", cwd=str(tmp_path),
                        command=[sys.executable, str(script), run_dir_arg],
                        restart_backoff_seconds=0.05, max_restarts=2)
        orch = Orchestrator(jobs=[spec], log_dir=tmp_path / "logs", status_path=tmp_path / "status.json",
                             config_path=config_path)
        job = orch.jobs[0]

        _poll_until(orch, job, lambda j: j.status == "pending" and j.restart_count == 1)
        capsys.readouterr()

        _poll_until(orch, job, lambda j: j.status == "pending" and j.restart_count == 2, timeout=10.0)

        out = capsys.readouterr().out
        assert "falling back to the in-memory spec" in out
        # Spec is untouched -- still the original command, job keeps retrying.
        assert job.spec.command == [sys.executable, str(script), run_dir_arg]
