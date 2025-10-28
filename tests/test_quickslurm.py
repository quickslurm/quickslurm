import os
import subprocess

import pytest
from quickslurm import Slurm, SlurmCommandError, SlurmError
from quickslurm.data import SubmitResult
from quickslurm.utils import _parse_job_id


class FakeCompletedProcess:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_parse_job_id_ok():
    assert _parse_job_id("Submitted batch job 12345\n") == 12345


def test_parse_job_id_fail():
    with pytest.raises(Exception):
        _parse_job_id("weird output")


def test_sbatch_success_wait_true(monkeypatch):
    calls = {"args": None}

    def fake_run(args, **kwargs):
        assert "sbatch" in args[0]
        calls["args"] = args
        return FakeCompletedProcess(returncode=0, stdout="Submitted batch job 42\n", stderr="")

    # Patch where the names are looked up: inside quickslurm.quickslurm module
    monkeypatch.setattr("quickslurm.quickslurm._slurm_wait", lambda job_id: None)
    monkeypatch.setattr("quickslurm.quickslurm._parse_result", lambda job_id: ("COMPLETED", 0, "job ok", ""))

    monkeypatch.setattr(subprocess, "run", fake_run)

    slurm = Slurm(logger=False)
    res = slurm.sbatch(script_path="script.sh", wait=True)

    assert isinstance(res, SubmitResult)
    assert res.job_id == 42
    assert res.state == "COMPLETED"
    assert res.returncode == 0
    assert res.stdout == "job ok"
    assert calls["args"][0].endswith("sbatch")
    assert "script.sh" in calls["args"]


def test_sbatch_failure_raises(monkeypatch):
    def fake_run(args, **kwargs):
        return FakeCompletedProcess(returncode=2, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run)

    slurm = Slurm(logger=False)
    with pytest.raises(SlurmCommandError) as exc:
        slurm.sbatch(script_path="oops.sh", check=True, wait=False)
    assert "boom" in str(exc.value)


def test_sbatch_check_false_returns_result(monkeypatch):
    def fake_run(args, **kwargs):
        return FakeCompletedProcess(returncode=5, stdout="stdout text", stderr="E: nope")

    monkeypatch.setattr(subprocess, "run", fake_run)

    slurm = Slurm(logger=False)
    res = slurm.sbatch(script_path="x.sh", check=False, wait=False)

    assert isinstance(res, SubmitResult)
    assert res.state == "UNKNOWN"
    assert res.returncode == 5
    assert res.job_id == 0
    assert res.stderr == "E: nope"
    assert res.stdout == "stdout text"


def test_sbatch_builds_flags(monkeypatch):
    captured = {"args": None}

    def fake_run(args, **kwargs):
        captured["args"] = args
        return FakeCompletedProcess(returncode=0, stdout="Submitted batch job 101\n", stderr="")

    monkeypatch.setattr("quickslurm.quickslurm._slurm_wait", lambda job_id: None)
    monkeypatch.setattr("quickslurm.quickslurm._parse_result", lambda job_id: ("COMPLETED", 0, "", ""))

    monkeypatch.setattr(subprocess, "run", fake_run)

    slurm = Slurm(logger=False)
    slurm.sbatch(
        script_path="train.sh",
        sbatch_options={"exclusive": True, "cpus_per_task": 4},
        wait=True,
    )

    args = " ".join(captured["args"])
    assert "--exclusive" in args
    assert "--cpus-per-task=4" in args
    assert "train.sh" in args


def test_submit_inline_calls_sbatch_and_cleans_up(monkeypatch, tmp_path):
    called = {"script_path": None}
    deleted = {"path": None}

    def fake_sbatch(self, script_path, script_args, sbatch_options, extra_env, timeout, check, wait):
        called["script_path"] = script_path
        return SubmitResult(99, "COMPLETED", 0, "ok", "", ["sbatch", script_path])

    def fake_unlink(path):
        deleted["path"] = path

    monkeypatch.setattr(Slurm, "sbatch", fake_sbatch, raising=False)
    monkeypatch.setattr(os, "unlink", fake_unlink)

    slurm = Slurm(logger=False)
    res = slurm.submit_inline(
        command=["python", "train.py", "--epochs", "1"],
        sbatch_options={"time": "00:01:00"},
        workdir=str(tmp_path),
        wait=True,
    )

    assert res.job_id == 99
    assert called["script_path"] is not None
    assert deleted["path"] == called["script_path"]


def test_srun_success_wait_true(monkeypatch):
    def fake_run(args, **kwargs):
        assert "srun" in args[0]
        return FakeCompletedProcess(returncode=0, stdout="Submitted batch job 7777\n", stderr="")

    monkeypatch.setattr("quickslurm.quickslurm._slurm_wait", lambda job_id: None)
    monkeypatch.setattr("quickslurm.quickslurm._parse_result", lambda job_id: ("COMPLETE", 0, "done", ""))

    monkeypatch.setattr(subprocess, "run", fake_run)

    slurm = Slurm(logger=False)
    res = slurm.srun(["echo", "hi"], srun_options={"ntasks": 1}, wait=True)

    assert res.job_id == 0
    assert res.state == "COMPLETE"
    # assert res.stdout == "done"


def test_scancel_success(monkeypatch):
    def fake_run(args, **kwargs):
        # scancel itself won't output a job id normally; just simulate success
        return FakeCompletedProcess(returncode=0, stdout="", stderr="")

    # Force parsing/flow inside quickslurm.quickslurm
    monkeypatch.setattr("quickslurm.quickslurm._parse_job_id", lambda s: 123)
    monkeypatch.setattr("quickslurm.quickslurm._slurm_wait", lambda job_id: None)
    monkeypatch.setattr("quickslurm.quickslurm._parse_result", lambda job_id: ("CANCELLED", 0, "", ""))

    monkeypatch.setattr(subprocess, "run", fake_run)

    slurm = Slurm(logger=False)
    res = slurm.scancel(123)

    assert res.job_id == 0
    assert res.state == "COMPLETE"
    assert res.returncode == 0


def test_run_timeout_raises(monkeypatch):
    def fake_run(args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args, timeout=1, output="out", stderr="err")

    monkeypatch.setattr(subprocess, "run", fake_run)

    slurm = Slurm(logger=False)
    with pytest.raises(SlurmCommandError) as exc:
        slurm.sbatch(script_path="train.sh", timeout=1)
    assert "timed out" in str(exc.value).lower()


def test_file_not_found_raises(monkeypatch):
    def fake_run(args, **kwargs):
        raise FileNotFoundError("no such file")

    monkeypatch.setattr(subprocess, "run", fake_run)

    slurm = Slurm(logger=False)
    with pytest.raises(SlurmError):
        slurm.sbatch(script_path="train.sh")


def test_env_merge(monkeypatch):
    seen = {"env": None}

    def fake_run(args, **kwargs):
        seen["env"] = kwargs.get("env") or {}
        return FakeCompletedProcess(returncode=0, stdout="Submitted batch job 333\n", stderr="")

    monkeypatch.setattr("quickslurm.quickslurm._slurm_wait", lambda job_id: None)
    monkeypatch.setattr("quickslurm.quickslurm._parse_result", lambda job_id: ("COMPLETED", 0, "", ""))

    monkeypatch.setattr(subprocess, "run", fake_run)

    slurm = Slurm(base_env={"BASE_ONLY": "1"}, logger=False)
    _ = slurm.sbatch(script_path="env.sh", extra_env={"PER_CALL": "yes"}, wait=True)

    assert seen["env"]["BASE_ONLY"] == "1"
    assert seen["env"]["PER_CALL"] == "yes"
