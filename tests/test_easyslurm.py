import os
import subprocess
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest
from easyslurm import Slurm
from easyslurm import Slurm, SlurmCommandError, _parse_job_id
.Path as sw


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


def test_submit_batch_success(monkeypatch, tmp_path):
    # Fake sbatch response
    def fake_run(args, **kwargs):
        assert args[0].endswith("sbatch")
        return FakeCompletedProcess(
            returncode=0, stdout="Submitted batch job 42\n", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    # log to tmp cwd fallback (exercise logger creation path)
    cwd = tmp_path / "proj"
    cwd.mkdir()
    os.chdir(cwd)

    slurm = Slurm(enable_logging=False)  # silent for test
    res = slurm.submit_batch(script_path="script.sh")
    assert res.job_id == 42


def test_srun_failure_raises(monkeypatch):
    def fake_run(args, **kwargs):
        assert args[0].endswith("srun")
        return FakeCompletedProcess(returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run)

    slurm = Slurm(enable_logging=False)
    with pytest.raises(SlurmCommandError) as exc:
        slurm.run(["hostname"])
    assert "boom" in str(exc.value)


def test_run_success(monkeypatch):
    def fake_run(args, **kwargs):
        return FakeCompletedProcess(returncode=0, stdout="hello\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    slurm = Slurm(enable_logging=False)
    out = slurm.run(["echo", "hello"])
    assert out.stdout.strip() == "hello"


def test_logging_falls_back_when_cwd_unwritable(monkeypatch, tmp_path):
    # 1) Make the module think CWD is tmp_path
    monkeypatch.setattr(sw.Path, "cwd", classmethod(lambda cls: tmp_path))

    # 2) Force the first touch() (in CWD) to fail so fallback kicks in
    orig_touch = sw.Path.touch

    def fake_touch(self, *a, **k):
        # Fail when trying to create the CWD log file specifically
        if self == tmp_path / "slurm_wrapper.log":
            raise PermissionError("nope")
        return orig_touch(self, *a, **k)

    monkeypatch.setattr(sw.Path, "touch", fake_touch)

    # 3) Construct -> should fall back to /tmp without crashing
    slurm = Slurm(enable_logging=True)

    # 4) Assert we ended up logging to /tmp
    file_handlers = [h for h in slurm.logger.handlers if isinstance(h, RotatingFileHandler)]
    assert file_handlers, "No file handler configured"
    assert any(h.baseFilename.startswith("/tmp/") for h in file_handlers)
