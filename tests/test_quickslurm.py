import os
import subprocess
from pathlib import Path

import pytest
from quickslurm import Slurm, SlurmCommandError
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

# def test_submit_batch_success(monkeypatch, tmp_path):
#     # Fake sbatch response
#     def fake_run(args, **kwargs):
#         assert args[0].endswith("sbatch")
#         return FakeCompletedProcess(
#             returncode=0, stdout="Submitted batch job 42\n", stderr=""
#         )
#     monkeypatch.setattr(subprocess, "run", fake_run)
#
#     # log to tmp cwd fallback (exercise logger creation path)
#     cwd = tmp_path / "proj"
#     cwd.mkdir()
#     os.chdir(cwd)
#
#     slurm = Slurm(enable_logging=False)  # silent for test
#     res = slurm.sbatch(script_path="script.sh")
#     assert res.job_id == 42

# def test_srun_failure_raises(monkeypatch):
#     def fake_run(args, **kwargs):
#         assert args[0].endswith("srun")
#         return FakeCompletedProcess(returncode=1, stdout="None FAILED thing Submitted batch job 42", stderr="boom")
#     monkeypatch.setattr(subprocess, "run", fake_run)

#     slurm = Slurm(enable_logging=False)
#     with pytest.raises(SlurmCommandError) as exc:
#         slurm.run(["hostname"])
#     assert "boom" in str(exc.value)

# def test_run_success(monkeypatch):
#     def fake_run(args, **kwargs):
#         return FakeCompletedProcess(returncode=0, stdout="None COMPLETED thing Submitted batch job 42", stderr="")
#     monkeypatch.setattr(subprocess, "run", fake_run)
#
#     slurm = Slurm(enable_logging=False)
#     out = slurm.srun(["echo", "hello"])
#     assert out.stdout.strip() == "None COMPLETED thing Submitted batch job 42"
#
# def test_logging_falls_back_when_cwd_unwritable(monkeypatch, tmp_path):
#     # Simulate unwritable CWD by raising on touch()
#     class DummyPath(Path):
#         _flavour = Path(".")._flavour
#
#         def touch(self, *a, **k):
#             raise PermissionError("nope")
#
#     # Patch Path.cwd() to return DummyPath
#     monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: DummyPath(tmp_path)))
#     slurm = Slurm(enable_logging=True)
#     # If we got here, logger initialized with /tmp fallback without crashing
