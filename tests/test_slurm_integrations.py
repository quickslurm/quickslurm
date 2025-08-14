import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest


def have_slurm():
    return shutil.which("sbatch") is not None and shutil.which("srun") is not None


skip_if_no_slurm = pytest.mark.skipif(
    not have_slurm(), reason="Slurm CLI not found on PATH"
)

slurm = pytest.mark.slurm


@slurm
@skip_if_no_slurm
def test_srun_echo():
    # Simple sanity check that the daemons are up and srun can execute on a node
    out = subprocess.check_output(["srun", "-N", "1", "bash", "-lc", "echo hello"])
    assert out.decode().strip().endswith("hello")


@slurm
@skip_if_no_slurm
def test_sbatch_submit_and_collect_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Submits a trivial batch job, waits for completion by polling squeue,
    and validates the output file (slurm-<jobid>.out).
    """

    # Run inside a writable temp dir so slurm-%j.out lands here
    monkeypatch.chdir(tmp_path)

    # Create a tiny batch script
    script = tmp_path / "hello.sh"
    script.write_text(
        "#!/bin/bash\n"
        "#SBATCH -N 1\n"
        "set -euo pipefail\n"
        'echo "hello from ${HOSTNAME}"\n'
    )
    script.chmod(0o755)

    # Submit job, capture job id from --parsable
    job_id = (
        subprocess.check_output(["sbatch", "--parsable", str(script)])
        .decode()
        .strip()
    )
    assert job_id.isdigit()

    # Poll squeue until job disappears (finished or failed)
    deadline = time.time() + 120  # 2 min timeout
    while time.time() < deadline:
        # -h = no header, -o %T = print state only; empty output means not in queue
        state = (
            subprocess.check_output(["squeue", "-j", job_id, "-h", "-o", "%T"])
            .decode()
            .strip()
        )
        if not state:
            break
        # Optionally log the current state for debugging
        print(f"Job {job_id} state: {state}")
        time.sleep(1)

    # By here the job should be out of the queue; check output file
    out_file = tmp_path / f"slurm-{job_id}.out"
    assert out_file.exists(), f"Expected {out_file} to exist"
    content = out_file.read_text().strip()
    assert "hello from" in content
