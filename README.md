# quickslurm
A lightweight Python wrapper around Slurm for:

- Submitting batch jobs (sbatch)
- Running commands (srun)
- Cancelling jobs (scancel)

It focuses on safe subprocess handling, simple ergonomics, and sensible defaults.

Full documentation can be found here: [quickslurm docs](https://quickslurm.github.io/quickslurm/)

## Features

- sbatch and srun helpers with consistent argument handling
- Optional built-in logging (file and stderr)
- Pass-through environment variables (base and per-call overrides)
- Convenience for inline sbatch scripts
- Optional wait for job completion after sbatch submission

---

## Installation

- Requires Python 3.8+
- Requires Slurm 23.02+ be installed on the system
- Assumes Slurm binaries (sbatch, srun, scancel, sacct if using wait) are on PATH

```shell script
pip install quickslurm
```

---

## Quick start

```python
from quickslurm import Slurm, SlurmError

slurm = Slurm(enable_logging=True)

# Submit an existing script
submit = slurm.sbatch(
    script_path="train.sh",
    sbatch_options={
        "job-name": "trainA",
        "time": "00:30:00",
        "partition": "short",
        "cpus-per-task": 4,
        "mem": "8G",
        "output": "slurm-%j.out",
    },
    script_args=["--epochs", "10"],
)
print("Job ID:", submit.job_id)

# Submit an inline command (temp script will be generated)
submit2 = slurm.submit_inline(
    command=["python", "train.py", "--epochs", "5"],
    sbatch_options={"time": "00:10:00", "job-name": "quick-train"},
)
print("Inline Job ID:", submit2.job_id)

# Run something via srun (non-interactive)
res = slurm.run(["hostname"], srun_options={"ntasks": 1})
print("Hostname:", res.stdout.strip())

# Cancel a job
slurm.cancel(submit2.job_id)
```

Sample output:

```
Job ID: 12300008
Inline Job ID: 12300009
Hostname: testnode1
```
