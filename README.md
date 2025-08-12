# quickslurm

A lightweight Python wrapper around Slurm for:

- Submitting batch jobs (sbatch)
- Running commands (srun)
- Cancelling jobs (scancel)

It focuses on safe subprocess handling, simple ergonomics, and sensible defaults.

## Features

- sbatch and srun helpers with consistent argument handling
- Optional built-in logging (file + stderr)
- Pass-through environment variables (base and per-call overrides)
- Convenience for inline sbatch scripts
- Optional wait for job completion after sbatch submission

---

## Installation

- Requires Python 3.8+
- Requires Slurm 20.02+ be installed on the system
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
submit = slurm.submit_batch(
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

---

## API

### Constructor: Slurm

```python
from quickslurm import Slurm

slurm = Slurm(
    sbatch_path="sbatch",
    srun_path="srun",
    default_timeout=None,
    base_env=None,
    enable_logging=False,
)
```

Args:

- sbatch_path: Path to the sbatch binary. Default: "sbatch".
- srun_path: Path to the srun binary. Default: "srun".
- default_timeout: Default timeout (seconds) for all subprocess calls if a per-call timeout isn’t supplied. Default:
  None (no timeout).
- base_env: Base environment variables applied to every call. Dict[str, str] or None.
- enable_logging:
    - True: enable module logger (rotating file + stderr)
    - False: no logging
    - logging.Logger: use a provided logger instance

Notes:

- base_env and per-call env are merged; per-call overrides win.
- Logging is module-local and won’t spam your root logger.

---

### submit_batch

```python
submit = slurm.submit_batch(
    script_path="path/to/script.sh",
    sbatch_options=None,
    script_args=None,
    extra_env=None,
    timeout=None,
    wait=False,
)
```

Submits an existing batch script via sbatch.

Args:

- script_path: Path to a shell script.
- sbatch_options: Mapping of sbatch options. Keys use either dashes or underscores; underscores are converted to dashes.
  Values:
    - bool True => flag without value (e.g., {"exclusive": True} -> --exclusive)
    - other types => "--key=value"
- script_args: Additional arguments appended to the script invocation.
- extra_env: Extra env vars merged on top of base_env for this call.
- timeout: Per-call timeout (seconds). Overrides default_timeout if set.
- wait: If True, waits for job completion by polling via sacct. Prints state transitions.

Returns SubmitResult:

- job_id: int
- stdout: str
- stderr: str
- args: list[str] (the executed command)

Raises:

- SlurmError if sbatch is missing
- SlurmCommandError on non-zero exit (includes result)

Example:

```python
submit = slurm.submit_batch(
    script_path="train.sh",
    sbatch_options={
        "job-name": "modelA",
        "time": "02:00:00",
        "partition": "gpu",
        "gres": "gpu:1",
        "cpus_per_task": 4,  # underscores are fine; becomes --cpus-per-task=4
        "exclusive": True,  # becomes --exclusive
    },
    script_args=["--epochs", "20", "--lr", "3e-4"],
    wait=True,  # will poll until job completes
)
print(submit.job_id)
```

Sample console output when wait=True:

```
waiting for slurm job 123457 to complete
Job 123457 finished with state: COMPLETED
```

---

### submit_inline

```python
submit = slurm.submit_inline(
    command=["python", "train.py", "--epochs", "5"],
    sbatch_options=None,
    shebang="#!/bin/bash -l",
    workdir=None,
    extra_env=None,
    timeout=None,
    wait=False,
)
```

Generates a temporary script that runs command and submits via sbatch.

Args:

- command: Sequence of program/args (no shell parsing required). Safely quoted.
- sbatch_options: Same behavior as submit_batch.
- shebang: Script header. Default: "#!/bin/bash -l".
- workdir: If provided, inserts a cd to this directory before command.
- extra_env, timeout, wait: As in submit_batch.

Returns SubmitResult (same shape as submit_batch).

Example:

```python
submit = slurm.submit_inline(
    command=["python", "inference.py", "--input", "data/images", "--out", "preds/"],
    sbatch_options={"time": "00:20:00", "partition": "short", "job-name": "infer"},
)
print("Submitted:", submit.job_id)
```

---

### run

```python
result = slurm.run(
    ["python", "-c", "print('hello from srun')"],
    srun_options=None,
    extra_env=None,
    timeout=None,
    check=True,
    wait=False,
)
```

Runs a command via srun (non-interactive). Useful inside an allocation or for short tasks.

Args:

- command: Sequence[str] for srun to execute.
- srun_options: Mapping converted to flags like sbatch_options.
- extra_env: Per-call env overrides.
- timeout: Per-call timeout (seconds).
- check: If True, raise SlurmCommandError on non-zero exit.
- wait: Ignored for srun; included for API symmetry (no queue wait performed).

Returns CommandResult:

- returncode: int
- stdout: str
- stderr: str
- args: list[str] (the executed command)

Example:

```python
res = slurm.run(["hostname"], srun_options={"ntasks": 1})
print(res.stdout.strip())
```

---

### cancel

```python
res = slurm.cancel(
    job_id,
    extra_env=None,
    timeout=None,
    scancel_path="scancel",
    check=True,
)
```

Cancels a job by ID via scancel.

Args:

- job_id: int or str
- extra_env, timeout: As above
- scancel_path: Path to scancel binary
- check: Raise on non-zero exit if True

Returns CommandResult.

Example:

```python
slurm.cancel(12345678)
```

---

## Option mapping details

When passing sbatch_options or srun_options:

- Keys are normalized: underscores become dashes. Examples:
    - {"cpus_per_task": 4} => --cpus-per-task=4
    - {"job-name": "x"} => --job-name=x
- Boolean True becomes a flag with no value:
    - {"exclusive": True} => --exclusive
- Boolean False is omitted.

---

## Handling environment

- The constructor’s base_env is applied to every call.
- Per-call extra_env is merged on top (stringified keys/values).
- The final environment is inherited from the process environment plus these overrides.

Example:

```python
slurm = Slurm(base_env={"WANDB_MODE": "offline"})
slurm.submit_inline(
    command=["python", "train.py"],
    extra_env={"CUDA_VISIBLE_DEVICES": "0"},
)
```

---

## Logging

- enable_logging=True will write rotating logs to quickslurm.log in the current working directory (fallback to
  /tmp/quickslurm.log) and also to stderr.
- You can pass your own logging.Logger if you want custom routing/formatting.

---

## Complete examples

1) Submit an existing script with arguments, wait for completion

```python
from quickslurm import Slurm

slurm = Slurm(enable_logging=True)
submit = slurm.submit_batch(
    script_path="scripts/train.sh",
    sbatch_options={
        "job-name": "resnet50",
        "partition": "gpu",
        "time": "04:00:00",
        "gres": "gpu:1",
        "cpus_per_task": 8,
        "mem": "24G",
        "output": "slurm-%j.out",
    },
    script_args=["--epochs", "50", "--batch-size", "128"],
    wait=True,
)
print("Job:", submit.job_id)
```

Possible slurm-%j.out snippet:

```
Epoch 1/50: loss=1.78 acc=0.45
...
Epoch 50/50: loss=0.67 acc=0.78
```

2) Submit an inline job with a working directory and environment

```python
from quickslurm import Slurm

slurm = Slurm(base_env={"PYTHONUNBUFFERED": "1"})
submit = slurm.submit_inline(
    command=["python", "tools/evaluate.py", "--ckpt", "ckpts/model.pt"],
    workdir="project",
    sbatch_options={"time": "00:30:00", "partition": "short", "job-name": "eval"},
    extra_env={"CUDA_VISIBLE_DEVICES": "0"},
)
print("Eval job:", submit.job_id)
```

3) Use srun for quick diagnostics within an allocation

```python
from quickslurm import Slurm, SlurmCommandError

slurm = Slurm()
try:
    res = slurm.run(["nvidia-smi"], srun_options={"ntasks": 1, "time": "00:02:00"})
    print(res.stdout)
except SlurmCommandError as e:
    print("nvidia-smi failed with code", e.result.returncode)
    print("stderr:", e.result.stderr)
```

4) Cancel a job

```python
from quickslurm import Slurm

slurm = Slurm()
slurm.cancel(12345678)
```

---

## Error handling

- SlurmError: Base class for Slurm-related failures (e.g., missing binaries).
- SlurmCommandError: Raised when a Slurm command exits non-zero (contains the CommandResult as .result, including
  stdout/stderr/args/returncode).

Example:

```python
from quickslurm import Slurm, SlurmCommandError

slurm = Slurm()
try:
    slurm.run(["bash", "-c", "exit 2"])
except SlurmCommandError as e:
    print("Failed:", e)
    print("Return code:", e.result.returncode)
    print("STDERR:", e.result.stderr)
```

---

## Notes and tips

- wait=True for submit_* will poll via sacct. Ensure sacct is available and your account can query job states.
- script_args for submit_batch are appended after the script path, in order.
- For GPU jobs, a typical sbatch option is gres=gpu:N and an appropriate partition; you can also set cpus-per-task and
  mem as needed.

---

## Minimal reference

Return types you’ll commonly use:

SubmitResult:

- job_id: int
- stdout: str
- stderr: str
- args: list[str]

CommandResult:

- returncode: int
- stdout: str
- stderr: str
- args: list[str]

---

## License
See `LICENSE` for more information.