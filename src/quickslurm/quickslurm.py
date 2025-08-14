"""

A lightweight Slurm wrapper for submitting batch jobs (sbatch) and running tasks (srun)
with robust subprocess handling.

Usage:
    from quickslurm import Slurm, SlurmError

    slurm = Slurm() # or Slurm(sbatch_path="/usr/bin/sbatch", srun_path="/usr/bin/srun")

    # Submit an existing script:
    sub = slurm.submit_batch(
        script_path="train.sh",
        sbatch_options={
            "job-name": "trainA",
            "time": "00:30:00",
            "partition": "short",
            "cpus-per-task": 4,
            "mem": "8G",
            "output": "slurm-%j.out",
        },
        script_args=["--epochs", "10"])
    print(sub.job_id)

    # Submit an inline command (auto-generates a temp script)
    sub2 = slurm.submit_inline(
        command=["python", "train.py", "--epochs", "5"],
        sbatch_options={"time": "00:10:00", "job-name": "quick-train"})

    # Run something interactively via srun (within an allocation or for short tasks)
    res = slurm.run(["hostname"], srun_options={"ntasks": 1})
    print(res.stdout)
"""

import logging
import os
import shlex
import subprocess
import tempfile
from logging import Logger
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

from .data import SubmitResult, CommandResult
from .utils import (
    SlurmCommandError, SlurmError,
    _build_flag_kv, _get_or_create_default_logger,
    _env_with, _parse_job_id, _slurm_wait
)


# ----------------- Main class -----------------

class Slurm:
    def __init__(
            self,
            *,
            sbatch_path: str = "sbatch",
            srun_path: str = "srun",
            default_timeout: Optional[float] = None,
            base_env: Optional[Mapping[str, str]] = None,
            enable_logging: Union[bool, Logger] = False,
    ):
        """
        Args:
            sbatch_path: Path to sbatch binary. The default is "sbatch".
            srun_path: Path to srun binary. The default is "srun".
            default_timeout: Default timeout (seconds) for subprocess runs. The default is None.
            base_env: Environment variables applied to every call. The default is None. Passed as a dict if used.
            enable_logging:
                - True: use the built-in logger (file in CWD, fallback /tmp, plus stderr).
                - False: use a NullHandler (silent).
                - logging.Logger object: use the provided logger. Note: this is only a log for the quickslurm module.
        """
        self.sbatch_path = sbatch_path
        self.srun_path = srun_path
        self.default_timeout = default_timeout
        self.base_env = _env_with(base_env)

        if isinstance(enable_logging, logging.Logger):
            self.logger = enable_logging
        elif enable_logging:
            self.logger = _get_or_create_default_logger()
        else:
            self.logger = logging.getLogger("quickslurm.null")
            self.logger.handlers.clear()
            self.logger.addHandler(logging.NullHandler())
            self.logger.propagate = False

    # ---------- Public API ----------

    def submit_batch(
            self,
            *,
            script_path: Union[str, Path],
            sbatch_options: Optional[Mapping[str, Union[str, int, float, bool]]] = None,
            script_args: Optional[Sequence[str]] = None,
            extra_env: Optional[Mapping[str, str]] = None,
            timeout: Optional[float] = None,
            wait: bool = False,
    ) -> SubmitResult:
        """
        Submit an existing batch script via sbatch.
        """
        if script_args is None:
            script_args = []

        cmd = [self.sbatch_path]
        if sbatch_options:
            cmd += _build_flag_kv(sbatch_options)
        cmd.append(str(script_path))
        cmd += [str(a) for a in script_args]

        result = self._run(cmd, env=_env_with(extra_env), timeout=timeout, wait=wait)
        job_id = _parse_job_id(result.stdout)
        return SubmitResult(job_id=job_id, stdout=result.stdout, stderr=result.stderr, args=result.args)

    def submit_inline(
            self,
            *,
            command: Sequence[str],
            sbatch_options: Optional[Mapping[str, Union[str, int, float, bool]]] = None,
            shebang: str = "#!/bin/bash -l",
            workdir: Optional[Union[str, Path]] = None,
            extra_env: Optional[Mapping[str, str]] = None,
            timeout: Optional[float] = None,
            wait: bool = False,
    ) -> SubmitResult:
        """
        Generate a temporary script containing `command` and submit it via sbatch.
        """
        cmd_line = " ".join(shlex.quote(str(c)) for c in command)
        parts = [shebang, "set -euo pipefail"]
        if workdir:
            parts.append(f"cd {shlex.quote(str(workdir))}")
        parts.append(cmd_line)
        script_text = "\n".join(parts) + "\n"

        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as tf:
            tf_path = tf.name
            tf.write(script_text)

        try:
            Path(tf_path).chmod(0o755)
            return self.submit_batch(
                script_path=tf_path,
                sbatch_options=sbatch_options,
                script_args=None,
                extra_env=extra_env,
                timeout=timeout,
                wait=wait
            )
        finally:
            try:
                os.unlink(tf_path)
            except OSError:
                pass

    def run(
            self,
            command: Sequence[str],
            *,
            srun_options: Optional[Mapping[str, Union[str, int, float, bool]]] = None,
            extra_env: Optional[Mapping[str, str]] = None,
            timeout: Optional[float] = None,
            check: bool = True,
            wait: bool = False,
    ) -> CommandResult:
        """
        Run a command via srun (non-interactive).
        """
        cmd = [self.srun_path]
        if srun_options:
            cmd += _build_flag_kv(srun_options)
        cmd += [str(c) for c in command]

        return self._run(cmd, env=_env_with(extra_env), timeout=timeout, check=check, wait=wait)

    def cancel(
            self,
            job_id: Union[int, str],
            *,
            extra_env: Optional[Mapping[str, str]] = None,
            timeout: Optional[float] = None,
            scancel_path: str = "scancel",
            check: bool = True,
    ) -> CommandResult:
        """
        Cancel a job by id using scancel.
        """
        cmd = [scancel_path, str(job_id)]
        return self._run(cmd, env=_env_with(extra_env), timeout=timeout, check=check)

    # ---------- Internal runner ----------

    def _run(
            self,
            args: Sequence[str],
            *,
            env: Optional[Mapping[str, str]] = None,
            timeout: Optional[float] = None,
            check: bool = True,
            input_text: Optional[str] = None,
            wait: bool = False
    ) -> CommandResult:
        merged_env = self.base_env.copy()
        if env:
            merged_env.update(env)

        self.logger.info("[Slurm] Running: %s", " ".join(shlex.quote(a) for a in args))

        try:
            cp = subprocess.run(
                list(map(str, args)),
                input=input_text,
                capture_output=True,
                text=True,
                env=merged_env,
                timeout=timeout if timeout is not None else self.default_timeout,
                check=False,
            )
            job_id = _parse_job_id(cp.stdout)
            if wait:
                _slurm_wait(job_id)
        except subprocess.TimeoutExpired as e:
            self.logger.error("[Slurm] Timeout after %ss", (timeout or self.default_timeout))
            raise SlurmCommandError(
                f"Command timed out after {timeout or self.default_timeout}s: {args}",
                CommandResult(-1, e.stdout or "", e.stderr or f"TimeoutExpired: {e}", list(map(str, args))),
            ) from e
        except FileNotFoundError as e:
            self.logger.error("[Slurm] Command not found: %s", args[0])
            raise SlurmError(f"Command not found: {args[0]!r}. Is Slurm on PATH?") from e
        except Exception as e:
            self.logger.error("[Slurm] Unexpected error: %s", e)
            raise

        from .utils import _check_exit
        result = CommandResult(_check_exit(job_id), cp.stdout, cp.stderr, list(map(str, args)))

        self.logger.debug("[Slurm] Return code: %s", cp.returncode)
        if cp.stdout.strip():
            self.logger.debug("[Slurm] STDOUT:\n%s", cp.stdout.strip())
        if cp.stderr.strip():
            self.logger.debug("[Slurm] STDERR:\n%s", cp.stderr.strip())
        print(result)
        if check and result.returncode > 0:
            raise SlurmCommandError(
                f"Command failed (exit {cp.returncode}): {args}\n{cp.stderr.strip()}",
                result,
            )

        return result
