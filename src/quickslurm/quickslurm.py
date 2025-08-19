"""

A lightweight Slurm wrapper for submitting batch jobs (sbatch) and running tasks (srun)
with robust subprocess handling.

Usage:
    # General initialization:
    ```
    from quickslurm import Slurm, SlurmError

    slurm = Slurm()
    #or
    slurm = Slurm(sbatch_path="/usr/bin/sbatch", srun_path="/usr/bin/srun")
    ```

    # Submit an existing script:
    ```
    sub = slurm.sbatch(
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
    ```

    # Submit an inline command (auto-generates a temp script)
    ```
    sub2 = slurm.submit_inline(
        command=["python", "train.py", "--epochs", "5"],
        sbatch_options={"time": "00:10:00", "job-name": "quick-train"})

    # Run something interactively via srun (within an allocation or for short tasks)
    res = slurm.srun(["hostname"], srun_options={"ntasks": 1})
    print(res.stdout)
    ```
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
    _env_with, _parse_job_id, _slurm_wait, _parse_result
)


# ----------------- Main class -----------------

class Slurm:
    def __init__(
            self,
            sbatch_path: str = "sbatch",
            srun_path: str = "srun",
            default_timeout: Optional[float] = None,
            base_env: Optional[Mapping[str, str]] = None,
            enable_logging: Union[bool, Logger] = False,
            config_file_path: str = 'quickslurm.cfg',
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
            config_file_path: quick slurm config file. "quickslurm.cfg" by default. (More in README)
                - this file can contain all of the key value pair 
        """
        self.sbatch_path = sbatch_path
        self.srun_path = srun_path
        self.default_timeout = default_timeout
        self.base_env = _env_with(base_env)

        if isinstance(enable_logging, logging.Logger):
            self.logger = enable_logging  # logging.getLogger(f"{enable_logging.name}.quickslurm")
        elif enable_logging:
            self.logger = _get_or_create_default_logger()
        else:
            self.logger = logging.getLogger("quickslurm.null")
            self.logger.handlers.clear()
            self.logger.addHandler(logging.NullHandler())
            self.logger.propagate = False

    # ---------- Public API ----------

    def sbatch(
            self,
            script_path: Union[str, Path],
            script_args: Optional[Sequence[str]] = None,
            sbatch_options: Optional[Mapping[str, Union[str, int, float, bool]]] = None,
            extra_env: Optional[Mapping[str, str]] = None,
            timeout: Optional[float] = None,
            check: bool = True,
            wait: bool = True,
    ) -> SubmitResult:
        """
        Submit an existing Slurm batch script using sbatch.

        This method builds and executes an sbatch command to submit the provided
        script file. Additional sbatch flags can be specified via a mapping
        (converted into --key=value or --flag style options), and positional
        arguments for the script can be appended. On success, the parsed Slurm
        job ID and raw command I/O are returned.

        If wait is True, the call blocks until the submitted job reaches a
        terminal state, and the resulting status is reflected in the returned
        object.

        If wait is False, the method will return a dummy `SubmitResult` object with
        the std_out and std_err of the calling subprocess, and the state will be 'UNKNOWN'.

        Args:
            script_path: Path to the batch script to submit.
            script_args: Optional sequence of additional arguments passed to the script
                after sbatch options.
            sbatch_options: Optional mapping of sbatch options (e.g., {"time": "00:10:00",
                "partition": "short", "cpus-per-task": 4}). Boolean values are treated
                as flags (True -> present, False -> omitted).
            extra_env: Optional environment variables to layer on top of the base
                environment for this invocation.
            timeout: Optional timeout in seconds for the underlying subprocess call.
                Falls back to the instance default_timeout if not provided.
            check: If True (default), raises SlurmCommandError on non-zero exit codes for calling subprocess.
            wait: If True, wait for the submitted job to finish before returning.

        Returns:
            SubmitResult: An object containing:
                - job_id: The parsed Slurm job ID (int or str).
                - state: The state of the job if wait=True, otherwise 'UNKNOWN'.
                - returncode: The exit code of the sbatch command if wait = True, otherwise subprocess return code.
                - stdout: Captured stdout from sbatch.
                - stderr: Captured stderr from sbatch.
                - args: The full command-line argument list that was executed.

        Raises:
            SlurmError: If the sbatch executable is not found or Slurm is not available.
            SlurmCommandError: If the sbatch command fails or times out, or if the job
                completes in an error state when wait=True.

        Example:
            ```
            slurm = Slurm()
            res = slurm.sbatch(
                 script_path="train.sh",
                 script_args=["--epochs", "5"],
                 sbatch_options={"job-name": "trainA", "time": "00:30:00"},
            )
            print(res.job_id)
            ```
        """

        if script_args is None:
            script_args = []

        cmd = [self.sbatch_path]
        if sbatch_options:
            cmd += _build_flag_kv(sbatch_options)
        cmd.append(str(script_path))
        cmd += [str(a) for a in script_args]

        return self._run(cmd, env=_env_with(extra_env), timeout=timeout, check=check, wait=wait)

    def submit_inline(
            self,
            command: Sequence[str],
            sbatch_options: Optional[Mapping[str, Union[str, int, float, bool]]] = None,
            shebang: str = "#!/bin/bash -l",
            workdir: Optional[Union[str, Path]] = None,
            extra_env: Optional[Mapping[str, str]] = None,
            timeout: Optional[float] = None,
            check: bool = True,
            wait: bool = True,
    ) -> SubmitResult:
        """
        Submit an inline command by generating a temporary sbatch script.

        This method creates a temporary shell script that:
        - Starts with the provided shebang.
        - Enables strict bash options (set -euo pipefail).
        - Optionally changes to the specified working directory.
        - Executes the given command (safely quoted).

        The temporary script is made executable, submitted via sbatch with the
        provided options, and then cleaned up. On success, the parsed job ID
        and captured I/O from the sbatch invocation are returned. If wait is
        True, the call blocks until the job reaches a terminal state.

        Args:
            command: The program and its arguments to run (e.g., ["python", "train.py", "--epochs", "5"]).
            sbatch_options: Optional mapping of sbatch options (e.g., {"time": "00:10:00", "partition": "short"}).
                Boolean values are treated as flags (True -> present, False -> omitted).
            shebang: Script interpreter line to place at the top of the generated script.
                Defaults to "#!/bin/bash -l".
            workdir: If provided, inserts a `cd` to this directory before running the command.
            extra_env: Environment variables layered on top of the instance base environment for this submission.
            timeout: Timeout in seconds for the underlying subprocess call. Falls back to the instance
                default if not provided.
            check: If True (default), raises SlurmCommandError on non-zero exit codes for calling subprocess.
            wait: If True, wait for the submitted job to finish before returning.

        Returns:
            SubmitResult: Contains:
                - job_id: The parsed Slurm job ID.
                - stdout: Captured stdout from sbatch.
                - stderr: Captured stderr from sbatch.
                - args: The executed command-line arguments.

        Raises:
            SlurmError: If the sbatch executable is not found or Slurm is unavailable.
            SlurmCommandError: If submission fails, times out, or the job completes in an error
                state when wait=True.

        Example:
            ```
            slurm = Slurm()
            sub = slurm.submit_inline(
                command=["python", "train.py", "--epochs", "5"],
                sbatch_options={"time": "00:10:00", "job-name": "quick-train"},
            )
            print(sub.job_id)
            ```
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
            return self.sbatch(
                script_path=tf_path,
                script_args=None,
                sbatch_options=sbatch_options,
                extra_env=extra_env,
                timeout=timeout,
                check=check,
                wait=wait
            )
        finally:
            try:
                os.unlink(tf_path)
            except OSError:
                pass

    def srun(
            self,
            command: Sequence[str],
            srun_options: Optional[Mapping[str, Union[str, int, float, bool]]] = None,
            extra_env: Optional[Mapping[str, str]] = None,
            timeout: Optional[float] = None,
            check: bool = True,
            wait: bool = True,
    ) -> SubmitResult:
        """
        Run a command via srun (non-interactive).

        Builds and executes an srun command using the provided options and command
        arguments. Environment variables can be layered on top of the instance base
        environment. By default, a non-zero exit code raises SlurmCommandError unless
        check is set to False.

        If wait is True, the method attempts to wait for the underlying Slurm job to
        finish when a job ID can be determined; otherwise, it returns immediately after
        the srun process completes.

        Args:
            command: Sequence of the program and its arguments to run via srun.
            srun_options: Optional mapping of srun options. Keys are normalized
                (underscores -> dashes) and values are converted to flags:
                - bool True => "--key" (flag present)
                - bool False => omitted
                - other types => "--key=value"
            extra_env: Optional environment variables to merge on top of the instance
                base environment for this call.
            timeout: Optional timeout in seconds for the subprocess call. Falls back to
                the instance default if not provided.
            check: If True (default), raise SlurmCommandError when the command exits
                with a non-zero status.
            wait: If True, attempt to wait for the job to reach a terminal state when a
                job ID can be parsed.

        Returns:
            CommandResult: Contains:
                - returncode: Process exit code.
                - stdout: Captured standard output.
                - stderr: Captured standard error.
                - args: The executed command-line arguments.

        Raises:
            SlurmError: If the srun executable is not found or Slurm is unavailable.
            SlurmCommandError: If the command fails, times out, or completes in an
                error state when check=True (or when wait=True and a terminal error is detected).
        """
        cmd = [self.srun_path]
        if srun_options:
            cmd += _build_flag_kv(srun_options)
        cmd += [str(c) for c in command]

        return self._run(cmd, env=_env_with(extra_env), timeout=timeout, check=check, wait=wait, batch=False)

    def scancel(
            self,
            job_id: Union[int, str],
            extra_env: Optional[Mapping[str, str]] = None,
            timeout: Optional[float] = None,
            scancel_path: str = "scancel",
            check: bool = True,
    ) -> SubmitResult:
        """
        Cancel a Slurm job by ID using scancel.

        Constructs and runs an scancel command to cancel the specified job. Environment
        overrides and a timeout can be supplied. By default, a non-zero exit code raises
        SlurmCommandError unless check is set to False.

        Args:
            job_id: The Slurm job identifier to cancel.
            extra_env: Optional environment variables merged on top of the instance
                base environment for this call.
            timeout: Optional timeout in seconds for the subprocess call. Falls back to
                the instance default if not provided.
            scancel_path: Path to the scancel executable. Defaults to "scancel".
            check: If True (default), raise SlurmCommandError when scancel exits
                with a non-zero status.

        Returns:
            CommandResult: Contains:
                - returncode: Process exit code.
                - stdout: Captured standard output.
                - stderr: Captured standard error.
                - args: The executed command-line arguments.

        Raises:
            SlurmError: If the scancel executable is not found or Slurm is unavailable.
            SlurmCommandError: If the scancel command fails or times out when check=True.
        """
        cmd = [scancel_path, str(job_id)]
        return self._run(cmd, env=_env_with(extra_env), timeout=timeout, check=check, batch=False, wait=False)

    # ---------- Internal runner ----------

    def _run(
            self,
            args: Sequence[str],
            *,
            env: Optional[Mapping[str, str]] = None,
            timeout: Optional[float] = None,
            check: bool = True,
            input_text: Optional[str] = None,
            wait: bool = True,
            batch: bool = True,
    ) -> SubmitResult:
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

        if cp.returncode != 0:
            if check and batch:
                raise SlurmCommandError(
                    f"Command failed (exit {cp.returncode}): {args}\n{cp.stderr.strip()}",
                    CommandResult(cp.returncode, cp.stdout, cp.stderr, list(map(str, args))),
                )
            else:
                return SubmitResult(00000, 'UNKNOWN', cp.returncode, cp.stdout, cp.stderr, list(map(str, args)))

        # if sbatch, we need to parse the job id and optionally wait
        if batch:
            job_id = _parse_job_id(cp.stdout)

            if wait:
                _slurm_wait(job_id)
                state, exit_code, std_out, std_err = _parse_result(job_id)
                self.logger.debug(f"[Slurm] Return code: {exit_code}")
                return SubmitResult(job_id, state, exit_code, std_out, std_err, list(map(str, args)))

            else:
                self.logger.debug(f"[Subprocess] Return code: {cp.returncode}", )
                return SubmitResult(job_id, 'UNKNOWN', cp.returncode, cp.stdout, cp.stderr, list(map(str, args)))

        # if srun return the output directly
        else:
            self.logger.debug(f"[srun] Return code: {cp.returncode}", )
            return SubmitResult(0, 'COMPLETE' if cp.returncode == 0 else 'FAILED', cp.returncode, cp.stdout, cp.stderr, list(map(str, args)))
