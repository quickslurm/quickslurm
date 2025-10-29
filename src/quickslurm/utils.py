import logging
import os
import re
from logging import Logger
from logging.handlers import RotatingFileHandler
from pathlib import Path
from subprocess import run, CalledProcessError
from time import sleep
from typing import Mapping, Optional, Union, List, Dict

from .data import CommandResult


# ----------------- Exceptions -----------------

class SlurmError(RuntimeError):
    pass


class SlurmCommandError(SlurmError):
    def __init__(self, message: str, result: CommandResult):
        super().__init__(message)
        self.result = result


class SlurmParseError(SlurmError):
    pass


# ----------------- Helpers -----------------
_JOB_ID_RE = re.compile(r"Submitted batch job\s+(\d+)")


def _env_with(overrides: Optional[Mapping[str, str]] = None) -> Mapping[str, str]:
    env = os.environ.copy()
    if overrides:
        env.update({str(k): str(v) for k, v in overrides.items()})
    return env


def _build_flag_kv(options: Mapping[str, Union[str, int, float, bool]]) -> List[str]:
    """
    Map {"job-name": "x", "exclusive": True} -> ["--job-name=x", "--exclusive"]
    """
    args: List[str] = []
    for k, v in options.items():
        key = str(k).strip().replace('_', '-')
        if isinstance(v, bool):
            if v:
                args.append(f"--{key}")
        else:
            args.append(f"--{key}={v}")
    return args


def _parse_job_id(sbatch_stdout: str) -> int:
    m = _JOB_ID_RE.search(sbatch_stdout)
    if not m:
        raise SlurmParseError(f"Could not parse job id from sbatch output:\n{sbatch_stdout}")
    return int(m.group(1))


def _default_log_path() -> Path:
    """Choose CWD/quickslurm.log, falling back to /tmp/quickslurm.log."""
    cwd_path = Path.cwd() / "quickslurm.log"
    try:
        # Touch to ensure we have perms; keep the file for reuse
        cwd_path.touch(exist_ok=True)
        return cwd_path
    except (OSError, PermissionError):
        tmp_path = Path("/tmp/quickslurm.log")
        tmp_path.touch(exist_ok=True)
        return tmp_path


def _get_or_create_default_logger() -> Logger:
    """
    Return a module-level logger configured once with:
      - Rotating file handler (5MB, 3 backups)
      - Stream handler (stderr)
    Prevents duplicate handlers if Slurm() is constructed multiple times.
    """
    logger = logging.getLogger("quickslurm")
    if getattr(logger, "_slurm_logger_configured", False):
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # keep logs from duplicating up the root chain

    log_path = _default_log_path()
    fh = RotatingFileHandler(str(log_path), maxBytes=5 * 1024 * 1024, backupCount=3)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger._slurm_logger_configured = True  # type: ignore[attr-defined]
    logger.info(f"[Slurm] Logging initialized at {log_path}")
    return logger


def _sacct_cmd(x, ops='JobID,State,ExitCode'):
    return run(
        f'sacct -j {x} --format={ops} --noheader --parsable2',
        timeout=10, shell=True, capture_output=True, text=True
    )


sacct_format = lambda x: x.strip().split('\n')[0].strip().split('|')


def _slurm_wait(job_id) -> None:
    print(f'waiting for slurm job {job_id} to complete')

    if job_id == 0:
        return

    while True:
        try:
            # sacct -j job_id --format=JobID,State --noheader --parsable2
            res = _sacct_cmd(job_id, 'State')

            state = sacct_format(res.stdout)
            if not state:
                sleep(10)
                continue

            job_state = state[0]
            if job_state in ['COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY']:
                print(f'Job {job_id} finished with state: {job_state}')
                return

            sleep(10)

        except CalledProcessError as e:
            print(f'Failed to check slurm status: {e}')
            if res:
                print(f'sacct output: {res.stdout}, {res.stderr}')
            sleep(10)
        except IndexError as e:
            print(f'Failed to parse sacct output: {e}')
            if res:
                print(f'sacct output: {res.stdout}, {res.stderr}')
            sleep(10)


def _parse_result(job_id):
    try:
        res = _sacct_cmd(job_id, ops='State,ExitCode,StdOut,StdErr')
        return sacct_format(res.stdout)[:4]
    except Exception as e:
        print(f'Warning: Failed to check exit status of job! {e}')
        return "UNKNOWN", 0, 'UNKNOWN', 'UNKNOWN'


# ----------------- Convenience preset -----------------

def default_gpu_options(
        gpus: int = 1,
        *,
        partition: Optional[str] = None,
        time: str = "01:00:00",
        mem: Optional[str] = None,
        cpus_per_task: Optional[int] = None,
        gres_type: str = "gpu",
) -> Dict[str, Union[str, int]]:
    """
    Quick helper to build common GPU sbatch options.
    """
    opts: Dict[str, Union[str, int]] = {"time": time, "gres": f"{gres_type}:{gpus}"}
    if partition:
        opts["partition"] = partition
    if mem:
        opts["mem"] = mem
    if cpus_per_task:
        opts["cpus-per-task"] = cpus_per_task
    return opts
