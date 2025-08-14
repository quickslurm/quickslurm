from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("quickslurm")  # must match [project].name in pyproject
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"


from .data import (
    CommandResult,
    SubmitResult,
)

from .utils import (
    SlurmParseError,
    SlurmError,
    SlurmCommandError,
    # _env_with,
    # _build_flag_kv,
    # _parse_job_id,
    # _default_log_path,
    # _get_or_create_default_logger,
    # _slurm_wait,
    default_gpu_options
)

# Re-export public API from the implementation module

from .quickslurm import (
    Slurm,
    # SlurmError,
    # SlurmCommandError,
    # CommandResult,
    # SubmitResult,
)

from . import quickslurm as quickslurm  # re-export submodule

__all__ = [
    "Slurm",
    "SlurmError",
    "SlurmCommandError",
    "CommandResult",
    "SubmitResult",
    "default_gpu_options",
    "__version__",
    "quickslurm"
]
