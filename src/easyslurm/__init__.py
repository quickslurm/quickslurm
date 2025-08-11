from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("easyslurm")  # keep in sync with [project].name
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

from .easyslurm import (
    Slurm,
    SlurmError,
    SlurmCommandError,
    SlurmParseError,
    CommandResult,
    SubmitResult,
    default_gpu_options,
)

__all__ = [
    "Slurm",
    "SlurmError",
    "SlurmCommandError",
    "SlurmParseError",
    "CommandResult",
    "SubmitResult",
    "default_gpu_options",
    "__version__",
]
