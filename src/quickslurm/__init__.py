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
    default_gpu_options
)

# Re-export public API from the implementation module

from .quickslurm import (
    Slurm,
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
