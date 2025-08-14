from dataclasses import dataclass
from typing import List


# ----------------- Data structures -----------------

@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    args: List[str]


@dataclass(frozen=True)
class SubmitResult:
    job_id: int
    state: str
    returncode: int
    stdout: str
    stderr: str
    args: List[str]