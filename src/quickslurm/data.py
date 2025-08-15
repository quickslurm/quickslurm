from dataclasses import dataclass
from typing import List


# ----------------- Data structures -----------------

@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    args: List[str]

submit_result_template = """
SubmitResult(
    job_id={id}, 
    state={state}, 
    returncode={code}, 
    stdout={out}, 
    stderr={err}, 
    args={args})
)
"""

@dataclass(frozen=True)
class SubmitResult:
    job_id: int
    state: str
    returncode: int
    stdout: str
    stderr: str
    args: List[str]

    def __str__(self):
        return submit_result_template.format(
            id=self.job_id,
            state=self.state,
            code=self.returncode,
            out=self.stdout,
            err=self.stderr,
            args=self.args
        )
    
    def __eq__(self, other):
        return isinstance(other, SubmitResult) and self.job_id == other.job_id 
    
    def __call__(self):
        # easy access to process return code
        rc = None
        if isinstance(self.returncode, int):
            rc = self.returncode  
        elif isinstance(self.returncode, str):
            if ':' in self.returncode:
                # handle cases like "0:0" or "1:0"
                rc = int(self.returncode.split(':')[0])
            elif self.returncode.isdigit():
                rc = int(self.returncode)
            else:
                rc = rc if rc is not None else 0
        
        return self.job_id, self.state, rc