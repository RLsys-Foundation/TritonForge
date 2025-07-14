import re
from pydantic import BaseModel


class KernelExecResult(BaseModel):
    """
    Single Kernel Execution
    """

    compiled: bool = False
    correctness: bool = False
    metadata: dict = {}
    runtime: float = -1.0  # in us, only recorded if we decide to measure performance
    runtime_stats: dict = {}  # only recorded if we decide to measure performance


class KernelEvalResult(BaseModel):
    """
    Kernel Evaluation Result
    """
    eval_status: str
    eval_response: str
    completed_at: str
    reward: float = 0.
    exec_result: KernelExecResult


def extract_last_code(output_string: str, code_language_types: list[str]=["python", "cpp"]) -> str | None:
    """
    Extract last code block from model output, specified by code_language_type
    """
    if not output_string or not output_string.strip():
        return None
        
    trimmed = output_string.strip()

    # Find all matches of code blocks
    code_matches = re.finditer(r"```(.*?)```", trimmed, re.DOTALL)
    
    # Get the last match by converting to list and taking the last element
    matches_list = list(code_matches)
    if matches_list:
        last_match = matches_list[-1]
        code = last_match.group(1).strip()

        # Remove language type headers
        for code_type in code_language_types:
            if code.startswith(code_type):
                code = code[len(code_type):].strip()

        return code
    
    return None
