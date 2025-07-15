import re
from typing import Optional, List, Union
from pydantic import BaseModel, Field


class KernelExecResult(BaseModel):
    """
    Single Kernel Execution Result
    
    Attributes:
        compiled: Whether the kernel compiled successfully
        correctness: Whether the kernel produced correct results
        metadata: Additional metadata about the execution
        runtime: Runtime in microseconds (-1.0 if not measured)
        runtime_stats: Detailed runtime statistics
    """
    compiled: bool = False
    correctness: bool = False
    metadata: dict = Field(default_factory=dict)
    runtime: float = -1.0  # in us, only recorded if we decide to measure performance
    runtime_stats: dict = Field(default_factory=dict)  # only recorded if we decide to measure performance


class KernelEvalResult(BaseModel):
    """
    Kernel Evaluation Result
    
    Attributes:
        eval_status: Status of the evaluation (e.g., "completed", "failed")
        eval_response: Human-readable response about the evaluation
        completed_at: Timestamp when evaluation completed
        reward: Numerical reward score (0.0 to 3.0)
        exec_result: Detailed execution result
    """
    eval_status: str
    eval_response: str
    completed_at: str
    reward: float = 0.0
    exec_result: KernelExecResult = Field(default_factory=KernelExecResult)


def extract_last_code(
    output_string: str, 
    code_language_types: Optional[List[str]] = None
) -> Optional[str]:
    """
    Extract the last code block from model output, specified by code_language_type
    
    Args:
        output_string: The output string from the model
        code_language_types: List of language types to look for (e.g., ["python", "cpp"])
        
    Returns:
        The extracted code string, or None if no code block found
        
    Examples:
        >>> extract_last_code("Here is code: ```python\nprint('hello')\n```")
        "print('hello')"
        
        >>> extract_last_code("No code here")
        None
    """
    if code_language_types is None:
        code_language_types = ["python", "cpp"]
    
    if not output_string or not output_string.strip():
        return None
        
    trimmed = output_string.strip()

    # Find all matches of code blocks with non-greedy matching
    code_matches = re.finditer(r"```(.*?)```", trimmed, re.DOTALL)
    
    # Get the last match by converting to list and taking the last element
    matches_list = list(code_matches)
    if matches_list:
        last_match = matches_list[-1]
        code = last_match.group(1).strip()

        # Remove language type headers if present
        for code_type in code_language_types:
            if code.startswith(code_type):
                code = code[len(code_type):].strip()
                break

        return code if code else None
    
    return None


def validate_kernel_code(code: str) -> bool:
    """
    Basic validation of kernel code
    
    Args:
        code: The kernel code to validate
        
    Returns:
        True if code appears to be valid, False otherwise
    """
    if not code or not code.strip():
        return False
    
    # Basic checks for common kernel patterns
    # This is a simple heuristic and can be extended
    stripped_code = code.strip()
    
    # Check if it's not just whitespace or comments
    lines = [line.strip() for line in stripped_code.split('\n') if line.strip()]
    non_comment_lines = [line for line in lines if not line.startswith('#') and not line.startswith('//')]
    
    return len(non_comment_lines) > 0
