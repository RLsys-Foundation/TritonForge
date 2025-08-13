import re
from typing import List, Optional, Tuple

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


def strip_thinking_tags(content: str, preserve_tags: bool = False) -> Tuple[str, str]:
    """
    Strip <think>...</think> tags from content while preserving the cleaned version.
    
    Args:
        content: The model output that may contain thinking tags
        preserve_tags: If True, returns both original and cleaned; if False, only cleaned
        
    Returns:
        Tuple of (cleaned_content, thinking_content)
        - cleaned_content: Content with think tags removed
        - thinking_content: Just the content that was inside think tags
    
    Examples:
        >>> strip_thinking_tags("<think>Planning...</think>Here's the code")
        ("Here's the code", "Planning...")
        
        >>> strip_thinking_tags("No thinking here")
        ("No thinking here", "")
    """
    if not content:
        return content, ""
    
    thinking_parts = []
    cleaned_parts = []
    
    # Pattern to match <think>...</think> blocks (including nested/multiline)
    # Using non-greedy matching to avoid consuming too much
    think_pattern = r'<think>(.*?)</think>'
    
    last_end = 0
    for match in re.finditer(think_pattern, content, re.DOTALL):
        # Add the part before the think tag to cleaned_parts
        cleaned_parts.append(content[last_end:match.start()])
        # Save the thinking content
        thinking_parts.append(match.group(1))
        last_end = match.end()
    
    # Add any remaining content after the last think tag
    cleaned_parts.append(content[last_end:])
    
    cleaned_content = ''.join(cleaned_parts).strip()
    thinking_content = '\n'.join(thinking_parts).strip()
    
    return cleaned_content, thinking_content


def extract_last_code(output_string: str, code_language_types: Optional[List[str]] = None, strip_think_tags: bool = True) -> Optional[str]:
    """
    Extract the last code block from model output, specified by code_language_type
    
    Now handles <think> tags intelligently - strips them before extraction by default.

    Args:
        output_string: The output string from the model
        code_language_types: List of language types to look for (e.g., ["python", "cpp"])
        strip_think_tags: Whether to strip <think> tags before extraction (default True)

    Returns:
        The extracted code string, or None if no code block found

    Examples:
        >>> extract_last_code("Here is code: ```python\nprint('hello')\n```")
        "print('hello')"
        
        >>> extract_last_code("<think>Let me think...</think>```python\nprint('hello')\n```")
        "print('hello')"

        >>> extract_last_code("No code here")
        None
    """
    if code_language_types is None:
        code_language_types = ["python", "cpp"]

    if not output_string or not output_string.strip():
        return None

    # Strip thinking tags if requested (default behavior)
    if strip_think_tags:
        output_string, _ = strip_thinking_tags(output_string)
    
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
                code = code[len(code_type) :].strip()
                break

        return code if code else None

    # If no code blocks found, check if the entire output looks like Python code
    # This handles cases where the model outputs code directly without markdown blocks
    if "import " in trimmed or "def " in trimmed or "@triton.jit" in trimmed:
        # Check if it starts with import or contains typical Python code patterns
        lines = trimmed.split('\n')
        # Look for the start of actual code (skip any non-code preamble)
        code_start = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('from ') or '@triton.jit' in line:
                code_start = i
                break
        
        # Return everything from the first import/decorator to the end
        if code_start < len(lines):
            return '\n'.join(lines[code_start:])
    
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
    lines = [line.strip() for line in stripped_code.split("\n") if line.strip()]
    non_comment_lines = [line for line in lines if not line.startswith("#") and not line.startswith("//")]

    return len(non_comment_lines) > 0
