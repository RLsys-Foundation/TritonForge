#!/usr/bin/env python3
"""
Enhanced Triton code extraction that handles thinking tags and various response formats
"""

import re
from typing import Optional, Tuple

def extract_thinking_and_code(response: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract thinking content and Triton code from model response.
    
    Returns:
        (thinking_content, triton_code)
    """
    thinking_content = None
    triton_code = None
    
    # Extract thinking tag content if present
    thinking_match = re.search(r'<thinking>(.*?)</thinking>', response, re.DOTALL)
    if thinking_match:
        thinking_content = thinking_match.group(1).strip()
    
    # Try multiple patterns for code extraction
    # Pattern 1: Standard markdown code blocks
    code_blocks = re.findall(r'```(?:python)?\n?(.*?)```', response, re.DOTALL)
    
    # Look for the code block with ModelNew class
    for block in code_blocks:
        if 'class ModelNew' in block and '@triton.jit' in block:
            triton_code = block.strip()
            break
    
    # If not found, try the last code block that contains Triton
    if not triton_code:
        for block in reversed(code_blocks):
            if '@triton.jit' in block or 'import triton' in block:
                triton_code = block.strip()
                break
    
    # Pattern 2: Code without markdown markers but with clear class definition
    if not triton_code:
        # Look for code starting with import statements and containing ModelNew
        import_pattern = r'(import torch.*?class ModelNew.*?def get_init_inputs\(\):.*?return.*?)(?=\n\n|\Z)'
        import_match = re.search(import_pattern, response, re.DOTALL)
        if import_match:
            triton_code = import_match.group(1).strip()
    
    return thinking_content, triton_code


def validate_triton_code(code: str) -> Tuple[bool, str]:
    """
    Validate that the extracted code contains necessary components.
    
    Returns:
        (is_valid, error_message)
    """
    if not code:
        return False, "No code extracted"
    
    required_components = {
        "import triton": "Missing triton import",
        "@triton.jit": "Missing Triton kernel definition",
        "class ModelNew": "Missing ModelNew class",
        "def forward": "Missing forward method",
        "def get_inputs": "Missing get_inputs function",
        "def get_init_inputs": "Missing get_init_inputs function"
    }
    
    missing = []
    for component, error_msg in required_components.items():
        if component not in code:
            missing.append(error_msg)
    
    if missing:
        return False, "; ".join(missing)
    
    return True, "Code validation passed"


def fix_common_issues(code: str) -> str:
    """
    Fix common issues in generated Triton code.
    """
    if not code:
        return code
    
    # Ensure proper imports at the beginning
    if not code.startswith("import"):
        required_imports = [
            "import torch",
            "import torch.nn as nn",
            "import torch.nn.functional as F",
            "import triton",
            "import triton.language as tl"
        ]
        
        # Check which imports are missing
        missing_imports = []
        for imp in required_imports:
            if imp not in code:
                missing_imports.append(imp)
        
        if missing_imports:
            code = "\n".join(missing_imports) + "\n\n" + code
    
    # Fix common CUDA/HIP compatibility issues
    code = code.replace(".cuda()", ".cuda() if torch.cuda.is_available() else x")
    code = code.replace("assert x.is_cuda", "assert x.is_cuda or x.device.type == 'cuda'")
    
    return code


def extract_and_validate_triton_code(response: str, verbose: bool = False) -> Optional[str]:
    """
    Main function to extract, validate, and fix Triton code from model response.
    """
    # Extract thinking and code
    thinking, code = extract_thinking_and_code(response)
    
    if verbose and thinking:
        print(f"Extracted thinking ({len(thinking)} chars):")
        print("-" * 60)
        print(thinking[:500] + "..." if len(thinking) > 500 else thinking)
        print("-" * 60)
    
    if not code:
        if verbose:
            print("No Triton code found in response")
        return None
    
    # Fix common issues
    code = fix_common_issues(code)
    
    # Validate code
    is_valid, message = validate_triton_code(code)
    
    if verbose:
        print(f"\nCode validation: {'✓' if is_valid else '✗'} {message}")
        if not is_valid:
            print(f"Extracted code preview (first 500 chars):")
            print("-" * 60)
            print(code[:500] if code else "No code")
            print("-" * 60)
    
    return code if is_valid else None


# Test the extractor
if __name__ == "__main__":
    # Test response with thinking tags
    test_response = """<thinking>
    I need to create a Triton kernel for this convolution operation.
    The key optimization will be to tile the computation efficiently.
    </thinking>
    
    Here's the optimized code with Triton kernels:
    
    ```python
    import torch
    import torch.nn as nn
    import triton
    import triton.language as tl
    
    @triton.jit
    def conv2d_kernel(...):
        # Kernel implementation
        pass
    
    class ModelNew(nn.Module):
        def __init__(self):
            super().__init__()
        
        def forward(self, x):
            # Use Triton kernel
            return x
    
    def get_inputs():
        return [torch.randn(1, 3, 32, 32).cuda()]
    
    def get_init_inputs():
        return []
    ```
    """
    
    code = extract_and_validate_triton_code(test_response, verbose=True)
    if code:
        print("\n✓ Successfully extracted Triton code")
    else:
        print("\n✗ Failed to extract valid Triton code")