#!/usr/bin/env python3
"""
Patch for utils.py to properly handle Qwen3-8B with SGLang
This creates a modified query_server function that works correctly with Qwen3
"""

import re
from typing import Optional
from openai import OpenAI


def extract_after_thinking(response: str) -> str:
    """Extract actual response after thinking tags."""
    if '<think>' in response and '</think>' in response:
        parts = response.split('</think>')
        if len(parts) > 1:
            return parts[1].strip()
    return response


def query_server_fixed_for_qwen(
    prompt: str,
    system_prompt: str = "You are a helpful assistant",
    temperature: float = 0.0,
    max_tokens: int = 8192,
    server_port: int = 30000,
    server_address: str = "localhost",
    verbose: bool = False
):
    """
    Fixed query function specifically for Qwen3-8B on SGLang.
    Uses chat completions API and handles thinking tags.
    """
    
    client = OpenAI(
        api_key="dummy-key",
        base_url=f"http://{server_address}:{server_port}/v1",
        timeout=None,
        max_retries=0
    )
    
    # Construct proper chat messages
    messages = []
    
    # Parse the prompt to extract system and user parts if they exist
    if prompt.startswith("System:"):
        parts = prompt.split("\n\n")
        for part in parts:
            if part.startswith("System:"):
                system_content = part.replace("System:", "").strip()
                messages.append({"role": "system", "content": system_content})
            elif part.startswith("User:"):
                user_content = part.replace("User:", "").strip()
                messages.append({"role": "user", "content": user_content})
            elif part.strip():
                # Any remaining content goes to user
                messages.append({"role": "user", "content": part.strip()})
    else:
        # Simple prompt - just use as user message
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    
    if verbose:
        print(f"Sending {len(messages)} messages to Qwen3-8B")
        print(f"Total prompt length: {sum(len(m['content']) for m in messages)} chars")
    
    try:
        # Use chat completions API for Qwen3
        response = client.chat.completions.create(
            model="Qwen/Qwen3-8B",  # Use actual model name
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=["<|im_end|>", "<|endoftext|>", "\n\n\n\n"]  # Qwen stop tokens
        )
        
        full_response = response.choices[0].message.content
        
        # Extract content after thinking tags
        actual_response = extract_after_thinking(full_response)
        
        if verbose and actual_response != full_response:
            print(f"Removed {len(full_response) - len(actual_response)} chars of thinking")
        
        return actual_response
        
    except Exception as e:
        if verbose:
            print(f"Error with chat API: {e}")
        raise e


def extract_triton_code_from_qwen_response(response: str, verbose: bool = False) -> Optional[str]:
    """
    Enhanced extraction specifically for Qwen3-8B responses.
    Handles thinking tags and various code formats.
    """
    
    # First remove thinking tags
    clean_response = extract_after_thinking(response)
    
    if verbose:
        print(f"Response after removing thinking: {len(clean_response)} chars")
    
    # Strategy 1: Look for markdown code blocks
    code_blocks = re.findall(r'```(?:python)?\n?(.*?)```', clean_response, re.DOTALL)
    
    # Find the best code block with most Triton components
    best_block = None
    best_score = 0
    
    for i, block in enumerate(code_blocks):
        score = 0
        
        # Essential components
        if 'import triton' in block:
            score += 3
        if '@triton.jit' in block:
            score += 3
        if 'class ModelNew' in block:
            score += 3
        
        # Supporting components
        if 'def forward' in block:
            score += 1
        if 'def get_inputs' in block:
            score += 1
        if 'def get_init_inputs' in block:
            score += 1
        if 'import torch' in block:
            score += 1
        
        if verbose and score > 0:
            print(f"Code block {i+1}: score={score}, length={len(block)}")
        
        if score > best_score:
            best_score = score
            best_block = block
    
    # Need at least the essential components
    if best_block and best_score >= 6:
        if verbose:
            print(f"✓ Found valid Triton code block with score {best_score}")
        return best_block.strip()
    
    # Strategy 2: Look for code without markdown markers
    # Pattern: starts with imports, ends with get_init_inputs
    pattern = r'(import torch.*?def get_init_inputs\(\).*?return.*?)(?=\n\n|\Z)'
    match = re.search(pattern, clean_response, re.DOTALL)
    
    if match:
        code = match.group(1).strip()
        # Verify it has Triton components
        if '@triton.jit' in code and 'class ModelNew' in code:
            if verbose:
                print("✓ Found Triton code without markdown markers")
            return code
    
    # Strategy 3: Look for incomplete code that we can fix
    if code_blocks:
        # Take the longest block that has at least some Triton
        for block in sorted(code_blocks, key=len, reverse=True):
            if 'triton' in block.lower() or 'ModelNew' in block:
                if verbose:
                    print("⚠ Found partial Triton code, attempting to use")
                return block.strip()
    
    if verbose:
        print("✗ No valid Triton code found")
        print(f"Response preview: {clean_response[:500]}")
    
    return None


def create_enhanced_prompt_for_qwen(original_prompt: str) -> str:
    """
    Enhance prompt to encourage Qwen3 to generate complete code after thinking.
    """
    
    # Add explicit instruction to generate code after thinking
    enhancement = """

After thinking through the problem, generate the complete Triton-optimized code.
The code must include:
1. All imports (torch, triton, triton.language)
2. The @triton.jit kernel function
3. A wrapper function that calls the kernel
4. The ModelNew class with forward method
5. The get_inputs() and get_init_inputs() functions

Output the complete code in a single Python code block using ```python markers."""
    
    # Check if prompt already has similar instructions
    if "Output the new code in codeblocks" not in original_prompt:
        return original_prompt + enhancement
    
    return original_prompt


# Test the fixes
if __name__ == "__main__":
    print("Testing Qwen3-8B fixes...")
    print("="*60)
    
    # Test extraction from response with thinking
    test_response = """<think>
    I need to create a Triton kernel for ReLU. Let me think about this...
    The ReLU function is max(0, x), which is simple to implement.
    </think>
    
    Here's the Triton-optimized implementation:
    
    ```python
    import torch
    import torch.nn as nn
    import triton
    import triton.language as tl
    
    @triton.jit
    def relu_kernel(
        x_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        output = tl.maximum(x, 0.0)
        tl.store(output_ptr + offsets, output, mask=mask)
    
    def triton_relu(x: torch.Tensor):
        output = torch.empty_like(x)
        n_elements = x.numel()
        BLOCK_SIZE = 1024
        grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
        relu_kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        return output
    
    class ModelNew(nn.Module):
        def __init__(self):
            super().__init__()
        
        def forward(self, x):
            return triton_relu(x)
    
    def get_inputs():
        return [torch.randn(16, 16384).cuda()]
    
    def get_init_inputs():
        return []
    ```
    """
    
    # Test extraction
    code = extract_triton_code_from_qwen_response(test_response, verbose=True)
    
    if code:
        print("\n✅ Successfully extracted Triton code!")
        print("Components found:")
        print(f"  - import triton: {'import triton' in code}")
        print(f"  - @triton.jit: {'@triton.jit' in code}")
        print(f"  - class ModelNew: {'class ModelNew' in code}")
        print(f"  - def forward: {'def forward' in code}")
        print(f"  - def get_inputs: {'def get_inputs' in code}")
    else:
        print("\n❌ Failed to extract code")
    
    print("\n" + "="*60)
    print("Testing actual API call...")
    
    try:
        # Test with actual SGLang server
        simple_prompt = "Write a Python function that adds two numbers. Just the code."
        response = query_server_fixed_for_qwen(
            simple_prompt,
            temperature=0.0,
            max_tokens=256,
            verbose=True
        )
        print(f"\nResponse preview: {response[:300]}")
    except Exception as e:
        print(f"Cannot test actual API: {e}")