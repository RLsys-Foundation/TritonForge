#!/usr/bin/env python3
"""
Fixed version of SGLang API calls that properly uses chat format for Qwen3-8B
"""

import os
import sys
import json
from openai import OpenAI

# Test proper chat API usage with Qwen3-8B

def test_sglang_chat_api():
    """Test the correct way to call Qwen3-8B through SGLang."""
    
    # Create client
    client = OpenAI(
        api_key="dummy-key",
        base_url="http://localhost:30000/v1",
        timeout=None,
        max_retries=0
    )
    
    # Simple test prompt using proper chat format
    messages = [
        {
            "role": "system",
            "content": "You are an expert programmer who writes clean, efficient code."
        },
        {
            "role": "user", 
            "content": "Write a Python function that calculates the factorial of a number. Just the code, no explanation."
        }
    ]
    
    print("Testing SGLang Chat API with Qwen3-8B")
    print("="*60)
    print("Messages being sent:")
    print(json.dumps(messages, indent=2))
    print("="*60)
    
    try:
        # Use chat.completions (NOT completions) for chat models
        response = client.chat.completions.create(
            model="Qwen/Qwen3-8B",  # Use actual model name from server
            messages=messages,
            temperature=0.0,
            max_tokens=256,
            stop=["<|im_end|>", "<|endoftext|>"]  # Qwen3 stop tokens
        )
        
        print("\nResponse:")
        print("-"*60)
        print(response.choices[0].message.content)
        print("-"*60)
        
        # Check if response contains code
        if "def" in response.choices[0].message.content:
            print("\n✓ Model generated code successfully!")
        else:
            print("\n⚠ No code found in response")
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nTrying completion API as fallback...")
        
        # If chat API fails, try completion API with Qwen format
        qwen_prompt = "<|im_start|>system\nYou are an expert programmer.<|im_end|>\n"
        qwen_prompt += "<|im_start|>user\nWrite a Python factorial function.<|im_end|>\n"
        qwen_prompt += "<|im_start|>assistant\n"
        
        response = client.completions.create(
            model="default",
            prompt=qwen_prompt,
            temperature=0.0,
            max_tokens=256,
            stop=["<|im_end|>"]
        )
        
        print("Completion API Response:")
        print("-"*60)
        print(response.choices[0].text)


def test_triton_generation():
    """Test Triton code generation with proper format."""
    
    client = OpenAI(
        api_key="dummy-key",
        base_url="http://localhost:30000/v1",
        timeout=None,
        max_retries=0
    )
    
    # Proper chat format for Triton task
    messages = [
        {
            "role": "system",
            "content": "You are an expert in writing Triton kernels for GPU programming."
        },
        {
            "role": "user",
            "content": """Write a Triton kernel for ReLU activation. The code should include:
1. Import statements (torch, triton, etc.)
2. A @triton.jit decorated kernel function
3. A wrapper function
4. A ModelNew class that uses the kernel

Here's the PyTorch version to optimize:
```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.relu(x)
```

Write the complete Triton-optimized version:"""
        }
    ]
    
    print("\n\nTesting Triton Generation")
    print("="*60)
    
    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen3-8B",
            messages=messages,
            temperature=0.0,
            max_tokens=1024,
            stop=["<|im_end|>", "<|endoftext|>", "```\n\n"]
        )
        
        content = response.choices[0].message.content
        print("Response length:", len(content))
        print("\nFirst 500 chars:")
        print("-"*60)
        print(content[:500])
        print("-"*60)
        
        # Check for Triton components
        has_import = "import triton" in content
        has_jit = "@triton.jit" in content
        has_model = "ModelNew" in content
        
        print(f"\n✓ Has triton import: {has_import}")
        print(f"✓ Has @triton.jit: {has_jit}")
        print(f"✓ Has ModelNew class: {has_model}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")


if __name__ == "__main__":
    # Test basic chat API
    test_sglang_chat_api()
    
    # Test Triton generation
    test_triton_generation()