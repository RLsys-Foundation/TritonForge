#!/usr/bin/env python3
"""
Quick test script to verify AMD MI300X + SGLang setup
"""

import os
import sys
import torch
import triton
import triton.language as tl
import requests

# Set environment variables
os.environ['ROCM_HOME'] = '/opt/rocm'
os.environ['HIP_PLATFORM'] = 'amd'
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx942'

print("=== KernelBench AMD MI300X + SGLang Setup Test ===\n")

# Test 1: PyTorch and AMD GPU
print("1. PyTorch and AMD GPU:")
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if hasattr(torch.version, 'hip'):
    print(f"   HIP version: {torch.version.hip}")
print(f"   Number of GPUs: {torch.cuda.device_count()}")
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    print(f"   GPU 0: {torch.cuda.get_device_name(0)}")
print()

# Test 2: Triton
print("2. Triton:")
print(f"   Triton version: {triton.__version__}")
print()

# Test 3: Environment variables
print("3. Environment Variables:")
print(f"   ROCM_HOME: {os.environ.get('ROCM_HOME', 'Not set')}")
print(f"   HIP_PLATFORM: {os.environ.get('HIP_PLATFORM', 'Not set')}")
print(f"   PYTORCH_ROCM_ARCH: {os.environ.get('PYTORCH_ROCM_ARCH', 'Not set')}")
print()

# Test 4: SGLang server
print("4. SGLang Server:")
try:
    response = requests.get("http://localhost:30000/v1/models", timeout=5)
    if response.status_code == 200:
        models = response.json()
        print(f"   ✓ Server running on port 30000")
        print(f"   Available models: {[m['id'] for m in models['data']]}")
    else:
        print(f"   ✗ Server responded with status {response.status_code}")
except Exception as e:
    print(f"   ✗ Server not accessible: {e}")
print()

# Test 5: Simple Triton kernel test
print("5. Simple Triton Kernel Test:")
try:
    @triton.jit
    def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)
    
    # Test on small tensors
    x = torch.randn(1024, device='cuda')
    y = torch.randn(1024, device='cuda')
    output = torch.empty_like(x)
    
    grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, x.numel(), BLOCK_SIZE=256)
    torch.cuda.synchronize()
    
    # Verify result
    expected = x + y
    if torch.allclose(output, expected, rtol=1e-5, atol=1e-5):
        print("   ✓ Triton kernel executed successfully")
    else:
        print("   ✗ Triton kernel produced incorrect results")
except Exception as e:
    print(f"   ✗ Triton kernel test failed: {e}")
print()

# Summary
print("=== Summary ===")
all_good = True

if not torch.cuda.is_available():
    print("✗ No GPU available")
    all_good = False
elif not hasattr(torch.version, 'hip'):
    print("✗ PyTorch not built with ROCm support")
    all_good = False
else:
    print("✓ AMD GPU and PyTorch ROCm support available")

try:
    response = requests.get("http://localhost:30000/v1/models", timeout=2)
    if response.status_code == 200:
        print("✓ SGLang server accessible")
    else:
        print("✗ SGLang server not responding correctly")
        all_good = False
except:
    print("✗ SGLang server not accessible")
    all_good = False

if all_good:
    print("\n✓ All systems ready! You can run the KernelBench evaluation.")
else:
    print("\n✗ Some issues detected. Please fix them before running the evaluation.")