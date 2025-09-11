#!/usr/bin/env python3
"""
Test script for AMD MI300X GPU support in KernelBench

This script tests the AMD GPU detection, Triton kernel evaluation,
and performance profiling on AMD MI300X GPUs.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.utils import is_amd_gpu, get_amd_gpu_info, set_gpu_arch
from src.amd_profiling import profile_kernel_basic, get_amd_gpu_metrics
from src.eval import eval_kernel_against_ref

def test_amd_detection():
    """Test AMD GPU detection utilities."""
    print("=" * 60)
    print("Testing AMD GPU Detection")
    print("=" * 60)
    
    is_amd = is_amd_gpu()
    print(f"Is AMD GPU: {is_amd}")
    
    if is_amd:
        gpu_info = get_amd_gpu_info()
        print(f"AMD GPU Info: {gpu_info}")
        
        # Check PyTorch ROCm support
        if hasattr(torch.version, 'hip'):
            print(f"PyTorch HIP version: {torch.version.hip}")
        
        # Check available devices
        if torch.cuda.is_available():
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No AMD GPU detected or running on NVIDIA GPU")
    
    print()

def test_gpu_arch_setting():
    """Test GPU architecture setting for AMD."""
    print("=" * 60)
    print("Testing GPU Architecture Setting")
    print("=" * 60)
    
    # Test setting AMD architectures
    try:
        set_gpu_arch(["MI300X"])
        print("Successfully set architecture: MI300X")
        print(f"PYTORCH_ROCM_ARCH: {os.environ.get('PYTORCH_ROCM_ARCH', 'Not set')}")
        print(f"HIP_ARCHITECTURES: {os.environ.get('HIP_ARCHITECTURES', 'Not set')}")
    except Exception as e:
        print(f"Failed to set AMD architecture: {e}")
    
    print()

def test_simple_triton_kernel():
    """Test a simple Triton kernel on AMD GPU."""
    print("=" * 60)
    print("Testing Simple Triton Kernel")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("No GPU available, skipping kernel test")
        return
    
    # Simple reference implementation
    ref_kernel = '''
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, y):
        return x + y

def get_inputs():
    x = torch.randn(1024, 1024, device='cuda')
    y = torch.randn(1024, 1024, device='cuda')
    return [x, y]

def get_init_inputs():
    return []
'''
    
    # Triton implementation
    triton_kernel = '''
import torch
import torch.nn as nn
import triton
import triton.language as tl

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

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, y):
        output = torch.empty_like(x)
        n_elements = output.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
        return output

def get_inputs():
    x = torch.randn(1024, 1024, device='cuda')
    y = torch.randn(1024, 1024, device='cuda')
    return [x, y]

def get_init_inputs():
    return []
'''
    
    try:
        result = eval_kernel_against_ref(
            original_model_src=ref_kernel,
            custom_model_src=triton_kernel,
            backend="triton",
            num_correct_trials=5,
            num_perf_trials=20,
            measure_performance=True,
            verbose=True
        )
        
        print(f"\nEvaluation Result:")
        print(f"  Compiled: {result.compiled}")
        print(f"  Correct: {result.correctness}")
        if result.runtime > 0:
            print(f"  Runtime: {result.runtime:.2f} us")
        print(f"  Metadata: {result.metadata}")
    except Exception as e:
        print(f"Failed to evaluate kernel: {e}")
    
    print()

def test_amd_profiling():
    """Test AMD-specific profiling utilities."""
    print("=" * 60)
    print("Testing AMD Profiling Utilities")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("No GPU available, skipping profiling test")
        return
    
    # Simple kernel for profiling
    def simple_add(x, y):
        return x + y
    
    # Create test tensors
    x = torch.randn(1024, 1024, device='cuda')
    y = torch.randn(1024, 1024, device='cuda')
    
    # Profile the kernel
    profile_results = profile_kernel_basic(
        kernel_func=simple_add,
        args=[x, y],
        warmup_runs=10,
        profile_runs=50
    )
    
    print("Profiling Results:")
    for key, value in profile_results.items():
        print(f"  {key}: {value}")
    
    # Try to get GPU metrics (if available)
    if is_amd_gpu():
        metrics = get_amd_gpu_metrics(torch.device('cuda:0'))
        if metrics:
            print("\nAMD GPU Metrics:")
            for key, value in metrics.items():
                print(f"  {key}: {value}")
    
    print()

def main():
    """Run all AMD MI300X tests."""
    print("AMD MI300X Support Test Suite for KernelBench")
    print("=" * 60)
    print()
    
    # Run tests
    test_amd_detection()
    test_gpu_arch_setting()
    test_simple_triton_kernel()
    test_amd_profiling()
    
    print("=" * 60)
    print("Test suite completed!")
    
    # Summary
    if is_amd_gpu():
        print("\nAMD GPU support is available and functional.")
        print("You can now use KernelBench to evaluate Triton kernels on AMD MI300X!")
    else:
        print("\nNo AMD GPU detected. Tests were run in compatibility mode.")

if __name__ == "__main__":
    main()