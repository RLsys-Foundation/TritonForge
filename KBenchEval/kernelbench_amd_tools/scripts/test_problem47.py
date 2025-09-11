#!/usr/bin/env python3
"""
Test subprocess isolation for problem 47 which caused memory fault
"""

import sys
import os
sys.path.insert(0, '/workspace/KernelBench')

from run_qwen3_evaluation_robust import evaluate_kernel_subprocess

# Read the problem 47 kernel that caused the crash
kernel_path = "/workspace/KernelBench/runs/qwen3_eval_20250831_150508/generated_kernels/level1_problem47.py"
ref_path = "/workspace/KernelBench/KernelBench/level_1/47_Sum_reduction_over_a_dimension.py"

if os.path.exists(kernel_path) and os.path.exists(ref_path):
    with open(kernel_path, 'r') as f:
        triton_code = f.read()
    
    with open(ref_path, 'r') as f:
        ref_code = f.read()
    
    print("Testing problem 47 with subprocess isolation...")
    print("=" * 60)
    
    # Test with subprocess (should catch the memory fault)
    result = evaluate_kernel_subprocess(ref_code, triton_code, timeout=30)
    
    print("Result:", result)
    
    if "error" in result:
        print(f"\n✓ Successfully caught error: {result['error']}")
        print("Subprocess isolation working correctly!")
    else:
        print(f"\nKernel evaluation result:")
        print(f"  Compiled: {result.get('compiled', False)}")
        print(f"  Correct: {result.get('correctness', False)}")
else:
    print("Cannot find kernel files. Please check paths.")