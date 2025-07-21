#!/bin/bash
# Launch script for KernelBench evaluation on AMD MI300X with SGLang

echo "=========================================="
echo "KernelBench AMD MI300X Evaluation Launcher"
echo "=========================================="
echo ""

# Set environment variables
export ROCM_HOME=/opt/rocm
export HIP_PLATFORM=amd
export PYTORCH_ROCM_ARCH=gfx942
export PATH=$ROCM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_HOME/lib:$LD_LIBRARY_PATH
export SGLANG_API_KEY=local-key

# Change to KernelBench directory
cd /workspace/KernelBench

# Run the evaluation script
python run_amd_mi300x_sglang.py