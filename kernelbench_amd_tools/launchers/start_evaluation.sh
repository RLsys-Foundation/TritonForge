#!/bin/bash
# KernelBench AMD MI300X Evaluation Launcher

echo "=============================================="
echo "KernelBench AMD MI300X Full Evaluation"
echo "=============================================="
echo ""

# Set AMD environment variables
export ROCM_HOME=/opt/rocm
export HIP_PLATFORM=amd
export PYTORCH_ROCM_ARCH=gfx942
export PATH=$ROCM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_HOME/lib:$LD_LIBRARY_PATH
export SGLANG_API_KEY=local-key
export PYTHONPATH=/workspace/KernelBench:$PYTHONPATH

# AMD optimizations
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export HSA_ENABLE_SDMA=0

# Change to KernelBench directory
cd /workspace/KernelBench

# Check SGLang server
echo "Checking SGLang server..."
if curl -s http://localhost:30000/v1/models > /dev/null 2>&1; then
    echo "✓ SGLang server is running"
    echo ""
else
    echo "✗ SGLang server not accessible on port 30000"
    echo "Please ensure the server is running with facebook/KernelLLM model"
    exit 1
fi

# Create logs directory
mkdir -p logs

echo "Starting evaluation of ~270 problems across 4 levels..."
echo "Progress saved to: runs/amd_mi300x_full_eval_*"
echo "Logs saved to: logs/evaluation_$(date +%Y%m%d_%H%M%S).log"
echo ""
echo "Monitor progress in another terminal:"
echo "  cd /workspace/KernelBench/kernelbench_amd_tools/scripts"
echo "  ./monitor_evaluation.py"
echo ""
echo "Press Ctrl+C to interrupt (progress will be saved)"
echo ""

# Run the evaluation
python /workspace/KernelBench/kernelbench_amd_tools/scripts/run_full_evaluation.py 2>&1 | tee logs/evaluation_$(date +%Y%m%d_%H%M%S).log