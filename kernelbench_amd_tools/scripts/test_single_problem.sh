#!/bin/bash

# Test script for single problem evaluation on AMD MI300X with Qwen3-8B on SGLang
# Use this to verify setup before running full evaluation

echo "=============================================="
echo "KernelBench AMD MI300X Single Problem Test"
echo "Model: Qwen3-8B on SGLang"
echo "=============================================="
echo ""

# Set AMD environment variables
export ROCM_HOME=/opt/rocm
export HIP_PLATFORM=amd
export PYTORCH_ROCM_ARCH=gfx942
export PATH=$ROCM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_HOME/lib:$LD_LIBRARY_PATH

# Set API keys for SGLang
export SGLANG_API_KEY="local-key"
export OPENAI_API_KEY="dummy-key"  # Required by OpenAI client library

# Set Python path
export PYTHONPATH=/workspace/KernelBench:$PYTHONPATH

# AMD optimizations
export HSA_ENABLE_SDMA=0

# Disable GPU core dumps
ulimit -c 0
export HSA_ENABLE_COREDUMP=0
export AMD_LOG_LEVEL=0
export ROCM_DISABLE_CRASH_DUMP=1
export HIP_ENABLE_COREDUMP=0

cd /workspace/KernelBench

# Check SGLang server
echo "Checking SGLang server..."
if curl -s http://localhost:30000/v1/models > /dev/null 2>&1; then
    echo "✓ SGLang server is running"
    echo "Model: $(curl -s http://localhost:30000/v1/models | python -c "import sys, json; data=json.load(sys.stdin); print(data['data'][0]['id'] if data['data'] else 'Unknown')")"
    echo ""
else
    echo "✗ SGLang server not accessible on port 30000"
    echo "Please ensure the server is running"
    exit 1
fi

# Parse command line arguments
LEVEL=${1:-1}
PROBLEM_ID=${2:-19}

echo "Testing single problem evaluation..."
echo "Level: $LEVEL, Problem: $PROBLEM_ID"
echo ""

# Create test results directory
TEST_DIR="runs/test_single_$(date +%Y%m%d_%H%M%S)"
mkdir -p $TEST_DIR

python scripts/generate_and_eval_single_sample.py \
    dataset_src=local \
    level=$LEVEL \
    problem_id=$PROBLEM_ID \
    gpu_arch='["MI300X"]' \
    backend=triton \
    server_type=sglang \
    eval_device=0 \
    verbose=True \
    log=True \
    log_generated_kernel=True \
    logdir="$TEST_DIR" \
    2>&1 | tee $TEST_DIR/evaluation.log

echo ""
echo "Results saved to: $TEST_DIR"
echo ""

# Check if evaluation was successful
if grep -q "compiled=True" $TEST_DIR/evaluation.log 2>/dev/null; then
    echo "✓ Kernel compiled successfully"
    if grep -q "correctness=True" $TEST_DIR/evaluation.log 2>/dev/null; then
        echo "✓ Kernel passed correctness check"
    else
        echo "✗ Kernel failed correctness check"
    fi
else
    echo "✗ Kernel compilation failed"
fi

echo ""
echo "If successful, run full evaluation with:"
echo "  cd /workspace/KernelBench/kernelbench_amd_tools/launchers"
echo "  ./start_evaluation.sh"