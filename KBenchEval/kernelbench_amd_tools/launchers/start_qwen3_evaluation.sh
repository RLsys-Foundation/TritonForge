#!/bin/bash

# =============================================================================
# Qwen3-8B Evaluation Launcher for AMD MI300X
# =============================================================================
# This script runs the complete evaluation pipeline for Qwen3-8B
# with proper handling of thinking tags and chat API
# =============================================================================

echo "=============================================================="
echo "KernelBench Evaluation with Qwen3-8B"
echo "Model: Qwen/Qwen3-8B on SGLang"
echo "GPU: AMD MI300X"
echo "=============================================================="
echo ""

# Set AMD environment variables
export ROCM_HOME=/opt/rocm
export HIP_PLATFORM=amd
export PYTORCH_ROCM_ARCH=gfx942
export PATH=$ROCM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_HOME/lib:$LD_LIBRARY_PATH

# Set API keys
export SGLANG_API_KEY="local-key"
export OPENAI_API_KEY="dummy-key"

# Set Python path
export PYTHONPATH=/workspace/KernelBench:$PYTHONPATH

# AMD optimizations
export HSA_ENABLE_SDMA=0

# Disable GPU core dumps completely
ulimit -c 0
export HSA_ENABLE_COREDUMP=0
export AMD_LOG_LEVEL=0
export ROCM_DISABLE_CRASH_DUMP=1
export HIP_ENABLE_COREDUMP=0

# Additional AMD debug settings to prevent crashes
export HSA_TOOLS_LIB=""
export HSA_TOOLS_REPORT_LOAD_FAILURE=0
export GPU_MAX_HW_QUEUES=1
export AMD_DIRECT_DISPATCH=0

cd /workspace/KernelBench

# Check SGLang server
echo "Step 1: Checking SGLang server..."
echo "---------------------------------"
if curl -s http://localhost:30000/v1/models > /dev/null 2>&1; then
    echo "✓ SGLang server is running"
    MODEL_INFO=$(curl -s http://localhost:30000/v1/models | python -c "import sys, json; data=json.load(sys.stdin); print(data['data'][0]['id'] if data['data'] else 'Unknown')" 2>/dev/null)
    echo "✓ Model: $MODEL_INFO"
    
    # Verify it's Qwen3
    if [[ "$MODEL_INFO" == *"Qwen"* ]]; then
        echo "✓ Qwen3 model confirmed"
    else
        echo "⚠ Warning: Expected Qwen3 model, found: $MODEL_INFO"
    fi
    echo ""
else
    echo "✗ SGLang server not accessible on port 30000"
    echo ""
    echo "Please start the server with:"
    echo "  HIP_VISIBLE_DEVICES=2,3 python3 -m sglang.launch_server \\"
    echo "    --model-path Qwen/Qwen3-8B \\"
    echo "    --tp 2 \\"
    echo "    --trust-remote-code \\"
    echo "    --host 0.0.0.0 \\"
    echo "    --port 30000"
    exit 1
fi

# Parse command line arguments
LEVELS=${1:-"1,2"}
MAX_PROBLEMS=${2:-""}
RUN_NAME=${3:-""}

echo "Step 2: Configuration"
echo "--------------------"
echo "Levels: $LEVELS"
if [ -n "$MAX_PROBLEMS" ]; then
    echo "Max problems per level: $MAX_PROBLEMS"
else
    echo "Max problems per level: All"
fi

if [ -n "$RUN_NAME" ]; then
    echo "Run name: $RUN_NAME"
else
    RUN_NAME="qwen3_eval_$(date +%Y%m%d_%H%M%S)"
    echo "Run name: $RUN_NAME (auto-generated)"
fi
echo ""

# Create output directory
OUTPUT_DIR="/workspace/KernelBench/runs/$RUN_NAME"
echo "Output directory: $OUTPUT_DIR"
echo ""

echo "Step 3: Starting Evaluation"
echo "--------------------------"
echo ""

# Build the command
CMD="python /workspace/KernelBench/kernelbench_amd_tools/scripts/run_qwen3_evaluation.py"
CMD="$CMD --levels $LEVELS"

if [ -n "$MAX_PROBLEMS" ]; then
    CMD="$CMD --max-problems $MAX_PROBLEMS"
fi

if [ -n "$RUN_NAME" ]; then
    CMD="$CMD --run-name $RUN_NAME"
fi

# Run the evaluation
$CMD 2>&1 | tee "$OUTPUT_DIR/evaluation.log"

echo ""
echo "=============================================================="
echo "Evaluation Complete!"
echo "=============================================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "View reports:"
echo "  Level 1: cat $OUTPUT_DIR/reports/level1_report.md"
echo "  Level 2: cat $OUTPUT_DIR/reports/level2_report.md"
echo "  Final:   cat $OUTPUT_DIR/reports/FINAL_REPORT.md"
echo ""
echo "View generated kernels:"
echo "  ls $OUTPUT_DIR/generated_kernels/"
echo ""
echo "View model responses (with thinking):"
echo "  ls $OUTPUT_DIR/responses/"
echo ""