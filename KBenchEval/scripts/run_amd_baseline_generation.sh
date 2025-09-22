#!/bin/bash
# Script to generate baseline timings for AMD MI300X GPUs
# Generates a combined JSON file with both Level 1 and Level 2 data (200 problems total)

set -e

echo "=================================================="
echo "AMD MI300X Baseline Timing Generation"
echo "=================================================="
echo ""
echo "This script will generate COMBINED baseline timings for:"
echo "  - Level 1: Single-kernel operators (100 problems)"
echo "  - Level 2: Simple fusion patterns (100 problems)"
echo "  - Total: 200 problems in a single JSON file"
echo "  - Modes: PyTorch Eager + torch.compile (limited)"
echo ""
echo "Expected duration: 30-60 minutes"
echo "=================================================="
echo ""

# Set AMD environment variables
export ROCM_HOME=/opt/rocm
export HIP_PLATFORM=amd
export PYTORCH_ROCM_ARCH=gfx942
export HSA_ENABLE_COREDUMP=0
export AMD_LOG_LEVEL=0
export ROCM_DISABLE_CRASH_DUMP=1
export HIP_ENABLE_COREDUMP=0
export HSA_ENABLE_SDMA=0
export GPU_MAX_HW_QUEUES=1

# Navigate to KBenchEval directory
cd /root/TritonForge/KBenchEval

# Quick test run first with both levels
echo "[Step 1/2] Running quick test with 5 problems per level (10 total)..."
python scripts/generate_baseline_time_amd.py \
  --test \
  --max-problems 5 \
  --level 1 2 \
  --hardware MI300X_rocm

echo ""
echo "[Test Complete] Test baseline generated successfully"
echo "Check the test output to verify both level1 and level2 are present in the JSON"
echo ""

# Show test results
if [ -f "results/timing/MI300X_rocm/baseline_time_torch_test.json" ]; then
    echo "Test JSON structure preview:"
    python -c "
import json
with open('results/timing/MI300X_rocm/baseline_time_torch_test.json', 'r') as f:
    data = json.load(f)
    for level in data.keys():
        print(f'  {level}: {len(data[level])} problems')
    "
    echo ""
fi

# Ask for confirmation before full run
read -p "Proceed with full baseline generation (200 problems)? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Exiting..."
    exit 1
fi

# Full baseline generation for BOTH Level 1 and Level 2 in a single run
echo ""
echo "[Step 2/2] Generating combined Level 1 + Level 2 baselines (200 problems total)..."
echo "This will create a single JSON file with both level1 and level2 sections"
python scripts/generate_baseline_time_amd.py \
  --level 1 2 \
  --hardware MI300X_rocm

echo ""
echo "=================================================="
echo "Baseline Generation Complete!"
echo "=================================================="
echo ""
echo "Results saved in:"
echo "  /root/TritonForge/KBenchEval/results/timing/MI300X_rocm/"
echo ""
echo "Files generated:"
ls -la /root/TritonForge/KBenchEval/results/timing/MI300X_rocm/
echo ""
echo "You can now use these baselines for performance comparison"
echo "in your KernelBench evaluations."
echo "=================================================="