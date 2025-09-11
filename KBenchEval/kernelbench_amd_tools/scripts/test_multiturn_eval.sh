#!/bin/bash

# Quick test script for multi-turn evaluation
# Tests with just 2 problems to verify functionality

echo "============================================"
echo "Testing Multi-Turn Evaluation for Qwen3-8B"
echo "============================================"
echo ""
echo "This will evaluate 2 problems with:"
echo "  - Max 3 turns per problem"
echo "  - Gamma = 0.4 (matching RL training)"
echo "  - Native template mode (matching RL training)"
echo "  - Subprocess isolation for robustness"
echo ""

cd /home/jinpan12/workspace/KernelBench/kernelbench_amd_tools/scripts

# Run evaluation with minimal problems for testing
python run_qwen3_evaluation_robust_multiturn.py \
    --levels 1 \
    --max-problems 2 \
    --max-turns 3 \
    --gamma 0.4 \
    --timeout 60 \
    --run-name "multiturn_test_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "Test complete! Check results in:"
echo "/home/jinpan12/workspace/KernelBench/runs/multiturn_test_*/"