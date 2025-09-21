#!/bin/bash

# AMD MI300X version with ROBUST evaluation server for Qwen3-8B SFT agent training
# SINGLE-TURN version with debugging batch sizes
# This script launches the training in tmux with proper AMD GPU settings
# Uses the robust eval server that handles memory faults gracefully

set -e

SESSION_NAME="slime_qwen3_sft_amd_singleturn"
WINDOW_1="slime"
WINDOW_2="buffer"
WINDOW_3="eval_server"

# Kill existing session if it exists
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "Killing existing tmux session: $SESSION_NAME"
    tmux kill-session -t $SESSION_NAME
fi

sleep 5

# Set AMD environment variables globally
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export HIP_PLATFORM=amd
export PYTORCH_ROCM_ARCH=gfx942
export ROCM_HOME=/opt/rocm

# Disable GPU core dumps to prevent crashes
export HSA_ENABLE_COREDUMP=0
export AMD_LOG_LEVEL=0
export ROCM_DISABLE_CRASH_DUMP=1
export HIP_ENABLE_COREDUMP=0

# Window 1: Main training script (SINGLE-TURN version)
tmux new-session -d -s $SESSION_NAME -n $WINDOW_1
tmux send-keys -t ${SESSION_NAME}:${WINDOW_1} "cd /root/TritonForge/SLIME" C-m
tmux send-keys -t ${SESSION_NAME}:${WINDOW_1} "export HIP_VISIBLE_DEVICES='0,1,2,3,4,5'" C-m
tmux send-keys -t ${SESSION_NAME}:${WINDOW_1} "bash ./scripts/agent-kbench-qwen3-8B-sft-amd-singleturn.sh |& tee /root/TritonForge/SLIME/logs/slime_qwen3_sft_amd_singleturn_train.log" C-m

# Window 2: Rollout buffer (will automatically use single-turn generator based on task type)
tmux new-window -t $SESSION_NAME -n $WINDOW_2
tmux send-keys -t ${SESSION_NAME}:${WINDOW_2} "sleep 30" C-m
tmux send-keys -t ${SESSION_NAME}:${WINDOW_2} "cd /root/TritonForge/SLIME/slime_plugins/rollout_buffer" C-m
tmux send-keys -t ${SESSION_NAME}:${WINDOW_2} "python buffer.py |& tee /root/TritonForge/SLIME/logs/buffer_qwen3_sft_amd_singleturn.log" C-m

# Window 3: ROBUST Evaluation server (using separate GPUs)
tmux new-window -t $SESSION_NAME -n $WINDOW_3
tmux send-keys -t ${SESSION_NAME}:${WINDOW_3} "sleep 30" C-m
tmux send-keys -t ${SESSION_NAME}:${WINDOW_3} "cd /root/TritonForge/KBenchEval" C-m

# Use HIP_VISIBLE_DEVICES for AMD GPUs (using GPUs 6,7 for evaluation)
tmux send-keys -t ${SESSION_NAME}:${WINDOW_3} "export HIP_VISIBLE_DEVICES='6,7'" C-m
tmux send-keys -t ${SESSION_NAME}:${WINDOW_3} "export CUDA_VISIBLE_DEVICES='6,7'" C-m  # Some scripts may still use CUDA_VISIBLE_DEVICES

# Disable core dumps for evaluation server too
tmux send-keys -t ${SESSION_NAME}:${WINDOW_3} "export HSA_ENABLE_COREDUMP=0" C-m
tmux send-keys -t ${SESSION_NAME}:${WINDOW_3} "export ROCM_DISABLE_CRASH_DUMP=1" C-m
tmux send-keys -t ${SESSION_NAME}:${WINDOW_3} "export HIP_ENABLE_COREDUMP=0" C-m
tmux send-keys -t ${SESSION_NAME}:${WINDOW_3} "export AMD_LOG_LEVEL=0" C-m

# Run ROBUST evaluation server with enhanced memory fault handling
tmux send-keys -t ${SESSION_NAME}:${WINDOW_3} "python scripts/eval_server_subprocess_robust.py |& tee /root/TritonForge/SLIME/logs/eval_server_qwen3_sft_amd_singleturn.log" C-m

# Create logs directory if it doesn't exist
mkdir -p /root/TritonForge/SLIME/logs

echo "============================================"
echo "AMD MI300X SINGLE-TURN Training Session Started"
echo "============================================"
echo ""
echo "Session: $SESSION_NAME"
echo "Windows:"
echo "  1. $WINDOW_1 - Main training (GPUs 0-5)"
echo "  2. $WINDOW_2 - Rollout buffer"
echo "  3. $WINDOW_3 - ROBUST Evaluation server (GPUs 6-7)"
echo ""
echo "Key Changes from Multi-turn:"
echo "  - Task type: kernelbench (single-turn)"
echo "  - No multi-turn iterations"
echo "  - Reduced batch sizes for debugging:"
echo "    * rollout-batch-size: 2 (was 4)"
echo "    * n-samples-per-prompt: 4 (was 8)"
echo "    * global-batch-size: 16 (was 32)"
echo "    * max-tokens-per-gpu: 2048 (was 4096)"
echo "    * num-rollout: 500 (was 1000)"
echo ""
echo "Features:"
echo "  - Enhanced memory fault handling"
echo "  - Automatic recovery from GPU crashes"
echo "  - Per-GPU fault tracking"
echo "  - Base64 encoding for code safety"
echo "  - Complete process isolation"
echo ""
echo "Logs:"
echo "  Training: /root/TritonForge/SLIME/logs/slime_qwen3_sft_amd_singleturn_train.log"
echo "  Buffer:   /root/TritonForge/SLIME/logs/buffer_qwen3_sft_amd_singleturn.log"
echo "  Eval:     /root/TritonForge/SLIME/logs/eval_server_qwen3_sft_amd_singleturn.log"
echo ""
echo "Monitor server health:"
echo "  curl http://localhost:18188/health"
echo "  curl http://localhost:18188/fault_statistics"
echo ""
echo "Attaching to tmux session..."
echo ""

# Attach to the session
tmux attach-session -t $SESSION_NAME