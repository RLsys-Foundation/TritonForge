#!/bin/bash

# AMD MI300X version with ROBUST evaluation server for Qwen3-8B SFT agent training
# This script launches the training in tmux with proper AMD GPU settings
# Uses the robust eval server that handles memory faults gracefully

set -e

SESSION_NAME="slime_qwen3_sft_amd_robust"
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

# Window 1: Main training script
tmux new-session -d -s $SESSION_NAME -n $WINDOW_1
tmux send-keys -t ${SESSION_NAME}:${WINDOW_1} "cd /workspace/slime" C-m
tmux send-keys -t ${SESSION_NAME}:${WINDOW_1} "export HIP_VISIBLE_DEVICES='0,1,2,3,4,5'" C-m
tmux send-keys -t ${SESSION_NAME}:${WINDOW_1} "bash ./scripts/agent-kbench-qwen3-8B-sft-amd.sh |& tee /workspace/slime/logs/slime_qwen3_sft_amd_robust_train.log" C-m

# Window 2: Rollout buffer
tmux new-window -t $SESSION_NAME -n $WINDOW_2
tmux send-keys -t ${SESSION_NAME}:${WINDOW_2} "sleep 30" C-m
tmux send-keys -t ${SESSION_NAME}:${WINDOW_2} "cd /workspace/slime/slime_plugins/rollout_buffer" C-m
tmux send-keys -t ${SESSION_NAME}:${WINDOW_2} "python buffer.py |& tee /workspace/slime/logs/buffer_qwen3_sft_amd_robust.log" C-m

# Window 3: ROBUST Evaluation server (using separate GPUs)
tmux new-window -t $SESSION_NAME -n $WINDOW_3
tmux send-keys -t ${SESSION_NAME}:${WINDOW_3} "sleep 30" C-m
tmux send-keys -t ${SESSION_NAME}:${WINDOW_3} "cd /workspace/KernelBench" C-m

# Use HIP_VISIBLE_DEVICES for AMD GPUs (using GPUs 6,7 for evaluation)
tmux send-keys -t ${SESSION_NAME}:${WINDOW_3} "export HIP_VISIBLE_DEVICES='6,7'" C-m
tmux send-keys -t ${SESSION_NAME}:${WINDOW_3} "export CUDA_VISIBLE_DEVICES='6,7'" C-m  # Some scripts may still use CUDA_VISIBLE_DEVICES

# Disable core dumps for evaluation server too
tmux send-keys -t ${SESSION_NAME}:${WINDOW_3} "export HSA_ENABLE_COREDUMP=0" C-m
tmux send-keys -t ${SESSION_NAME}:${WINDOW_3} "export ROCM_DISABLE_CRASH_DUMP=1" C-m
tmux send-keys -t ${SESSION_NAME}:${WINDOW_3} "export HIP_ENABLE_COREDUMP=0" C-m
tmux send-keys -t ${SESSION_NAME}:${WINDOW_3} "export AMD_LOG_LEVEL=0" C-m

# Run ROBUST evaluation server with enhanced memory fault handling
tmux send-keys -t ${SESSION_NAME}:${WINDOW_3} "python scripts/eval_server_subprocess_robust.py |& tee /workspace/slime/logs/eval_server_qwen3_sft_amd_robust.log" C-m

# Create logs directory if it doesn't exist
mkdir -p /workspace/slime/logs

echo "============================================"
echo "AMD MI300X ROBUST Training Session Started"
echo "============================================"
echo ""
echo "Session: $SESSION_NAME"
echo "Windows:"
echo "  1. $WINDOW_1 - Main training (GPUs 0-5)"
echo "  2. $WINDOW_2 - Rollout buffer"
echo "  3. $WINDOW_3 - ROBUST Evaluation server (GPUs 6-7)"
echo ""
echo "Features:"
echo "  - Enhanced memory fault handling"
echo "  - Automatic recovery from GPU crashes"
echo "  - Per-GPU fault tracking"
echo "  - Base64 encoding for code safety"
echo "  - Complete process isolation"
echo ""
echo "Logs:"
echo "  Training: /workspace/slime/logs/slime_qwen3_sft_amd_robust_train.log"
echo "  Buffer:   /workspace/slime/logs/buffer_qwen3_sft_amd_robust.log"
echo "  Eval:     /workspace/slime/logs/eval_server_qwen3_sft_amd_robust.log"
echo ""
echo "Monitor server health:"
echo "  curl http://localhost:18188/health"
echo "  curl http://localhost:18188/fault_statistics"
echo ""
echo "Attaching to tmux session..."
echo ""

# Attach to the session
tmux attach-session -t $SESSION_NAME