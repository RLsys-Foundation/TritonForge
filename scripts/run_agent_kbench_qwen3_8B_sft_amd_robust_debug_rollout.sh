#!/bin/bash

# DEBUG ROLLOUT ONLY VERSION - AMD MI300X
# This script launches the rollout-only debugging in tmux
# Only runs rollout generation without training to debug multi-turn issues

set -e

SESSION_NAME="slime_qwen3_sft_amd_debug_rollout"
WINDOW_1="rollout_debug"
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

# Window 1: Main rollout debug script (using only 2 GPUs)
tmux new-session -d -s $SESSION_NAME -n $WINDOW_1
tmux send-keys -t ${SESSION_NAME}:${WINDOW_1} "cd /home/jinpan12/workspace/slime" C-m
tmux send-keys -t ${SESSION_NAME}:${WINDOW_1} "export HIP_VISIBLE_DEVICES='0,1'" C-m  # Only 2 GPUs for rollout
tmux send-keys -t ${SESSION_NAME}:${WINDOW_1} "bash ./scripts/agent-kbench-qwen3-8B-sft-amd-debug-rollout.sh |& tee /home/jinpan12/workspace/slime/logs/slime_qwen3_sft_amd_debug_rollout.log" C-m

# Window 2: Rollout buffer
tmux new-window -t $SESSION_NAME -n $WINDOW_2
tmux send-keys -t ${SESSION_NAME}:${WINDOW_2} "sleep 30" C-m
tmux send-keys -t ${SESSION_NAME}:${WINDOW_2} "cd /home/jinpan12/workspace/slime/slime_plugins/rollout_buffer" C-m
tmux send-keys -t ${SESSION_NAME}:${WINDOW_2} "python buffer.py |& tee /home/jinpan12/workspace/slime/logs/buffer_qwen3_sft_amd_debug_rollout.log" C-m

# Window 3: ROBUST Evaluation server (using separate GPUs)
tmux new-window -t $SESSION_NAME -n $WINDOW_3
tmux send-keys -t ${SESSION_NAME}:${WINDOW_3} "sleep 30" C-m
tmux send-keys -t ${SESSION_NAME}:${WINDOW_3} "cd /home/jinpan12/workspace/KernelBench" C-m

# Use HIP_VISIBLE_DEVICES for AMD GPUs (using GPUs 2,3 for evaluation)
tmux send-keys -t ${SESSION_NAME}:${WINDOW_3} "export HIP_VISIBLE_DEVICES='2,3'" C-m
tmux send-keys -t ${SESSION_NAME}:${WINDOW_3} "export CUDA_VISIBLE_DEVICES='2,3'" C-m  # Some scripts may still use CUDA_VISIBLE_DEVICES

# Disable core dumps for evaluation server too
tmux send-keys -t ${SESSION_NAME}:${WINDOW_3} "export HSA_ENABLE_COREDUMP=0" C-m
tmux send-keys -t ${SESSION_NAME}:${WINDOW_3} "export ROCM_DISABLE_CRASH_DUMP=1" C-m
tmux send-keys -t ${SESSION_NAME}:${WINDOW_3} "export HIP_ENABLE_COREDUMP=0" C-m
tmux send-keys -t ${SESSION_NAME}:${WINDOW_3} "export AMD_LOG_LEVEL=0" C-m

# Run ROBUST evaluation server with enhanced memory fault handling
tmux send-keys -t ${SESSION_NAME}:${WINDOW_3} "python scripts/eval_server_subprocess_robust.py |& tee /home/jinpan12/workspace/slime/logs/eval_server_qwen3_sft_amd_debug_rollout.log" C-m

# Create logs directory if it doesn't exist
mkdir -p /home/jinpan12/workspace/slime/logs

echo "============================================"
echo "AMD MI300X DEBUG ROLLOUT-ONLY MODE"
echo "============================================"
echo ""
echo "Session: $SESSION_NAME"
echo "Windows:"
echo "  1. $WINDOW_1 - Rollout debugging (GPUs 0-1)"
echo "  2. $WINDOW_2 - Rollout buffer"
echo "  3. $WINDOW_3 - ROBUST Evaluation server (GPUs 2-3)"
echo ""
echo "Mode: DEBUG ROLLOUT-ONLY"
echo "  - NO Megatron loading"
echo "  - NO training or weight updates"
echo "  - ONLY rollout generation with SGLang"
echo "  - Used to debug multi-turn inference issues"
echo ""
echo "GPU Allocation:"
echo "  - Rollout: GPUs 0-1 (2 GPUs)"
echo "  - Eval Server: GPUs 2-3 (2 GPUs)"
echo "  - Total: 4 GPUs"
echo ""
echo "Logs:"
echo "  Rollout:  /home/jinpan12/workspace/slime/logs/slime_qwen3_sft_amd_debug_rollout.log"
echo "  Buffer:   /home/jinpan12/workspace/slime/logs/buffer_qwen3_sft_amd_debug_rollout.log"
echo "  Eval:     /home/jinpan12/workspace/slime/logs/eval_server_qwen3_sft_amd_debug_rollout.log"
echo ""
echo "Monitor server health:"
echo "  curl http://localhost:18188/health"
echo "  curl http://localhost:18188/fault_statistics"
echo ""
echo "Attaching to tmux session..."
echo ""

# Attach to the session
tmux attach-session -t $SESSION_NAME