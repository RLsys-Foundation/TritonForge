#!/bin/bash

set -e

# Project root directory - change this if TritonForge is in a different location
PROJECT_ROOT="/root/TritonForge"

# Ensure logs directory exists
mkdir -p ${PROJECT_ROOT}/SLIME/logs

SESSION_NAME="slime_qwen3_sft_multi_turn_run"
WINDOW_1="slime"
WINDOW_2="buffer"
WINDOW_3="eval_server"

if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "Killing existing tmux session: $SESSION_NAME"
    tmux kill-session -t $SESSION_NAME
fi

sleep 5

tmux new-session -d -s $SESSION_NAME -n $WINDOW_1
tmux send-keys -t ${SESSION_NAME}:${WINDOW_1} "cd ${PROJECT_ROOT}" C-m
tmux send-keys -t ${SESSION_NAME}:${WINDOW_1} "bash ./SLIME/scripts/agent-example-kbench-qwen3-8B-sft-nv-multi-turn.sh |& tee ${PROJECT_ROOT}/SLIME/logs/slime_qwen3_sft_multi_turn_train.log" C-m

tmux new-window -t $SESSION_NAME -n $WINDOW_2
tmux send-keys -t ${SESSION_NAME}:${WINDOW_2} "sleep 30 && cd ${PROJECT_ROOT}/SLIME/slime_plugins/rollout_buffer && python buffer.py |& tee ${PROJECT_ROOT}/SLIME/logs/buffer_qwen3_sft_multi_turn.log" C-m

tmux new-window -t $SESSION_NAME -n $WINDOW_3
tmux send-keys -t ${SESSION_NAME}:${WINDOW_3} "sleep 30 && cd ${PROJECT_ROOT}/KBenchEval && source .venv/bin/activate && CUDA_VISIBLE_DEVICES=6,7 python scripts/eval_server_subprocess.py |& tee ${PROJECT_ROOT}/SLIME/logs/eval_server_qwen3_sft_multi_turn.log" C-m

tmux attach-session -t $SESSION_NAME