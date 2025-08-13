#!/bin/bash

set -e

SESSION_NAME="slime_qwen3_run"
WINDOW_1="slime"
WINDOW_2="buffer"
WINDOW_3="eval_server"

if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "Killing existing tmux session: $SESSION_NAME"
    tmux kill-session -t $SESSION_NAME
fi

sleep 5

tmux new-session -d -s $SESSION_NAME -n $WINDOW_1
tmux send-keys -t ${SESSION_NAME}:${WINDOW_1} "cd $(pwd)" C-m
tmux send-keys -t ${SESSION_NAME}:${WINDOW_1} "bash ./scripts/agent-example-kbench-qwen3-8B.sh |& tee /root/slime/slime_qwen3_train.log" C-m

tmux new-window -t $SESSION_NAME -n $WINDOW_2
tmux send-keys -t ${SESSION_NAME}:${WINDOW_2} "sleep 30 && cd slime_plugins/rollout_buffer && python buffer.py |& tee /root/slime/buffer_qwen3.log" C-m

tmux new-window -t $SESSION_NAME -n $WINDOW_3
tmux send-keys -t ${SESSION_NAME}:${WINDOW_3} "sleep 30 && cd /root/KernelBench && source .venv/bin/activate && CUDA_VISIBLE_DEVICES=6,7 python scripts/eval_server_subprocess.py |& tee /root/slime/eval_server_qwen3.log" C-m

tmux attach-session -t $SESSION_NAME