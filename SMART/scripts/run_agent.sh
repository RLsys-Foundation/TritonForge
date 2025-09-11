
#!/bin/bash

set -e

SESSION_NAME="slime_run"
WINDOW_1="slime"
WINDOW_2="buffer"

if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "Killing existing tmux session: $SESSION_NAME"
    tmux kill-session -t $SESSION_NAME
fi

tmux new-session -d -s $SESSION_NAME -n $WINDOW_1
tmux send-keys -t ${SESSION_NAME}:${WINDOW_1} "cd $(pwd)" C-m
tmux send-keys -t ${SESSION_NAME}:${WINDOW_1} "bash ./scripts/agent-example.sh" C-m

tmux new-window -t $SESSION_NAME -n $WINDOW_2
tmux send-keys -t ${SESSION_NAME}:${WINDOW_2} "sleep 30 && cd slime_plugins/rollout_buffer && python buffer.py" C-m

tmux attach-session -t $SESSION_NAME