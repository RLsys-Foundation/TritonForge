#!/bin/bash
# Resume script for generating remaining 700 samples (301-1000)
# Includes CUDA error handling and tmux setup

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}   Resume Generation: Samples 301-1000 (700 samples)${NC}"
echo -e "${BLUE}================================================${NC}"

# Configuration for remaining samples
export INPUT_FILE="${INPUT_FILE:-/root/kernel_book/kernelbook_sft_format.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-/root/KernelBench/multi_turn_sft_outputs}"
export START_SAMPLE=301      # Starting from sample 301
export MAX_SAMPLES=700       # Remaining 700 samples  
export BATCH_SIZE=10         # Reduced batch size to avoid memory issues
export MAX_TURNS=3           # 3 turns per conversation
export MODEL="claude-sonnet-4-20250514"

# CUDA error handling
export CUDA_LAUNCH_BLOCKING=1         # Better error reporting
export TORCH_USE_CUDA_DSA=1          # Enable device-side assertions
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Memory management

# Check for API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo -e "${RED}[ERROR]${NC} ANTHROPIC_API_KEY environment variable not set"
    echo ""
    echo "Please set your API key:"
    echo -e "${YELLOW}  export ANTHROPIC_API_KEY=your_api_key${NC}"
    echo -e "${YELLOW}  ./resume_generation_700.sh${NC}"
    exit 1
fi

# Create output directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="$OUTPUT_DIR/resume_${TIMESTAMP}_700samples"
mkdir -p "$RUN_DIR"

# Extract samples 301-1000 from input file
echo -e "${GREEN}[INFO]${NC} Extracting samples 301-1000 from input file..."
tail -n +301 "$INPUT_FILE" | head -n 700 > "$RUN_DIR/input_subset.jsonl"

# Save configuration
cat > "$RUN_DIR/config.json" << EOF
{
    "input_file": "$RUN_DIR/input_subset.jsonl",
    "original_input": "$INPUT_FILE",
    "start_sample": $START_SAMPLE,
    "max_samples": $MAX_SAMPLES,
    "batch_size": $BATCH_SIZE,
    "max_turns": $MAX_TURNS,
    "model": "$MODEL",
    "timestamp": "$TIMESTAMP",
    "start_time": "$(date -Iseconds)",
    "cuda_settings": {
        "CUDA_LAUNCH_BLOCKING": "1",
        "TORCH_USE_CUDA_DSA": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512"
    }
}
EOF

echo -e "${GREEN}[INFO]${NC} Configuration:"
echo "  - Starting from sample: $START_SAMPLE"
echo "  - Samples to generate: $MAX_SAMPLES"
echo "  - Batch size: $BATCH_SIZE (reduced for stability)"
echo "  - Max turns: $MAX_TURNS"
echo "  - Model: $MODEL"
echo "  - CUDA error handling: ENABLED"
echo ""
echo -e "${YELLOW}[INFO]${NC} Estimated time: 6-8 hours"
echo ""

# Create tmux session script
cat > "$RUN_DIR/run_in_tmux.sh" << 'SCRIPT'
#!/bin/bash
# This script runs inside tmux

# Set CUDA environment
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Function to handle CUDA errors
handle_cuda_error() {
    echo "[ERROR] CUDA error detected, attempting recovery..."
    sleep 5
    # Clear CUDA cache
    python -c "import torch; torch.cuda.empty_cache()"
    nvidia-smi
}

# Trap errors
trap 'handle_cuda_error' ERR

# Run the generation with error recovery
while true; do
    python /root/KernelBench/claude_multi_turn_sft_with_recovery.py \
        --input "$1/input_subset.jsonl" \
        --api-key "$ANTHROPIC_API_KEY" \
        --output-dir "$1" \
        --max-samples 700 \
        --batch-size 10 \
        --max-turns 3 \
        --model "claude-sonnet-4-20250514" \
        --start-idx 301 \
        --verbose 2>&1 | tee -a "$1/generation.log"
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Generation completed successfully!"
        break
    elif [ $EXIT_CODE -eq 99 ]; then
        # Custom exit code for CUDA errors - retry
        echo "CUDA error encountered, retrying in 30 seconds..."
        sleep 30
        python -c "import torch; torch.cuda.empty_cache()"
    else
        echo "Generation failed with exit code: $EXIT_CODE"
        break
    fi
done

echo "Generation finished at: $(date)"
SCRIPT

chmod +x "$RUN_DIR/run_in_tmux.sh"

# Instructions for tmux
echo -e "${BLUE}================================================${NC}"
echo -e "${GREEN}Starting tmux session for generation...${NC}"
echo ""
echo -e "${YELLOW}Tmux Commands:${NC}"
echo "  • Detach from session: Ctrl+b, then d"
echo "  • List sessions: tmux ls"
echo "  • Reattach to session: tmux attach -t sft_generation"
echo "  • Kill session: tmux kill-session -t sft_generation"
echo ""
echo -e "${YELLOW}Monitoring:${NC}"
echo "  • In tmux: View live output"
echo "  • Outside: tail -f $RUN_DIR/generation.log"
echo "  • Progress: python monitor_generation.py"
echo ""
echo -e "${BLUE}================================================${NC}"

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} tmux is not installed"
    echo "Install with: apt-get update && apt-get install -y tmux"
    exit 1
fi

# Kill any existing session
tmux kill-session -t sft_generation 2>/dev/null || true

# Start tmux session
echo -e "${GREEN}[INFO]${NC} Starting tmux session 'sft_generation'..."
tmux new-session -d -s sft_generation \
    "cd /root/KernelBench && \
     export ANTHROPIC_API_KEY='$ANTHROPIC_API_KEY' && \
     python claude_multi_turn_sft.py \
        --input '$RUN_DIR/input_subset.jsonl' \
        --api-key '$ANTHROPIC_API_KEY' \
        --output-dir '$RUN_DIR' \
        --max-samples 700 \
        --batch-size 10 \
        --max-turns 3 \
        --model 'claude-sonnet-4-20250514' \
        --verbose 2>&1 | tee '$RUN_DIR/generation.log'"

echo ""
echo -e "${GREEN}✓ Tmux session 'sft_generation' started!${NC}"
echo ""
echo -e "${YELLOW}To attach to the session:${NC}"
echo "  tmux attach -t sft_generation"
echo ""
echo -e "${YELLOW}To monitor without attaching:${NC}"
echo "  tail -f $RUN_DIR/generation.log"
echo ""
echo -e "${GREEN}[INFO]${NC} Generation is running in background"
echo -e "${GREEN}[INFO]${NC} Output directory: $RUN_DIR"