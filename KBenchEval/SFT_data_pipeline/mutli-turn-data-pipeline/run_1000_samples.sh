#!/bin/bash
# Production script for generating 1000 multi-turn SFT samples
# Optimized parameters for large-scale generation with Claude API

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}   Production Run: 1000 Multi-Turn SFT Samples${NC}"
echo -e "${BLUE}================================================${NC}"

# Production configuration for 1000 samples
export INPUT_FILE="${INPUT_FILE:-/root/kernel_book/kernelbook_sft_format.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-/root/KernelBench/multi_turn_sft_outputs}"
export MAX_SAMPLES=1000       # Generate 1000 samples
export BATCH_SIZE=20          # Process 20 at a time for efficiency
export MAX_TURNS=3            # 3 turns per conversation (initial + 2 improvements)
export MODEL="claude-sonnet-4-20250514"  # Latest Claude 3.5 Sonnet

# Performance settings
export NUM_CORRECT_TRIALS=5   # Number of correctness trials
export NUM_PERF_TRIALS=20     # Number of performance trials

# Check for API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo -e "${RED}[ERROR]${NC} ANTHROPIC_API_KEY environment variable not set"
    echo ""
    echo "Please set your API key:"
    echo -e "${YELLOW}  export ANTHROPIC_API_KEY=your_api_key${NC}"
    echo -e "${YELLOW}  ./run_1000_samples.sh${NC}"
    exit 1
fi

# Estimate time and cost
echo -e "${GREEN}[INFO]${NC} Configuration:"
echo "  - Input: $INPUT_FILE"
echo "  - Output: $OUTPUT_DIR"
echo "  - Samples: $MAX_SAMPLES"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Max turns: $MAX_TURNS"
echo "  - Model: $MODEL"
echo ""

# Calculate estimates
BATCHES=$((MAX_SAMPLES / BATCH_SIZE))
AVG_TIME_PER_SAMPLE=30  # seconds (conservative estimate)
TOTAL_TIME=$((MAX_SAMPLES * AVG_TIME_PER_SAMPLE))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))

echo -e "${YELLOW}[ESTIMATE]${NC} Time required:"
echo "  - Batches to process: $BATCHES"
echo "  - Estimated time: ${HOURS}h ${MINUTES}m"
echo "  - API calls: ~$((MAX_SAMPLES * MAX_TURNS)) (up to 3000 calls)"
echo ""

# Prompt for confirmation
read -p "Do you want to proceed? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}[INFO]${NC} Generation cancelled"
    exit 0
fi

# Create output directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="$OUTPUT_DIR/run_${TIMESTAMP}_1000samples"
mkdir -p "$RUN_DIR"

# Save configuration
cat > "$RUN_DIR/config.json" << EOF
{
    "input_file": "$INPUT_FILE",
    "max_samples": $MAX_SAMPLES,
    "batch_size": $BATCH_SIZE,
    "max_turns": $MAX_TURNS,
    "model": "$MODEL",
    "timestamp": "$TIMESTAMP",
    "start_time": "$(date -Iseconds)",
    "estimated_hours": $HOURS,
    "estimated_minutes": $MINUTES
}
EOF

echo -e "${GREEN}[INFO]${NC} Starting generation at $(date)"
echo -e "${GREEN}[INFO]${NC} Output will be saved to: $RUN_DIR"
echo ""

# Run with screen for long-running process
if command -v screen &> /dev/null; then
    echo -e "${GREEN}[INFO]${NC} Running in screen session 'sft_generation'"
    echo "  To detach: Ctrl+A, D"
    echo "  To reattach: screen -r sft_generation"
    echo ""
    screen -dmS sft_generation bash -c "
        python /root/KernelBench/claude_multi_turn_sft.py \
            --input '$INPUT_FILE' \
            --api-key '$ANTHROPIC_API_KEY' \
            --output-dir '$RUN_DIR' \
            --max-samples $MAX_SAMPLES \
            --batch-size $BATCH_SIZE \
            --max-turns $MAX_TURNS \
            --model '$MODEL' \
            --verbose 2>&1 | tee '$RUN_DIR/generation.log'
        echo 'Generation complete at: \$(date)' >> '$RUN_DIR/generation.log'
    "
    echo -e "${GREEN}[INFO]${NC} Generation started in background"
    echo "  Monitor progress: tail -f $RUN_DIR/generation.log"
else
    # Run directly if screen not available
    python /root/KernelBench/claude_multi_turn_sft.py \
        --input "$INPUT_FILE" \
        --api-key "$ANTHROPIC_API_KEY" \
        --output-dir "$RUN_DIR" \
        --max-samples $MAX_SAMPLES \
        --batch-size $BATCH_SIZE \
        --max-turns $MAX_TURNS \
        --model "$MODEL" \
        --verbose 2>&1 | tee "$RUN_DIR/generation.log"
fi