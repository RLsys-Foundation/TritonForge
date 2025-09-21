#!/bin/bash
# Generate multi-turn SFT data from KernelBook using Claude API
# This script processes samples and creates training-ready conversations

# Configuration
INPUT_FILE="${INPUT_FILE:-/root/kernel_book/kernelbook_sft_format.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/KernelBench/multi_turn_sft_outputs}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"  # Default to 100 samples
BATCH_SIZE="${BATCH_SIZE:-5}"      # Process 20 samples at a time
MAX_TURNS="${MAX_TURNS:-3}"         # Maximum 3 turns per conversation
MODEL="${MODEL:-claude-sonnet-4-20250514}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check for API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
    print_error "ANTHROPIC_API_KEY environment variable not set"
    echo "Usage: ANTHROPIC_API_KEY=your_key ./generate_multiturn_sft.sh"
    echo ""
    echo "Options (via environment variables):"
    echo "  MAX_SAMPLES   - Number of samples to process (default: 100)"
    echo "  BATCH_SIZE    - Batch size for processing (default: 20)"
    echo "  MAX_TURNS     - Maximum turns per conversation (default: 3)"
    echo "  OUTPUT_DIR    - Output directory (default: ./multi_turn_sft_outputs)"
    exit 1
fi

# Create output directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="$OUTPUT_DIR/run_${TIMESTAMP}_${MAX_SAMPLES}samples"
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
    "start_time": "$(date -Iseconds)"
}
EOF

# Print configuration
echo "================================================"
echo "   Multi-Turn SFT Data Generation Pipeline"
echo "================================================"
print_info "Input file: $INPUT_FILE"
print_info "Output directory: $RUN_DIR"
print_info "Number of samples: $MAX_SAMPLES"
print_info "Batch size: $BATCH_SIZE"
print_info "Max turns: $MAX_TURNS"
print_info "Model: $MODEL"
print_info "Start time: $(date)"
echo "================================================"

# Check if virtual environment exists
if [ -d "/root/KernelBench/.venv" ]; then
    print_info "Activating virtual environment..."
    source /root/KernelBench/.venv/bin/activate
else
    print_warning "Virtual environment not found, using system Python"
fi

# Check dependencies
print_info "Checking dependencies..."
python -c "import anthropic, torch, pandas, pyarrow" 2>/dev/null
if [ $? -ne 0 ]; then
    print_warning "Some dependencies might be missing"
fi

# Run the generation pipeline
print_info "Starting generation pipeline..."
# Use pipefail to capture Python script exit status even with tee
set -o pipefail
python /root/KernelBench/claude_multi_turn_sft.py \
    --input "$INPUT_FILE" \
    --api-key "$ANTHROPIC_API_KEY" \
    --output-dir "$RUN_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --batch-size "$BATCH_SIZE" \
    --max-turns "$MAX_TURNS" \
    --model "$MODEL" \
    --verbose 2>&1 | tee "$RUN_DIR/generation.log"

GENERATION_STATUS=$?
set +o pipefail

# Check results
if [ $GENERATION_STATUS -eq 0 ]; then
    echo ""
    echo "================================================"
    print_info "Generation completed successfully!"
    echo "================================================"
    
    # The run directory is now the output directory itself
    ACTUAL_RUN_DIR="$RUN_DIR"
    
    if [ -d "$ACTUAL_RUN_DIR" ]; then
        # Count and display results
        if [ -f "$ACTUAL_RUN_DIR/multi-turn-sft.jsonl" ]; then
            NUM_GENERATED=$(wc -l < "$ACTUAL_RUN_DIR/multi-turn-sft.jsonl")
            print_info "Generated $NUM_GENERATED multi-turn conversations"
            
            # Show file sizes
            echo ""
            print_info "Output files:"
            ls -lh "$ACTUAL_RUN_DIR"/*.jsonl 2>/dev/null | while read line; do
                echo "  $line"
            done
            
            if [ -f "$ACTUAL_RUN_DIR/multi-turn-sft.parquet" ]; then
                ls -lh "$ACTUAL_RUN_DIR"/*.parquet | while read line; do
                    echo "  $line"
                done
            fi
            
            # Show summary if exists
            if [ -f "$ACTUAL_RUN_DIR/summary.json" ]; then
                echo ""
                print_info "Summary statistics:"
                python -c "
import json
with open('$ACTUAL_RUN_DIR/summary.json') as f:
    data = json.load(f)
    if 'statistics' in data:
        stats = data['statistics']
        print(f'  Average turns: {stats.get(\"avg_turns\", 0):.1f}')
        print(f'  Compilation rate: {stats.get(\"final_compilation_rate\", 0):.1%}')
        print(f'  Correctness rate: {stats.get(\"final_correctness_rate\", 0):.1%}')
" 2>/dev/null
            fi
        fi
        
        echo ""
        print_info "Results saved to: $ACTUAL_RUN_DIR"
        print_info "Main SFT file: $ACTUAL_RUN_DIR/multi-turn-sft.jsonl"
        print_info "Detailed conversations: $ACTUAL_RUN_DIR/test_output_conversation.jsonl"
        
        if [ -f "$ACTUAL_RUN_DIR/multi-turn-sft.parquet" ]; then
            print_info "Parquet file: $ACTUAL_RUN_DIR/multi-turn-sft.parquet"
        fi
    fi
    
    # Save completion time
    echo "$(date -Iseconds)" > "$RUN_DIR/completion_time.txt"
    
else
    echo ""
    echo "================================================"
    print_error "Generation failed with status code: $GENERATION_STATUS"
    echo "================================================"
    print_error "Check the log file for details: $RUN_DIR/generation.log"
    exit 1
fi

echo ""
print_info "End time: $(date)"
echo "================================================"