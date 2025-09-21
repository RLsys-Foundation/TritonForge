#!/bin/bash
# Test script for multi-turn SFT data generation
# This demonstrates how to run the pipeline with small sample for testing

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}   Test Run: Multi-Turn SFT Data Generation${NC}"
echo -e "${BLUE}================================================${NC}"

# Test with just 2 samples to verify setup
export MAX_SAMPLES=2
export BATCH_SIZE=1
export MAX_TURNS=3
export OUTPUT_DIR=/root/KernelBench/multi_turn_sft_outputs/test_run

# Check for API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo -e "${YELLOW}[WARNING]${NC} ANTHROPIC_API_KEY not set"
    echo "To run this test:"
    echo "  export ANTHROPIC_API_KEY=your_api_key"
    echo "  ./test_run_multiturn_generation.sh"
    exit 1
fi

echo -e "${GREEN}[INFO]${NC} Running with test configuration:"
echo "  - Samples: $MAX_SAMPLES (for quick test)"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Max turns: $MAX_TURNS"
echo "  - Output: $OUTPUT_DIR"

# Run the generation
./generate_multiturn_sft.sh

echo -e "${BLUE}================================================${NC}"
echo -e "${GREEN}Test run complete!${NC}"
echo ""
echo "To generate 1000 samples, use:"
echo -e "${YELLOW}  export ANTHROPIC_API_KEY=your_api_key${NC}"
echo -e "${YELLOW}  export MAX_SAMPLES=1000${NC}"
echo -e "${YELLOW}  export BATCH_SIZE=20${NC}"
echo -e "${YELLOW}  ./generate_multiturn_sft.sh${NC}"