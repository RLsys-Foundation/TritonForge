# Multi-Turn SFT Data Generation Pipeline Guide

## Overview
This pipeline generates multi-turn supervised fine-tuning (SFT) data using Claude API for optimizing Triton kernels. It creates conversations where Claude iteratively improves kernel implementations based on evaluation feedback.

## Key Components

### 1. Main Script: `claude_multi_turn_sft.py`
- **Purpose**: Core generation logic using Claude API
- **Features**:
  - Multi-turn conversation generation (up to 3 turns)
  - Automatic evaluation using KernelBench framework
  - Iterative improvement based on compilation/correctness feedback
  - Automatic fixes for common model issues

### 2. Shell Scripts
- **`generate_multiturn_sft.sh`**: Main generation script
- **`test_run_multiturn_generation.sh`**: Test with 2 samples
- **`run_1000_samples.sh`**: Production script for 1000 samples

### 3. Monitoring Tool: `monitor_generation.py`
- Real-time progress tracking
- Statistics and completion estimates
- Live updates every 10 seconds

## How It Works

### Generation Process
1. **Initial Turn**: Claude generates first Triton kernel implementation
2. **Evaluation**: Kernel is evaluated for compilation and correctness
3. **Feedback Loop**: Evaluation results are provided to Claude
4. **Improvement**: Claude generates improved version based on feedback
5. **Repeat**: Process continues for up to 3 turns or until correctness achieved

### Automatic Fixes Applied
The pipeline automatically fixes common issues in original models:
- **Embedding inputs**: Converts `torch.rand()` to `torch.randint()` for embedding models
- **Initialization bounds**: Fixes `nn.init.uniform_()` to use proper range
- **CUDA placement**: Adds `.cuda()` to tensor creations

## Quick Start

### 1. Test Run (2 samples)
```bash
# Set your API key
export ANTHROPIC_API_KEY=your_api_key_here

# Run test
./test_run_multiturn_generation.sh
```

### 2. Production Run (1000 samples)
```bash
# Set your API key
export ANTHROPIC_API_KEY=your_api_key_here

# Run generation
./run_1000_samples.sh

# Monitor progress in another terminal
python monitor_generation.py
```

## Configuration Parameters

### Essential Parameters
- **MAX_SAMPLES**: Number of samples to generate (default: 100)
- **BATCH_SIZE**: Samples per batch (default: 20 for production)
- **MAX_TURNS**: Maximum conversation turns (default: 3)
- **MODEL**: Claude model to use (default: claude-sonnet-4-20250514)

### Performance Parameters
- **NUM_CORRECT_TRIALS**: Correctness verification trials (default: 5)
- **NUM_PERF_TRIALS**: Performance measurement trials (default: 20)

### Recommended Settings for 1000 Samples
```bash
export MAX_SAMPLES=1000      # Target samples
export BATCH_SIZE=20         # Efficient batch size
export MAX_TURNS=3           # 3 turns for good coverage
export MODEL="claude-sonnet-4-20250514"  # Latest Sonnet
```

## Time and Cost Estimates

### For 1000 Samples:
- **Estimated Time**: 8-10 hours
- **API Calls**: Up to 3000 (1000 samples × 3 turns)
- **Processing Rate**: ~30 seconds per sample (conservative)

### Factors Affecting Time:
1. Model complexity (simple models evaluate faster)
2. Compilation failures (require retries)
3. API response time
4. GPU availability for evaluation

## Output Files

### Directory Structure
```
multi_turn_sft_outputs/
└── run_YYYYMMDD_HHMMSS_1000samples/
    ├── config.json                    # Run configuration
    ├── generation.log                 # Detailed logs
    ├── batch_XXXX.jsonl              # Intermediate batch results
    ├── multi-turn-sft.jsonl          # Final SFT data (main output)
    ├── test_output_conversation.jsonl # Detailed conversations
    ├── multi-turn-sft.parquet        # Parquet format (optional)
    └── summary.json                  # Statistics summary
```

### Main Output: `multi-turn-sft.jsonl`
- Format: JSONL with conversation messages
- Ready for fine-tuning
- Each line contains a complete multi-turn conversation

## Monitoring and Debugging

### Live Monitoring
```bash
# Start monitoring tool
python monitor_generation.py

# Or watch logs directly
tail -f multi_turn_sft_outputs/run_*/generation.log
```

### Using Screen for Long Runs
The production script automatically uses `screen` if available:
```bash
# Detach from screen: Ctrl+A, D
# Reattach to screen
screen -r sft_generation

# List screen sessions
screen -ls
```

### Common Issues and Solutions

1. **API Key Error**
   ```bash
   export ANTHROPIC_API_KEY=your_key_here
   ```

2. **CUDA Not Available**
   - Ensure you're on a GPU machine
   - Check CUDA with: `nvidia-smi`

3. **Compilation Lock Issues**
   - Script automatically retries once
   - If persistent, restart the generation

4. **Memory Issues**
   - Reduce BATCH_SIZE (e.g., to 10 or 5)
   - Monitor GPU memory: `nvidia-smi -l 1`

## Data Quality Metrics

### Success Rates (from previous runs)
- **Compilation Rate**: ~95%
- **Correctness Rate**: ~60-70% (after 3 turns)
- **Average Turns**: 2.3

### Quality Indicators
- **Good**: Achieved correctness in 1-2 turns
- **Moderate**: Compiled but not correct after 3 turns
- **Poor**: Compilation failures across all turns

## Advanced Usage

### Custom Configuration
```bash
# Override any parameter
export MAX_SAMPLES=500
export BATCH_SIZE=10
export MAX_TURNS=5
export OUTPUT_DIR=/custom/path

./generate_multiturn_sft.sh
```

### Resume from Interruption
If generation is interrupted:
1. Check completed batches in output directory
2. Adjust MAX_SAMPLES to remaining count
3. Restart generation (will create new run directory)

### Parallel Generation
For faster processing with multiple API keys:
1. Split input data into chunks
2. Run multiple instances with different API keys
3. Merge results afterward

## Analysis Tools

### Load Generated Data
```python
import json

# Load SFT data
with open('multi-turn-sft.jsonl', 'r') as f:
    conversations = [json.loads(line) for line in f]

# Analyze conversations
for conv in conversations:
    messages = conv['messages']
    print(f"Turns: {len([m for m in messages if m['role'] == 'assistant'])}")
```

### Example Usage Script
See `load_sft_data_example.py` for loading and analyzing generated data.

## Best Practices

1. **Start Small**: Test with 2-10 samples first
2. **Monitor Progress**: Use monitoring tool for long runs
3. **Save API Key Securely**: Don't commit to version control
4. **Batch Processing**: Use appropriate batch size (10-20)
5. **Error Handling**: Check logs for failed samples
6. **Backup Results**: Copy successful runs before new attempts

## Support and Troubleshooting

### Log Files
- **generation.log**: Main execution log
- **batch_XXXX.jsonl**: Individual batch results
- Check for error messages and stack traces

### Performance Optimization
- Adjust BATCH_SIZE based on available memory
- Use latest Claude model for best results
- Ensure stable internet connection

### Getting Help
1. Check FIXES_SUMMARY.md for recent improvements
2. Review generation.log for specific errors
3. Verify CUDA and dependencies are installed