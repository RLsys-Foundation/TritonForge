# Multi-Turn Training Logging Documentation

## Overview

The multi-turn kernel generator now includes comprehensive logging functionality to track the progression and effects of each turn during training. This helps analyze:
- How models improve across turns
- Which turns achieve correctness vs performance improvements  
- The effectiveness of the multi-turn approach

## Features

### 1. Turn-by-Turn Logging
Each turn in the multi-turn dialogue is logged with:
- Generated kernel code
- Evaluation results (compilation, correctness, runtime)
- Reward received
- Error messages if any
- Speedup compared to baseline

### 2. Trajectory Logging
Complete trajectories are logged with:
- Full conversation history
- All turn rewards and aggregated return
- Final execution details
- Performance progression across turns

### 3. Console Output
Real-time console output shows:
- Concise turn summaries with ✓/✗ status indicators
- Performance metrics (runtime, speedup, reward)
- Final trajectory summaries

## Log File Structure

### Log Directory
All logs are saved to: `/workspace/slime/multi_turn_logs/`

### File Naming Convention
- Turn logs: `turn_{instance_id}_t{turn_idx}_{timestamp}.json`
- Trajectory logs: `trajectory_{instance_id}_{timestamp}.json`

### Turn Log Format
```json
{
  "timestamp": "20250807_084801_411804",
  "instance_id": "kernelbench_1",
  "turn_idx": 0,
  "turn_data": {
    "prompt": [...],
    "response": "...",
    "kernel_code": "...",
    "eval_result": {
      "compiled": true,
      "correctness": false,
      "runtime": 5.234,
      "speedup": 0.8,
      "error_message": "..."
    },
    "reward": 0.5,
    "extra_info": {...}
  }
}
```

### Trajectory Log Format
```json
{
  "timestamp": "20250807_084801_412128",
  "instance_id": "kernelbench_1",
  "final_reward": 2.5,
  "num_turns": 3,
  "turn_rewards": [0.5, 1.0, 3.0],
  "aggregated_return": 2.5,
  "history": [...],
  "messages": [...],
  "execution_details": {
    "final_compiled": true,
    "final_correctness": true,
    "final_runtime": 1.567,
    "final_speedup": 2.5
  },
  "extra_info": {...}
}
```

## Console Output Examples

### Turn Summary
```
[Turn 1] kernelbench_1: Compile:✓ Correct:✗ Runtime:5.23ms Speedup:0.80x Reward:0.500
[Turn 2] kernelbench_1: Compile:✓ Correct:✓ Runtime:3.12ms Speedup:1.20x Reward:1.200
[Turn 3] kernelbench_1: Compile:✓ Correct:✓ Runtime:1.57ms Speedup:2.50x Reward:3.000
```

### Trajectory Summary
```
=== Final Trajectory Summary for kernelbench_1 ===
Total turns: 3
Turn rewards: [0.5, 1.2, 3.0]
Aggregated return: 2.5000
Final correctness: True
Final speedup: 2.50x
```

## Usage

### Enable/Disable Logging
Edit `/workspace/slime/slime_plugins/rollout_buffer/generator/multi_turn_kernel_generator.py`:
```python
ENABLE_DETAILED_LOGGING = True  # Set to False to disable
```

### Change Log Directory
```python
LOG_DIR = "/workspace/slime/multi_turn_logs"  # Change path as needed
```

### Run Training with Logging
```bash
# Standard multi-turn training (logs enabled by default)
cd /workspace/slime
./scripts/run_agent_kbench_kernelllm_8B.sh
```

### Analyze Logs
```bash
# Analyze all logs in directory
python scripts/analyze_multi_turn_logs.py

# Analyze specific directory
python scripts/analyze_multi_turn_logs.py --log-dir /path/to/logs

# Save analysis to specific file
python scripts/analyze_multi_turn_logs.py --output analysis.json
```

## Analysis Output

The analysis script provides:
1. **Summary Statistics**
   - Average turns per trajectory
   - Average aggregated return
   - Final correctness rate
   - Improvement rate
   - Average speedup

2. **Turn-by-Turn Analysis**
   - Average reward by turn
   - Success rates per turn
   - Performance progression

3. **Trajectory Details**
   - Top performing trajectories
   - Correctness progression
   - Reward improvements

## Testing

Run the test script to verify logging functionality:
```bash
python scripts/test_multi_turn_logging.py
```

## Integration with Existing Code

The logging is integrated into:
1. `multi_turn_kernel_generator.py` - Core logging implementation
2. `rollout_multi_turn_trajectory()` - Logs each turn
3. `worker_process_multi_turn()` - Logs final trajectories

## Benefits

1. **Training Insights**: Understand how models improve across turns
2. **Debugging**: Identify where failures occur in multi-turn dialogue
3. **Performance Analysis**: Track speedup improvements
4. **Research**: Analyze effectiveness of multi-turn vs single-turn approaches
5. **Reproducibility**: Complete record of training progression

## Notes

- Logs are append-only (won't overwrite existing logs)
- Timestamps include microseconds to ensure unique filenames
- JSON format for easy parsing and analysis
- Minimal performance overhead (~1-2ms per log write)
- Automatic directory creation if it doesn't exist