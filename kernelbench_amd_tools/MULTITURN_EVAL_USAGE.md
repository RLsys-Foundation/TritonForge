# Multi-Turn Evaluation for Qwen3-8B

## Overview
The multi-turn evaluation script (`run_qwen3_evaluation_robust_multiturn.py`) mimics the exact multi-turn generation logic used in SLIME RL training. This allows for accurate debugging and evaluation of how the model performs during multi-turn refinement.

## Key Features

### 1. Exact RL Training Replication
- **Message Accumulation**: Uses native template mode where messages accumulate across turns
- **Improvement Instructions**: Adds user messages between turns with specific improvement feedback
- **Reward Calculation**: Same reward structure as training (compilation: 0.1, correctness: 1.0, performance bonus: up to 2.0)
- **Aggregated Return**: Uses gamma=0.4 discount factor like in training

### 2. Multi-Turn Flow (Matching RL Training)
```
Turn 1: Initial generation from prompt
↓ Evaluate → Calculate reward
↓ Add improvement instruction (if not satisfactory)
Turn 2: Generate improved version based on feedback
↓ Evaluate → Calculate reward  
↓ Add improvement instruction (if needed)
Turn 3: Final refinement attempt
↓ Calculate aggregated return: R = r₁ + 0.4*r₂ + 0.16*r₃
```

### 3. Early Termination
- Stops if reward ≥ 2.0 (correctness + good performance)
- Matches the RL training's early termination logic

## Usage

### Basic Usage
```bash
cd /home/jinpan12/workspace/KernelBench/kernelbench_amd_tools/scripts

# Make executable
chmod +x run_qwen3_evaluation_robust_multiturn.py

# Run with default settings (3 turns, gamma=0.4)
python run_qwen3_evaluation_robust_multiturn.py --levels 1 --max-problems 10
```

### Advanced Options
```bash
# Custom configuration matching specific RL training
python run_qwen3_evaluation_robust_multiturn.py \
    --levels 1,2 \
    --max-problems 50 \
    --max-turns 3 \
    --gamma 0.4 \
    --timeout 60 \
    --run-name "debug_rl_training"

# Start from specific problem (useful for resuming)
python run_qwen3_evaluation_robust_multiturn.py \
    --levels 1 \
    --start-from 11 \
    --max-problems 20
```

### Command-Line Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--levels` | "1,2" | Comma-separated problem levels |
| `--max-problems` | None | Max problems per level (None = all) |
| `--start-from` | None | Start from problem number |
| `--run-name` | Auto | Custom run name for results |
| `--max-turns` | 3 | Maximum refinement turns |
| `--gamma` | 0.4 | Discount factor for aggregated return |
| `--timeout` | 60 | Evaluation timeout per turn (seconds) |
| `--no-subprocess` | False | Disable subprocess isolation (not recommended) |
| `--no-native-template` | False | Disable native template mode (not recommended) |

## Output Structure

### Directory Layout
```
/home/jinpan12/workspace/KernelBench/runs/<run_name>/
├── reports/
│   └── MULTI_TURN_REPORT.md      # Comprehensive analysis
├── generated_kernels/
│   ├── level1_problem1_turn0.py  # Generated code per turn
│   ├── level1_problem1_turn1.py
│   └── ...
├── responses/
│   ├── level1_problem1_turn0.txt # Raw model responses
│   └── ...
├── turns/
│   ├── level1_problem1_turn0.json # Detailed turn data
│   └── ...
├── results.json                   # Complete results
└── progress.json                  # Progress tracking
```

### Results Analysis

The multi-turn report includes:
1. **Overall Success Metrics**: Compilation, correctness, multi-turn success rates
2. **Turn Analysis**: Average reward by turn position, improvement rates
3. **Best Performers**: Problems with highest aggregated returns
4. **Error Analysis**: Memory faults, timeouts, other errors

## Comparing with RL Training

### Message Template Consistency
The script uses the same JSONL templates from:
```
/home/jinpan12/workspace/slime/data/kernel_bench/kernel_bench_triton_level_1_2.jsonl
```

### Improvement Message Format (Exact Match)
```python
# Turn 2+ improvement instruction (same as RL training)
"Based on the previous attempt above, generate an improved kernel that:
1. Fixes the compilation errors  # OR "Fixes correctness issues" OR "Maintains correctness"
2. Improves performance if possible
3. Maintains the same functionality as required

Error from previous attempt: <error_message>

Please generate the improved kernel code:"
```

### Reward Calculation (Identical)
```python
reward = 0.0
if compiled:
    reward = 0.1  # Compilation reward
    if correct:
        reward = 1.0  # Correctness reward
        if runtime > 0 and baseline_runtime:
            speedup = baseline_runtime / runtime
            performance_bonus = min(max(speedup - 1.0, 0.0), 2.0)
            reward += performance_bonus
```

## Debugging RL Training Issues

### 1. Check Message Accumulation
```bash
# Look at turn JSON files to verify message structure
cat runs/*/turns/level1_problem1_turn*.json | jq '.messages | length'
```

### 2. Analyze Reward Progression
```bash
# Extract turn rewards for analysis
cat runs/*/results.json | jq '.problems | .[] | {problem: .problem_name, rewards: .turn_rewards}'
```

### 3. Monitor Thinking Tag Behavior
```bash
# Check if thinking tags appear and are handled correctly
grep -l "<think>" runs/*/responses/*.txt | wc -l
```

### 4. Compare with Single-Turn
Run both single-turn and multi-turn evaluation on the same problems:
```bash
# Single-turn
python run_qwen3_evaluation_robust.py --levels 1 --max-problems 10 --run-name "single_turn_test"

# Multi-turn
python run_qwen3_evaluation_robust_multiturn.py --levels 1 --max-problems 10 --run-name "multi_turn_test"

# Compare results
diff runs/single_turn_test/results.json runs/multi_turn_test/results.json
```

## Key Differences from Single-Turn

| Aspect | Single-Turn | Multi-Turn |
|--------|-------------|------------|
| Attempts | 1 generation | Up to 3 refinements |
| Message History | Reset each problem | Accumulates across turns |
| Reward | Direct reward | Aggregated with gamma discount |
| Feedback | None | Improvement instructions between turns |
| Early Stop | N/A | Stops if reward ≥ 2.0 |
| Chat Template | Simple prompt | Native template with accumulation |

## Troubleshooting

### SGLang Not Running
```bash
# Check if SGLang is running
curl http://localhost:30000/health

# Start SGLang if needed
cd /path/to/sglang
python -m sglang.launch_server --model-path Qwen/Qwen3-8B --port 30000
```

### Memory Issues
```bash
# Reduce concurrent evaluations
export CUDA_VISIBLE_DEVICES=0  # Use single GPU
python run_qwen3_evaluation_robust_multiturn.py --timeout 30
```

### Subprocess Crashes
```bash
# Check error details in turn files
cat runs/*/turns/*_turn*.json | jq '.history_entry.eval_result.error_message'
```

## Integration with RL Training

This evaluation script is designed to exactly replicate the behavior in:
- `/home/jinpan12/workspace/slime/slime_plugins/rollout_buffer/generator/multi_turn_kernel_generator.py`

Key matching points:
1. `rollout_multi_turn_trajectory()` function logic
2. `construct_multi_turn_prompt()` message building  
3. `calculate_aggregated_return()` with gamma
4. Early termination conditions
5. Improvement message construction

Use this script to debug issues with multi-turn RL training by comparing:
- Message accumulation patterns
- Reward calculations
- Turn-to-turn improvements
- Early termination behavior