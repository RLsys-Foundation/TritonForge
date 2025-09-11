# Single-Turn vs Multi-Turn Kernel Generation Configuration

## Key Parameter Changes for Single-Turn Training

### 1. Task Type Changes
**Multi-turn:**
```bash
--rm-type kernelbench_multiturn
--rollout-task-type kernelbench_multiturn
```

**Single-turn:**
```bash
--rm-type kernelbench
--rollout-task-type kernelbench
```

### 2. Multi-turn Specific Parameters (REMOVED)
```bash
# These parameters are ONLY for multi-turn and should be removed:
--max-turns 3      # Number of improvement iterations
--gamma 0.4        # Discount factor for aggregated rewards
```

### 3. Reduced Batch Sizes for Debugging
| Parameter | Multi-turn Value | Single-turn Debug Value | Purpose |
|-----------|------------------|------------------------|----------|
| `--num-rollout` | 1000 | 500 | Total rollouts per epoch |
| `--rollout-batch-size` | 4 | 2 | Batch size for rollout generation |
| `--n-samples-per-prompt` | 8 | 4 | Samples generated per prompt |
| `--global-batch-size` | 32 | 16 | Global training batch size |
| `--max-tokens-per-gpu` | 4096 | 2048 | Max tokens per GPU for dynamic batching |

## How the System Automatically Selects the Generator

The buffer server (`slime_plugins/rollout_buffer/buffer.py`) automatically selects the correct generator based on the `task_type`:

1. When `--rollout-task-type kernelbench` is specified, the buffer server:
   - Registers task type as `"kernelbench"`
   - Imports and uses `KernelGenerator` from `kernel_generator.py`
   - Performs single-turn generation with direct reward

2. When `--rollout-task-type kernelbench_multiturn` is specified:
   - Registers task type as `"kernelbench_multiturn"`
   - Imports and uses `MultiTurnKernelGenerator` from `multi_turn_kernel_generator.py`
   - Performs multi-turn generation with aggregated rewards

## File Locations and Usage

### Scripts Created for Single-Turn:
- **Training Script**: `/home/jinpan12/workspace/slime/scripts/agent-kbench-qwen3-8B-sft-amd-singleturn.sh`
- **Launcher Script**: `/home/jinpan12/workspace/slime/scripts/run_agent_kbench_qwen3_8B_sft_amd_singleturn.sh`

### Generator Files:
- **Single-turn**: `slime_plugins/rollout_buffer/generator/kernel_generator.py`
  - Class: `KernelGenerator`
  - Function: `rollout_one_trajectory()`
  - Reward: Direct from single evaluation

- **Multi-turn**: `slime_plugins/rollout_buffer/generator/multi_turn_kernel_generator.py`
  - Class: `MultiTurnKernelGenerator`
  - Function: `rollout_multi_turn_trajectory()`
  - Reward: Aggregated with discount factor

## Running the Single-Turn Training

```bash
# Make scripts executable
chmod +x /home/jinpan12/workspace/slime/scripts/run_agent_kbench_qwen3_8B_sft_amd_singleturn.sh
chmod +x /home/jinpan12/workspace/slime/scripts/agent-kbench-qwen3-8B-sft-amd-singleturn.sh

# Run the launcher
bash /home/jinpan12/workspace/slime/scripts/run_agent_kbench_qwen3_8B_sft_amd_singleturn.sh
```

## Monitoring and Debugging

### Log Files:
- Training: `/home/jinpan12/workspace/slime/logs/slime_qwen3_sft_amd_singleturn_train.log`
- Buffer: `/home/jinpan12/workspace/slime/logs/buffer_qwen3_sft_amd_singleturn_log`
- Eval Server: `/home/jinpan12/workspace/slime/logs/eval_server_qwen3_sft_amd_singleturn.log`

### Health Checks:
```bash
# Check evaluation server health
curl http://localhost:18188/health

# Check fault statistics
curl http://localhost:18188/fault_statistics

# Monitor Ray cluster
ray status
```

## Differences in Reward Calculation

### Single-turn:
- Simple reward based on compilation, correctness, and performance
- No iteration or improvement attempts
- Faster but potentially lower quality

### Multi-turn:
- Aggregated return: `R = Σ(γ^t * r_t)` where γ=0.4, t=turn index
- Allows iterative improvement based on feedback
- Higher computational cost but potentially better results

## Troubleshooting

If you encounter issues:

1. **OOM Errors**: Further reduce batch sizes:
   ```bash
   --rollout-batch-size 1
   --n-samples-per-prompt 2
   --global-batch-size 8
   ```

2. **Generator Not Found**: Verify task type matches:
   - Check `--rollout-task-type` in training script
   - Ensure it's either `kernelbench` or `kernelbench_multiturn`

3. **Evaluation Failures**: Check eval server is running:
   ```bash
   curl http://localhost:18188/health
   ```

4. **Ray Issues**: Restart Ray cluster:
   ```bash
   ray stop --force
   ray start --head --num-gpus 6
   ```