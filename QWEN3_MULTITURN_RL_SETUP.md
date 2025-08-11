# Qwen3-8B Multi-turn RL Training Setup

## Overview
This document describes the setup for multi-turn reinforcement learning (RL) post-training of a fine-tuned Qwen3-8B model on KernelBook data for PyTorch → Triton kernel generation.

## Key Features
- **Model**: Qwen3-8B fine-tuned on KernelBook dataset  
- **Training Method**: Multi-turn GRPO (Group Relative Policy Optimization)
- **Task**: Generate optimized Triton kernels from PyTorch operations
- **Multi-turn Config**: 
  - Max turns: 3
  - Discount factor (γ): 0.4
  - Context accumulation from previous attempts
  - Early termination on success or repeated failures

## Prerequisites

### 1. Model Checkpoints
- **Megatron checkpoint**: `/root/Megatron-models/qwen3-8b-kernelbook-sft-megatron` (your fine-tuned model)
- **HuggingFace checkpoint**: `/root/Megatron-models/qwen3-8b-kernelbook-sft-hf` (will be created from Megatron)
- **Original Qwen3-8B**: `/root/Qwen3-8B` (needed for tokenizer and config)

### 2. Data
- KernelBench dataset at `/root/slime/data/kernel_bench/kernel_bench_triton_level_1_2.jsonl`

### 3. Environment
- Docker image: `zhuzilin/slime:latest`
- At least 6 GPUs (4 for training, 2 for rollout generation)
- Ray cluster for distributed execution

## Setup Steps

### Step 1: Verify Setup
Run the verification script to check all components:
```bash
cd /root/slime
python scripts/verify_qwen3_setup.py
```

This will check:
- Model checkpoints existence
- Data files availability  
- Multi-turn support components
- GPU availability
- Script permissions

### Step 2: Convert Megatron to HuggingFace Format
SGLang requires HuggingFace format for inference. Convert your Megatron checkpoint:
```bash
bash scripts/convert_qwen3_megatron_to_hf.sh
```

This creates the HF model at `/root/Megatron-models/qwen3-8b-kernelbook-sft-hf`

### Step 3: Start Training
Launch the complete training pipeline with tmux session management:
```bash
bash scripts/run_agent_kbench_qwen3_8B.sh
```

This starts three tmux windows:
1. **slime**: Main training process
2. **buffer**: Rollout buffer server
3. **eval_server**: KernelBench evaluation server

## Configuration Details

### Model Architecture (Qwen3-8B)
```bash
--num-layers 36
--hidden-size 4096
--ffn-hidden-size 12288
--num-attention-heads 32
--num-query-groups 8  # GQA
--vocab-size 151936
--qk-layernorm  # Qwen3 specific
--rotary-base 1000000
```

### Parallelism Configuration
```bash
--tensor-model-parallel-size 2 (TP)
--pipeline-model-parallel-size 1 (PP)
--context-parallel-size 2 (CP)
# Total: 2 * 1 * 2 = 4 GPUs for training
```

### Multi-turn RL Parameters
```bash
--rm-type kernelbench_multiturn
--rollout-task-type kernelbench_multiturn
--max-turns 3
--gamma 0.4
--loss-mask-type qwen  # Qwen-specific loss masking
--use-native-chat-template  # Use Qwen's native template
```

### Training Hyperparameters
```bash
--lr 1e-6  # Conservative for fine-tuned model
--kl-loss-coef 0.01  # Prevent divergence from SFT
--entropy-coef 0.001  # Small exploration bonus
--global-batch-size 16
--rollout-batch-size 4
--n-samples-per-prompt 4  # Generate 4 attempts per problem
```

## Key Differences from KernelLLM Setup

1. **Loss Mask Type**: Changed from `kernelllm` to `qwen`
2. **Model Architecture**: Qwen3-specific features (qk-layernorm, larger vocab)
3. **Chat Template**: Added `--use-native-chat-template` for Qwen format
4. **Checkpoint Paths**: Using your fine-tuned Megatron checkpoint
5. **Learning Rate**: More conservative (1e-6 vs 5e-7) since starting from fine-tuned model

## Monitoring Training

### Tmux Session Management
```bash
# Attach to running session
tmux attach -t slime_qwen3_run

# Navigate windows
Ctrl+b, n  # Next window
Ctrl+b, p  # Previous window
Ctrl+b, 0/1/2  # Jump to window by number

# Detach from session (keeps running)
Ctrl+b, d
```

### Log Files
- Training log: `/root/slime/qwen3_train.log`
- Buffer log: `/root/slime/qwen3_buffer.log`
- Eval server log: `/root/slime/qwen3_eval_server.log`
- Multi-turn details: `/workspace/slime/multi_turn_logs/`

### WandB Monitoring
Configure your WandB key in the script:
```bash
export WANDB_KEY="your_key_here"
```
Project: `slime-multiturn-qwen3-8B`

## Multi-turn Training Flow

1. **Turn 1**: Model generates initial kernel from problem description
2. **Evaluation**: Kernel is tested for compilation, correctness, and performance
3. **Turn 2**: If needed, model sees previous attempt + errors, generates improved kernel
4. **Turn 3**: Final attempt with full context of previous failures
5. **Reward Aggregation**: R = r₁ + γ*r₂ + γ²*r₃ (γ=0.4)

### Reward Structure
- Compilation success: 0.1
- Correctness: 0.3  
- Performance: Additional reward based on speedup vs baseline
- Early termination if correctness + good performance achieved

## Troubleshooting

### Common Issues

1. **"HuggingFace model not found"**
   - Run the conversion script: `bash scripts/convert_qwen3_megatron_to_hf.sh`

2. **"Megatron checkpoint not found"**
   - Ensure your fine-tuned checkpoint is at `/root/Megatron-models/qwen3-8b-kernelbook-sft-megatron`

3. **Ray cluster issues**
   - Kill existing processes: `pkill -9 ray; pkill -9 python`
   - Restart Ray: `ray start --head --num-gpus 6`

4. **Out of memory**
   - Reduce `--rollout-batch-size` (currently 4)
   - Reduce `--global-batch-size` (currently 16)
   - Reduce `--max-tokens-per-gpu` (currently 4096)

5. **Evaluation server not responding**
   - Check KernelBench installation
   - Verify CUDA devices in eval server command
   - Check eval server log for errors

## Output Checkpoints

Trained checkpoints will be saved at:
```
/root/Megatron-models/qwen3-8b-kernelbook-rl-megatron/
```

Checkpoints are saved every 50 iterations (configurable via `--save-interval`).

## Next Steps

After training:
1. Convert final checkpoint to HF format for deployment
2. Evaluate on held-out KernelBench problems
3. Compare performance against baseline and single-turn models
4. Fine-tune hyperparameters based on results

## Summary

This setup implements multi-turn RL post-training for Qwen3-8B, building on your SFT checkpoint to improve kernel generation through iterative refinement. The system learns from compilation errors and performance feedback to generate increasingly optimized Triton kernels.