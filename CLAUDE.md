# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

**slime** is an advanced LLM post-training framework for reinforcement learning scaling, bridging high-performance distributed training (Megatron-LM) with flexible data generation workflows (SGLang). Currently specialized for multi-turn reinforcement learning in PyTorch → Triton kernel generation tasks.

## Common Development Commands

### Environment Setup
```bash
# Using Docker (recommended)
docker run --rm --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -it zhuzilin/slime:latest /bin/bash

# Install slime
git clone https://github.com/THUDM/slime.git
cd slime
pip install -e .
```

### Running Tests
```bash
# Run pytest tests
pytest tests/

# Run specific test script (e.g., Qwen3 0.6B test)
bash tests/test_qwen3_0.6B.sh

# Test multi-turn kernel generation
python scripts/test_multi_turn_kernel.py
```

### Code Quality
```bash
# Format code with black
black . --line-length 119

# Run linting with ruff
ruff check . --line-length 119

# Pre-commit hooks (install once)
apt install pre-commit -y
pre-commit install
```

### Training Jobs
```bash
# Start Ray cluster (on head node)
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats

# Submit training job
ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json='{"env_vars": {"PYTHONPATH": "/root/Megatron-LM"}}' \
  -- python3 train.py [args...]

# Run kernel generation agent training
bash scripts/run_agent_kbench_kernelllm_8B.sh
```

### Checkpoint Conversion
```bash
# HF → Megatron torch_dist
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
  --hf-checkpoint /path/to/hf_model \
  --save /path/to/torch_dist_model

# Megatron → HF
PYTHONPATH=/root/Megatron-LM python tools/convert_torch_dist_to_hf.py \
  --input-dir /path/to/torch_dist/iter_xxx/ \
  --output-dir /path/to/hf_output \
  --origin-hf-dir /path/to/original_hf_model
```

## Architecture Overview

### Core Framework Structure
- **slime/**: Core framework with backend integrations and utilities
  - `backends/megatron_utils/`: Megatron-LM integration for distributed training
  - `backends/sglang_utils/`: SGLang integration for efficient inference
  - `ray/`: Ray actors, placement groups, and rollout orchestration
  - `utils/`: Types, arguments, masks, and various utilities
- **slime_plugins/**: Plugin system for models, rollout buffers, and generators
  - `models/`: Model-specific implementations (GLM, Qwen, KernelLLM)
  - `rollout_buffer/`: Data buffer and generator implementations
  - `rollout_buffer/generator/`: Task-specific generators including multi-turn kernel generation

### Key Entry Points
- **train.py**: Synchronous training orchestrator
- **train_async.py**: Asynchronous training for better GPU utilization
- Both coordinate Ray-based distributed training with placement groups

### Multi-Turn Kernel Generation System
The framework features a sophisticated multi-turn RL system for kernel generation:
- **Multi-turn dialogue** with configurable horizon (default 3 turns via `--max-turns`)
- **Aggregated return calculation** with discount factor (γ = 0.4 via `--gamma`)
- **Context construction** from previous attempts including compilation status, correctness, and performance
- **Smart early termination** based on success/failure conditions

Key files:
- `slime_plugins/rollout_buffer/generator/multi_turn_kernel_generator.py`: Multi-turn rollout logic
- `slime_plugins/rollout_buffer/generator/kernel_generator.py`: Single-turn generation
- `slime_plugins/rollout_buffer/generator/reward_utils/kernel_utils.py`: Kernel evaluation

### Distributed Training Architecture
- **Ray** for job orchestration and resource management
- **Megatron-LM** for large-scale distributed training (tensor/pipeline/context parallelism)
- **SGLang** for high-throughput inference during rollout generation
- **Custom placement groups** for optimal GPU allocation

### Training Features
- **GRPO (Group Relative Policy Optimization)** with advantage estimation
- **KL divergence loss** for policy regularization
- **Dynamic batch sizing** based on token count
- **CPU optimizer offloading** for memory efficiency
- **Gradient checkpointing** with configurable granularity

## Key Technical Details

### Argument Categories
1. **Megatron arguments**: Configure via standard Megatron flags (e.g., `--tensor-model-parallel-size`)
2. **SGLang arguments**: Prefix with `--sglang-` (e.g., `--sglang-mem-fraction-static`)
3. **slime-specific arguments**: See `slime/utils/arguments.py`

### Model Support
- **GLM-4-9B**: GLM family with sandwich normalization
- **Qwen3 series**: 4B, 8B, 30B-A3B MoE variants
- **KernelLLM-8B**: Llama 3.1 based, specialized for kernel generation

### Data Pipeline
- **KernelBench datasets** in `data/kernel_bench/` (JSONL format)
- **Multi-turn support** with history and context tracking
- **Dynamic sampling** with configurable filters

### Logging & Monitoring
- **WandB integration** via `--use-wandb`
- **Multi-turn logging** in `MULTI_TURN_LOGGING.md`
- **JSON-based logs** for analysis and reproducibility