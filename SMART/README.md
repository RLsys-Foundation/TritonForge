# TritonForge

**TritonForge** is an advanced reinforcement learning framework for training large language models to automatically convert PyTorch code into optimized Triton GPU kernels. Built on top of the SLIME framework, TritonForge enables multi-turn iterative kernel improvement through compilation feedback and performance metrics.

## 🎯 Key Features

- **🔄 Multi-Turn Kernel Generation**: Iteratively improve kernels through up to 3 turns of refinement based on compilation errors and performance feedback
- **⚡ Cross-Platform GPU Support**: Optimized for both NVIDIA (CUDA) and AMD (ROCm) GPUs
- **🎮 Custom Rollout System**: Flexible single-turn and multi-turn kernel generators with configurable reward functions
- **📊 Performance-Driven**: Rewards based on compilation success, correctness, and speedup metrics
- **🔧 Production Ready**: Comprehensive testing, monitoring, and debugging capabilities

## 🏗️ Architecture

TritonForge combines three powerful systems:

1. **Megatron-LM**: Distributed training with tensor/pipeline/data parallelism
2. **SGLang**: High-performance inference serving for rollout generation  
3. **Ray**: Orchestration of distributed training and rollout actors

### Core Components

```
tritonforge/
├── slime_plugins/rollout_buffer/     # Custom rollout and reward system
│   ├── generator/
│   │   ├── kernel_generator.py       # Single-turn kernel generation
│   │   ├── multi_turn_kernel_generator.py  # Multi-turn with iterative improvement
│   │   └── kernelbench_config.py     # Reward configuration
│   └── buffer.py                      # Trajectory management server
├── scripts/                           # Launch scripts for training
│   ├── run_agent_kbench_qwen3_8B_sft_fixed.sh  # NVIDIA training
│   └── run_agent_kbench_qwen3_8B_sft_amd.sh    # AMD training
└── slime/                            # Core SLIME framework
```

## 🚀 Quick Start

### Prerequisites

- Docker with GPU support
- NVIDIA GPUs (A100/H100) or AMD GPUs (MI300X)
- At least 80GB GPU memory for 8B models
- Python 3.10+

### Environment Setup

#### For NVIDIA GPUs:
```bash
docker run --rm --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -it zhuzilin/slime:latest /bin/bash

git clone https://github.com/[your-org]/TritonForge.git
cd TritonForge
pip install -e .
```

#### For AMD GPUs:
```bash
docker run --rm -it \
  --device /dev/dri --device /dev/kfd \
  --group-add video --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined --privileged \
  --shm-size 128G \
  yushengsuthu/slime-amd:slime_ubuntu22.04_rocm6.3.4-patch-numa_vllm0.8.5-patch_sglang0.4.7_megatron-core-patch_ray0.47-patch \
  /bin/bash

git clone https://github.com/[your-org]/TritonForge.git
cd TritonForge
pip install -e .
```

## 🎓 Training Models

### Single-Turn Kernel Generation

For basic PyTorch to Triton conversion without iterative refinement:

```bash
# NVIDIA GPUs
bash scripts/run_agent_kbench_qwen3_8B_sft.sh

# AMD GPUs  
bash scripts/run_agent_kbench_qwen3_8B_sft_amd_singleturn.sh
```

### Multi-Turn Kernel Generation (Recommended)

For iterative kernel improvement through compilation feedback:

```bash
# NVIDIA GPUs - Qwen3-8B with multi-turn refinement
bash scripts/run_agent_kbench_qwen3_8B_sft_fixed.sh

# AMD GPUs - Qwen3-8B with multi-turn refinement
bash scripts/run_agent_kbench_qwen3_8B_sft_amd.sh
```

The multi-turn scripts will automatically:
1. Launch the training process with Ray
2. Start the rollout buffer server for trajectory management
3. Initialize the kernel evaluation server
4. Enable iterative improvement with discount factor γ=0.4

## ⚙️ Configuration

### Key Training Parameters

```bash
# Multi-turn configuration
--max-turns 3              # Maximum refinement iterations
--gamma 0.4                # Discount factor for rewards
--rollout-task-type kernelbench_multiturn  # Task type

# Model parallelism (adjust based on GPU count)
--tensor-model-parallel-size 2
--context-parallel-size 2
--pipeline-model-parallel-size 1

# Rollout configuration
--rollout-batch-size 4
--n-samples-per-prompt 8
--rollout-max-response-len 8192
--rollout-temperature 0.8
```

### Custom Rollout Functions

TritonForge supports custom rollout and reward functions:

```python
# In your training script
ROLLOUT_ARGS=(
   --rollout-function-path slime.rollout.agent_rollout.generate_rollout
   --rm-type kernelbench_multiturn  # or 'kernelbench' for single-turn
   --custom-generate-function-path your_module.custom_generate
   --custom-rm-path your_module.custom_reward_func
)
```

## 🔍 Kernel Generation Pipeline

### Single-Turn Generation
1. **Input**: PyTorch operation code
2. **Output**: Triton kernel implementation
3. **Evaluation**: Compilation, correctness, performance
4. **Reward**: 0.1 (compile) + 0.3 (correct) + performance bonus

### Multi-Turn Generation
1. **Turn 0**: Initial kernel generation attempt
2. **Turn 1**: Refinement based on compilation errors or performance gaps
3. **Turn 2**: Further optimization for speedup
4. **Aggregated Return**: R_t = Σ(γ^(i-t) * reward_i)

Example multi-turn trajectory:
```python
# Turn 0: Initial attempt
prompt: "Convert this PyTorch matmul to Triton..."
response: "@triton.jit\ndef matmul_kernel(...)"
eval: compiled=True, correct=False, error="dimension mismatch"
reward: 0.1

# Turn 1: Fix correctness issues  
prompt: "Previous attempt had dimension mismatch. Fix the kernel..."
response: "@triton.jit\ndef matmul_kernel_v2(...)"
eval: compiled=True, correct=True, speedup=1.2x
reward: 0.9

# Turn 2: Optimize for performance
prompt: "Kernel is correct but slow. Optimize for better performance..."
response: "@triton.jit\ndef matmul_kernel_v3(...)"
eval: compiled=True, correct=True, speedup=2.1x  
reward: 1.4
```

## 📊 Monitoring & Debugging

### Training Logs
```bash
# Monitor main training
tail -f slime_qwen3_sft_fixed_train.log

# Check rollout buffer
tail -f buffer_qwen3_sft_fixed.log  

# Watch evaluation server
tail -f eval_server_qwen3_sft_fixed.log
```

### Multi-Turn Trajectories
```bash
# View detailed multi-turn data
ls -la multi_turn_logs/trajectory_*.json
cat multi_turn_logs/trajectory_latest.json | jq '.'
```

### Performance Metrics
- **Compilation Rate**: Percentage of kernels that compile
- **Correctness Rate**: Percentage passing functional tests
- **Average Speedup**: Performance vs PyTorch baseline
- **Turn Efficiency**: Success rate by turn number

## 🛠️ Advanced Features

### Custom Reward Functions

Create your own reward model in `slime_plugins/rollout_buffer/generator/`:

```python
def custom_reward_func(eval_result: dict) -> float:
    reward = 0.0
    if eval_result['compiled']:
        reward += 0.1
    if eval_result['correctness']:
        reward += 0.3
    if eval_result['speedup'] > 1.0:
        reward += min(eval_result['speedup'] - 1.0, 1.0)
    return reward
```

### Kernel Validation

TritonForge includes comprehensive validation:
- Syntax checking for Triton decorators
- Import validation for required libraries
- Performance benchmarking against PyTorch
- Correctness testing with random inputs

## 🤝 Contributing

We welcome contributions! Areas of interest:
- Support for additional GPU architectures
- New kernel optimization strategies
- Enhanced multi-turn dialogue strategies
- Performance profiling tools

## 📚 Documentation

- [Architecture Overview](docs/architecture.md)
- [Multi-Turn Training Guide](docs/multi_turn.md)
- [AMD Setup Guide](docs/amd_setup.md)
- [Custom Generators](docs/custom_generators.md)

## 🙏 Acknowledgments

TritonForge builds upon:
- [SLIME](https://github.com/THUDM/slime) - The foundational RL framework
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) - Distributed training
- [SGLang](https://github.com/sgl-project/sglang) - High-performance inference
- [Triton](https://github.com/openai/triton) - GPU kernel language

## 📄 License

Apache 2.0 - See LICENSE file for details

## 📧 Contact

For questions and support:
- Issue Tracker: [GitHub Issues](https://github.com/[your-org]/TritonForge/issues)
- Discussions: [GitHub Discussions](https://github.com/[your-org]/TritonForge/discussions)

---

**TritonForge** - Forging optimal GPU kernels through reinforcement learning 🔥⚡