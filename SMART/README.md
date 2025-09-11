# SMART (SLIME-based Multi-turn Adaptive Reinforcement Training)

**SMART** is an advanced reinforcement learning framework for training Large Language Models to generate optimized GPU kernels. Built on top of [SLIME (Scalable Language model Improvement by Merit-based Exploration)](https://github.com/THUDM/slime), SMART enables multi-turn iterative kernel improvement through compilation feedback and performance metrics.

## 🎯 Overview

SMART extends SLIME's capabilities to specifically target GPU kernel generation, implementing:

- **🔄 Multi-Turn Refinement**: Iteratively improve kernels through up to 3 turns based on compilation errors and performance feedback
- **📊 Custom Reward Functions**: Tailored rewards for compilation success, functional correctness, and performance speedup
- **⚡ Cross-Platform Support**: Optimized for both NVIDIA (CUDA) and AMD (ROCm) GPUs
- **🎮 Flexible Rollout System**: Single-turn and multi-turn kernel generators with configurable reward functions

## 🏗️ Architecture

SMART leverages SLIME's three-component architecture:

1. **[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)**: Distributed training with tensor/pipeline/data parallelism
2. **[SGLang](https://github.com/sgl-project/sglang)**: High-performance inference serving for rollout generation
3. **[Ray](https://www.ray.io/)**: Orchestration of distributed training and rollout actors

### Core Components

```
SMART/
├── slime/                              # Core SLIME framework
│   ├── backends/                       # Training and inference backends
│   │   ├── megatron_utils/            # Distributed training utilities
│   │   └── sglang_utils/              # Inference serving utilities
│   ├── ray/                           # Distributed orchestration
│   └── rollout/                       # Rollout and reward computation
├── slime_plugins/                      # SMART-specific extensions
│   ├── rollout_buffer/                # Custom kernel generation
│   │   ├── generator/                 # Kernel generators
│   │   │   ├── kernel_generator.py              # Single-turn generation
│   │   │   ├── multi_turn_kernel_generator.py   # Multi-turn generation
│   │   │   └── kernelbench_config.py           # Reward configuration
│   │   └── buffer.py                  # Trajectory management server
│   └── models/                        # Model-specific implementations
├── scripts/                           # Training launch scripts
│   ├── run_agent_kbench_*.sh         # NVIDIA training scripts
│   └── run_agent_kbench_*_amd.sh     # AMD training scripts
└── tools/                             # Model conversion utilities
```

## 🚀 Installation

### Prerequisites

- Docker with GPU support
- NVIDIA GPUs (A100/H100) or AMD GPUs (MI300X)
- At least 80GB GPU memory for 8B models
- Python 3.10+

### Setup

<details>
<summary><b>NVIDIA Environment</b></summary>

```bash
# Launch Docker container
docker pull zhuzilin/slime:20250706-v2
docker run --rm --gpus all --ipc=host --shm-size=128g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $HOME:$HOME \
  -it zhuzilin/slime:20250706-v2 /bin/bash

# Install SMART
cd SMART
pip install -e .
```

</details>

<details>
<summary><b>AMD Environment</b></summary>

```bash
# Launch Docker container
docker pull rlsys/april:slime_ubuntu22.04_rocm6.3.4-patch-numa_vllm0.8.5-patch_sglang0.4.7_megatron-core-patch_ray0.47-patch_apex_vim

docker run -it \
  --device /dev/dri --device /dev/kfd \
  --group-add video --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined --privileged \
  --shm-size 128G \
  rlsys/april:slime_ubuntu22.04_rocm6.3.4-patch-numa_vllm0.8.5-patch_sglang0.4.7_megatron-core-patch_ray0.47-patch_apex_vim \
  /bin/bash

# Clone and setup
git clone git@github.com:SwordFaith/slime.git SMART
cd SMART
git checkout dev-Azure
pip install -e .

# Set AMD environment variables
export ROCM_HOME=/opt/rocm
export HIP_PLATFORM=amd
export PYTORCH_ROCM_ARCH=gfx942
export HSA_ENABLE_SDMA=0
```

</details>

## 🎓 Training Pipeline

### Stage 1: Supervised Fine-Tuning (SFT)

Before reinforcement learning, we perform supervised fine-tuning using SLIME:

```bash
# Dataset: facebook/KernelLLM
# Base Model: Qwen/Qwen3-8B
# Output: JinnP/Qwen3-8B-Kernelbook-SFT-filtered

python train.py \
  --model-name Qwen3-8B \
  --dataset kernelllm \
  --training-mode sft \
  --output-dir models/Qwen3-8B-Kernelbook-SFT
```

The SFT stage uses high-quality PyTorch-to-kernel conversion examples from [facebook/KernelLLM](https://huggingface.co/datasets/facebook/KernelLLM) to establish baseline kernel generation capabilities.

### Stage 2: Reinforcement Learning (RL)

After SFT, we apply reinforcement learning for further improvement:

#### Single-Turn Training

Basic PyTorch to Triton conversion without iterative refinement:

```bash
# NVIDIA GPUs
bash scripts/run_agent_kbench_qwen3_8B_sft.sh

# AMD GPUs
bash scripts/run_agent_kbench_qwen3_8B_sft_amd_singleturn.sh
```

#### Multi-Turn Training (Recommended)

Iterative kernel improvement through compilation feedback:

```bash
# NVIDIA GPUs
bash scripts/run_agent_kbench_qwen3_8B_sft_fixed.sh

# AMD GPUs
bash scripts/run_agent_kbench_qwen3_8B_sft_amd.sh
```

The multi-turn scripts automatically:
1. Launch Ray cluster for distributed training
2. Start rollout buffer server for trajectory management
3. Initialize kernel evaluation server
4. Enable iterative improvement with discount factor γ=0.4

## ⚙️ Configuration

### Key Training Parameters

```bash
# Multi-turn configuration
--max-turns 3                          # Maximum refinement iterations
--gamma 0.4                            # Discount factor for rewards
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

# Training hyperparameters
--learning-rate 1e-5
--micro-batch-size 1
--global-batch-size 128
--seq-length 8192
```

### Custom Rollout Functions

SMART supports custom rollout and reward functions:

```python
# In slime_plugins/rollout_buffer/generator/kernelbench_config.py

def custom_reward_func(eval_result: dict) -> float:
    """Custom reward function for kernel generation"""
    reward = 0.0
    
    # Compilation success
    if eval_result.get('compiled', False):
        reward += 0.1
    
    # Functional correctness
    if eval_result.get('correctness', False):
        reward += 0.3
    
    # Performance speedup
    speedup = eval_result.get('speedup', 0.0)
    if speedup > 1.0:
        reward += min(speedup - 1.0, 1.0)
    
    return reward
```

## 🔍 Multi-Turn Generation Pipeline

### How It Works

1. **Turn 0 - Initial Generation**:
   - Input: PyTorch operation code
   - Output: Initial Triton kernel attempt
   - Evaluation: Compilation, correctness, performance

2. **Turn 1 - Error Correction**:
   - Input: Previous attempt + compilation errors/feedback
   - Output: Refined kernel addressing issues
   - Evaluation: Re-evaluate all metrics

3. **Turn 2 - Performance Optimization**:
   - Input: Working kernel + performance metrics
   - Output: Optimized kernel for better speedup
   - Evaluation: Final performance assessment

### Example Trajectory

```python
# Turn 0: Initial attempt
prompt: "Convert this PyTorch matmul to Triton..."
response: "@triton.jit\ndef matmul_kernel(...)"
eval: compiled=True, correct=False, error="dimension mismatch"
reward: 0.1

# Turn 1: Fix correctness
prompt: "Previous attempt had dimension mismatch. Fix the kernel..."
response: "@triton.jit\ndef matmul_kernel_v2(...)"
eval: compiled=True, correct=True, speedup=1.2x
reward: 0.9

# Turn 2: Optimize performance
prompt: "Kernel is correct but slow. Optimize for better performance..."
response: "@triton.jit\ndef matmul_kernel_v3(...)"
eval: compiled=True, correct=True, speedup=2.1x
reward: 1.4

# Aggregated return with γ=0.4
return_t0 = 0.1 + 0.4*0.9 + 0.16*1.4 = 0.684
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

# Ray dashboard
ray dashboard  # Access at http://localhost:8265
```

### Multi-Turn Trajectories

```bash
# View detailed multi-turn data
ls -la multi_turn_logs/trajectory_*.json
cat multi_turn_logs/trajectory_latest.json | jq '.'

# Analyze reward distribution
python slime_plugins/rollout_buffer/tools/analyze_rollout_data.py \
  --log-dir multi_turn_logs \
  --output-stats reward_stats.json
```

### Performance Metrics

Monitor key metrics during training:
- **Compilation Rate**: Percentage of kernels that compile successfully
- **Correctness Rate**: Percentage passing functional tests
- **Average Speedup**: Performance improvement vs PyTorch baseline
- **Turn Efficiency**: Success rate improvement by turn number

## 🛠️ Advanced Features

### Model Conversion

Convert between HuggingFace and Megatron formats:

```bash
# HuggingFace to Megatron
python tools/convert_hf_to_torch_dist.py \
  --hf-path models/Qwen3-8B \
  --output-path models/Qwen3-8B_torch_dist

# Megatron to HuggingFace
python tools/convert_torch_dist_to_hf.py \
  --torch-dist-path models/Qwen3-8B-Kernelbook-SFT-filtered \
  --output-path models/Qwen3-8B-Kernelbook-SFT-HF
```

### Custom Generators

Implement custom kernel generators in `slime_plugins/rollout_buffer/generator/`:

```python
from .base_generator import BaseGenerator

class CustomKernelGenerator(BaseGenerator):
    def generate_single_turn(self, prompt: str) -> str:
        # Custom single-turn generation logic
        pass
    
    def generate_multi_turn(self, history: List[dict]) -> str:
        # Custom multi-turn generation logic
        pass
```

## 🔬 Results

*[Results section to be added after experiments complete]*

## 🤝 Contributing

We welcome contributions! Areas of particular interest:
- Support for additional GPU architectures
- New kernel optimization strategies
- Enhanced multi-turn dialogue strategies
- Performance profiling and analysis tools

## 📚 References

- **SLIME Framework**: [THUDM/slime](https://github.com/THUDM/slime) - The foundational RL framework
- **KernelLLM Dataset**: [facebook/KernelLLM](https://huggingface.co/datasets/facebook/KernelLLM) - SFT training data
- **Trained Models**: [JinnP/Qwen3-8B-Kernelbook-SFT-filtered](https://huggingface.co/JinnP/Qwen3-8B-Kernelbook-SFT-filtered)

## 📄 License

Apache 2.0 - See LICENSE file for details

## 📧 Contact

For questions about SMART:
- Issue Tracker: [GitHub Issues](https://github.com/RLsys-Foundation/SMART/issues)
- Original SLIME Framework: [THUDM/slime](https://github.com/THUDM/slime)