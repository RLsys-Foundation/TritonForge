# SLIME (Scalable Language model Improvement by Merit-based Exploration)

**Honest Disclosure:** This is a fixed and improved version of the original [SLIME](https://github.com/THUDM/slime) framework. We believe in transparency - this repository contains SLIME with essential bug fixes, optimizations, and enhancements that make it production-ready for GPU kernel generation tasks. All credit for the foundational framework goes to the original SLIME authors.

**SLIME** is an advanced reinforcement learning framework for training Large Language Models to generate optimized GPU kernels, enabling multi-turn iterative kernel improvement through compilation feedback and performance metrics.

## 🎯 Overview

This fixed version of SLIME includes improvements and optimizations specifically for GPU kernel generation, implementing:

- **🔄 Multi-Turn Refinement**: Iteratively improve kernels through up to 3 turns based on compilation errors and performance feedback
- **📊 Custom Reward Functions**: Tailored rewards for compilation success, functional correctness, and performance speedup
- **⚡ Cross-Platform Support**: Optimized for both NVIDIA (CUDA) and AMD (ROCm) GPUs
- **🎮 Flexible Rollout System**: Single-turn and multi-turn kernel generators with configurable reward functions

## 🏗️ Architecture

SLIME leverages a three-component architecture:

1. **[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)**: Distributed training with tensor/pipeline/data parallelism
2. **[SGLang](https://github.com/sgl-project/sglang)**: High-performance inference serving for rollout generation
3. **[Ray](https://www.ray.io/)**: Orchestration of distributed training and rollout actors

### Core Components

```
SLIME/
├── slime/                              # Core SLIME framework
│   ├── backends/                       # Training and inference backends
│   │   ├── megatron_utils/            # Distributed training utilities
│   │   └── sglang_utils/              # Inference serving utilities
│   ├── ray/                           # Distributed orchestration
│   └── rollout/                       # Rollout and reward computation
├── slime_plugins/                      # Task-specific extensions
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

# Install SLIME
cd SLIME
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
git clone git@github.com:SwordFaith/slime.git SLIME
cd SLIME
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
# NVIDIA GPUs (run from /root/TritonForge)
cd /root/TritonForge
bash SLIME/scripts/run_agent_kbench_qwen3_8B_sft_nv_single_turn.sh

# AMD GPUs
bash scripts/run_agent_kbench_qwen3_8B_sft_amd_singleturn.sh
```

#### Multi-Turn Training (Recommended)

Iterative kernel improvement through compilation feedback:

```bash
# NVIDIA GPUs (run from /root/TritonForge)
cd /root/TritonForge
bash SLIME/scripts/run_agent_kbench_qwen3_8B_sft_nv_multi_turn.sh

# AMD GPUs
bash scripts/run_agent_kbench_qwen3_8B_sft_amd.sh
```

The multi-turn scripts automatically:
1. Launch Ray cluster for distributed training
2. Start rollout buffer server for trajectory management
3. Initialize kernel evaluation server
4. Enable iterative improvement with discount factor γ=0.4

**Note**: If you've cloned TritonForge to a different location, update the `PROJECT_ROOT` variable at the top of each script:
```bash
# In SLIME/scripts/run_agent_kbench_*.sh
PROJECT_ROOT="/your/path/to/TritonForge"

# In SLIME/scripts/agent-example-kbench-*.sh
PROJECT_ROOT=/your/path
```

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

SLIME supports custom rollout and reward functions:

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

## 🚀 ROADMAP - Q4 2025 & Beyond

*Current Date: September 27, 2025*

TritonForge is evolving rapidly! Here's our priority-based roadmap starting from October 2025, organized from easiest to most complex implementations.

### 📅 Monthly Milestones - Q4 2025

#### 🟢 **October 2025** - Foundation & Quick Wins
*Priority: High | Difficulty: Easy*

- **AMD Multi-turn Stability Fix** 🔧
  - Test with ROCm 6.5+ versions
  - Implement crash recovery mechanisms
  - Document workarounds for known issues

- **Basic GUI Dashboard (v0.1)** 📊
  - Simple web interface for training metrics
  - Real-time loss curves and basic stats
  - Checkpoint listing and management

- **KernelBench Data Collection** 📈
  - Start collecting more kernel examples
  - Categorize by complexity levels
  - Set up automated testing pipeline

#### 🟡 **November 2025** - Scaling & Optimization
*Priority: High | Difficulty: Medium*

- **4+4+2 Architecture Implementation** 🏗️
  - Expand from 4+2+2 to 4+4+2 for multi-turn
  - Optimize resource allocation
  - Test on single-node first
  - Document configuration best practices

- **GUI Dashboard v0.5** 🎨
  - Add rollout trajectory visualization
  - Implement reward distribution charts
  - Basic task queue monitoring

- **Initial MOE Model Testing** 🤖
  - Start experiments with smaller MOE models
  - Profile memory and compute requirements
  - Baseline performance benchmarks

#### 🔴 **December 2025** - Advanced Features
*Priority: Medium | Difficulty: Medium-Hard*

- **Qwen3-30B-A3B Integration** 🚀
  - Full [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) support
  - Optimize sparse activation patterns
  - Performance tuning for MOE architecture

- **Tool Calling Framework (Phase 1)** 🛠️
  - PyTorch profiler integration
  - Basic operation cost analysis
  - Simple bottleneck detection

- **GUI Dashboard v1.0** 💻
  - Complete monitoring suite
  - Multi-turn trajectory inspection
  - Performance analytics dashboard
  - Error tracking and debugging tools

### 🎯 2026 Roadmap - Priority Order

#### 🏆 **High Priority** (Q1 2026)
*Essential for core functionality*

1. **FSDP Backend Integration** - Once SLIME upstream supports it
2. **KernelBench v0.1 Release** - Comprehensive benchmark suite
3. **Tool Calling Phase 2** - Documentation access & search capabilities
4. **Multi-node Support** - Distributed training across nodes

#### ⭐ **Medium Priority** (Q2 2026)
*Enhanced capabilities*

5. **Multi-DSL Support** - Start with CUDA, then HIP/ROCm
6. **Advanced Tool Calling** - Terminal execution, A/B testing
7. **70B+ Model Support** - Scale beyond 30B parameters
8. **Production Features** - Enterprise authentication, versioning

#### 💫 **Future Enhancements** (H2 2026)
*Nice-to-have features*

9. **Cross-DSL Translation** - Convert between kernel languages
10. **Academic Integration** - Paper search and citation
11. **Community Platform** - Kernel sharing and collaboration
12. **SaaS Deployment** - Cloud-hosted solution

### 🔧 Implementation Priority Matrix

| Feature | Effort | Impact | Priority | Target |
|---------|--------|--------|----------|--------|
| AMD Stability Fix | Low | High | 🔴 Critical | Oct 2025 |
| Basic GUI | Low | Medium | 🟠 High | Oct 2025 |
| 4+4+2 Architecture | Medium | High | 🟠 High | Nov 2025 |
| MOE Models (30B) | Medium | High | 🟠 High | Dec 2025 |
| Tool Calling (Basic) | Medium | Medium | 🟡 Medium | Dec 2025 |
| FSDP Support | High | High | 🟡 Medium | Q1 2026 |
| Multi-DSL | High | Medium | 🟢 Low | Q2 2026 |
| 70B+ Models | High | Low | 🟢 Low | Q2 2026 |

### 🚦 Quick Wins for October 2025

1. **Week 1-2**: AMD stability patches and testing
2. **Week 2-3**: Basic GUI implementation (FastAPI + React)
3. **Week 3-4**: KernelBench data pipeline setup
4. **Week 4**: Documentation and testing

### 🎯 Success Metrics

- **Performance**: 2-3x speedup over hand-written kernels
- **Reliability**: 99%+ compilation success rate
- **Coverage**: Support for 95%+ of common DL operations
- **Scale**: Efficient training on 100B+ parameter models
- **Adoption**: Integration with major ML frameworks

## 🤝 Contributing

We welcome contributions! Areas of particular interest:
- Support for additional GPU architectures
- New kernel optimization strategies
- Enhanced multi-turn dialogue strategies
- Performance profiling and analysis tools

## 📚 References

- **Original SLIME Framework**: [THUDM/slime](https://github.com/THUDM/slime) - The original framework this is based on
- **KernelLLM Dataset**: [facebook/KernelLLM](https://huggingface.co/datasets/facebook/KernelLLM) - SFT training data
- **Trained Models**: [JinnP/Qwen3-8B-Kernelbook-SFT-filtered](https://huggingface.co/JinnP/Qwen3-8B-Kernelbook-SFT-filtered)

## 📄 License

Apache 2.0 - See LICENSE file for details

## 📧 Contact

For questions about this fixed version of SLIME:
- Issue Tracker: [GitHub Issues](https://github.com/RLsys-Foundation/TritonForge/issues)
- Original SLIME Framework: [THUDM/slime](https://github.com/THUDM/slime)