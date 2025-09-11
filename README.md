# TritonForge

**TritonForge** is an advanced machine learning framework that trains Large Language Models (LLMs) to automatically convert PyTorch operations into optimized Triton GPU kernels. The project combines supervised fine-tuning (SFT) followed by reinforcement learning (RL) to achieve state-of-the-art kernel generation performance.

## 🎯 Overview

TritonForge consists of two main components:

- **[SMART](SMART/)**: A reinforcement learning training framework built on top of [SLIME](https://github.com/THUDM/slime), enabling multi-turn iterative kernel improvement through compilation feedback and performance metrics
- **[KBenchEval](KBenchEval/)**: A comprehensive benchmark suite for evaluating GPU kernel generation quality and performance

## 🚀 Quick Start

### Prerequisites

- Docker with GPU support
- NVIDIA GPUs (A100/H100) or AMD GPUs (MI300X)
- Python 3.10+
- At least 80GB GPU memory for 8B models

### Installation

The setup process differs between NVIDIA and AMD environments. Follow the appropriate guide below:

<details>
<summary><b>📗 NVIDIA Setup</b></summary>

#### 1. Launch Docker Container

```bash
docker pull zhuzilin/slime:20250706-v2

docker run --rm --gpus all --ipc=host --shm-size=128g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $HOME:$HOME \
  -it zhuzilin/slime:20250706-v2 /bin/bash
```

#### 2. Clone Repository

```bash
git clone https://github.com/RLsys-Foundation/TritonForge.git
cd TritonForge
```

#### 3. Setup KBenchEval

```bash
cd KBenchEval

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -e .
```

#### 4. Setup SMART

```bash
cd ../SMART
pip install -e .
```

#### 5. Download Models

```bash
# Create models directory
mkdir -p models

# Hugging Face format of fine-tuned Qwen3-8B model (for evaluation)
huggingface-cli download JinnP/Qwen3-8B-Kernelbook-SFT-HF --local-dir models/Qwen3-8B-Kernelbook-SFT-HF

# Megatron format of fine-tuned Qwen3-8B model (for continued training)
huggingface-cli download JinnP/Qwen3-8B-Kernelbook-SFT-filtered --local-dir models/Qwen3-8B-Kernelbook-SFT-filtered

# Base Qwen3-8B model (HuggingFace format)
huggingface-cli download Qwen/Qwen3-8B --local-dir models/Qwen3-8B

# Base Qwen3-8B model (Megatron format)
huggingface-cli download zyzshishui0627/Qwen3-8B_torch_dist --local-dir models/Qwen3-8B_torch_dist
```

</details>

<details>
<summary><b>📕 AMD Setup</b></summary>

#### 1. Allocate Compute Node (Azure)

```bash
tmux new-session -d -s kernel_agent_node_0

salloc --nodes=1 --exclusive --gres=gpu:8 \
          --time=120-00:00:00 \
          --nodelist=pdfc-aig-00001N \
          --job-name=Kernel-Agent
```

#### 2. Launch Docker Container

```bash
docker pull rlsys/april:slime_ubuntu22.04_rocm6.3.4-patch-numa_vllm0.8.5-patch_sglang0.4.7_megatron-core-patch_ray0.47-patch_apex_vim

docker run -it \
  --device /dev/dri \
  --device /dev/kfd \
  -p 8265:8265 \
  --group-add video \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --privileged \
  -v $HOME/.ssh:/root/.ssh \
  -v $HOME:$HOME \
  --shm-size 128G \
  --name slime_dev \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -w $PWD \
  rlsys/april:slime_ubuntu22.04_rocm6.3.4-patch-numa_vllm0.8.5-patch_sglang0.4.7_megatron-core-patch_ray0.47-patch_apex_vim \
  /bin/bash
```

#### 3. Clone Repository

```bash
# Clone SMART (SLIME fork)
git clone git@github.com:SwordFaith/slime.git SMART
cd SMART
git checkout dev-Azure

# Clone KBenchEval  
cd ..
git clone git@github.com:SwordFaith/KernelBench.git KBenchEval
cd KBenchEval
git checkout AMD-ver
```

#### 4. Set AMD Environment Variables

```bash
# Set AMD environment variables
export ROCM_HOME=/opt/rocm
export HIP_PLATFORM=amd
export PYTORCH_ROCM_ARCH=gfx942
export PATH=$ROCM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_HOME/lib:$LD_LIBRARY_PATH
export SGLANG_API_KEY=local-key
export PYTHONPATH=/workspace/KernelBench:$PYTHONPATH

# AMD optimizations
export HSA_ENABLE_SDMA=0

# Prevent GPU core dumps
export HSA_ENABLE_COREDUMP=0
export AMD_LOG_LEVEL=0
export ROCM_DISABLE_CRASH_DUMP=1
export HIP_ENABLE_COREDUMP=0
export HSA_TOOLS_LIB=/opt/rocm/lib/librocm-debug-agent.so.2:0
export GPU_MAX_HW_QUEUES=1
```

#### 5. Setup KBenchEval

```bash
cd KBenchEval

# Install additional dependencies
pip install pydra_config==0.0.15
cd /usr/local/lib/python3.12/dist-packages && ln -sf pydra_config pydra
cd -

pip install together google-generativeai
pip install -e .
```

#### 6. Setup SMART

```bash
cd ../SMART
pip install -e .
```

#### 7. Download Models

```bash
# Create models directory
mkdir -p models

# Download the same models as NVIDIA setup
huggingface-cli download JinnP/Qwen3-8B-Kernelbook-SFT-HF --local-dir models/Qwen3-8B-Kernelbook-SFT-HF
huggingface-cli download JinnP/Qwen3-8B-Kernelbook-SFT-filtered --local-dir models/Qwen3-8B-Kernelbook-SFT-filtered
huggingface-cli download Qwen/Qwen3-8B --local-dir models/Qwen3-8B
huggingface-cli download zyzshishui0627/Qwen3-8B_torch_dist --local-dir models/Qwen3-8B_torch_dist
```

</details>

## 🎓 Training Pipeline

TritonForge employs a two-stage training approach:

### Stage 1: Supervised Fine-Tuning (SFT)

We first fine-tune the base Qwen3-8B model using the [facebook/KernelLLM](https://huggingface.co/datasets/facebook/KernelLLM) dataset, which contains high-quality PyTorch-to-kernel conversion examples. The resulting model is available at [JinnP/Qwen3-8B-Kernelbook-SFT-filtered](https://huggingface.co/JinnP/Qwen3-8B-Kernelbook-SFT-filtered).

### Stage 2: Reinforcement Learning (RL)

We then apply reinforcement learning using SMART to further improve the model's kernel generation capabilities:

- **Training Data**: KernelBench Level 1-2 (200 problems)
- **Approach**: Multi-turn iterative refinement with compilation and performance feedback
- **Reward Signal**: Compilation success + functional correctness + speedup metrics

## 📊 Quick Evaluation

### Test Single Problem

<details>
<summary><b>NVIDIA</b></summary>

```bash
cd KBenchEval
source .venv/bin/activate

python scripts/generate_and_eval_single_sample.py \
  dataset_src="huggingface" \
  level=1 \
  problem_id=19 \
  verbose_logging=true
```

</details>

<details>
<summary><b>AMD</b></summary>

```bash
cd KBenchEval

export OPENAI_API_KEY="dummy-key"
python scripts/generate_and_eval_single_sample.py \
  dataset_src=local \
  level=1 \
  problem_id=19 \
  gpu_arch='["MI300X"]' \
  backend=triton \
  server_type=sglang \
  eval_device=0 \
  verbose=True
```

</details>

### Run Full Evaluation

<details>
<summary><b>NVIDIA</b></summary>

```bash
cd SMART
# Multi-turn kernel generation training
bash scripts/run_agent_kbench_qwen3_8B_sft_fixed.sh
```

</details>

<details>
<summary><b>AMD</b></summary>

```bash
# Terminal 1: Launch SGLang server
cd KBenchEval
HIP_VISIBLE_DEVICES=2,3 python3 -m sglang.launch_server \
  --model-path models/Qwen3-8B-Kernelbook-SFT-HF \
  --tp 2 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 30000

# Terminal 2: Run evaluation
cd KBenchEval
python kernelbench_amd_tools/scripts/run_qwen3_evaluation_robust.py --levels 1,2
```

</details>

## 📁 Project Structure

```
TritonForge/
├── SMART/                      # RL training framework (based on SLIME)
│   ├── slime/                  # Core SLIME framework
│   ├── slime_plugins/          # Custom generators and reward functions
│   └── scripts/                # Training launch scripts
├── KBenchEval/                 # Kernel evaluation framework
│   ├── KernelBench/            # Benchmark problems (Level 1-3)
│   ├── src/                    # Evaluation logic
│   └── scripts/                # Evaluation scripts
└── models/                     # Downloaded model checkpoints
```

## 🔬 Results

*[Results section to be added after experiments complete]*

## 🙏 Acknowledgments

- **[SLIME](https://github.com/THUDM/slime)**: The foundational RL framework that SMART is built upon
- **[Meta AI](https://ai.meta.com/)**: For contributing Triton backend support to KernelBench evaluation
- **[facebook/KernelLLM](https://huggingface.co/datasets/facebook/KernelLLM)**: High-quality SFT dataset for kernel generation
- **[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)**: Distributed training infrastructure
- **[SGLang](https://github.com/sgl-project/sglang)**: High-performance inference serving
- **[Triton](https://github.com/openai/triton)**: GPU kernel programming language

## 📄 License

Apache 2.0 - See LICENSE file for details

## 📧 Contact

For questions and support:
- Issue Tracker: [GitHub Issues](https://github.com/RLsys-Foundation/TritonForge/issues)
- Discussions: [GitHub Discussions](https://github.com/RLsys-Foundation/TritonForge/discussions)