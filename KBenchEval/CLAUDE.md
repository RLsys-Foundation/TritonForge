# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KernelBench is a benchmark for evaluating LLMs' ability to generate efficient GPU kernels. The project involves transpiling PyTorch operators to CUDA/Triton kernels and evaluating their correctness and performance.

## Key Commands

### Setup and Environment
```bash
# Create and activate conda environment
conda create --name kernel-bench python=3.10
conda activate kernel-bench

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Running Evaluations

**Single Problem Evaluation:**
```bash
# Run level 2, problem 40 from HuggingFace dataset
python3 scripts/generate_and_eval_single_sample.py dataset_src="huggingface" level=2 problem_id=40

# Use verbose logging for debugging
python3 scripts/generate_and_eval_single_sample.py dataset_src="huggingface" level=2 problem_id=40 verbose_logging=true
```

**Batch Generation and Evaluation:**
```bash
# Generate responses for all level 1 problems
python3 scripts/generate_samples.py run_name=test_hf_level_1 dataset_src=huggingface level=1 num_workers=50 server_type=deepseek model_name=deepseek-chat temperature=0

# Evaluate generated kernels
python3 scripts/eval_from_generations.py run_name=test_hf_level_1 dataset_src=local level=1 num_gpu_devices=8 timeout=300

# Optional: Build compilation cache first
python3 scripts/eval_from_generations.py run_name=test_hf_level_1 dataset_src=local level=1 num_gpu_devices=8 timeout=300 build_cache=True num_cpu_workers=32
```

**Performance Analysis:**
```bash
# Compute benchmark metrics (fast_p scores)
python3 scripts/benchmark_eval_analysis.py run_name=test_hf_level_1 level=1 hardware=L40S_matx3 baseline=baseline_time_torch

# Generate baseline times for your hardware
python3 scripts/generate_baseline_time.py level=1
```

### Testing and Validation

**Run single kernel test:**
```bash
python3 scripts/run_and_check.py --source_code path/to/kernel.py --ref_code path/to/reference.py
```

**AMD GPU Testing:**
```bash
python3 tests/test_amd_mi300x.py
```

### Evaluation Server

**Start evaluation server (standard):**
```bash
python scripts/simple_eval_server.py
```

**Start evaluation server with CUDA fixes (for Triton kernels with resource issues):**
```bash
python scripts/simple_eval_server_cuda_fix.py
```

## Architecture Overview

### Core Components

1. **Benchmark Dataset** (`KernelBench/`):
   - Level 1: Single-kernel operators (100 problems)
   - Level 2: Simple fusion patterns (100 problems)  
   - Level 3: Full model architectures (50 problems)
   - Level 4: HuggingFace models

2. **Evaluation Pipeline** (`src/eval.py`):
   - Correctness checking against PyTorch reference
   - Performance measurement and speedup calculation
   - Support for both CUDA and Triton kernels
   - AMD GPU support via ROCm/HIP

3. **Prompt Construction** (`src/prompt_constructor*.py`):
   - Templates for generating CUDA kernels
   - Templates for generating Triton kernels
   - Problem description formatting

4. **Kernel Compilation** (`src/compile.py`):
   - JIT compilation of generated kernels
   - Error handling and validation
   - Resource usage checking

5. **AMD Support** (`src/amd_profiling.py`, `kernelbench_amd_tools/`):
   - AMD MI300X GPU detection and profiling
   - ROCm/HIP compatibility layer
   - AMD-specific kernel evaluation

### Key Evaluation Metrics

- **fast_0**: Fraction of correct kernels (correctness rate)
- **fast_1**: Fraction of kernels that are correct AND faster than PyTorch
- **fast_2**: Fraction of kernels that are correct AND at least 2x faster than PyTorch

### Important Files

- `src/eval.py`: Core evaluation logic for correctness and performance
- `src/dataset.py`: Dataset loading and problem management
- `src/utils.py`: GPU detection, model inference, and utilities
- `scripts/generate_and_eval_single_sample.py`: Single problem testing entry point
- `scripts/eval_from_generations.py`: Batch evaluation of generated kernels
- `scripts/benchmark_eval_analysis.py`: Performance metric computation

### Working with Generated Kernels

Generated kernels are stored in `runs/{run_name}/` with the structure:
```
runs/{run_name}/
├── generations/
│   └── {problem_id}.json  # Raw LLM responses
└── evaluations/
    └── {problem_id}.json  # Evaluation results
```

### Environment Variables

For LLM API access:
- `OPENAI_API_KEY`: OpenAI API key
- `ANTHROPIC_API_KEY`: Anthropic API key
- `DEEPSEEK_API_KEY`: DeepSeek API key
- `GOOGLE_API_KEY`: Google API key

For debugging:
- `CUDA_LAUNCH_BLOCKING=1`: Synchronous CUDA operations
- `TORCH_USE_CUDA_DSA=1`: Device-side assertions

### Common Issues and Solutions

1. **CUDA Illegal Memory Access**: Use `scripts/simple_eval_server_cuda_fix.py` instead of the standard eval server
2. **Shared Memory Limits**: The CUDA fix server includes pre-validation for Triton kernels
3. **AMD GPU Support**: Ensure ROCm/PyTorch-ROCm is installed for AMD GPUs

### Development Notes

- Always check existing implementations before creating new kernels
- Use `pydra` configuration system for experiment management
- Evaluation results are cached to avoid redundant computation
- Modal integration available for cloud GPU evaluation