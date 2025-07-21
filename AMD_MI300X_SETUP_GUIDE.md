# KernelBench AMD MI300X + SGLang Setup Guide

## Overview

This guide documents the setup for running KernelBench evaluation on AMD MI300X GPUs using:
- **Triton backend** for kernel generation
- **SGLang server** with facebook/KernelLLM model on port 30000
- **AMD ROCm** environment with PyTorch HIP support

## Environment Setup

### 1. System Information
- **GPU**: 8x AMD Instinct MI300X (gfx942 architecture)
- **PyTorch**: 2.7.0a0+git295f2ed with HIP 6.3.42134
- **Triton**: 3.2.0
- **ROCm**: 6.3.4 (installed at /opt/rocm)

### 2. Environment Variables
The following environment variables are automatically set by our scripts:
```bash
export ROCM_HOME=/opt/rocm
export HIP_PLATFORM=amd
export PYTORCH_ROCM_ARCH=gfx942
export PATH=$ROCM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_HOME/lib:$LD_LIBRARY_PATH
export SGLANG_API_KEY=local-key
```

### 3. SGLang Server
The SGLang server is running on `localhost:30000` with the `facebook/KernelLLM` model.

## Running KernelBench Evaluation

### Quick Test
To verify your setup:
```bash
cd /workspace/KernelBench
python test_setup.py
```

### Main Evaluation Script
We've created a comprehensive script for running evaluations:
```bash
cd /workspace/KernelBench
python run_amd_mi300x_sglang.py
```

This script offers two modes:
1. **Single Sample Evaluation** - Quick test on one problem
2. **Batch Generation and Evaluation** - Full evaluation on multiple problems

### Script Features

#### Single Sample Mode
- Evaluates a single problem (default: Level 1, Problem 1)
- Uses Triton backend for AMD GPU
- Connects to SGLang server on port 30000
- Performs correctness and performance evaluation

#### Batch Mode
- Generates solutions for multiple problems in parallel
- Evaluates all generated solutions
- Produces comprehensive results

### Configuration Options

The script uses the following key configurations:

```python
# AMD GPU Configuration
config.gpu_arch = ["MI300X"]
config.backend = "triton"

# SGLang Server Configuration
config.server_type = "sglang"
config.server_port = 30000
config.server_address = "localhost"

# Generation Parameters
config.max_tokens = 4096
config.temperature = 0.0  # Deterministic generation

# Evaluation Parameters
config.n_correctness = 5   # Correctness trials
config.n_trial = 100      # Performance trials
config.timeout = 300      # Kernel compilation timeout
```

## AMD-Specific Optimizations

### 1. Triton Kernels
KernelBench automatically generates Triton kernels optimized for AMD GPUs:
- Uses AMD's 64-wide wavefronts (vs NVIDIA's 32-wide warps)
- Optimized block sizes (typically 64 or 256 for MI300X)
- Leverages AMD's larger L2 cache

### 2. Example Triton Kernel Structure
```python
@triton.jit
def kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Optimized for AMD MI300X
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # ... kernel logic ...
```

## Troubleshooting

### Common Issues

1. **SGLang Server Not Running**
   - Verify server is running: `curl http://localhost:30000/v1/models`
   - Check that facebook/KernelLLM model is loaded

2. **Triton Compilation Errors**
   - Increase timeout in configuration
   - Check PYTORCH_ROCM_ARCH is set correctly

3. **Out of Memory**
   - Reduce batch size in generation
   - Use fewer parallel workers

### Verification Commands

```bash
# Check AMD GPUs
rocm-smi --showproductname

# Check PyTorch HIP support
python -c "import torch; print(torch.version.hip)"

# Check Triton
python -c "import triton; print(triton.__version__)"

# Check SGLang server
curl http://localhost:30000/v1/models
```

## Performance Tips

1. **Block Sizes**: For MI300X, use block sizes of 64, 128, or 256
2. **Memory Access**: Optimize for AMD's memory hierarchy
3. **Kernel Fusion**: Consider fusing operations to reduce memory bandwidth

## Results

Evaluation results are saved in:
- `runs/{run_name}/` - Generated kernels
- `runs/{run_name}/eval_results.json` - Evaluation metrics

Use the analysis script to compute benchmark performance:
```bash
python scripts/benchmark_eval_analysis.py run_name=amd_mi300x_sglang_test level=1 hardware=MI300X baseline=baseline_time_torch
```

## Next Steps

1. Start with Level 1 problems to verify setup
2. Progress to higher levels for more complex optimizations
3. Analyze results to understand performance characteristics
4. Fine-tune generation parameters for better results

## Support

For issues specific to AMD GPU support, refer to:
- `/workspace/KernelBench/docs/AMD_SUPPORT.md`
- AMD ROCm documentation
- Triton documentation for AMD GPUs