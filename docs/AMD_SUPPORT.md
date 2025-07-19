# AMD GPU Support in KernelBench

This document describes the AMD GPU support in KernelBench, specifically for evaluating Triton kernels on AMD MI300X and other CDNA/RDNA architectures.

## Overview

KernelBench now supports AMD GPUs through ROCm and PyTorch's HIP backend. This allows you to:
- Evaluate Triton kernels on AMD GPUs (MI300X, MI250X, MI100)
- Profile kernel performance using AMD-specific tools
- Generate and optimize kernels for AMD architectures

## Prerequisites

### 1. ROCm Installation
Ensure ROCm is installed on your system:
```bash
# Check ROCm installation
rocm-smi --version

# Set ROCm path (if not already set)
export ROCM_HOME=/opt/rocm
export PATH=$ROCM_HOME/bin:$PATH
```

### 2. PyTorch with ROCm Support
Install PyTorch with ROCm support:
```bash
# For ROCm 5.7
pip install torch --index-url https://download.pytorch.org/whl/rocm5.7

# For ROCm 6.0
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
```

### 3. Triton Installation
Install Triton with ROCm support:
```bash
pip install triton
```

## Supported AMD GPUs

The following AMD GPU architectures are supported:
- **MI300X** (gfx942) - Latest CDNA 3 architecture
- **MI250X** (gfx90a) - CDNA 2 architecture
- **MI100** (gfx908) - CDNA 1 architecture
- Additional architectures: gfx906, gfx940, gfx941

## Configuration

### GPU Architecture Setting

When configuring KernelBench for AMD GPUs, specify the architecture in your configuration:

```python
# For evaluation scripts
config.gpu_arch = ["MI300X"]  # or ["gfx942"] for explicit architecture code

# Multiple architectures can be specified
config.gpu_arch = ["MI300X", "MI250X"]
```

### Backend Selection

For AMD GPUs, always use the Triton backend:
```python
config.backend = "triton"
```

Note: If you specify `backend="cuda"` on an AMD system, it will automatically switch to "triton" with a warning.

## Usage Examples

### 1. Single Sample Evaluation

```python
from scripts.generate_and_eval_single_sample import EvalConfig, main

config = EvalConfig()
config.dataset_src = "local"
config.level = 1
config.problem_id = 1
config.gpu_arch = ["MI300X"]
config.backend = "triton"

main(config)
```

### 2. Batch Evaluation

```python
from scripts.eval_from_generations import EvalConfig, main

config = EvalConfig()
config.run_name = "my_amd_run"
config.dataset_src = "local"
config.level = 1
config.gpu_arch = ["MI300X"]
config.backend = "triton"

main(config)
```

### 3. Running Tests

Test AMD GPU support:
```bash
python tests/test_amd_mi300x.py
```

Run example evaluation:
```bash
python examples/run_amd_evaluation.py
```

## AMD-Specific Features

### 1. GPU Detection

KernelBench automatically detects AMD GPUs:
```python
from src.utils import is_amd_gpu, get_amd_gpu_info

if is_amd_gpu():
    print("AMD GPU detected")
    gpu_info = get_amd_gpu_info()
    print(f"GPU: {gpu_info}")
```

### 2. Performance Profiling

Use AMD-specific profiling utilities:
```python
from src.amd_profiling import profile_kernel_basic, get_amd_gpu_metrics

# Profile a kernel
results = profile_kernel_basic(
    kernel_func=my_kernel,
    args=[x, y],
    warmup_runs=10,
    profile_runs=100
)

# Get GPU metrics
metrics = get_amd_gpu_metrics(device)
```

### 3. Optimization Hints

Get optimization suggestions for AMD GPUs:
```python
from src.amd_profiling import get_amd_optimization_hints

hints = get_amd_optimization_hints("matmul")
for hint in hints:
    print(hint)
```

## Performance Considerations

### 1. Block Sizes
AMD GPUs have different optimal block sizes compared to NVIDIA:
- AMD GPUs use 64-wide wavefronts (vs NVIDIA's 32-wide warps)
- Optimal block sizes are often multiples of 64
- For MI300X, consider block sizes of 64, 128, or 256

### 2. Memory Hierarchy
- AMD MI300X has larger L2 cache than most NVIDIA GPUs
- Optimize memory access patterns accordingly
- Use LDS (Local Data Share) for shared memory operations

### 3. Triton Kernel Optimization
When writing Triton kernels for AMD:
```python
@triton.jit
def kernel(x_ptr, y_ptr, output_ptr, n_elements, 
           BLOCK_SIZE: tl.constexpr):
    # Use block sizes optimized for AMD
    # BLOCK_SIZE = 64 or 256 for MI300X
    ...
```

## Troubleshooting

### 1. ROCm Not Detected
If ROCm is not detected:
```bash
# Check environment variables
echo $ROCM_HOME
echo $HIP_PLATFORM

# Set if needed
export HIP_PLATFORM=amd
export ROCM_HOME=/opt/rocm
```

### 2. PyTorch HIP Support
Verify PyTorch ROCm support:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"HIP version: {torch.version.hip if hasattr(torch.version, 'hip') else 'Not available'}")
```

### 3. Kernel Compilation Errors
If you encounter HIP compilation errors:
- Increase timeout in configuration: `config.timeout = 300`
- Check that all required ROCm libraries are in LD_LIBRARY_PATH
- Ensure compatible versions of ROCm, PyTorch, and Triton

### 4. Performance Issues
If performance is lower than expected:
- Verify you're using the correct architecture setting
- Check that kernels are optimized for AMD wavefront size
- Use profiling tools to identify bottlenecks

## Environment Variables

Important environment variables for AMD GPUs:
- `ROCM_HOME`: ROCm installation directory
- `HIP_PLATFORM`: Set to "amd" for AMD GPUs
- `HIP_VISIBLE_DEVICES`: Control which AMD GPUs are visible
- `PYTORCH_ROCM_ARCH`: Target architectures for kernel compilation
- `HSA_OVERRIDE_GFX_VERSION`: Override detected GPU architecture (use with caution)

## Known Limitations

1. Some CUDA-specific optimizations may not translate directly to AMD
2. ROCm profiling tools integration is still in development
3. Certain Triton features may have different performance characteristics on AMD

## Future Enhancements

Planned improvements for AMD support:
1. Deep integration with rocprof for detailed profiling
2. Automatic kernel optimization for AMD architectures
3. Support for AMD-specific Triton extensions
4. Integration with AMD's composable kernel library

## References

- [ROCm Documentation](https://rocm.docs.amd.com/)
- [PyTorch ROCm Support](https://pytorch.org/get-started/locally/)
- [Triton Documentation](https://triton-lang.org/)
- [AMD Instinct MI300X](https://www.amd.com/en/products/accelerators/instinct/mi300.html)