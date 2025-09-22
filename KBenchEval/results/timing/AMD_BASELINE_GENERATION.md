# AMD MI300X Baseline Timing Generation

## Overview

This guide explains how to generate baseline timings for KernelBench problems on AMD MI300X GPUs. These baselines are used to measure the performance of generated kernels against reference PyTorch implementations.

## Quick Start

### Automated Generation

Run the provided script for automated baseline generation:

```bash
cd /root/TritonForge/KBenchEval
bash scripts/run_amd_baseline_generation.sh
```

This will generate baselines for:
- Level 1: 100 single-kernel operators
- Level 2: 100 fusion patterns
- PyTorch Eager execution mode
- torch.compile with inductor backend (limited support)

### Manual Generation

For more control over the process:

```bash
cd /root/TritonForge/KBenchEval

# Test with single problem
python scripts/generate_baseline_time_amd.py --problem 1 --problem-level 1

# Test mode with 5 problems
python scripts/generate_baseline_time_amd.py --test --max-problems 5 --level 1

# Full generation for specific levels
python scripts/generate_baseline_time_amd.py --level 1 2 --hardware MI300X_rocm
```

## Script Options

### `generate_baseline_time_amd.py`

| Option | Description | Default |
|--------|-------------|---------|
| `--test` | Run in test mode with limited problems | False |
| `--level` | Levels to process (space-separated) | 1 2 |
| `--problem` | Test specific problem ID | None |
| `--problem-level` | Level for specific problem test | 1 |
| `--hardware` | Hardware name for saving results | Auto-detect |
| `--max-problems` | Maximum problems per level (for testing) | All |

## Environment Setup

The script automatically configures AMD ROCm/HIP environment:

```bash
# ROCm configuration
export ROCM_HOME=/opt/rocm
export HIP_PLATFORM=amd
export PYTORCH_ROCM_ARCH=gfx942

# Stability settings (disable core dumps)
export HSA_ENABLE_COREDUMP=0
export AMD_LOG_LEVEL=0
export ROCM_DISABLE_CRASH_DUMP=1
export HIP_ENABLE_COREDUMP=0

# Performance optimization
export HSA_ENABLE_SDMA=0
export GPU_MAX_HW_QUEUES=1
```

## Output Format

Baseline timings are saved as JSON files in:
```
/root/TritonForge/KBenchEval/results/timing/MI300X_rocm/
```

### File Structure

```
MI300X_rocm/
├── baseline_time_torch.json                    # PyTorch Eager execution (200 problems: level1+level2)
├── baseline_time_torch_compile_inductor_default.json  # torch.compile (if supported, 200 problems)
└── baseline_time_torch_test.json              # Test results
```

### JSON Format

Each JSON file contains BOTH level1 and level2 data in a single file:

```json
{
  "level1": {
    "1_Square_matrix_multiplication_.py": {
      "mean": 0.19,      // Mean execution time (ms)
      "std": 0.0123,     // Standard deviation
      "min": 0.176,      // Minimum time
      "max": 0.28,       // Maximum time
      "num_trials": 100, // Number of trials
      "hardware": "AMD GPU",
      "gpu_type": "AMD",
      "device": "cuda:0"
    },
    ... // 99 more level1 problems (100 total)
  },
  "level2": {
    "1_Conv2D_ReLU_BiasAdd.py": {
      "mean": 0.079,
      "std": 0.0135,
      "min": 0.071,
      "max": 0.206,
      "num_trials": 100,
      "hardware": "AMD GPU",
      "gpu_type": "AMD",
      "device": "cuda:0"
    },
    ... // 99 more level2 problems (100 total)
  }
}
```

**Total**: 200 problems per JSON file (100 from level1 + 100 from level2)
```

## Differences from NVIDIA

### Supported Features
- ✅ PyTorch Eager execution
- ✅ Basic torch.compile with inductor backend
- ✅ ROCm/HIP kernel execution
- ✅ AMD GPU profiling

### Limitations
- ❌ cudagraphs backend (NVIDIA-specific)
- ⚠️ Limited torch.compile modes (some may crash)
- ⚠️ max-autotune-no-cudagraphs not supported

## Troubleshooting

### Common Issues

1. **GPU Not Detected**
   ```bash
   rocm-smi  # Check GPU visibility
   ```

2. **Memory Errors**
   - Reduce batch sizes in problem definitions
   - Use fewer trials: `--max-problems 10`

3. **Compilation Failures**
   - Some torch.compile modes may not work on ROCm
   - Fall back to eager execution

4. **Performance Variations**
   - MI300X has different characteristics than NVIDIA GPUs
   - Baseline times will differ significantly from H100/A100

## Performance Notes

- MI300X has 192GB HBM3 memory (vs 80GB on H100)
- Different memory bandwidth characteristics
- ROCm compilation may take longer than CUDA
- Warmup iterations are crucial for stable measurements

## Integration with KernelBench

Once generated, these baselines can be used for:

1. **Performance Comparison**
   ```python
   # In evaluation scripts
   baseline = "baseline_time_torch"
   hardware = "MI300X_rocm"
   ```

2. **Speedup Calculation**
   - Compare generated kernel times against baseline
   - Calculate fast_0, fast_1, fast_2 metrics

3. **Cross-Platform Analysis**
   - Compare AMD vs NVIDIA performance
   - Identify platform-specific optimizations

## Next Steps

After generating baselines:

1. Run kernel evaluations:
   ```bash
   python scripts/generate_and_eval_single_sample.py \
     --backend triton \
     --baseline MI300X_rocm
   ```

2. Analyze performance:
   ```bash
   python scripts/benchmark_eval_analysis.py \
     --hardware MI300X_rocm \
     --baseline baseline_time_torch
   ```

## Support

For issues specific to AMD baseline generation:
- Check ROCm installation: `rocm-smi`
- Verify PyTorch ROCm support: `python -c "import torch; print(torch.cuda.is_available())"`
- See logs in: `/root/TritonForge/KBenchEval/results/timing/MI300X_rocm/`

---

**Note**: This is part of the TritonForge project's effort to support both NVIDIA and AMD GPUs for kernel optimization research.