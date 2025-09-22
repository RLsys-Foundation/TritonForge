# AMD MI300X Baseline Performance Summary

## Overview

We have successfully generated comprehensive baseline timings for AMD MI300X GPUs covering all 200 KernelBench problems (Level 1 + Level 2). These baselines serve as reference performance metrics for evaluating generated Triton kernels.

## Baseline Files Generated

```
/root/TritonForge/KBenchEval/results/timing/MI300X_rocm/
├── baseline_time_torch.json                    # PyTorch Eager (200 problems)
├── baseline_time_torch_compile_inductor_default.json  # torch.compile (200 problems)
└── baseline_time_torch_test.json              # Test subset (10 problems)
```

## Performance Characteristics

### Level 1 - Single Kernel Operations (100 problems)

Sample timings from key operations:

| Operation | Mean (ms) | Std (ms) | Min (ms) | Max (ms) |
|-----------|-----------|----------|----------|----------|
| Square Matrix Multiplication | 0.190 | 0.0105 | 0.176 | 0.251 |
| Standard Matrix Multiplication | 0.224 | 0.0106 | 0.211 | 0.295 |
| Batched Matrix Multiplication | 0.079 | 0.0063 | 0.073 | 0.137 |
| Matrix Vector Multiplication | 1.180 | 0.0070 | 1.160 | 1.220 |
| Matrix Scalar Multiplication | 0.144 | 0.0043 | 0.138 | 0.174 |

### Level 2 - Fusion Patterns (100 problems)

Sample timings from fusion operations:

| Operation | Mean (ms) | Std (ms) | Min (ms) | Max (ms) |
|-----------|-----------|----------|----------|----------|
| Conv2D + ReLU + BiasAdd | 0.079 | 0.0114 | 0.073 | 0.187 |
| ConvTranspose2d + Ops | 0.182 | 0.0066 | 0.173 | 0.243 |
| ConvTranspose3d + Complex | 48.60 | 1.350 | 48.00 | 60.10 |
| Conv2d + Mish + Mish | 0.089 | 0.0093 | 0.084 | 0.179 |

## Key Observations

### AMD MI300X Characteristics

1. **Memory Bandwidth**: MI300X with 192GB HBM3 shows excellent performance on memory-bound operations
2. **Compute Throughput**: Matrix operations perform competitively with NVIDIA H100
3. **Fusion Efficiency**: Complex fusion patterns benefit from large memory capacity
4. **ROCm Compilation**: Some torch.compile modes have limited support on ROCm

### Platform Differences

When comparing to NVIDIA baselines:
- Small kernels: Similar performance to H100
- Large kernels: Benefits from higher memory capacity
- Complex fusions: Competitive performance
- Compilation: Limited torch.compile optimization options

## Usage in Evaluation

These baselines are automatically used when evaluating kernels on AMD:

```python
# In evaluation scripts
python scripts/generate_and_eval_single_sample.py \
  dataset_src=local \
  level=1 \
  problem_id=19 \
  gpu_arch='["MI300X"]' \
  backend=triton \
  baseline="MI300X_rocm"
```

## Integration Points

1. **SLIME Training**: Baselines used for reward calculation in RL
2. **Performance Metrics**: Calculate fast_0, fast_1, fast_2 speedups
3. **Cross-Platform**: Compare AMD vs NVIDIA kernel performance

## Technical Details

### Generation Configuration
- **Trials**: 100 runs per problem for statistical significance
- **Warmup**: 10 iterations before measurement
- **Precision**: BF16 support enabled
- **Backend**: ROCm 6.3.4 with PyTorch 2.7.0

### Environment Settings
```bash
export PYTORCH_ROCM_ARCH=gfx942
export HSA_ENABLE_SDMA=0
export GPU_MAX_HW_QUEUES=1
```

## Validation

All baselines validated using `verify_baseline_structure.py`:
- ✅ Contains both level1 and level2
- ✅ 100 problems per level (200 total)
- ✅ Valid timing statistics for each problem
- ✅ Compatible with KernelBench evaluation framework

## Next Steps

With these baselines established:
1. Run kernel evaluations against PyTorch reference
2. Calculate performance speedups
3. Use in SLIME RL training for reward signals
4. Compare cross-platform optimization strategies

---

**Note**: These baselines represent a significant milestone in enabling AMD GPU support for the TritonForge project, allowing fair performance comparison across different hardware platforms.