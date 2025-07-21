# KernelBench AMD MI300X Complete Guide

## Overview

This guide provides everything you need to run KernelBench evaluations on AMD MI300X GPUs using Triton backend with SGLang server.

## System Requirements

- **Hardware**: AMD MI300X GPUs (8x available, gfx942 architecture)
- **Software**: 
  - PyTorch 2.7.0+ with HIP support
  - Triton 3.2.0+
  - ROCm 6.3.4+
  - SGLang server with facebook/KernelLLM model

## SGLang Server Setup

Launch the SGLang server on GPUs 2,3 with tensor parallelism:
```bash
HIP_VISIBLE_DEVICES=2,3 python3 -m sglang.launch_server \
    --model-path facebook/KernelLLM \
    --tp 2 \
    --host 0.0.0.0 \
    --port 30000
```

## Quick Start

### 1. Single Problem Evaluation
```bash
export PYTHONPATH=/workspace/KernelBench:$PYTHONPATH
export SGLANG_API_KEY=local-key

python scripts/generate_and_eval_single_sample.py \
    dataset_src=local level=1 problem_id=19 \
    gpu_arch='["MI300X"]' backend=triton \
    server_type=sglang
```

### 2. Full Batch Evaluation (270 problems)
```bash
cd /workspace/KernelBench/kernelbench_amd_tools/launchers
./start_evaluation.sh
```

## GPU Device Allocation

| GPU | Purpose | Notes |
|-----|---------|-------|
| GPU 0 | Evaluation | Primary device for kernel evaluation |
| GPU 1 | Evaluation | Secondary device (can run parallel evaluations) |
| GPU 2-3 | SGLang Server | Hosts facebook/KernelLLM with TP=2 |
| GPU 4-7 | Available | Can be used for additional evaluations |

## Environment Configuration

```bash
# Required
export ROCM_HOME=/opt/rocm
export HIP_PLATFORM=amd
export PYTORCH_ROCM_ARCH=gfx942
export SGLANG_API_KEY=local-key
export PYTHONPATH=/workspace/KernelBench:$PYTHONPATH

# Optimizations
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export HSA_ENABLE_SDMA=0
```

## Monitoring Tools

### Progress Monitor
```bash
cd /workspace/KernelBench/kernelbench_amd_tools/scripts
./monitor_evaluation.py
```
Shows: completion rate, success statistics, recent failures

### GPU Monitor
```bash
cd /workspace/KernelBench/kernelbench_amd_tools/scripts
./monitor_gpu_usage.py
```
Shows: GPU utilization, memory usage, device assignments

## Results Analysis

### After Evaluation
```bash
cd /workspace/KernelBench/kernelbench_amd_tools/scripts

# Analyze AMD-specific issues
./amd_issue_handler.py /workspace/KernelBench/runs/amd_mi300x_full_eval_*/

# View final report
cat /workspace/KernelBench/runs/amd_mi300x_full_eval_*/reports/FINAL_REPORT.md
```

## Expected Results

### Success Rates by Level
- **Level 1** (100 problems): 20-40% - Simple operators
- **Level 2** (100 problems): 10-30% - Fused operations  
- **Level 3** (50 problems): 5-15% - Full architectures
- **Level 4** (20 problems): 0-10% - HuggingFace models

### Common AMD Issues
1. **Memory Allocation**: "HIP out of memory" - Large allocations
2. **Type Errors**: "expected Tensor()" - Scalar/tensor mismatches
3. **Block Size**: Configuration issues with AMD wavefront sizes
4. **Compilation**: Triton JIT compilation failures

## Output Structure

```
runs/amd_mi300x_full_eval_TIMESTAMP/
├── progress.json              # Progress tracking
├── results.json               # Evaluation results
├── reports/                   # Analysis reports
│   ├── level1_report.md
│   ├── level2_report.md
│   ├── level3_report.md
│   ├── level4_report.md
│   └── FINAL_REPORT.md
├── generated_kernels/         # All Triton kernels
│   └── level{1-4}_problem{N}.py
└── logs/                      # Detailed logs
    └── level{1-4}/
```

## Troubleshooting

### SGLang Server Issues
```bash
# Check server status
curl http://localhost:30000/v1/models

# Should return:
# {"data":[{"id":"facebook/KernelLLM",...}]}
```

### GPU Memory Issues
```bash
# Check memory usage
rocm-smi --showmeminfo vram

# Clear GPU memory if needed
rocm-smi --resetgpustats
```

### Resume Interrupted Evaluation
The system automatically saves progress. Just run the same command again:
```bash
./start_evaluation.sh
```
It will skip completed problems and continue from where it stopped.

## Performance Notes

- **Duration**: 6-9 hours for full evaluation
- **Timeout**: 10 minutes per problem
- **Memory**: Monitor GPU 0 memory usage (SGLang server)
- **Storage**: ~1-2GB for kernels and logs

## Support

For issues specific to AMD GPU support:
- Check generated kernel syntax in `generated_kernels/`
- Review error patterns in AMD issue analysis report
- Common fixes involve adjusting block sizes for AMD (64, 128, 256)

---
*This guide consolidates all information needed for KernelBench AMD MI300X evaluation.*