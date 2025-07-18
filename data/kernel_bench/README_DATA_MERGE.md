# KernelBench Data Merge Documentation

## Overview

The KernelBench Triton data has been merged from 4 separate level files into a single comprehensive dataset to improve data distribution during training.

## Data Files

### Original Files (per level)
- `kernel_bench_triton_level_1.jsonl` - 100 entries (basic operators)
- `kernel_bench_triton_level_2.jsonl` - 100 entries (fusion patterns)
- `kernel_bench_triton_level_3.jsonl` - 50 entries (full architectures)
- `kernel_bench_triton_level_4.jsonl` - 20 entries (HuggingFace models)

### Merged File
- `kernel_bench_triton_all_levels.jsonl` - 270 entries (all levels combined and shuffled)

## Distribution

The merged dataset contains:
- **Level 1**: 100 entries (37.0%) - Single kernel operators
- **Level 2**: 100 entries (37.0%) - Operator fusion patterns
- **Level 3**: 50 entries (18.5%) - Complete neural network architectures
- **Level 4**: 20 entries (7.4%) - Real-world HuggingFace models

Total: 270 unique training examples across 269 unique problems.

## Benefits of Merged Data

1. **Balanced Training**: Model sees problems of varying difficulty in each batch
2. **Better Generalization**: Exposure to all problem types throughout training
3. **Progressive Learning**: Natural curriculum from simple to complex problems
4. **Improved Performance**: Better handling of both basic and advanced kernels

## Usage

The training scripts have been updated to use the merged data:

```bash
export PROMPT_DATA=/workspace/slime/data/kernel_bench/kernel_bench_triton_all_levels.jsonl
```

## Verification

Run the verification script to ensure data integrity:

```bash
cd /workspace/slime/data/kernel_bench
python verify_merged_data.py
```

## Regenerating Merged Data

If you need to regenerate the merged file:

```bash
cd /workspace/slime/data/kernel_bench
python merge_triton_levels.py

# Options:
# --no-shuffle    : Keep original order (not recommended for training)
# --seed N        : Use different random seed for shuffling
# --output FILE   : Specify different output filename
```

## Data Format

Each entry maintains the original format:
```json
{
  "prompt": [...],          // System and user messages
  "label": "...",          // Reference implementation
  "instance_id": "...",    // Unique identifier
  "extra_info": {
    "level": 1-4,          // Difficulty level
    "problem_id": N,       // Problem number within level
    "problem_name": "...", // Descriptive problem name
    "data_source": "kernel_bench_triton",
    "task_type": "kernelbench"
  }
}
```

## Notes

- The data is shuffled by default with seed=42 for reproducibility
- All 270 entries are preserved without modification
- The merged file is ~2MB in size
- Compatible with all existing Slime/KernelBench infrastructure