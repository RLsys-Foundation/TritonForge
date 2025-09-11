# TritonForge Experimental Results

This directory contains experimental results and performance metrics from TritonForge training and evaluation.

## Files

### qwen3-8b-multi-turn-nvidia-h100.png
- **Model**: Qwen3-8B fine-tuned with RL
- **Hardware**: NVIDIA H100 GPUs
- **Training Mode**: Multi-turn iterative refinement
- **WandB Log**: [View Training Progress](https://wandb.ai/jhinpan-university-of-michigan/slime-multiturn-qwen3-8B-sft-filtered/runs/5o347842?nw=nwuserjhinpan)
- **Description**: Shows training progress and performance metrics for multi-turn kernel generation on NVIDIA H100

### qwen3-8b-single-turn-amd-mi300x.png
- **Model**: Qwen3-8B fine-tuned
- **Hardware**: AMD MI300X GPUs
- **Training Mode**: Single-turn generation
- **Description**: Performance metrics for single-turn kernel generation on AMD MI300X platform

## Additional Results

### Single-Turn Baseline (NVIDIA H100)
- **Model**: KernelLLM
- **Report**: [Detailed Analysis](https://tar-gazelle-668.notion.site/Kernel-Agent-Single-Turn-Experiment-Result-235651cb22e580d989cde0dc1fac5c8d)

### Multi-Turn AMD MI300X (Coming Soon)
- Currently under training
- Results will be added upon completion

## Metrics Explained

- **fast_0**: Percentage of kernels that compile and are functionally correct
- **fast_1**: Percentage of kernels that are correct AND faster than PyTorch baseline
- **fast_2**: Percentage of kernels that are correct AND at least 2x faster than PyTorch

## Update History

- 2024-01: Initial results from NVIDIA H100 experiments
- 2024-02: Added AMD MI300X single-turn results
- Ongoing: Multi-turn AMD experiments in progress