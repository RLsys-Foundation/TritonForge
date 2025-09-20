# Example: Qwen3-30B-A3B Model

[中文版](../../zh/models/qwen3-30B-A3B.md)

## Environment Preparation

The environment setup, model download, data, and checkpoint conversion are the same as for the Qwen3-4B model. You can refer to [Example: Qwen3-4B Model](./qwen3-4B.md), replacing mentions of Qwen3-4B with Qwen3-30B-A3B.

## Run Training

Execute the training script:

```bash
cd /root/slime
bash scripts/run-qwen3-30B-A3B.sh
```

### Parameter Introduction

Here, we will briefly introduce the MoE-related parts in the [run-qwen3-30B-A3B.sh](../../../scripts/run-qwen3-30B-A3B.sh) script.

1.  To support running Qwen3-30B-A3B in an 8xH800 environment, we need to enable Megatron's CPU Adam to save GPU memory. The corresponding configuration is:

    ```bash
    OPTIMIZER_ARGS=(
       ...
       --optimizer-cpu-offload
       --overlap-cpu-optimizer-d2h-h2d
       --use-precision-aware-optimizer
    )
    ```

2.  Enable MoE optimization supported by Megatron. The current configuration is tp4, ep8:

    ```bash
    PERF_ARGS=(
       --tensor-model-parallel-size 4
       --sequence-parallel
       --pipeline-model-parallel-size 1
       --context-parallel-size 1
       --expert-model-parallel-size 8
       --expert-tensor-parallel-size 1
       ...
    )
    ```

3.  Enable MoE optimization supported by SGLang. The current configuration is ep8:

    ```bash
    SGLANG_ARGS=(
       --rollout-num-gpus-per-engine 8
       --sglang-mem-fraction-static 0.5
       --sglang-enable-ep-moe
       --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
    )
    ```

    Similarly, you can also add DP attention, for example, by configuring:

    ```bash
       --sglang-enable-dp-attention
       --sglang-dp-size 8
    ```
