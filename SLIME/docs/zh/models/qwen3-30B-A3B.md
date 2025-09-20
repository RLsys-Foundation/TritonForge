# 示例：Qwen3-30B-A3B 模型

[English](../../en/models/qwen3-30B-A3B.md)

## 环境准备

搭建环境、下载模型、数据与 ckpt 转换均与 Qwen3-4B 模型相同，可以参考 [示例：Qwen3-4B 模型](./qwen3-4B.md)，将文中 Qwen3-4B 的部分转换为 Qwen3-30B-A3B 即可。

## 执行训练

执行训练：

```bash
cd /root/slime
bash scripts/run-qwen3-30B-A3B.sh
```

### 参数简介

这里我们简单介绍一下脚本 [run-qwen3-30B-A3B.sh](../../../scripts/run-qwen3-30B-A3B.sh) 中与 MoE 相关的部分。

1. 为了支持在 8xH800 环境中运行 Qwen3-30B-A3B，我们需要开启 megatron 的 CPU Adam 以节省显存，对应配置为：

   ```bash
   OPTIMIZER_ARGS=(
      ...
      --optimizer-cpu-offload
      --overlap-cpu-optimizer-d2h-h2d
      --use-precision-aware-optimizer
   )
   ```

2. 开启 megatron 支持的 moe 优化，当前配置为 tp4, ep8：

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

3. 开启 sglang 支持的 moe 优化，当前配置为 ep8：

   ```bash
   SGLANG_ARGS=(
      --rollout-num-gpus-per-engine 8
      --sglang-mem-fraction-static 0.5
      --sglang-enable-ep-moe
      --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
   )
   ```

   类似地，也可以加入 dp attention，例如配置上：

   ```bash
      --sglang-enable-dp-attention
      --sglang-dp-size 8
   ```
