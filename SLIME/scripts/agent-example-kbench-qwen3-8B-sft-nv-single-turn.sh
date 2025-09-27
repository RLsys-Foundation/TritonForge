#!/bin/bash

# Kernel Code Generation Agent Training Script - Qwen3-8B-SFT
# This script trains Qwen3-8B-slime-kernelbook-sft to generate optimized CUDA kernels

# Clean up previous runs
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

export PYTHONBUFFERED=16

# Configure your WandB key if available
export WANDB_KEY=${WANDB_KEY:-"0db9fd073cc9e49c8bcec2b0a6929792ecb64e4e"}

# Model parallelism configuration - Fixed for 4 training GPUs
export TP_SIZE=2    # Tensor parallelism
export PP_SIZE=1    # Pipeline parallelism
export CP_SIZE=2    # Context parallelism (total_model_size = 2*1*2 = 4, matches 4 GPUs)

# Model paths - Updated for Qwen3-8B
PROJECT_ROOT=/root
export HF_MODEL_PATH="${PROJECT_ROOT}/models/Qwen3-8B"
export MCORE_MODEL_PATH="${PROJECT_ROOT}/models/Qwen3-8B-Kernelbook-SFT-filtered"
export PROMPT_DATA="${PROJECT_ROOT}/TritonForge/SLIME/data/kernel_bench/kernel_bench_triton_level_1_2.jsonl"
export MCORE_MODEL_PATH_SAVE="${PROJECT_ROOT}/models/Qwen3-8B-Kernelbook-SFT-filtered_save"

# Qwen3-8B model architecture parameters
MODEL_ARGS=(
   --swiglu
   --num-layers 36
   --hidden-size 4096
   --ffn-hidden-size 12288
   --num-attention-heads 32
   --group-query-attention
   --num-query-groups 8
   --use-rotary-position-embeddings
   --disable-bias-linear
   --normalization "RMSNorm"
   --norm-epsilon 1e-6
   --rotary-base 1000000
   --vocab-size 151936
   --kv-channels 128
   --qk-layernorm
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --untie-embeddings-and-output-weights
   --attention-dropout 0.0
   --hidden-dropout 0.0
)

CKPT_ARGS=(
  # Load both actor and reference from your SFT checkpoint
  --load ${MCORE_MODEL_PATH}
  --ref-load ${MCORE_MODEL_PATH}

  # Save RL-updated weights here
  --save ${MCORE_MODEL_PATH_SAVE}
  --save-interval 200

  # Load weights only (avoid stale optimizer/RNG states)
  --no-load-optim
  --no-load-rng

  # Optional fallback: if --load isn't found, try HF path
  --hf-checkpoint ${HF_MODEL_PATH}
)

ROLLOUT_ARGS=(
   --rollout-function-path slime.rollout.agent_rollout.generate_rollout
   --rm-type kernelbench
   --prompt-data ${PROMPT_DATA}
   --input-key prompt
   --label-key label
   --num-rollout 1000
   --rollout-batch-size 4  # Reduced for faster debugging and lower memory usage
   --rollout-max-response-len 8192  # Extended for multi-turn context accumulation
   --rollout-temperature 1.0  # Higher for code diversity
   --rollout-shuffle
   --n-samples-per-prompt 8  # Generate 8 responses per prompt for pass@8
   --global-batch-size 32  
   --balance-data
   --max-turns 3  # Multi-turn dialogue horizon
   --gamma 0.4  # Discount factor for aggregated return
)

PERF_ARGS=(
   --tensor-model-parallel-size ${TP_SIZE}
   --sequence-parallel
   --pipeline-model-parallel-size ${PP_SIZE}
   --context-parallel-size ${CP_SIZE}
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --grad-reduce-in-bf16
   # --micro-batch-size 1
   # --ref-micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project TF-NV-singleturn-qwen3-8B-sft
   --wandb-group TF-Qwen3-8B-SFT-KBench-SingleTurn
   --wandb-key ${WANDB_KEY}
)

# Launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-"12345"}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 6 --disable-usage-stats

# Wait for Ray to be ready
sleep 5

# Check Ray status (use GCS address, not HTTP URL)
echo "Checking Ray cluster status..."
ray status

# Submit the training job
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "/root/Megatron-LM/",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_CUMEM_ENABLE": "0"
     }
   }' \
   -- python3 SLIME/train_async.py \
   --num-epoch 1000 \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --rollout-num-gpus 2 \
   --rollout-num-gpus-per-engine 1 \
   --sglang-mem-fraction-static 0.8 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   --agent-rollout-buffer-url http://${MASTER_ADDR}:8889 \
   --disable-rewards-normalization \
   --offload-old-actor \
   --offload-ref \
   --loss-mask-type qwen \
   --sglang-log-level error \
   --input-key prompt \
   --log-passrate \
   --rollout-task-type kernelbench