#!/bin/bash

# Kernel Code Generation Agent Training Script - KernelLLM
# This script trains KernelLLM to generate optimized CUDA kernels

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
export TP_SIZE=2    # Reduced from 4 to 2 (sufficient for 8B model)
export PP_SIZE=1
export CP_SIZE=2    # Reduced from 4 to 2 (total_model_size = 2*1*2 = 4, matches 4 GPUs)

# Model paths
PROJECT_ROOT=/workspace
export HF_MODEL_PATH=/workspace/hf_models/facebook--KernelLLM
export MCORE_MODEL_PATH=/workspace/megatron_model/KernelLLM-8B-25.02
export PROMPT_DATA=/workspace/slime/data/kernel_bench/kernel_bench_triton_level_1.jsonl
export MCORE_MODEL_PATH_SAVE=/workspace/megatron_model/KernelLLM-8B-25.02_save

# KernelLLM-8B model architecture (Llama 3.1 based)
MODEL_ARGS=(
   --swiglu
   --num-layers 32
   --hidden-size 4096
   --ffn-hidden-size 14336
   --num-attention-heads 32
   --group-query-attention
   --num-query-groups 8
   # --max-position-embeddings 131072
   # --seq-length 12288
   --use-rotary-position-embeddings
   --disable-bias-linear
   --normalization "RMSNorm"
   --norm-epsilon 1e-05
   --rotary-base 500000
   --vocab-size 128256
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --untie-embeddings-and-output-weights
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --use-rope-scaling
   --rope-scaling-factor 8.0
)

CKPT_ARGS=(
   --hf-checkpoint ${HF_MODEL_PATH}
   --ref-load ${MCORE_MODEL_PATH}
   --load ${MCORE_MODEL_PATH_SAVE}
   --save-interval 20  # Save more frequently for kernel training
   --save ${MCORE_MODEL_PATH_SAVE}
)

ROLLOUT_ARGS=(
   --rollout-function-path slime.rollout.agent_rollout.generate_rollout
   --rm-type kernelbench
   --prompt-data ${PROMPT_DATA}
   --input-key prompt
   --label-key label
   --num-rollout 1000  # Reduced for kernel tasks
   --rollout-batch-size 16  # Reduced for small dataset
   --rollout-max-response-len 11264  # Larger for CUDA code
   --rollout-temperature 0.8  # Higher for code diversity
   --rollout-shuffle
   --n-samples-per-prompt 8  # Reduced to 4 responses per prompt
   --global-batch-size 64  # Reduced to match smaller dataset (16 * 4)
   --balance-data
)

# EVAL_ARGS=(
#    --eval-interval 20
#    --eval-prompt-data kernelbench ${PROMPT_DATA}
#    --n-samples-per-eval-prompt 16
#    --eval-max-response-len 12288
#    --eval-top-p 0.95
# )

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
   --max-tokens-per-gpu 3072
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
   --lr 5e-7
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
   --wandb-project slime-dev-atlas-agent
   --wandb-group KernelLLM-8B-KBench-Triton-Level1
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
   -- python3 train_async.py \
   --num-epoch 100 \
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
   --loss-mask-type distill_qwen \
   --sglang-log-level error \
   --input-key prompt \
   --log-passrate \
   --rollout-task-type kernelbench