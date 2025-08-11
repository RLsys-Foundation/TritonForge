#!/bin/bash

# Kernel Code Generation Agent Training Script - Qwen3-8B
# This script trains Qwen3-8B (fine-tuned on KernelBook) for multi-turn kernel generation

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

# Model parallelism configuration - Adjust based on your GPU setup
export TP_SIZE=2    # Tensor parallelism
export PP_SIZE=1    # Pipeline parallelism  
export CP_SIZE=2    # Context parallelism (total GPUs = TP * PP * CP = 4)

# Model paths
PROJECT_ROOT=/root
# Using Qwen3-8B SFT model (fine-tuned on KernelBook data, iteration 566)
export MCORE_MODEL_PATH=/root/Megatron-models/qwen3-8b-kernelbook-sft-megatron
export HF_MODEL_PATH=/root/Huggingface-models/qwen3-8b-kernelbook-sft-hf
export PROMPT_DATA=/root/slime/data/kernel_bench/kernel_bench_triton_level_1_2.jsonl
export MCORE_MODEL_PATH_SAVE=/root/Megatron-models/qwen3-8b-kernelbook-rl-megatron

# Qwen3-8B model architecture
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
   --vocab-size 152576
   --kv-channels 128
   --qk-layernorm
   --untie-embeddings-and-output-weights
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --attention-dropout 0.0
   --hidden-dropout 0.0
)

CKPT_ARGS=(
   --hf-checkpoint ${HF_MODEL_PATH}
   --ref-load ${MCORE_MODEL_PATH}
   --load ${MCORE_MODEL_PATH}  # Start from fine-tuned checkpoint
   --save-interval 50  # Save every 50 iterations
   --save ${MCORE_MODEL_PATH_SAVE}
)

ROLLOUT_ARGS=(
   --rollout-function-path slime.rollout.agent_rollout.generate_rollout
   --rm-type kernelbench_multiturn
   --prompt-data ${PROMPT_DATA}
   --input-key prompt
   --label-key label
   --num-rollout 2000  # Total rollouts to generate
   --rollout-batch-size 4  # Batch size for rollout generation
   --rollout-max-response-len 12288  # Extended for multi-turn context
   --rollout-temperature 0.8  # Temperature for diversity
   --rollout-shuffle
   --n-samples-per-prompt 4  # Generate 4 responses per prompt
   --global-batch-size 16  # Global batch size for training
   --balance-data
   --max-turns 3  # Multi-turn dialogue horizon
   --gamma 0.4  # Discount factor for aggregated return
)

EVAL_ARGS=(
   --eval-interval 50
   --eval-prompt-data kernelbench ${PROMPT_DATA}
   --n-samples-per-eval-prompt 8
   --eval-max-response-len 12288
   --eval-temperature 0.7
   --eval-top-p 0.95
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
   --recompute-num-layers 2

   # Dynamic batching for efficient GPU usage
   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.01  # Small KL penalty to prevent divergence
   --kl-loss-type low_var_kl
   --entropy-coef 0.001  # Small entropy bonus for exploration
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6  # Conservative learning rate for fine-tuned model
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --clip-grad 1.0  # Gradient clipping for stability

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-multiturn-qwen3-8B
   --wandb-group Qwen3-8B-KernelBook-MultiTurn
   --wandb-key ${WANDB_KEY}
)

# Launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-"12345"}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 6 --disable-usage-stats

# Wait for Ray to be ready
sleep 5

# Check Ray status
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
   ${EVAL_ARGS[@]} \
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
   --rollout-task-type kernelbench_multiturn