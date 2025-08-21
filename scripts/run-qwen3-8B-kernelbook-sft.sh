#!/bin/bash

# ---- clean up prior runs ----
pkill -9 sglang || true
sleep 3
ray stop --force || true
pkill -9 ray || true
pkill -9 python || true
sleep 3
pkill -9 ray || true
pkill -9 python || true

set -ex

# ---- runtime basics ----
export PYTHONBUFFERED=16
# optional: export WANDB_KEY=xxxxx
export WANDB_KEY=${WANDB_KEY:-"0db9fd073cc9e49c8bcec2b0a6929792ecb64e4e"}

# Enhanced memory management to prevent OOM
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.8"
export CUDA_LAUNCH_BLOCKING=0  # Async kernel launches for better memory management

# Single-node NVLink detection
NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l || echo 0)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-8B.sh"     # defines MODEL_ARGS

# ---- checkpoints ----
# RESUMING FROM CHECKPOINT: Will auto-detect and resume from step 174
CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-8B/
   --ref-load      /root/Qwen3-8B_torch_dist
   --load          /root/Qwen3-8B-slime-kernelbook-sft/
   --save          /root/Qwen3-8B-slime-kernelbook-sft/
   --save-interval 175
   # NOTE: --finetune flag removed - it resets iteration to 0!
   # Checkpoint will be auto-loaded since latest_checkpointed_iteration.txt exists
)

# ---- SFT task (KernelBook, PyTorch→Triton) ----
# Using original batch size for checkpoint compatibility
SFT_ARGS=(
   --rollout-function-path slime.rollout.sft_example.generate_rollout
   --prompt-data /root/kernel_book/kernelbook_sft_final_combined.parquet
   --input-key messages
   --rollout-shuffle
   --num-epoch 1
   --rollout-batch-size 32    # Original batch size for checkpoint compatibility
   --global-batch-size 32      # Original batch size to match optimizer state

   --loss-type sft_loss
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --debug-train-only
)

# ---- Performance / parallelism: TP=2, CP=4, PP=1 ⇒ DP=1 ----
PERF_ARGS=(
   --tensor-model-parallel-size 2
   --pipeline-model-parallel-size 1
   --sequence-parallel
   --context-parallel-size 4

   --bf16

   # Recompute: full+uniform to minimize peak activations during RS
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 12

   # Use sample-based microbatching first for stability
   --micro-batch-size 1

   # Keep overlaps off to reduce temp buffers
   # --tp-comm-overlap
   # --overlap-grad-reduce
   --use-distributed-optimizer
)

# ---- Optimizer / schedule ----
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-decay-style cosine
   --min-lr 1e-6
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.95

   --lr-warmup-fraction 0.03
   --clip-grad 1.0
)

# ---- WandB (enable if desired) ----
WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-kernelbook-sft-combined
   --wandb-group qwen3-8B-kernelbook-sft-combined
   --wandb-key ${WANDB_KEY}
)

# ---- Misc runtime ----
MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0

   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32

   # IMPORTANT: CP>1 requires the CP-aware attention path; do NOT force flash
   # --attention-backend flash
)


# define empty arrays if not provided elsewhere
DISTRIBUTED_ARGS=()
EVAL_ARGS=()

# ---- Ray launcher (single node, 8 GPUs) ----
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats

# prefer NVLink, disable IB on single node; also limit CUDA connections for determinism
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"NCCL_IB_DISABLE\": \"1\"
  }
}"

echo "==== Parallelism ===="
echo "TP=2, PP=1, CP=4 → world=8 ⇒ DP=1"
echo "BF16; full recompute (12 layers); batch_size=32"
echo "RESUMING FROM CHECKPOINT: iter_0000174 (will continue from step 175)"
echo "Memory optimizations: gradient recomputation, enhanced CUDA allocation"
echo "====================="

echo "[DEBUG] MODEL_ARGS:"; printf ' %q' "${MODEL_ARGS[@]}"; echo
echo "[DEBUG] PERF_ARGS :" ; printf ' %q' "${PERF_ARGS[@]}"; echo

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${SFT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${DISTRIBUTED_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${MISC_ARGS[@]}"