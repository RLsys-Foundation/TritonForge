#!/bin/bash

# Script to convert Qwen3-8B Megatron checkpoint to HuggingFace format
# This is needed for SGLang to use the model during rollout generation

set -ex

# Model paths
MEGATRON_CKPT="/root/Megatron-models/qwen3-8b-kernelbook-sft-megatron"
HF_OUTPUT="/root/Huggingface-models/qwen3-8b-kernelbook-sft-hf"
ORIGINAL_HF_MODEL="/root/Qwen3-8B"  # Original Qwen3-8B HF model for config

# Check if Megatron checkpoint exists
if [ ! -d "$MEGATRON_CKPT" ]; then
    echo "Error: Megatron checkpoint not found at $MEGATRON_CKPT"
    exit 1
fi

# Check if original HF model exists (needed for config)
if [ ! -d "$ORIGINAL_HF_MODEL" ]; then
    echo "Error: Original HF model not found at $ORIGINAL_HF_MODEL"
    echo "Please download the original Qwen3-8B model from HuggingFace first"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p $(dirname $HF_OUTPUT)

# Method 1: Using the convert_to_hf.py tool with torch_dist format
# This assumes the checkpoint is in torch_dist format
if [ -f "$MEGATRON_CKPT/latest_checkpointed_iteration.txt" ]; then
    # It's a regular Megatron checkpoint, need to use torchrun method
    echo "Converting Megatron checkpoint to HuggingFace format..."
    
    # Get the latest iteration
    ITER=$(cat $MEGATRON_CKPT/latest_checkpointed_iteration.txt)
    CKPT_PATH="$MEGATRON_CKPT/iter_$(printf "%07d" $ITER)"
    
    torchrun --nproc_per_node 4 tools/convert_to_hf.py \
        --load $CKPT_PATH \
        --output-dir $HF_OUTPUT \
        --swiglu \
        --num-layers 36 \
        --hidden-size 4096 \
        --ffn-hidden-size 12288 \
        --num-attention-heads 32 \
        --group-query-attention \
        --num-query-groups 8 \
        --use-rotary-position-embeddings \
        --disable-bias-linear \
        --normalization "RMSNorm" \
        --norm-epsilon 1e-6 \
        --rotary-base 1000000 \
        --vocab-size 151936 \
        --kv-channels 128 \
        --qk-layernorm \
        --untie-embeddings-and-output-weights \
        --tensor-model-parallel-size 2 \
        --pipeline-model-parallel-size 1 \
        --context-parallel-size 2
else
    # It might be a torch_dist format already
    echo "Converting torch_dist checkpoint to HuggingFace format..."
    
    cd /root/slime
    PYTHONPATH=/root/Megatron-LM python tools/convert_torch_dist_to_hf.py \
        --input-dir $MEGATRON_CKPT \
        --output-dir $HF_OUTPUT \
        --origin-hf-dir $ORIGINAL_HF_MODEL
fi

# Verify the conversion
if [ -f "$HF_OUTPUT/config.json" ] && [ -f "$HF_OUTPUT/model.safetensors.index.json" -o -f "$HF_OUTPUT/pytorch_model.bin" -o -f "$HF_OUTPUT/model.safetensors" ]; then
    echo "Conversion successful! HuggingFace model saved at: $HF_OUTPUT"
    
    # Copy tokenizer files from original model if not present
    if [ ! -f "$HF_OUTPUT/tokenizer.json" ]; then
        echo "Copying tokenizer files from original model..."
        cp $ORIGINAL_HF_MODEL/tokenizer* $HF_OUTPUT/ 2>/dev/null || true
        cp $ORIGINAL_HF_MODEL/merges.txt $HF_OUTPUT/ 2>/dev/null || true
        cp $ORIGINAL_HF_MODEL/vocab.json $HF_OUTPUT/ 2>/dev/null || true
        cp $ORIGINAL_HF_MODEL/special_tokens_map.json $HF_OUTPUT/ 2>/dev/null || true
        cp $ORIGINAL_HF_MODEL/generation_config.json $HF_OUTPUT/ 2>/dev/null || true
    fi
    
    echo "Model is ready for use with SGLang!"
else
    echo "Error: Conversion may have failed. Please check the output directory."
    exit 1
fi