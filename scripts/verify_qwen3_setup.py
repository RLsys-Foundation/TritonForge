#!/usr/bin/env python3
"""
Verification script for Qwen3-8B multi-turn RL training setup.
Checks all necessary components before starting training.
"""

import os
import sys
import json
from pathlib import Path


def check_path_exists(path, description):
    """Check if a path exists and report status."""
    if os.path.exists(path):
        print(f"✓ {description}: {path}")
        return True
    else:
        print(f"✗ {description} NOT FOUND: {path}")
        return False


def check_megatron_checkpoint(path):
    """Check if Megatron checkpoint is valid."""
    if not os.path.exists(path):
        print(f"✗ Megatron checkpoint NOT FOUND: {path}")
        return False
    
    # Check for checkpoint structure
    iter_file = os.path.join(path, "latest_checkpointed_iteration.txt")
    if os.path.exists(iter_file):
        with open(iter_file, 'r') as f:
            iteration = f.read().strip()
        iter_dir = os.path.join(path, f"iter_{int(iteration):07d}")
        if os.path.exists(iter_dir):
            print(f"✓ Megatron checkpoint found (iteration {iteration}): {path}")
            return True
    
    # Check if it's a torch_dist format
    if os.path.exists(os.path.join(path, "model.pt")) or \
       os.path.exists(os.path.join(path, "pytorch_model.bin")):
        print(f"✓ Torch dist checkpoint found: {path}")
        return True
    
    print(f"✗ Invalid Megatron checkpoint structure: {path}")
    return False


def check_hf_model(path):
    """Check if HuggingFace model is valid."""
    if not os.path.exists(path):
        print(f"✗ HuggingFace model NOT FOUND: {path}")
        return False
    
    config_path = os.path.join(path, "config.json")
    if not os.path.exists(config_path):
        print(f"✗ HuggingFace model missing config.json: {path}")
        return False
    
    # Check for model weights
    has_weights = (
        os.path.exists(os.path.join(path, "pytorch_model.bin")) or
        os.path.exists(os.path.join(path, "model.safetensors")) or
        os.path.exists(os.path.join(path, "pytorch_model-00001-of-00002.bin"))
    )
    
    if not has_weights:
        print(f"✗ HuggingFace model missing weights: {path}")
        return False
    
    # Check tokenizer
    has_tokenizer = (
        os.path.exists(os.path.join(path, "tokenizer.json")) or
        os.path.exists(os.path.join(path, "tokenizer_config.json"))
    )
    
    if not has_tokenizer:
        print(f"⚠ HuggingFace model missing tokenizer files: {path}")
    
    print(f"✓ HuggingFace model found: {path}")
    return True


def check_data_files():
    """Check if training data files exist."""
    data_path = "/root/slime/data/kernel_bench/kernel_bench_triton_level_1_2.jsonl"
    if not os.path.exists(data_path):
        # Try alternative path
        data_path = "/workspace/slime/data/kernel_bench/kernel_bench_triton_level_1_2.jsonl"
    
    return check_path_exists(data_path, "Training data")


def check_multi_turn_support():
    """Check if multi-turn components are properly configured."""
    print("\n=== Multi-turn Support ===")
    
    # Check multi-turn kernel generator
    generator_path = "/root/slime/slime_plugins/rollout_buffer/generator/multi_turn_kernel_generator.py"
    if not check_path_exists(generator_path, "Multi-turn generator"):
        return False
    
    # Check loss mask utilities
    mask_utils_path = "/root/slime/slime/utils/mask_utils.py"
    if not check_path_exists(mask_utils_path, "Loss mask utilities"):
        return False
    
    # Verify Qwen support in mask utils
    with open(mask_utils_path, 'r') as f:
        content = f.read()
        if 'tokenizer_type == "qwen"' in content:
            print("✓ Qwen tokenizer type supported in loss mask")
        else:
            print("✗ Qwen tokenizer type not found in loss mask utilities")
            return False
    
    return True


def check_scripts():
    """Check if all necessary scripts exist and are executable."""
    print("\n=== Scripts ===")
    
    scripts = [
        "/root/slime/scripts/agent-example-kbench-qwen3-8B.sh",
        "/root/slime/scripts/run_agent_kbench_qwen3_8B.sh",
        "/root/slime/scripts/convert_qwen3_megatron_to_hf.sh"
    ]
    
    all_good = True
    for script in scripts:
        if check_path_exists(script, f"Script {os.path.basename(script)}"):
            # Check if executable
            if not os.access(script, os.X_OK):
                print(f"  ⚠ Script not executable, fixing...")
                os.chmod(script, 0o755)
        else:
            all_good = False
    
    return all_good


def verify_gpu_setup():
    """Check GPU availability."""
    print("\n=== GPU Setup ===")
    try:
        import torch
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"✓ CUDA available with {num_gpus} GPUs")
            for i in range(min(num_gpus, 8)):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("✗ CUDA not available")
            return False
    except ImportError:
        print("✗ PyTorch not installed")
        return False


def main():
    """Main verification function."""
    print("=" * 60)
    print("Qwen3-8B Multi-turn RL Training Setup Verification")
    print("=" * 60)
    
    all_checks_passed = True
    
    # Check model paths
    print("\n=== Model Checkpoints ===")
    megatron_path = "/root/Megatron-models/qwen3-8b-kernelbook-sft-megatron"
    hf_path = "/root/Megatron-models/qwen3-8b-kernelbook-sft-hf"
    
    if not check_megatron_checkpoint(megatron_path):
        all_checks_passed = False
        print("  → Please ensure the Megatron checkpoint exists")
    
    if not check_hf_model(hf_path):
        print("  → HF model not found. Need to convert Megatron checkpoint.")
        print("  → Run: bash scripts/convert_qwen3_megatron_to_hf.sh")
        all_checks_passed = False
    
    # Check data files
    print("\n=== Data Files ===")
    if not check_data_files():
        all_checks_passed = False
        print("  → Please ensure KernelBench data is available")
    
    # Check multi-turn support
    if not check_multi_turn_support():
        all_checks_passed = False
    
    # Check scripts
    if not check_scripts():
        all_checks_passed = False
    
    # Check GPU setup
    if not verify_gpu_setup():
        all_checks_passed = False
    
    # Final summary
    print("\n" + "=" * 60)
    if all_checks_passed:
        print("✓ All checks passed! Ready to start training.")
        print("\nTo start training:")
        print("1. If HF model doesn't exist, run:")
        print("   bash scripts/convert_qwen3_megatron_to_hf.sh")
        print("2. Start training with:")
        print("   bash scripts/run_agent_kbench_qwen3_8B.sh")
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        sys.exit(1)
    
    # Print configuration summary
    print("\n=== Configuration Summary ===")
    print("Model: Qwen3-8B (fine-tuned on KernelBook)")
    print("Training mode: Multi-turn RL (GRPO)")
    print("Max turns: 3")
    print("Discount factor (γ): 0.4")
    print("Loss mask type: qwen")
    print("Rollout task: kernelbench_multiturn")
    print("Parallelism: TP=2, PP=1, CP=2 (4 GPUs for training)")
    print("Rollout GPUs: 2")
    print("=" * 60)


if __name__ == "__main__":
    main()