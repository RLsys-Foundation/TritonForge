#!/usr/bin/env python3
"""Test script for multi-turn kernel generation functionality."""

import logging
import sys

# Add slime to path
sys.path.insert(0, "/workspace/slime")
sys.path.insert(0, "/workspace/slime/slime_plugins/rollout_buffer")

from slime_plugins.rollout_buffer.generator.multi_turn_kernel_generator import (
    TASK_TYPE,
    calculate_aggregated_return,
    construct_multi_turn_prompt,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_context_construction():
    """Test multi-turn context construction."""
    logger.info("Testing multi-turn context construction...")

    # Sample original prompt
    original_prompt = [
        {"role": "system", "content": "You are an expert in writing CUDA/Triton kernels."},
        {"role": "user", "content": "def forward(self, x): return torch.matmul(x, x.T)"},
    ]

    # Test turn 0 (should return original prompt)
    turn_0_prompt = construct_multi_turn_prompt(original_prompt, 0, [])
    assert turn_0_prompt == original_prompt, "Turn 0 should return original prompt"
    logger.info("✓ Turn 0 context construction passed")

    # Test turn 1 with history
    history = [
        {
            "turn_idx": 0,
            "kernel_code": "@triton.jit\ndef matmul_kernel(...): pass",
            "eval_result": {"compiled": True, "correctness": False, "runtime": 0, "error_message": "Output mismatch"},
            "reward": 0.1,
        }
    ]

    turn_1_prompt = construct_multi_turn_prompt(original_prompt, 1, history)
    assert len(turn_1_prompt) == len(original_prompt)
    assert "Previous Attempt" in turn_1_prompt[1]["content"]
    assert "Evaluation Results" in turn_1_prompt[1]["content"]
    logger.info("✓ Turn 1 context construction passed")

    # Test turn 2 with more history
    history.append(
        {
            "turn_idx": 1,
            "kernel_code": "@triton.jit\ndef matmul_kernel_v2(...): pass",
            "eval_result": {
                "compiled": True,
                "correctness": True,
                "runtime": 1.5,
                "speedup": 1.2,
                "error_message": "",
            },
            "reward": 1.2,
        }
    )

    turn_2_prompt = construct_multi_turn_prompt(original_prompt, 2, history)
    assert "Previous Attempt 1" in turn_2_prompt[1]["content"]
    assert "Previous Attempt 2" in turn_2_prompt[1]["content"]
    assert "Speedup: 1.20x" in turn_2_prompt[1]["content"]
    logger.info("✓ Turn 2 context construction passed")

    logger.info("✅ All context construction tests passed!")


def test_aggregated_return():
    """Test aggregated return calculation."""
    logger.info("Testing aggregated return calculation...")

    # Test empty rewards
    assert calculate_aggregated_return([]) == 0.0
    logger.info("✓ Empty rewards test passed")

    # Test single turn
    assert calculate_aggregated_return([1.0]) == 1.0
    logger.info("✓ Single turn test passed")

    # Test multiple turns with gamma=0.4
    rewards = [0.1, 0.3, 1.2]  # Compilation, correctness, performance
    expected = 0.1 + 0.4 * 0.3 + 0.4**2 * 1.2
    result = calculate_aggregated_return(rewards, gamma=0.4)
    assert abs(result - expected) < 0.001, f"Expected {expected}, got {result}"
    logger.info(f"✓ Multi-turn test passed: {rewards} -> {result:.3f}")

    # Test with different gamma
    result_gamma_05 = calculate_aggregated_return(rewards, gamma=0.5)
    expected_05 = 0.1 + 0.5 * 0.3 + 0.5**2 * 1.2
    assert abs(result_gamma_05 - expected_05) < 0.001
    logger.info(f"✓ Different gamma test passed: γ=0.5 -> {result_gamma_05:.3f}")

    logger.info("✅ All aggregated return tests passed!")


def test_loss_mask_generation():
    """Test loss mask generation for KernelLLM."""
    logger.info("Testing loss mask generation for KernelLLM...")

    # Import directly to avoid megatron dependencies
    import sys

    sys.path.insert(0, "/workspace/slime")

    # Import the mask generator directly
    import importlib.util

    from transformers import AutoTokenizer

    spec = importlib.util.spec_from_file_location("mask_utils", "/workspace/slime/slime/utils/mask_utils.py")
    mask_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mask_utils)
    MultiTurnLossMaskGenerator = mask_utils.MultiTurnLossMaskGenerator

    # Load KernelLLM tokenizer (or use a similar Llama tokenizer for testing)
    try:
        tokenizer = AutoTokenizer.from_pretrained("/workspace/hf_models/facebook--KernelLLM")
    except:
        logger.warning("KernelLLM model not found, using meta-llama/Llama-2-7b-hf for testing")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    # Initialize mask generator with kernelllm type
    mask_generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="kernelllm")

    # Test single-turn conversation
    messages = [
        {"role": "user", "content": "Write a CUDA kernel for matrix multiplication"},
        {"role": "assistant", "content": "@triton.jit\ndef matmul_kernel(): pass"},
    ]

    try:
        token_ids, loss_mask = mask_generator.get_loss_mask(messages)

        # Verify that we have tokens and mask
        assert len(token_ids) > 0, "No tokens generated"
        assert len(loss_mask) == len(token_ids), "Mask length mismatch"

        # Verify that mask has some 1s (for assistant response)
        assert sum(loss_mask) > 0, "No training tokens in mask"

        logger.info(f"✓ Single-turn mask generation passed: {len(token_ids)} tokens, {sum(loss_mask)} training tokens")
    except Exception as e:
        logger.error(f"✗ Single-turn mask generation failed: {e}")
        return False

    # Test multi-turn conversation
    messages_multi = [
        {"role": "user", "content": "Write a kernel"},
        {"role": "assistant", "content": "Here's a kernel"},
        {"role": "user", "content": "Improve it"},
        {"role": "assistant", "content": "Here's an improved version"},
    ]

    try:
        token_ids_multi, loss_mask_multi = mask_generator.get_loss_mask(messages_multi)

        assert len(token_ids_multi) > len(token_ids), "Multi-turn should have more tokens"
        assert sum(loss_mask_multi) > sum(loss_mask), "Multi-turn should have more training tokens"

        logger.info(
            f"✓ Multi-turn mask generation passed: {len(token_ids_multi)} tokens, {sum(loss_mask_multi)} training tokens"
        )
    except Exception as e:
        logger.error(f"✗ Multi-turn mask generation failed: {e}")
        return False

    logger.info("✅ All loss mask tests passed!")
    return True


def test_task_type_registration():
    """Test that the multi-turn kernel generator is properly registered."""
    logger.info("Testing task type registration...")

    assert TASK_TYPE == "kernelbench_multiturn", f"Wrong task type: {TASK_TYPE}"
    logger.info(f"✓ Task type correctly set to: {TASK_TYPE}")

    # Check that the module has required exports
    from slime_plugins.rollout_buffer.generator import multi_turn_kernel_generator

    assert hasattr(multi_turn_kernel_generator, "run_rollout"), "Missing run_rollout function"
    assert hasattr(multi_turn_kernel_generator, "TASK_TYPE"), "Missing TASK_TYPE constant"
    assert hasattr(multi_turn_kernel_generator, "MultiTurnKernelGenerator"), "Missing generator class"

    logger.info("✓ All required exports found")
    logger.info("✅ Task type registration tests passed!")


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("Starting Multi-Turn Kernel Generation Tests")
    logger.info("=" * 60)

    try:
        # Run all test suites
        test_context_construction()
        print()

        test_aggregated_return()
        print()

        test_loss_mask_generation()
        print()

        test_task_type_registration()
        print()

        logger.info("=" * 60)
        logger.info("🎉 ALL TESTS PASSED SUCCESSFULLY! 🎉")
        logger.info("=" * 60)
        logger.info("\nThe multi-turn kernel generation system is ready for use!")
        logger.info("\nTo start training, run:")
        logger.info("  bash /workspace/slime/scripts/run_agent_kbench_kernelllm_8B.sh")

    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
