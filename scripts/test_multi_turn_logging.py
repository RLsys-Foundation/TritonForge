#!/usr/bin/env python3
"""
Test script to verify multi-turn logging functionality.
This script creates a minimal test case to ensure logging works correctly.
"""

import json
import os
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, "/workspace/slime")

from slime_plugins.rollout_buffer.generator.multi_turn_kernel_generator import (
    LOG_DIR,
    log_turn_summary,
    save_multi_turn_data_to_local,
)


def test_turn_logging():
    """Test logging individual turns."""
    print("Testing turn logging...")

    # Create test data for a turn
    test_turn_data = {
        "instance_id": "test_kernel_001",
        "turn_idx": 0,
        "prompt": [
            {"role": "system", "content": "You are an expert CUDA kernel writer."},
            {"role": "user", "content": "Write a matrix multiplication kernel."},
        ],
        "response": "```python\n@triton.jit\ndef matmul_kernel(...):\n    pass\n```",
        "kernel_code": "@triton.jit\ndef matmul_kernel(...):\n    pass",
        "eval_result": {
            "compiled": True,
            "correctness": False,
            "runtime": 5.234,
            "speedup": 0.8,
            "error_message": "Output mismatch",
        },
        "reward": 0.5,
        "extra_info": {"level": 1, "problem_name": "matrix_multiplication", "problem_id": 1},
    }

    # Log the turn
    save_multi_turn_data_to_local(test_turn_data, turn_idx=0, is_final=False)

    # Test turn summary logging
    log_turn_summary(
        0,
        test_turn_data["instance_id"],
        {"eval_result": test_turn_data["eval_result"], "reward": test_turn_data["reward"]},
    )

    print(f"✓ Turn logging test completed")


def test_trajectory_logging():
    """Test logging final trajectory."""
    print("\nTesting trajectory logging...")

    # Create test trajectory data
    test_trajectory_data = {
        "instance_id": "test_kernel_001",
        "reward": 2.5,
        "num_turns": 3,
        "turn_rewards": [0.5, 1.0, 3.0],
        "aggregated_return": 2.5,
        "history": [
            {
                "turn_idx": 0,
                "kernel_code": "# Turn 0 kernel",
                "eval_result": {"compiled": True, "correctness": False, "runtime": 5.234, "speedup": 0.8},
                "reward": 0.5,
            },
            {
                "turn_idx": 1,
                "kernel_code": "# Turn 1 kernel",
                "eval_result": {"compiled": True, "correctness": True, "runtime": 3.123, "speedup": 1.2},
                "reward": 1.0,
            },
            {
                "turn_idx": 2,
                "kernel_code": "# Turn 2 kernel",
                "eval_result": {"compiled": True, "correctness": True, "runtime": 1.567, "speedup": 2.5},
                "reward": 3.0,
            },
        ],
        "messages": [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "User prompt"},
            {"role": "assistant", "content": "Final kernel code"},
        ],
        "execution_details": {
            "final_compiled": True,
            "final_correctness": True,
            "final_runtime": 1.567,
            "final_speedup": 2.5,
        },
        "extra_info": {"level": 1, "problem_name": "matrix_multiplication"},
    }

    # Log the trajectory
    save_multi_turn_data_to_local(test_trajectory_data, is_final=True)

    print(f"✓ Trajectory logging test completed")


def verify_log_files():
    """Verify that log files were created correctly."""
    print("\nVerifying log files...")

    log_dir = Path(LOG_DIR)
    if not log_dir.exists():
        print(f"✗ Log directory {log_dir} does not exist!")
        return False

    # Check for turn files
    turn_files = list(log_dir.glob("turn_test_kernel_001_*.json"))
    if not turn_files:
        print(f"✗ No turn log files found!")
        return False

    print(f"✓ Found {len(turn_files)} turn log file(s)")

    # Check for trajectory files
    traj_files = list(log_dir.glob("trajectory_test_kernel_001_*.json"))
    if not traj_files:
        print(f"✗ No trajectory log files found!")
        return False

    print(f"✓ Found {len(traj_files)} trajectory log file(s)")

    # Verify content of a log file
    if turn_files:
        with open(turn_files[0], "r") as f:
            data = json.load(f)
            if "instance_id" in data and "turn_data" in data:
                print(f"✓ Turn log file has correct structure")
            else:
                print(f"✗ Turn log file missing expected fields")
                return False

    if traj_files:
        with open(traj_files[0], "r") as f:
            data = json.load(f)
            if "instance_id" in data and "history" in data:
                print(f"✓ Trajectory log file has correct structure")
            else:
                print(f"✗ Trajectory log file missing expected fields")
                return False

    return True


def cleanup_test_files():
    """Clean up test log files."""
    print("\nCleaning up test files...")

    log_dir = Path(LOG_DIR)
    if log_dir.exists():
        # Remove test files
        for pattern in ["turn_test_kernel_001_*.json", "trajectory_test_kernel_001_*.json"]:
            for file in log_dir.glob(pattern):
                file.unlink()
                print(f"  Removed {file.name}")


def main():
    print("=" * 60)
    print("MULTI-TURN LOGGING TEST")
    print("=" * 60)

    # Ensure log directory exists
    os.makedirs(LOG_DIR, exist_ok=True)
    print(f"Log directory: {LOG_DIR}")

    # Run tests
    test_turn_logging()
    test_trajectory_logging()

    # Give filesystem time to write
    time.sleep(0.5)

    # Verify files
    success = verify_log_files()

    # Cleanup
    cleanup_test_files()

    print("\n" + "=" * 60)
    if success:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 60)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
