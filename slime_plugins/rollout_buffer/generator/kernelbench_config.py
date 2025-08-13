"""Configuration for KernelBench reward and evaluation settings."""

# Reward thresholds
KERNELBENCH_REWARDS = {
    "compilation": 0.1,  # Reward for successful compilation
    "correctness": 0.3,  # Reward for correct implementation
    "max_reward": 2.3,  # Maximum valid reward (0.3 correctness + 2.0 max performance)
}

# Pass@k calculation thresholds
KERNELBENCH_PASSRATE_THRESHOLDS = {
    "compilation": 0.1,  # Threshold for counting as successful compilation
    "correctness": 0.3,  # Threshold for counting as correct implementation
}

# Validation settings
KERNELBENCH_VALIDATION = {
    "require_triton_jit": True,  # Require @triton.jit decorator
    "require_triton_ops": False,  # Loosened: Don't require actual Triton operations for now
    "allow_torch_in_kernel": True,  # Allow torch ops since we're not requiring tl.* ops
}
