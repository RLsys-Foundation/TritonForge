#!/usr/bin/env python3
"""
Example script for running KernelBench evaluation on AMD MI300X GPUs

This script demonstrates how to:
1. Configure KernelBench for AMD GPUs
2. Run single sample evaluation with Triton backend
3. Run batch evaluation from existing generations
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.generate_and_eval_single_sample import EvalConfig as SingleEvalConfig, main as single_eval_main
from scripts.eval_from_generations import EvalConfig as BatchEvalConfig, main as batch_eval_main
from src.utils import is_amd_gpu, get_amd_gpu_info

def run_single_sample_amd():
    """Example: Evaluate a single sample on AMD GPU."""
    print("=" * 60)
    print("Running Single Sample Evaluation on AMD MI300X")
    print("=" * 60)
    
    # Create configuration for AMD
    config = SingleEvalConfig()
    
    # Basic configuration
    config.dataset_src = "local"  # or "huggingface"
    config.level = 1
    config.problem_id = 1
    
    # AMD-specific configuration
    config.gpu_arch = ["MI300X"]  # Specify AMD MI300X architecture
    config.backend = "triton"     # Use Triton backend for AMD
    
    # Model configuration (adjust based on your needs)
    config.server_type = "deepseek"
    config.model_name = "deepseek-coder"
    config.max_tokens = 4096
    config.temperature = 0.0
    
    # Enable verbose logging to see AMD-specific information
    config.verbose = True
    config.verbose_logging()
    
    print(f"Configuration: {config}")
    print()
    
    try:
        # Run the evaluation
        single_eval_main(config)
        print("\nSingle sample evaluation completed successfully!")
    except Exception as e:
        print(f"\nError during evaluation: {e}")

def run_batch_evaluation_amd():
    """Example: Run batch evaluation on AMD GPU."""
    print("\n" + "=" * 60)
    print("Running Batch Evaluation on AMD MI300X")
    print("=" * 60)
    
    # Create configuration for batch evaluation
    config = BatchEvalConfig()
    
    # Specify the run to evaluate
    config.run_name = "amd_test_run"  # Change this to your run name
    config.dataset_src = "local"      # or "huggingface"
    config.level = 1
    
    # AMD-specific configuration
    config.gpu_arch = ["MI300X"]
    config.backend = "triton"
    
    # Evaluation settings
    config.num_correct_trials = 5
    config.num_perf_trials = 100
    config.measure_performance = True
    
    # For AMD, you might want to adjust these based on your system
    config.num_gpu_devices = 1  # Number of AMD GPUs to use
    config.timeout = 300       # Increased timeout for compilation
    
    print(f"Configuration: {config}")
    print()
    
    try:
        # Run the batch evaluation
        batch_eval_main(config)
        print("\nBatch evaluation completed successfully!")
    except Exception as e:
        print(f"\nError during batch evaluation: {e}")

def check_amd_environment():
    """Check and display AMD GPU environment information."""
    print("AMD GPU Environment Check")
    print("=" * 60)
    
    # Check if AMD GPU is available
    if is_amd_gpu():
        print("✓ AMD GPU detected")
        
        gpu_info = get_amd_gpu_info()
        if gpu_info:
            print(f"  GPU Information: {', '.join(gpu_info)}")
        
        # Check PyTorch ROCm support
        import torch
        if hasattr(torch.version, 'hip'):
            print(f"✓ PyTorch ROCm support available (HIP version: {torch.version.hip})")
        else:
            print("✗ PyTorch ROCm support not detected")
            print("  Please install PyTorch with ROCm support:")
            print("  pip install torch --index-url https://download.pytorch.org/whl/rocm5.7")
        
        # Check environment variables
        rocm_home = os.environ.get('ROCM_HOME')
        if rocm_home:
            print(f"✓ ROCM_HOME set to: {rocm_home}")
        else:
            print("! ROCM_HOME not set (optional)")
        
        # Check if Triton is available
        try:
            import triton
            print(f"✓ Triton available (version: {triton.__version__})")
        except ImportError:
            print("✗ Triton not installed")
            print("  Please install Triton: pip install triton")
    else:
        print("✗ No AMD GPU detected")
        print("  This script is designed for AMD GPUs (MI300X)")
        print("  For NVIDIA GPUs, use the standard configuration")
    
    print()

def main():
    """Main function to demonstrate AMD GPU usage."""
    print("KernelBench AMD MI300X Evaluation Example")
    print("=" * 60)
    print()
    
    # First, check the environment
    check_amd_environment()
    
    if not is_amd_gpu():
        print("This example requires an AMD GPU. Exiting.")
        return
    
    # Ask user what they want to do
    print("What would you like to do?")
    print("1. Run single sample evaluation")
    print("2. Run batch evaluation from existing generations")
    print("3. Run both")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        run_single_sample_amd()
    elif choice == "2":
        run_batch_evaluation_amd()
    elif choice == "3":
        run_single_sample_amd()
        run_batch_evaluation_amd()
    else:
        print("Invalid choice. Please run the script again.")

if __name__ == "__main__":
    main()