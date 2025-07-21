#!/usr/bin/env python3
"""
Script for running KernelBench evaluation on AMD MI300X GPUs with SGLang server
Using facebook/KernelLLM model hosted on port 30000
"""

import os
import sys

# Set up environment variables for AMD MI300X
os.environ['ROCM_HOME'] = '/opt/rocm'
os.environ['HIP_PLATFORM'] = 'amd'
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx942'
os.environ['PATH'] = f"{os.environ.get('ROCM_HOME', '/opt/rocm')}/bin:{os.environ.get('PATH', '')}"
os.environ['LD_LIBRARY_PATH'] = f"{os.environ.get('ROCM_HOME', '/opt/rocm')}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

# Set SGLang API key (can be any string for local deployment)
os.environ['SGLANG_API_KEY'] = 'local-key'

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import directly from scripts directory by executing them
import importlib.util

# Load the scripts as modules
def load_script(script_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load required scripts
generate_and_eval_single = load_script(os.path.join(os.path.dirname(__file__), "scripts/generate_and_eval_single_sample.py"), "generate_and_eval_single")
generate_samples = load_script(os.path.join(os.path.dirname(__file__), "scripts/generate_samples.py"), "generate_samples")
eval_from_generations = load_script(os.path.join(os.path.dirname(__file__), "scripts/eval_from_generations.py"), "eval_from_generations")

# Import from these modules
SingleEvalConfig = generate_and_eval_single.EvalConfig
single_eval_main = generate_and_eval_single.main
GenerateConfig = generate_samples.GenerationConfig  # Fixed class name
generate_main = generate_samples.main
BatchEvalConfig = eval_from_generations.EvalConfig
batch_eval_main = eval_from_generations.main

from src.utils import is_amd_gpu, get_amd_gpu_info
import torch
import triton

def check_environment():
    """Check and display AMD GPU environment information."""
    print("AMD MI300X Environment Check")
    print("=" * 60)
    
    # Check if AMD GPU is available
    if is_amd_gpu():
        print("✓ AMD GPU detected")
        
        gpu_info = get_amd_gpu_info()
        if gpu_info:
            print(f"  GPU Information: {', '.join(gpu_info)}")
        
        # Check PyTorch ROCm support
        if hasattr(torch.version, 'hip'):
            print(f"✓ PyTorch ROCm support available (HIP version: {torch.version.hip})")
        else:
            print("✗ PyTorch ROCm support not detected")
            return False
        
        # Check environment variables
        print(f"✓ ROCM_HOME: {os.environ.get('ROCM_HOME')}")
        print(f"✓ HIP_PLATFORM: {os.environ.get('HIP_PLATFORM')}")
        print(f"✓ PYTORCH_ROCM_ARCH: {os.environ.get('PYTORCH_ROCM_ARCH')}")
        
        # Check Triton
        print(f"✓ Triton available (version: {triton.__version__})")
        
        # Check SGLang server
        import requests
        try:
            response = requests.get("http://localhost:30000/v1/models", timeout=5)
            if response.status_code == 200:
                models = response.json()
                print(f"✓ SGLang server running on port 30000")
                print(f"  Available models: {[m['id'] for m in models['data']]}")
            else:
                print("✗ SGLang server not responding correctly")
                return False
        except Exception as e:
            print(f"✗ SGLang server not accessible: {e}")
            return False
    else:
        print("✗ No AMD GPU detected")
        return False
    
    print()
    return True

def run_single_sample_evaluation():
    """Run evaluation on a single problem."""
    print("\n" + "=" * 60)
    print("Running Single Sample Evaluation")
    print("=" * 60)
    
    config = SingleEvalConfig()
    
    # Basic configuration
    config.dataset_src = "local"  # Use local dataset
    config.level = 1              # Start with level 1
    config.problem_id = 1         # First problem
    
    # AMD-specific configuration
    config.gpu_arch = ["MI300X"]  # AMD MI300X architecture
    config.backend = "triton"     # Use Triton backend for AMD
    
    # SGLang server configuration
    config.server_type = "sglang"
    config.server_port = 30000
    config.server_address = "localhost"
    config.model_name = "facebook/KernelLLM"  # This will be ignored by SGLang, it uses the loaded model
    
    # Generation parameters
    config.max_tokens = 4096
    config.temperature = 0.0  # Deterministic generation
    config.num_samples = 1
    
    # Evaluation parameters
    config.n_correctness = 5   # Number of correctness trials
    config.n_trial = 100      # Number of performance trials
    config.timeout = 300      # Timeout for kernel compilation/execution
    
    # Enable verbose logging
    config.verbose = True
    config.verbose_logging()
    
    print(f"\nConfiguration:")
    print(f"  Dataset: {config.dataset_src}")
    print(f"  Level: {config.level}")
    print(f"  Problem ID: {config.problem_id}")
    print(f"  GPU Architecture: {config.gpu_arch}")
    print(f"  Backend: {config.backend}")
    print(f"  Server: {config.server_type} at {config.server_address}:{config.server_port}")
    print()
    
    try:
        single_eval_main(config)
        print("\n✓ Single sample evaluation completed successfully!")
    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

def run_batch_generation_and_evaluation():
    """Run generation and evaluation on multiple problems."""
    print("\n" + "=" * 60)
    print("Running Batch Generation and Evaluation")
    print("=" * 60)
    
    run_name = "amd_mi300x_sglang_test"
    level = 1  # Start with level 1
    
    # Step 1: Generate samples
    print("\nStep 1: Generating samples...")
    gen_config = GenerateConfig()
    gen_config.run_name = run_name
    gen_config.dataset_src = "local"
    gen_config.level = level
    gen_config.num_workers = 10  # Parallel workers for generation
    gen_config.server_type = "sglang"
    gen_config.server_port = 30000
    gen_config.server_address = "localhost"
    gen_config.model_name = "facebook/KernelLLM"
    gen_config.temperature = 0.0
    gen_config.max_tokens = 4096
    gen_config.backend = "triton"  # Specify Triton backend
    
    try:
        generate_main(gen_config)
        print(f"\n✓ Generation completed for {run_name}")
    except Exception as e:
        print(f"\n✗ Error during generation: {e}")
        return
    
    # Step 2: Evaluate generated samples
    print("\nStep 2: Evaluating generated samples...")
    eval_config = BatchEvalConfig()
    eval_config.run_name = run_name
    eval_config.dataset_src = "local"
    eval_config.level = level
    eval_config.gpu_arch = ["MI300X"]
    eval_config.backend = "triton"
    eval_config.num_correct_trials = 5
    eval_config.num_perf_trials = 100
    eval_config.measure_performance = True
    eval_config.num_gpu_devices = 1  # Use 1 GPU for evaluation
    eval_config.timeout = 300
    
    try:
        batch_eval_main(eval_config)
        print(f"\n✓ Evaluation completed for {run_name}")
    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")

def main():
    """Main function."""
    print("KernelBench AMD MI300X + SGLang Evaluation")
    print("=" * 60)
    print(f"Using facebook/KernelLLM model on port 30000")
    print()
    
    # Check environment
    if not check_environment():
        print("\nEnvironment check failed. Please ensure:")
        print("1. AMD MI300X GPU is available")
        print("2. PyTorch with ROCm support is installed")
        print("3. Triton is installed")
        print("4. SGLang server is running on port 30000 with facebook/KernelLLM model")
        return
    
    # Menu
    print("\nWhat would you like to do?")
    print("1. Run single sample evaluation (quick test)")
    print("2. Run batch generation and evaluation (full test)")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        run_single_sample_evaluation()
    elif choice == "2":
        run_batch_generation_and_evaluation()
    elif choice == "3":
        print("Exiting...")
    else:
        print("Invalid choice. Please run the script again.")

if __name__ == "__main__":
    main()