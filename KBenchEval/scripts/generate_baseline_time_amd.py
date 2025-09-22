#!/usr/bin/env python3
"""
Generate baseline time for KernelBench on AMD MI300X GPUs
This profiles the wall clock time for each KernelBench reference problem

Modified version of generate_baseline_time.py for AMD ROCm/HIP support
"""

import torch
import numpy as np
import os
import json
import sys
from tqdm import tqdm

# AMD GPU Environment Setup
if os.path.exists('/opt/rocm'):
    # AMD GPU settings
    os.environ['ROCM_HOME'] = os.environ.get('ROCM_HOME', '/opt/rocm')
    os.environ['HIP_PLATFORM'] = 'amd'
    os.environ['PYTORCH_ROCM_ARCH'] = os.environ.get('PYTORCH_ROCM_ARCH', 'gfx942')

    # Disable GPU core dumps for stability
    os.environ['HSA_ENABLE_COREDUMP'] = '0'
    os.environ['AMD_LOG_LEVEL'] = '0'
    os.environ['ROCM_DISABLE_CRASH_DUMP'] = '1'
    os.environ['HIP_ENABLE_COREDUMP'] = '0'

    # Performance optimization
    os.environ['HSA_ENABLE_SDMA'] = '0'
    os.environ['GPU_MAX_HW_QUEUES'] = '1'

    print(f"[AMD Setup] ROCm environment configured")
    print(f"[AMD Setup] PYTORCH_ROCM_ARCH={os.environ.get('PYTORCH_ROCM_ARCH')}")
    IS_AMD_GPU = True
else:
    IS_AMD_GPU = False
    print("[NVIDIA Setup] Using CUDA backend")

# Add project root to path
REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)
sys.path.insert(0, REPO_TOP_PATH)

from src.eval import (
    load_original_model_and_inputs,
    time_execution_with_cuda_event,
    get_timing_stats,
    set_seed,
)
from src.dataset import construct_problem_dataset_from_problem_dir
from src.utils import read_file

# Import and set GPU architecture for AMD
if IS_AMD_GPU:
    from src.utils import set_gpu_arch
    set_gpu_arch(["MI300X", "gfx942"])

KERNEL_BENCH_PATH = os.path.join(REPO_TOP_PATH, "KernelBench")
TIMING_DIR = os.path.join(REPO_TOP_PATH, "results", "timing")


def fetch_ref_arch_from_dataset(dataset: list[str],
                                problem_id: int) -> tuple[str, str, str]:
    """
    Fetch the reference architecture from the problem directory
    problem_id should be logical index (1-indexed), matching the problem_id in the problem_name

    Returns:
        ref_arch_path: str, the path to the reference architecture
        ref_arch_name: str, the name of the reference architecture
        ref_arch_src: str, the source code of the reference architecture
    """
    ref_arch_path = None

    for file in dataset:
        if file.split("/")[-1].split("_")[0] == str(problem_id):
            ref_arch_path = file
            break
    if ref_arch_path is None:
        raise ValueError(f"No reference architecture found for problem_id {problem_id}")

    ref_arch_src = read_file(ref_arch_path)

    ref_arch_name = ref_arch_path.split("/")[-1]
    return (ref_arch_path, ref_arch_name, ref_arch_src)


def measure_program_time(
        ref_arch_name: str,
        ref_arch_src: str,
        num_trials: int = 100,
        use_torch_compile: bool = False,
        torch_compile_backend: str="inductor",
        torch_compile_options: str="default",
        device: torch.device="cuda:0",
        verbose: bool = False,
) -> dict:
    """
    Measure the time of a KernelBench reference architecture
    Supports both CUDA and ROCm/HIP backends
    """
    context = {}
    Model, get_init_inputs, get_inputs = load_original_model_and_inputs(
        ref_arch_src, context
    )
    try:
        with torch.no_grad():
            torch.cuda.synchronize(device=device)
            set_seed(42)
            inputs = get_inputs()
            set_seed(42)
            init_inputs = get_init_inputs()
            inputs = [
                x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                for x in inputs
            ]
            init_inputs = [
                x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                for x in init_inputs
            ]

            # Initialize PyTorch model, use this for eager mode execution
            model = Model(*init_inputs)

            if use_torch_compile:
                # For AMD GPUs, some compile modes may not be fully supported
                if IS_AMD_GPU and torch_compile_options in ["max-autotune-no-cudagraphs"]:
                    print(f"[AMD] Skipping {torch_compile_options} mode (not fully supported on ROCm)")
                    return {"error": f"Mode {torch_compile_options} not supported on AMD"}

                # Skip cudagraphs backend on AMD
                if IS_AMD_GPU and torch_compile_backend == "cudagraphs":
                    print(f"[AMD] Skipping cudagraphs backend (not supported on ROCm)")
                    return {"error": "cudagraphs backend not supported on AMD"}

                print(f"Using torch.compile to compile model {ref_arch_name} with {torch_compile_backend} backend and {torch_compile_options} mode")
                model = torch.compile(model, backend=torch_compile_backend, mode=torch_compile_options)
            else:
                print(f"Using PyTorch Eager Execution on {ref_arch_name}")

            model = model.cuda(device=device)
            torch.cuda.synchronize(device=device)

            # Warmup runs for stable measurements
            print(f"Running warmup iterations...")
            for _ in range(min(10, num_trials // 10)):
                _ = model(*inputs)
            torch.cuda.synchronize(device=device)

            # Actual timing
            elapsed_times = time_execution_with_cuda_event(
                model, *inputs, num_trials=num_trials, verbose=verbose, device=device
            )
            runtime_stats = get_timing_stats(elapsed_times, device=device)

            if verbose:
                print(f"{ref_arch_name} {runtime_stats}")

            return runtime_stats
    except Exception as e:
        error_msg = str(e)
        print(f"[Eval] Error in Measuring Performance: {error_msg}")

        # Return error dict for tracking
        return {"error": error_msg[:200]}


def record_baseline_times(use_torch_compile: bool = False,
                          torch_compile_backend: str="inductor",
                          torch_compile_options: str="default",
                          file_name: str="baseline_time.json",
                          levels: list[int] = [1, 2],  # Default to level 1 and 2
                          max_problems_per_level: int = None):  # Option to limit problems for testing
    """
    Generate baseline time for KernelBench on AMD/NVIDIA GPUs
    Creates a single JSON file with both level1 and level2 data
    """
    device = torch.device("cuda:0")
    json_results = {}

    total_problems = 0

    # Process all specified levels and combine into single JSON
    for level in levels:
        PROBLEM_DIR = os.path.join(KERNEL_BENCH_PATH, "level" + str(level))
        dataset = construct_problem_dataset_from_problem_dir(PROBLEM_DIR)
        json_results[f"level{level}"] = {}

        num_problems = len(dataset)
        if max_problems_per_level:
            num_problems = min(num_problems, max_problems_per_level)

        print(f"\n[Level {level}] Processing {num_problems} problems...")

        successful_count = 0
        failed_count = 0

        for problem_id in tqdm(range(1, num_problems + 1), desc=f"Level {level}"):
            try:
                ref_arch_path, ref_arch_name, ref_arch_src = fetch_ref_arch_from_dataset(dataset, problem_id)
                runtime_stats = measure_program_time(
                    ref_arch_name=ref_arch_name,
                    ref_arch_src=ref_arch_src,
                    use_torch_compile=use_torch_compile,
                    torch_compile_backend=torch_compile_backend,
                    torch_compile_options=torch_compile_options,
                    device=device,
                    verbose=False  # Set to True for debugging
                )

                # Check if we got an error dict from measure_program_time
                if "error" not in runtime_stats:
                    json_results[f"level{level}"][ref_arch_name] = runtime_stats
                    successful_count += 1
                else:
                    json_results[f"level{level}"][ref_arch_name] = runtime_stats
                    failed_count += 1

            except Exception as e:
                print(f"[Error] Level {level}, Problem {problem_id} failed: {str(e)}")
                json_results[f"level{level}"][f"problem_{problem_id}_error"] = {"error": str(e)[:200]}
                failed_count += 1

        total_problems += successful_count + failed_count
        print(f"[Level {level}] Complete: {successful_count} successful, {failed_count} failed")

    # Save combined results
    save_path = os.path.join(TIMING_DIR, file_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(json_results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"[Complete] Baseline generation finished")
    print(f"Total problems processed: {total_problems}")
    print(f"Levels included: {levels}")
    print(f"Results saved to: {save_path}")

    # Count total valid baselines
    valid_count = 0
    for level_key, level_data in json_results.items():
        valid_count += sum(1 for v in level_data.values() if "error" not in v)

    print(f"Total valid baselines: {valid_count}")
    print(f"{'='*60}")

    return json_results


def test_single_problem(level_num: int, problem_id: int):
    """
    Test measure_program_time on a particular problem
    Useful for debugging specific issues
    """
    device = torch.device("cuda:0")

    PROBLEM_DIR = os.path.join(KERNEL_BENCH_PATH, "level" + str(level_num))
    dataset = construct_problem_dataset_from_problem_dir(PROBLEM_DIR)

    ref_arch_path, ref_arch_name, ref_arch_src = fetch_ref_arch_from_dataset(dataset, problem_id)

    print(f"\n[Test] Running problem: {ref_arch_name}")
    print(f"[Test] Level: {level_num}, Problem ID: {problem_id}")

    # Test eager execution
    print("\n[Test] Testing PyTorch Eager Execution...")
    exec_stats = measure_program_time(
        ref_arch_name=ref_arch_name,
        ref_arch_src=ref_arch_src,
        use_torch_compile=False,
        device=device,
        verbose=True,
        num_trials=10  # Fewer trials for testing
    )
    print(f"Eager execution stats: {exec_stats}")

    # Test torch.compile if not on AMD or if supported mode
    if not IS_AMD_GPU:
        print("\n[Test] Testing torch.compile (inductor, default)...")
        compile_stats = measure_program_time(
            ref_arch_name=ref_arch_name,
            ref_arch_src=ref_arch_src,
            use_torch_compile=True,
            torch_compile_backend="inductor",
            torch_compile_options="default",
            device=device,
            verbose=True,
            num_trials=10
        )
        print(f"Compile stats: {compile_stats}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate baseline timings for KernelBench")
    parser.add_argument("--test", action="store_true", help="Run in test mode with limited problems")
    parser.add_argument("--level", type=int, nargs='+', default=[1, 2], help="Levels to process (default: 1 2)")
    parser.add_argument("--problem", type=int, help="Test specific problem ID")
    parser.add_argument("--problem-level", type=int, default=1, help="Level for specific problem test")
    parser.add_argument("--hardware", type=str, help="Hardware name for saving results")
    parser.add_argument("--max-problems", type=int, help="Maximum problems per level (for testing)")

    args = parser.parse_args()

    # Detect hardware if not specified
    if not args.hardware:
        if IS_AMD_GPU:
            hardware_name = "MI300X_rocm"
        else:
            # Try to detect NVIDIA GPU type
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                if "H100" in gpu_name:
                    hardware_name = "H100"
                elif "A100" in gpu_name:
                    hardware_name = "A100"
                elif "V100" in gpu_name:
                    hardware_name = "V100"
                else:
                    hardware_name = gpu_name.replace(" ", "_")
            else:
                hardware_name = "unknown"
    else:
        hardware_name = args.hardware

    print(f"\n{'='*60}")
    print(f"KernelBench Baseline Timing Generator")
    print(f"{'='*60}")
    print(f"Hardware: {hardware_name}")
    print(f"Platform: {'AMD ROCm/HIP' if IS_AMD_GPU else 'NVIDIA CUDA'}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA/ROCm available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"{'='*60}\n")

    # Test specific problem if requested
    if args.problem:
        test_single_problem(args.problem_level, args.problem)
        sys.exit(0)

    # Test mode - run with limited problems
    if args.test:
        print("[Test Mode] Running with limited problems for testing...")
        record_baseline_times(
            use_torch_compile=False,
            file_name=f"{hardware_name}/baseline_time_torch_test.json",
            levels=args.level,
            max_problems_per_level=args.max_problems or 5
        )
        sys.exit(0)

    # Full baseline generation
    input(f"\nYou are about to start recording baseline times for {hardware_name}.\nLevels to process: {args.level} (200 total problems for level 1+2)\nThis will take considerable time. Press Enter to continue...")

    # Check if directory exists
    if os.path.exists(os.path.join(TIMING_DIR, hardware_name)):
        response = input(f"\nDirectory {hardware_name} already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            sys.exit(0)

    print(f"\n[Starting] Generating baseline timings for levels: {args.level}")
    print(f"This will generate a combined JSON file with {len(args.level) * 100} total problems")

    # 1. Record PyTorch Eager Execution - COMBINED for all levels
    print("\n[Phase 1] Recording PyTorch Eager Execution baseline for all levels...")
    print(f"Generating combined baseline for Level {', '.join(map(str, args.level))}")
    record_baseline_times(
        use_torch_compile=False,
        file_name=f"{hardware_name}/baseline_time_torch.json",
        levels=args.level,
        max_problems_per_level=args.max_problems
    )

    # 2. Record Torch Compile using Inductor (if supported) - COMBINED for all levels
    if not IS_AMD_GPU:  # Full torch.compile support on NVIDIA
        print("\n[Phase 2] Recording torch.compile baselines for all levels...")
        for torch_compile_mode in ["default", "reduce-overhead", "max-autotune"]:
            print(f"\n  Mode: {torch_compile_mode} (Levels {', '.join(map(str, args.level))})")
            record_baseline_times(
                use_torch_compile=True,
                torch_compile_backend="inductor",
                torch_compile_options=torch_compile_mode,
                file_name=f"{hardware_name}/baseline_time_torch_compile_inductor_{torch_compile_mode}.json",
                levels=args.level,
                max_problems_per_level=args.max_problems
            )
    else:  # Limited torch.compile support on AMD
        print("\n[Phase 2] Recording torch.compile baselines (AMD limited mode) for all levels...")
        # Only test default mode on AMD
        record_baseline_times(
            use_torch_compile=True,
            torch_compile_backend="inductor",
            torch_compile_options="default",
            file_name=f"{hardware_name}/baseline_time_torch_compile_inductor_default.json",
            levels=args.level,
            max_problems_per_level=args.max_problems
        )

    print(f"\n{'='*60}")
    print(f"Baseline timing generation complete!")
    print(f"Results saved in: {TIMING_DIR}/{hardware_name}/")
    print(f"{'='*60}")