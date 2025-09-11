#!/usr/bin/env python3
"""
Fixed evaluation script for AMD MI300X that properly handles:
1. JSONL template loading
2. Thinking tag extraction
3. Triton code validation
"""

import os
import sys
import json
import time
from pathlib import Path

# Add paths
sys.path.insert(0, '/workspace/KernelBench')
sys.path.insert(0, '/workspace/KernelBench/kernelbench_amd_tools/scripts')

from fixed_triton_extractor import extract_and_validate_triton_code
from src.eval import eval_kernel_against_ref
from src.utils import create_inference_server_from_presets, set_gpu_arch
from src.dataset import construct_kernelbench_dataset

# Environment setup
os.environ['SGLANG_API_KEY'] = 'local-key'
os.environ['OPENAI_API_KEY'] = 'dummy-key'
os.environ['ROCM_HOME'] = '/opt/rocm'
os.environ['HIP_PLATFORM'] = 'amd'
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx942'
os.environ['PYTHONPATH'] = '/workspace/KernelBench:' + os.environ.get('PYTHONPATH', '')

# Paths
JSONL_PATH = "/workspace/slime/data/kernel_bench/kernel_bench_triton_level_1_2.jsonl"
JSONL_FALLBACK_PATH = "/workspace/slime/data/kernel_bench/kernel_bench_triton_level_1.jsonl"


class TritonEvaluator:
    def __init__(self, use_jsonl=True, verbose=True):
        self.use_jsonl = use_jsonl
        self.verbose = verbose
        self.jsonl_data = {}
        
        # Load JSONL data if requested
        if use_jsonl:
            self.load_jsonl_templates()
        
        # Set GPU architecture for MI300X
        set_gpu_arch(["MI300X", "gfx942"])
        
        # Create inference server
        self.inference_server = create_inference_server_from_presets(
            server_type="sglang",
            temperature=0.0,
            max_tokens=8192,
            verbose=verbose
        )
        
        # Load dataset
        self.dataset = construct_kernelbench_dataset(1)  # Level 1
    
    def load_jsonl_templates(self):
        """Load JSONL templates for better prompt formatting."""
        jsonl_path = JSONL_PATH if os.path.exists(JSONL_PATH) else JSONL_FALLBACK_PATH
        
        if not os.path.exists(jsonl_path):
            print(f"Warning: JSONL file not found at {jsonl_path}")
            self.use_jsonl = False
            return
        
        print(f"Loading JSONL templates from: {jsonl_path}")
        with open(jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                problem_id = data['extra_info']['problem_id']
                self.jsonl_data[problem_id] = data
        
        print(f"Loaded {len(self.jsonl_data)} templates")
    
    def get_prompt_for_problem(self, problem_id: int, ref_code: str) -> str:
        """Get prompt either from JSONL template or generate it."""
        problem_id_str = str(problem_id)
        
        if self.use_jsonl and problem_id_str in self.jsonl_data:
            # Use JSONL template
            template = self.jsonl_data[problem_id_str]
            
            # Build chat-style prompt
            messages = template['prompt']
            
            # If using chat format, send as messages
            if isinstance(self.inference_server, dict):
                return messages
            
            # Otherwise, concatenate into single prompt
            prompt = ""
            for msg in messages:
                if msg['role'] == 'system':
                    prompt += f"System: {msg['content']}\n\n"
                elif msg['role'] == 'user':
                    # Replace the architecture in the template with our ref_code
                    user_content = msg['content']
                    # Find and replace the last code block with our ref_code
                    import re
                    # Find the position of "You are given the following architecture:"
                    split_marker = "You are given the following architecture:"
                    if split_marker in user_content:
                        parts = user_content.split(split_marker)
                        # Keep everything before and add our code
                        prompt += parts[0] + split_marker + "\n```python\n" + ref_code + "\n```\n\n"
                        prompt += "Optimize the architecture named Model with custom Triton kernels! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code!"
                    else:
                        prompt += f"User: {user_content}\n\n"
            
            return prompt
        else:
            # Fallback to default prompt generation
            from src.prompt_constructor_triton import prompt_generate_custom_triton_from_prompt_template
            return prompt_generate_custom_triton_from_prompt_template(ref_code)
    
    def evaluate_single_problem(self, problem_id: int, save_dir: str = None):
        """Evaluate a single problem with proper extraction."""
        print(f"\n{'='*60}")
        print(f"Evaluating Problem {problem_id}")
        print(f"{'='*60}")
        
        # Get problem code
        problem_idx = problem_id - 1  # 0-indexed
        if problem_idx >= len(self.dataset):
            print(f"Problem {problem_id} not found in dataset")
            return None
        
        problem_path = self.dataset[problem_idx]
        problem_name = os.path.basename(problem_path)
        
        with open(problem_path, 'r') as f:
            ref_code = f.read()
        
        print(f"Problem: {problem_name}")
        
        # Get prompt
        prompt = self.get_prompt_for_problem(problem_id, ref_code)
        
        if self.verbose:
            print(f"Prompt length: {len(prompt) if isinstance(prompt, str) else 'chat format'}")
        
        # Generate response
        print("Generating Triton code...")
        start_time = time.time()
        
        try:
            response = self.inference_server(prompt)
            generation_time = time.time() - start_time
            print(f"Generation took {generation_time:.2f} seconds")
        except Exception as e:
            print(f"Error during generation: {e}")
            return None
        
        # Extract and validate Triton code
        print("Extracting Triton code...")
        triton_code = extract_and_validate_triton_code(response, verbose=self.verbose)
        
        if not triton_code:
            print("Failed to extract valid Triton code")
            if self.verbose:
                print("\nFull response (first 2000 chars):")
                print("-"*60)
                print(response[:2000])
                print("-"*60)
            return {
                "problem_id": problem_id,
                "problem_name": problem_name,
                "generated": False,
                "error": "Failed to extract valid Triton code"
            }
        
        # Save generated code
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            code_path = os.path.join(save_dir, f"problem_{problem_id}_{problem_name}.py")
            with open(code_path, 'w') as f:
                f.write(triton_code)
            print(f"Saved generated code to: {code_path}")
        
        # Evaluate the kernel
        print("Evaluating kernel...")
        try:
            result = eval_kernel_against_ref(
                ref_code,
                triton_code,
                verbose=self.verbose,
                measure_performance=True,
                num_correct_trials=5,
                num_perf_trials=100,
                backend="triton",
                device=0  # Use GPU 0
            )
            
            return {
                "problem_id": problem_id,
                "problem_name": problem_name,
                "generated": True,
                "compiled": result.compiled,
                "correct": result.correctness,
                "runtime_ms": result.runtime if result.runtime > 0 else None,
                "metadata": result.metadata
            }
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return {
                "problem_id": problem_id,
                "problem_name": problem_name,
                "generated": True,
                "compiled": False,
                "error": str(e)
            }
    
    def run_batch_evaluation(self, problem_ids: list, save_dir: str = None):
        """Run evaluation on multiple problems."""
        results = []
        
        for problem_id in problem_ids:
            result = self.evaluate_single_problem(problem_id, save_dir)
            if result:
                results.append(result)
                
                # Print summary
                print(f"\nProblem {problem_id} Summary:")
                print(f"  Generated: {result.get('generated', False)}")
                print(f"  Compiled: {result.get('compiled', False)}")
                print(f"  Correct: {result.get('correct', False)}")
                if result.get('runtime_ms'):
                    print(f"  Runtime: {result['runtime_ms']:.3f} ms")
        
        return results


def main():
    """Main entry point for testing."""
    import argparse
    parser = argparse.ArgumentParser(description="Triton evaluation with fixed extraction")
    parser.add_argument("--problems", type=str, default="19", 
                       help="Comma-separated problem IDs or 'all' for all problems")
    parser.add_argument("--use-jsonl", action="store_true", default=True,
                       help="Use JSONL templates for prompts")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Verbose output")
    parser.add_argument("--save-dir", type=str, 
                       default=f"runs/triton_eval_{time.strftime('%Y%m%d_%H%M%S')}",
                       help="Directory to save generated code")
    
    args = parser.parse_args()
    
    # Parse problem IDs
    if args.problems == "all":
        problem_ids = list(range(1, 101))  # All Level 1 problems
    else:
        problem_ids = [int(p) for p in args.problems.split(",")]
    
    print("AMD MI300X Triton Evaluation")
    print(f"Problems: {problem_ids}")
    print(f"Using JSONL: {args.use_jsonl}")
    print(f"Save directory: {args.save_dir}")
    print()
    
    # Create evaluator
    evaluator = TritonEvaluator(use_jsonl=args.use_jsonl, verbose=args.verbose)
    
    # Run evaluation
    results = evaluator.run_batch_evaluation(problem_ids, args.save_dir)
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    total = len(results)
    generated = sum(1 for r in results if r.get('generated'))
    compiled = sum(1 for r in results if r.get('compiled'))
    correct = sum(1 for r in results if r.get('correct'))
    
    print(f"Total problems: {total}")
    print(f"Generated: {generated}/{total} ({100*generated/total:.1f}%)")
    print(f"Compiled: {compiled}/{total} ({100*compiled/total:.1f}%)")
    print(f"Correct: {correct}/{total} ({100*correct/total:.1f}%)")
    
    # Save results
    results_file = os.path.join(args.save_dir, "results.json")
    with open(results_file, 'w') as f:
        json.dump({
            "summary": {
                "total": total,
                "generated": generated,
                "compiled": compiled,
                "correct": correct
            },
            "results": results
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()