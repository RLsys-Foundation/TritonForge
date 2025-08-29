#!/usr/bin/env python3
"""
Robust evaluation script that handles cases where model doesn't generate code
Continues evaluation even when extraction fails
"""

import os
import sys
import json
import time
import traceback
from datetime import datetime
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

# Disable GPU core dumps
os.environ['HSA_ENABLE_COREDUMP'] = '0'
os.environ['AMD_LOG_LEVEL'] = '0'
os.environ['ROCM_DISABLE_CRASH_DUMP'] = '1'
os.environ['HIP_ENABLE_COREDUMP'] = '0'


class RobustEvaluator:
    def __init__(self, run_name=None, use_jsonl=True):
        """Initialize robust evaluator with comprehensive error handling."""
        
        # Create run name
        if run_name is None:
            run_name = f"amd_mi300x_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.run_name = run_name
        self.use_jsonl = use_jsonl
        
        # Setup directories
        self.results_dir = f"/workspace/KernelBench/runs/{run_name}"
        self.report_dir = f"{self.results_dir}/reports"
        self.kernels_dir = f"{self.results_dir}/generated_kernels"
        self.logs_dir = f"{self.results_dir}/logs"
        self.failed_dir = f"{self.results_dir}/failed_responses"
        
        for dir_path in [self.results_dir, self.report_dir, self.kernels_dir, 
                         self.logs_dir, self.failed_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize results tracking
        self.results = {
            "metadata": {
                "run_name": run_name,
                "start_time": datetime.now().isoformat(),
                "gpu": "AMD MI300X",
                "architecture": "gfx942",
                "backend": "triton",
                "model": "Qwen3-8B",
                "server": "SGLang on localhost:30000"
            },
            "summary": {
                "total_attempted": 0,
                "code_generated": 0,
                "no_code_generated": 0,
                "compiled": 0,
                "correct": 0,
                "runtime_errors": 0,
                "extraction_failures": 0
            },
            "problems": {}
        }
        
        # Load JSONL templates if requested
        self.jsonl_data = {}
        if use_jsonl:
            self.load_jsonl_templates()
        
        # Set GPU architecture
        set_gpu_arch(["MI300X", "gfx942"])
        
        # Create inference server
        self.inference_server = create_inference_server_from_presets(
            server_type="sglang",
            temperature=0.0,
            max_tokens=8192,
            verbose=False
        )
        
        # Load dataset
        self.datasets = {
            1: construct_kernelbench_dataset(1),
            2: construct_kernelbench_dataset(2),
            3: construct_kernelbench_dataset(3),
            4: construct_kernelbench_dataset(4)
        }
        
        # Progress tracking
        self.progress_file = f"{self.results_dir}/progress.json"
        self.results_file = f"{self.results_dir}/results.json"
        self.load_progress()
    
    def load_jsonl_templates(self):
        """Load JSONL templates for all levels."""
        jsonl_files = {
            1: "/workspace/slime/data/kernel_bench/kernel_bench_triton_level_1_2.jsonl",
            2: "/workspace/slime/data/kernel_bench/kernel_bench_triton_level_2.jsonl",
            3: "/workspace/slime/data/kernel_bench/kernel_bench_triton_level_3.jsonl",
            4: "/workspace/slime/data/kernel_bench/kernel_bench_triton_level_4.jsonl"
        }
        
        for level, jsonl_path in jsonl_files.items():
            if not os.path.exists(jsonl_path):
                # Try alternate path
                alt_path = f"/workspace/slime/data/kernel_bench/kernel_bench_triton_level_{level}.jsonl"
                if os.path.exists(alt_path):
                    jsonl_path = alt_path
                else:
                    print(f"Warning: JSONL file not found for level {level}")
                    continue
            
            self.jsonl_data[level] = {}
            with open(jsonl_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    problem_id = data['extra_info']['problem_id']
                    self.jsonl_data[level][problem_id] = data
            
            print(f"Loaded {len(self.jsonl_data[level])} templates for level {level}")
    
    def load_progress(self):
        """Load previous progress if resuming."""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                self.progress = json.load(f)
        else:
            self.progress = {
                "completed": [],
                "failed": [],
                "no_code": [],
                "current": None
            }
    
    def save_progress(self):
        """Save current progress."""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def save_results(self):
        """Save accumulated results."""
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def get_prompt(self, level, problem_id, ref_code):
        """Get prompt from JSONL or generate default."""
        problem_id_str = str(problem_id)
        
        if self.use_jsonl and level in self.jsonl_data and problem_id_str in self.jsonl_data[level]:
            # Use JSONL template
            template = self.jsonl_data[level][problem_id_str]
            messages = template['prompt']
            
            # Build single string prompt
            prompt = ""
            for msg in messages:
                if msg['role'] == 'system':
                    prompt += f"System: {msg['content']}\n\n"
                elif msg['role'] == 'user':
                    user_content = msg['content']
                    # Replace architecture in template
                    if "You are given the following architecture:" in user_content:
                        parts = user_content.split("You are given the following architecture:")
                        prompt += parts[0] + "You are given the following architecture:\n```python\n"
                        prompt += ref_code + "\n```\n\n"
                        prompt += "Optimize the architecture named Model with custom Triton kernels! "
                        prompt += "Name your optimized output architecture ModelNew. "
                        prompt += "Output the new code in codeblocks. Please generate real code, NOT pseudocode, "
                        prompt += "make sure the code compiles and is fully functional. "
                        prompt += "Just output the new model code, no other text, and NO testing code!"
                    else:
                        prompt += f"User: {user_content}\n\n"
            return prompt
        else:
            # Fallback to default prompt generation
            from src.prompt_constructor_triton import prompt_generate_custom_triton_from_prompt_template
            return prompt_generate_custom_triton_from_prompt_template(ref_code)
    
    def evaluate_single_problem(self, level, problem_id):
        """Evaluate a single problem with robust error handling."""
        
        problem_key = f"level{level}_problem{problem_id}"
        
        # Skip if already completed
        if problem_key in self.progress["completed"]:
            print(f"  Skipping {problem_key} - already completed")
            return None
        
        # Get problem
        dataset = self.datasets[level]
        problem_idx = problem_id - 1
        if problem_idx >= len(dataset):
            print(f"  Problem {problem_id} not in level {level} dataset")
            return None
        
        problem_path = dataset[problem_idx]
        problem_name = os.path.basename(problem_path)
        
        with open(problem_path, 'r') as f:
            ref_code = f.read()
        
        # Initialize result
        result = {
            "level": level,
            "problem_id": problem_id,
            "problem_name": problem_name,
            "problem_key": problem_key,
            "generated": False,
            "code_extracted": False,
            "compiled": False,
            "correct": False,
            "error": None,
            "generation_time": 0,
            "evaluation_time": 0
        }
        
        # Update progress
        self.progress["current"] = problem_key
        self.save_progress()
        
        try:
            # Generate prompt
            prompt = self.get_prompt(level, problem_id, ref_code)
            
            # Generate response
            print(f"  Generating code...", end=" ")
            start_time = time.time()
            response = self.inference_server(prompt)
            generation_time = time.time() - start_time
            result["generation_time"] = generation_time
            print(f"({generation_time:.1f}s)")
            
            # Try to extract code
            print(f"  Extracting Triton code...", end=" ")
            triton_code = extract_and_validate_triton_code(response, verbose=False)
            
            if not triton_code:
                # No valid code extracted - this is expected for some models
                print("NO CODE GENERATED")
                result["error"] = "Model did not generate valid Triton code"
                result["generated"] = False
                
                # Save failed response for analysis
                failed_file = os.path.join(self.failed_dir, f"{problem_key}.txt")
                with open(failed_file, 'w') as f:
                    f.write(f"PROMPT:\n{prompt[:1000]}...\n\n")
                    f.write(f"RESPONSE:\n{response[:5000]}...\n")
                
                # Track as no_code
                self.progress["no_code"].append(problem_key)
                self.results["summary"]["no_code_generated"] += 1
                
            else:
                print("OK")
                result["generated"] = True
                result["code_extracted"] = True
                self.results["summary"]["code_generated"] += 1
                
                # Save generated code
                kernel_file = os.path.join(self.kernels_dir, f"{problem_key}.py")
                with open(kernel_file, 'w') as f:
                    f.write(triton_code)
                
                # Evaluate the kernel
                print(f"  Evaluating kernel...", end=" ")
                eval_start = time.time()
                
                try:
                    eval_result = eval_kernel_against_ref(
                        ref_code,
                        triton_code,
                        verbose=False,
                        measure_performance=True,
                        num_correct_trials=5,
                        num_perf_trials=100,
                        backend="triton",
                        device=0
                    )
                    
                    result["evaluation_time"] = time.time() - eval_start
                    result["compiled"] = eval_result.compiled
                    result["correct"] = eval_result.correctness
                    
                    if eval_result.compiled:
                        self.results["summary"]["compiled"] += 1
                        print(f"COMPILED", end=" ")
                    else:
                        print(f"COMPILE_FAIL", end=" ")
                    
                    if eval_result.correctness:
                        self.results["summary"]["correct"] += 1
                        result["runtime_ms"] = eval_result.runtime
                        print(f"✓ CORRECT ({eval_result.runtime:.2f}ms)")
                    else:
                        print(f"✗ INCORRECT")
                    
                    if eval_result.metadata:
                        result["metadata"] = eval_result.metadata
                        
                except Exception as e:
                    print(f"EVAL_ERROR: {str(e)[:50]}")
                    result["error"] = f"Evaluation error: {str(e)}"
                    self.results["summary"]["runtime_errors"] += 1
            
            # Mark as completed
            self.progress["completed"].append(problem_key)
            
        except Exception as e:
            print(f"  ERROR: {str(e)[:100]}")
            result["error"] = str(e)
            self.progress["failed"].append({
                "problem": problem_key,
                "error": str(e),
                "time": datetime.now().isoformat()
            })
        
        finally:
            # Update summary
            self.results["summary"]["total_attempted"] += 1
            
            # Save result
            self.results["problems"][problem_key] = result
            
            # Save progress
            self.save_progress()
            self.save_results()
        
        return result
    
    def run_level_evaluation(self, level, max_problems=None):
        """Run evaluation for an entire level."""
        
        print(f"\n{'='*60}")
        print(f"LEVEL {level} EVALUATION")
        print(f"{'='*60}")
        
        dataset = self.datasets[level]
        num_problems = min(len(dataset), max_problems) if max_problems else len(dataset)
        
        level_stats = {
            "total": 0,
            "generated": 0,
            "no_code": 0,
            "compiled": 0,
            "correct": 0,
            "errors": 0
        }
        
        for problem_id in range(1, num_problems + 1):
            print(f"\nProblem {problem_id}/{num_problems}:")
            
            result = self.evaluate_single_problem(level, problem_id)
            
            if result:
                level_stats["total"] += 1
                if result["generated"]:
                    level_stats["generated"] += 1
                    if result["compiled"]:
                        level_stats["compiled"] += 1
                    if result["correct"]:
                        level_stats["correct"] += 1
                elif result["error"] and "did not generate" in result["error"]:
                    level_stats["no_code"] += 1
                else:
                    level_stats["errors"] += 1
            
            # Print running statistics every 10 problems
            if problem_id % 10 == 0:
                print(f"\nLevel {level} Progress: {problem_id}/{num_problems}")
                print(f"  Generated: {level_stats['generated']}/{level_stats['total']}")
                print(f"  No Code: {level_stats['no_code']}/{level_stats['total']}")
                print(f"  Compiled: {level_stats['compiled']}/{level_stats['total']}")
                print(f"  Correct: {level_stats['correct']}/{level_stats['total']}")
            
            # Small delay between problems
            time.sleep(1)
        
        # Generate level report
        self.generate_level_report(level, level_stats)
        
        return level_stats
    
    def generate_level_report(self, level, stats):
        """Generate report for a level."""
        report_file = f"{self.report_dir}/level{level}_report.md"
        
        with open(report_file, 'w') as f:
            f.write(f"# Level {level} Evaluation Report\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Model: Qwen3-8B on SGLang\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"- Total Problems: {stats['total']}\n")
            f.write(f"- Code Generated: {stats['generated']}\n")
            f.write(f"- No Code Generated: {stats['no_code']}\n")
            f.write(f"- Compiled Successfully: {stats['compiled']}\n")
            f.write(f"- Passed Correctness: {stats['correct']}\n")
            f.write(f"- Errors: {stats['errors']}\n\n")
            
            if stats['total'] > 0:
                f.write("## Success Rates\n\n")
                f.write(f"- Generation Rate: {stats['generated']/stats['total']*100:.1f}%\n")
                f.write(f"- No Code Rate: {stats['no_code']/stats['total']*100:.1f}%\n")
                f.write(f"- Compilation Rate: {stats['compiled']/stats['total']*100:.1f}%\n")
                f.write(f"- Correctness Rate: {stats['correct']/stats['total']*100:.1f}%\n\n")
            
            f.write("## Notes\n\n")
            if stats['no_code'] > 0:
                f.write(f"- {stats['no_code']} problems resulted in no generated code ")
                f.write("(model may need fine-tuning for this task)\n")
    
    def generate_final_report(self):
        """Generate comprehensive final report."""
        report_file = f"{self.report_dir}/FINAL_REPORT.md"
        
        summary = self.results["summary"]
        
        with open(report_file, 'w') as f:
            f.write("# KernelBench AMD MI300X Evaluation Report\n\n")
            f.write(f"**Run Name**: {self.run_name}\n")
            f.write(f"**Model**: Qwen3-8B on SGLang\n")
            f.write(f"**GPU**: AMD MI300X (gfx942)\n")
            f.write(f"**Backend**: Triton\n")
            f.write(f"**Start Time**: {self.results['metadata']['start_time']}\n")
            f.write(f"**End Time**: {datetime.now().isoformat()}\n\n")
            
            f.write("## Overall Summary\n\n")
            f.write(f"- **Total Attempted**: {summary['total_attempted']}\n")
            f.write(f"- **Code Generated**: {summary['code_generated']}\n")
            f.write(f"- **No Code Generated**: {summary['no_code_generated']}\n")
            f.write(f"- **Compiled**: {summary['compiled']}\n")
            f.write(f"- **Correct**: {summary['correct']}\n")
            f.write(f"- **Runtime Errors**: {summary['runtime_errors']}\n\n")
            
            if summary['total_attempted'] > 0:
                f.write("## Success Metrics\n\n")
                total = summary['total_attempted']
                f.write(f"- **Generation Rate**: {summary['code_generated']/total*100:.1f}%\n")
                f.write(f"- **No Code Rate**: {summary['no_code_generated']/total*100:.1f}%\n")
                f.write(f"- **Compilation Rate**: {summary['compiled']/total*100:.1f}%\n")
                f.write(f"- **Correctness Rate**: {summary['correct']/total*100:.1f}%\n")
                f.write(f"- **Overall Success**: {summary['correct']/total*100:.1f}%\n\n")
            
            f.write("## Key Findings\n\n")
            if summary['no_code_generated'] > 0:
                f.write(f"1. **Model Generation Issues**: {summary['no_code_generated']} problems ")
                f.write("resulted in no valid Triton code being generated.\n")
                f.write("   - The model appears to repeat instructions instead of generating code.\n")
                f.write("   - This suggests the model needs fine-tuning for Triton code generation.\n\n")
            
            if summary['code_generated'] > 0:
                f.write(f"2. **When Code is Generated**:\n")
                if summary['compiled'] > 0:
                    compile_rate = summary['compiled']/summary['code_generated']*100
                    f.write(f"   - Compilation success rate: {compile_rate:.1f}%\n")
                if summary['correct'] > 0:
                    correct_rate = summary['correct']/summary['code_generated']*100
                    f.write(f"   - Correctness rate (when compiled): {correct_rate:.1f}%\n")
            
            f.write("\n## Recommendations\n\n")
            f.write("1. **Model Fine-tuning**: The model needs training on Triton code generation tasks\n")
            f.write("2. **Prompt Engineering**: May need adjusted prompts for Qwen3-8B\n")
            f.write("3. **Alternative Models**: Consider using specialized code models\n")
            f.write("4. **Post-processing**: Add fallback strategies for when no code is generated\n")
    
    def run_full_evaluation(self, levels=None, max_problems_per_level=None):
        """Run complete evaluation across specified levels."""
        
        if levels is None:
            levels = [1, 2, 3, 4]
        
        print("="*80)
        print("KernelBench Robust Evaluation")
        print(f"Run Name: {self.run_name}")
        print(f"Model: Qwen3-8B on SGLang")
        print(f"Levels: {levels}")
        print(f"Results Directory: {self.results_dir}")
        print("="*80)
        
        try:
            for level in levels:
                if level not in self.datasets:
                    print(f"Skipping level {level} - dataset not available")
                    continue
                
                level_stats = self.run_level_evaluation(level, max_problems_per_level)
                
                # Save after each level
                self.save_results()
        
        except KeyboardInterrupt:
            print("\n\nEvaluation interrupted by user!")
        
        except Exception as e:
            print(f"\n\nFatal error: {e}")
            traceback.print_exc()
        
        finally:
            # Generate final report
            print("\n\nGenerating final report...")
            self.generate_final_report()
            
            print(f"\nEvaluation complete!")
            print(f"Results saved to: {self.results_dir}")
            print(f"Final report: {self.report_dir}/FINAL_REPORT.md")
            
            # Print quick summary
            summary = self.results["summary"]
            print(f"\nQuick Summary:")
            print(f"  Total Attempted: {summary['total_attempted']}")
            print(f"  Code Generated: {summary['code_generated']}")
            print(f"  No Code Generated: {summary['no_code_generated']}")
            print(f"  Compiled: {summary['compiled']}")
            print(f"  Correct: {summary['correct']}")


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Robust KernelBench evaluation")
    parser.add_argument("--levels", type=str, default="1",
                       help="Comma-separated levels to evaluate (e.g., '1,2' or 'all')")
    parser.add_argument("--max-problems", type=int, default=None,
                       help="Max problems per level (for testing)")
    parser.add_argument("--run-name", type=str, default=None,
                       help="Custom run name")
    
    args = parser.parse_args()
    
    # Parse levels
    if args.levels == "all":
        levels = [1, 2, 3, 4]
    else:
        levels = [int(l) for l in args.levels.split(",")]
    
    # Create evaluator
    evaluator = RobustEvaluator(run_name=args.run_name)
    
    # Run evaluation
    evaluator.run_full_evaluation(levels=levels, max_problems_per_level=args.max_problems)


if __name__ == "__main__":
    main()