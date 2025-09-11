#!/usr/bin/env python3
"""
Complete evaluation script for Qwen3-8B on AMD MI300X
Handles thinking tags, uses proper chat API, and extracts Triton code correctly
"""

import os
import sys
import json
import time
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
from openai import OpenAI

# Add paths
sys.path.insert(0, '/workspace/KernelBench')

from src.eval import eval_kernel_against_ref
from src.dataset import construct_kernelbench_dataset

# Environment setup
os.environ['SGLANG_API_KEY'] = 'local-key'
os.environ['OPENAI_API_KEY'] = 'dummy-key'
os.environ['ROCM_HOME'] = '/opt/rocm'
os.environ['HIP_PLATFORM'] = 'amd'
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx942'
os.environ['PYTHONPATH'] = '/workspace/KernelBench:' + os.environ.get('PYTHONPATH', '')

# Disable GPU core dumps to prevent crashes
os.environ['HSA_ENABLE_COREDUMP'] = '0'
os.environ['AMD_LOG_LEVEL'] = '0'
os.environ['ROCM_DISABLE_CRASH_DUMP'] = '1'
os.environ['HIP_ENABLE_COREDUMP'] = '0'

# Also disable core dumps at system level
import resource
try:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
except:
    pass  # May not have permission in some environments


class Qwen3TritonExtractor:
    """Handles extraction of Triton code from Qwen3-8B responses."""
    
    @staticmethod
    def remove_thinking_tags(response: str) -> str:
        """Remove <think>...</think> tags and return actual response."""
        if '<think>' in response and '</think>' in response:
            parts = response.split('</think>')
            if len(parts) > 1:
                return parts[1].strip()
        return response
    
    @staticmethod
    def extract_code_blocks(text: str) -> list:
        """Extract all code blocks from text."""
        return re.findall(r'```(?:python)?\n?(.*?)```', text, re.DOTALL)
    
    @staticmethod
    def score_code_block(code: str) -> int:
        """Score a code block based on Triton components."""
        score = 0
        
        # Essential components (high score)
        if 'import triton' in code:
            score += 3
        if '@triton.jit' in code:
            score += 3
        if 'class ModelNew' in code:
            score += 3
        
        # Required functions (medium score)
        if 'def forward' in code:
            score += 2
        if 'def get_inputs' in code:
            score += 2
        if 'def get_init_inputs' in code:
            score += 2
        
        # Supporting components (low score)
        if 'import torch' in code:
            score += 1
        if 'triton.language as tl' in code:
            score += 1
        if 'def triton_' in code or 'def ' in code and '_kernel[' in code:
            score += 1
        
        return score
    
    def extract_triton_code(self, response: str, verbose: bool = False) -> Optional[str]:
        """Extract valid Triton code from Qwen3-8B response."""
        
        # Step 1: Remove thinking tags
        clean_response = self.remove_thinking_tags(response)
        
        if verbose:
            if len(clean_response) < len(response):
                print(f"Removed {len(response) - len(clean_response)} chars of thinking")
        
        # Step 2: Extract code blocks
        code_blocks = self.extract_code_blocks(clean_response)
        
        if verbose:
            print(f"Found {len(code_blocks)} code blocks")
        
        # Step 3: Find best code block
        best_block = None
        best_score = 0
        
        for i, block in enumerate(code_blocks):
            score = self.score_code_block(block)
            if verbose and score > 0:
                print(f"  Block {i+1}: score={score}")
            
            if score > best_score:
                best_score = score
                best_block = block
        
        # Need minimum score of 9 (has essential components)
        if best_block and best_score >= 9:
            if verbose:
                print(f"✓ Found valid Triton code (score: {best_score})")
            return best_block.strip()
        
        # Step 4: Try to find code without markdown
        if not best_block:
            # Look for code pattern without ```
            pattern = r'(import torch.*?import triton.*?class ModelNew.*?def get_init_inputs.*?return.*?)(?=\n\n\n|\Z)'
            match = re.search(pattern, clean_response, re.DOTALL)
            if match:
                code = match.group(1).strip()
                score = self.score_code_block(code)
                if score >= 9:
                    if verbose:
                        print(f"✓ Found Triton code without markdown (score: {score})")
                    return code
        
        if verbose:
            print(f"✗ No valid Triton code found (best score: {best_score})")
            if best_block:
                print("Best block preview:")
                print(best_block[:300])
        
        return None


class Qwen3Evaluator:
    """Evaluator specifically configured for Qwen3-8B model."""
    
    def __init__(self, run_name=None):
        """Initialize evaluator for Qwen3-8B."""
        
        if run_name is None:
            run_name = f"qwen3_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.run_name = run_name
        self.extractor = Qwen3TritonExtractor()
        
        # Setup directories
        self.results_dir = f"/workspace/KernelBench/runs/{run_name}"
        self.report_dir = f"{self.results_dir}/reports"
        self.kernels_dir = f"{self.results_dir}/generated_kernels"
        self.logs_dir = f"{self.results_dir}/logs"
        self.responses_dir = f"{self.results_dir}/responses"
        
        for dir_path in [self.results_dir, self.report_dir, self.kernels_dir, 
                         self.logs_dir, self.responses_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize OpenAI client for SGLang
        self.client = OpenAI(
            api_key="dummy-key",
            base_url="http://localhost:30000/v1",
            timeout=None,
            max_retries=0
        )
        
        # Load datasets
        self.datasets = {
            1: construct_kernelbench_dataset(1),
            2: construct_kernelbench_dataset(2)
        }
        
        # Load JSONL templates
        self.jsonl_data = {}
        self.load_jsonl_templates()
        
        # Results tracking
        self.results = {
            "metadata": {
                "run_name": run_name,
                "start_time": datetime.now().isoformat(),
                "model": "Qwen/Qwen3-8B",
                "gpu": "AMD MI300X",
                "backend": "triton"
            },
            "summary": {
                "total": 0,
                "thinking_detected": 0,
                "code_generated": 0,
                "compiled": 0,
                "correct": 0
            },
            "problems": {}
        }
        
        # Progress tracking
        self.progress_file = f"{self.results_dir}/progress.json"
        self.results_file = f"{self.results_dir}/results.json"
        self.load_progress()
    
    def load_jsonl_templates(self):
        """Load JSONL templates for prompt construction."""
        jsonl_files = {
            1: "/workspace/slime/data/kernel_bench/kernel_bench_triton_level_1_2.jsonl",
            2: "/workspace/slime/data/kernel_bench/kernel_bench_triton_level_2.jsonl"
        }
        
        for level, jsonl_path in jsonl_files.items():
            if not os.path.exists(jsonl_path):
                alt_path = f"/workspace/slime/data/kernel_bench/kernel_bench_triton_level_{level}.jsonl"
                if os.path.exists(alt_path):
                    jsonl_path = alt_path
                else:
                    continue
            
            self.jsonl_data[level] = {}
            with open(jsonl_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    problem_id = data['extra_info']['problem_id']
                    self.jsonl_data[level][problem_id] = data
            
            print(f"Loaded {len(self.jsonl_data[level])} templates for level {level}")
    
    def load_progress(self):
        """Load previous progress."""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                self.progress = json.load(f)
        else:
            self.progress = {"completed": [], "failed": []}
    
    def save_progress(self):
        """Save current progress."""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def save_results(self):
        """Save results."""
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def construct_messages(self, level: int, problem_id: str, ref_code: str) -> list:
        """Construct chat messages from JSONL template or default."""
        
        # Try to use JSONL template
        if level in self.jsonl_data and problem_id in self.jsonl_data[level]:
            template = self.jsonl_data[level][problem_id]
            messages = []
            
            for msg in template['prompt']:
                if msg['role'] == 'system':
                    messages.append(msg)
                elif msg['role'] == 'user':
                    # Replace the architecture in the template
                    content = msg['content']
                    if "You are given the following architecture:" in content:
                        parts = content.split("You are given the following architecture:")
                        new_content = parts[0] + "You are given the following architecture:\n```python\n"
                        new_content += ref_code + "\n```\n\n"
                        new_content += "Optimize the architecture named Model with custom Triton kernels! "
                        new_content += "Name your optimized output architecture ModelNew. "
                        new_content += "Output the new code in codeblocks. Please generate real code, NOT pseudocode, "
                        new_content += "make sure the code compiles and is fully functional. "
                        new_content += "Just output the new model code, no other text, and NO testing code!\n\n"
                        new_content += "After thinking through the optimization, provide the complete code."
                        messages.append({"role": "user", "content": new_content})
                    else:
                        messages.append(msg)
            
            return messages
        
        # Default messages if no template
        return [
            {
                "role": "system",
                "content": "You are an expert in writing Triton kernels for efficient GPU programming."
            },
            {
                "role": "user",
                "content": f"""Optimize this PyTorch model with Triton kernels:

```python
{ref_code}
```

Create a ModelNew class that uses Triton kernels for optimization.
Include all necessary imports, the @triton.jit kernel, wrapper functions, and the ModelNew class.
Output the complete code in a Python code block."""
            }
        ]
    
    def query_model(self, messages: list, max_tokens: int = 8192) -> str:
        """Query Qwen3-8B with proper chat API."""
        
        response = self.client.chat.completions.create(
            model="Qwen/Qwen3-8B",  # Use actual model name
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
            stop=["<|im_end|>", "<|endoftext|>"]
        )
        
        return response.choices[0].message.content
    
    def evaluate_problem(self, level: int, problem_id: int):
        """Evaluate a single problem."""
        
        problem_key = f"level{level}_problem{problem_id}"
        
        # Skip if completed
        if problem_key in self.progress["completed"]:
            print(f"  Skipping {problem_key} (already completed)")
            return None
        
        # Get problem
        dataset = self.datasets[level]
        if problem_id - 1 >= len(dataset):
            return None
        
        problem_path = dataset[problem_id - 1]
        problem_name = os.path.basename(problem_path)
        
        with open(problem_path, 'r') as f:
            ref_code = f.read()
        
        print(f"\n{'='*60}")
        print(f"Evaluating {problem_key}: {problem_name}")
        print(f"{'='*60}")
        
        result = {
            "level": level,
            "problem_id": problem_id,
            "problem_name": problem_name,
            "has_thinking": False,
            "code_generated": False,
            "compiled": False,
            "correct": False
        }
        
        try:
            # Construct messages
            messages = self.construct_messages(level, str(problem_id), ref_code)
            
            # Query model
            print("  Querying Qwen3-8B...", end=" ")
            start_time = time.time()
            response = self.query_model(messages)
            generation_time = time.time() - start_time
            result["generation_time"] = generation_time
            print(f"({generation_time:.1f}s)")
            
            # Save response
            response_file = f"{self.responses_dir}/{problem_key}.txt"
            with open(response_file, 'w') as f:
                f.write(response)
            
            # Check for thinking
            if '<think>' in response:
                result["has_thinking"] = True
                self.results["summary"]["thinking_detected"] += 1
                print("  ✓ Thinking detected")
            
            # Extract Triton code
            print("  Extracting Triton code...", end=" ")
            triton_code = self.extractor.extract_triton_code(response, verbose=False)
            
            if triton_code:
                print("✓")
                result["code_generated"] = True
                self.results["summary"]["code_generated"] += 1
                
                # Save code
                code_file = f"{self.kernels_dir}/{problem_key}.py"
                with open(code_file, 'w') as f:
                    f.write(triton_code)
                
                # Evaluate
                print("  Evaluating kernel...", end=" ")
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
                
                result["compiled"] = eval_result.compiled
                result["correct"] = eval_result.correctness
                
                if eval_result.compiled:
                    self.results["summary"]["compiled"] += 1
                    print("COMPILED", end=" ")
                
                if eval_result.correctness:
                    self.results["summary"]["correct"] += 1
                    result["runtime_ms"] = eval_result.runtime
                    print(f"✓ CORRECT ({eval_result.runtime:.2f}ms)")
                else:
                    print("✗ INCORRECT")
                
                if eval_result.metadata:
                    result["eval_metadata"] = eval_result.metadata
            else:
                print("✗ NO CODE")
                result["error"] = "No valid Triton code extracted"
        
        except Exception as e:
            print(f"  ERROR: {str(e)[:100]}")
            result["error"] = str(e)
            self.progress["failed"].append(problem_key)
        
        finally:
            self.results["summary"]["total"] += 1
            self.results["problems"][problem_key] = result
            self.progress["completed"].append(problem_key)
            self.save_progress()
            self.save_results()
        
        return result
    
    def run_evaluation(self, levels: list, max_problems: Optional[int] = None):
        """Run evaluation for specified levels."""
        
        print(f"\n{'='*70}")
        print(f"Qwen3-8B Triton Code Generation Evaluation")
        print(f"Model: Qwen/Qwen3-8B on SGLang")
        print(f"GPU: AMD MI300X")
        print(f"Levels: {levels}")
        print(f"{'='*70}\n")
        
        for level in levels:
            if level not in self.datasets:
                continue
            
            dataset = self.datasets[level]
            num_problems = min(len(dataset), max_problems) if max_problems else len(dataset)
            
            print(f"\nLevel {level}: {num_problems} problems")
            print("-"*40)
            
            level_stats = {
                "total": 0,
                "thinking": 0,
                "generated": 0,
                "compiled": 0,
                "correct": 0
            }
            
            for problem_id in range(1, num_problems + 1):
                result = self.evaluate_problem(level, problem_id)
                
                if result:
                    level_stats["total"] += 1
                    if result.get("has_thinking"):
                        level_stats["thinking"] += 1
                    if result.get("code_generated"):
                        level_stats["generated"] += 1
                    if result.get("compiled"):
                        level_stats["compiled"] += 1
                    if result.get("correct"):
                        level_stats["correct"] += 1
                
                # Progress update every 5 problems
                if problem_id % 5 == 0:
                    print(f"\n  Progress: {problem_id}/{num_problems}")
                    print(f"    Generated: {level_stats['generated']}/{level_stats['total']}")
                    print(f"    Compiled: {level_stats['compiled']}/{level_stats['total']}")
                    print(f"    Correct: {level_stats['correct']}/{level_stats['total']}")
            
            # Level summary
            self.generate_level_report(level, level_stats)
        
        # Final report
        self.generate_final_report()
    
    def generate_level_report(self, level: int, stats: dict):
        """Generate report for a level."""
        report_file = f"{self.report_dir}/level{level}_report.md"
        
        with open(report_file, 'w') as f:
            f.write(f"# Level {level} Evaluation Report\n\n")
            f.write(f"**Model**: Qwen/Qwen3-8B\n")
            f.write(f"**Generated**: {datetime.now().isoformat()}\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"- Total Problems: {stats['total']}\n")
            f.write(f"- With Thinking: {stats['thinking']}\n")
            f.write(f"- Code Generated: {stats['generated']}\n")
            f.write(f"- Compiled: {stats['compiled']}\n")
            f.write(f"- Correct: {stats['correct']}\n\n")
            
            if stats['total'] > 0:
                f.write("## Success Rates\n\n")
                f.write(f"- Thinking Rate: {stats['thinking']/stats['total']*100:.1f}%\n")
                f.write(f"- Generation Rate: {stats['generated']/stats['total']*100:.1f}%\n")
                f.write(f"- Compilation Rate: {stats['compiled']/stats['total']*100:.1f}%\n")
                f.write(f"- Correctness Rate: {stats['correct']/stats['total']*100:.1f}%\n")
    
    def generate_final_report(self):
        """Generate final report."""
        report_file = f"{self.report_dir}/FINAL_REPORT.md"
        summary = self.results["summary"]
        
        with open(report_file, 'w') as f:
            f.write("# Qwen3-8B Triton Evaluation Report\n\n")
            f.write(f"**Run**: {self.run_name}\n")
            f.write(f"**Model**: Qwen/Qwen3-8B\n")
            f.write(f"**Start**: {self.results['metadata']['start_time']}\n")
            f.write(f"**End**: {datetime.now().isoformat()}\n\n")
            
            f.write("## Overall Results\n\n")
            f.write(f"- Total Problems: {summary['total']}\n")
            f.write(f"- Thinking Detected: {summary['thinking_detected']}\n")
            f.write(f"- Code Generated: {summary['code_generated']}\n")
            f.write(f"- Compiled: {summary['compiled']}\n")
            f.write(f"- Correct: {summary['correct']}\n\n")
            
            if summary['total'] > 0:
                f.write("## Success Metrics\n\n")
                f.write(f"- Thinking Rate: {summary['thinking_detected']/summary['total']*100:.1f}%\n")
                f.write(f"- Generation Rate: {summary['code_generated']/summary['total']*100:.1f}%\n")
                f.write(f"- Compilation Rate: {summary['compiled']/summary['total']*100:.1f}%\n")
                f.write(f"- Correctness Rate: {summary['correct']/summary['total']*100:.1f}%\n")
                f.write(f"- End-to-End Success: {summary['correct']/summary['total']*100:.1f}%\n\n")
            
            f.write("## Key Findings\n\n")
            f.write(f"1. Model shows reasoning with <think> tags in {summary['thinking_detected']} problems\n")
            f.write(f"2. Successfully generated Triton code for {summary['code_generated']} problems\n")
            f.write(f"3. {summary['compiled']} kernels compiled successfully\n")
            f.write(f"4. {summary['correct']} kernels passed correctness tests\n")
        
        print(f"\n{'='*70}")
        print(f"Evaluation Complete!")
        print(f"Results saved to: {self.results_dir}")
        print(f"Final report: {report_file}")
        print(f"{'='*70}")


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Qwen3-8B Triton evaluation")
    parser.add_argument("--levels", type=str, default="1,2",
                       help="Comma-separated levels (e.g., '1,2')")
    parser.add_argument("--max-problems", type=int, default=None,
                       help="Max problems per level")
    parser.add_argument("--run-name", type=str, default=None,
                       help="Custom run name")
    
    args = parser.parse_args()
    
    # Parse levels
    levels = [int(l) for l in args.levels.split(",")]
    
    # Set GPU architecture
    from src.utils import set_gpu_arch
    set_gpu_arch(["MI300X", "gfx942"])
    
    # Create evaluator
    evaluator = Qwen3Evaluator(run_name=args.run_name)
    
    # Run evaluation
    evaluator.run_evaluation(levels, max_problems=args.max_problems)


if __name__ == "__main__":
    main()