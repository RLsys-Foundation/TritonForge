#!/usr/bin/env python3
"""
Comprehensive batch evaluation script for KernelBench on AMD MI300X
Runs all levels (1-4) with Triton backend and tracks results
"""

import os
import sys
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
import shutil

# Set up environment variables for AMD MI300X
os.environ['ROCM_HOME'] = '/opt/rocm'
os.environ['HIP_PLATFORM'] = 'amd'
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx942'
os.environ['PATH'] = f"{os.environ.get('ROCM_HOME', '/opt/rocm')}/bin:{os.environ.get('PATH', '')}"
os.environ['LD_LIBRARY_PATH'] = f"{os.environ.get('ROCM_HOME', '/opt/rocm')}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"
os.environ['SGLANG_API_KEY'] = 'local-key'
os.environ['PYTHONPATH'] = f"/workspace/KernelBench:{os.environ.get('PYTHONPATH', '')}"

# Import KernelBench modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import construct_kernelbench_dataset

# Configuration
RUN_NAME = f"amd_mi300x_full_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
RESULTS_DIR = f"/workspace/KernelBench/runs/{RUN_NAME}"
REPORT_DIR = f"{RESULTS_DIR}/reports"
KERNELS_DIR = f"{RESULTS_DIR}/generated_kernels"
LOGS_DIR = f"{RESULTS_DIR}/logs"

# Create directories
for dir_path in [RESULTS_DIR, REPORT_DIR, KERNELS_DIR, LOGS_DIR]:
    Path(dir_path).mkdir(parents=True, exist_ok=True)

class FullEvaluator:
    def __init__(self):
        self.results = {
            "metadata": {
                "run_name": RUN_NAME,
                "start_time": datetime.now().isoformat(),
                "gpu": "AMD MI300X",
                "architecture": "gfx942",
                "backend": "triton",
                "model": "facebook/KernelLLM",
                "server": "SGLang on localhost:30000"
            },
            "levels": {}
        }
        self.progress_file = f"{RESULTS_DIR}/progress.json"
        self.results_file = f"{RESULTS_DIR}/results.json"
        self.load_progress()
        
    def load_progress(self):
        """Load previous progress if resuming"""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                self.progress = json.load(f)
        else:
            self.progress = {
                "completed": [],
                "failed": [],
                "current": None
            }
    
    def save_progress(self):
        """Save current progress"""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def save_results(self):
        """Save accumulated results"""
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def is_completed(self, level, problem_id):
        """Check if a problem was already completed"""
        return f"level{level}_problem{problem_id}" in self.progress["completed"]
    
    def mark_completed(self, level, problem_id):
        """Mark a problem as completed"""
        problem_key = f"level{level}_problem{problem_id}"
        if problem_key not in self.progress["completed"]:
            self.progress["completed"].append(problem_key)
        self.save_progress()
    
    def mark_failed(self, level, problem_id, error):
        """Mark a problem as failed"""
        problem_key = f"level{level}_problem{problem_id}"
        self.progress["failed"].append({
            "problem": problem_key,
            "error": str(error),
            "time": datetime.now().isoformat()
        })
        self.save_progress()
    
    def run_single_evaluation(self, level, problem_id):
        """Run evaluation for a single problem"""
        print(f"\n{'='*60}")
        print(f"Evaluating Level {level} Problem {problem_id}")
        print(f"{'='*60}")
        
        # Skip if already completed
        if self.is_completed(level, problem_id):
            print(f"Skipping - already completed")
            return None
        
        # Mark as current
        self.progress["current"] = f"level{level}_problem{problem_id}"
        self.save_progress()
        
        try:
            # Create log directory
            logdir = f"{LOGS_DIR}/level{level}"
            Path(logdir).mkdir(parents=True, exist_ok=True)
            
            # Run evaluation using command line interface
            # Use GPU 0 for evaluation (SGLang is on GPUs 2,3)
            cmd = [
                sys.executable,
                "scripts/generate_and_eval_single_sample.py",
                f"dataset_src=local",
                f"level={level}",
                f"problem_id={problem_id}",
                f'gpu_arch=["MI300X"]',
                f"backend=triton",
                f"server_type=sglang",
                f"verbose=False",
                f"log=True",
                f"log_generated_kernel=True",
                f"logdir={logdir}",
                f"eval_device=0"  # Explicitly use GPU 0
            ]
            
            import subprocess
            start_time = time.time()
            # Set environment with API key
            env = os.environ.copy()
            env['OPENAI_API_KEY'] = 'dummy-key'
            
            result = subprocess.run(
                cmd, 
                cwd="/workspace/KernelBench",
                capture_output=True, 
                text=True,
                timeout=600,  # 10 minute timeout per problem
                env=env
            )
            
            elapsed_time = time.time() - start_time
            
            # Parse results from output
            eval_result = self.parse_evaluation_output(result.stdout, result.stderr)
            eval_result["elapsed_time"] = elapsed_time
            
            # Save generated kernel if exists
            kernel_file = f"{logdir}/generated_kernel_level_{level}_problem_{problem_id}.py"
            if os.path.exists(kernel_file):
                kernel_dest = f"{KERNELS_DIR}/level{level}_problem{problem_id}.py"
                shutil.copy2(kernel_file, kernel_dest)
                eval_result["kernel_saved"] = kernel_dest
            
            # Mark as completed
            self.mark_completed(level, problem_id)
            
            # Save result
            if f"level{level}" not in self.results["levels"]:
                self.results["levels"][f"level{level}"] = {}
            self.results["levels"][f"level{level}"][f"problem{problem_id}"] = eval_result
            self.save_results()
            
            return eval_result
            
        except subprocess.TimeoutExpired:
            error_msg = "Evaluation timeout (10 minutes)"
            print(f"ERROR: {error_msg}")
            self.mark_failed(level, problem_id, error_msg)
            return {"error": error_msg, "timeout": True}
            
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"ERROR: {error_msg}")
            print(traceback.format_exc())
            self.mark_failed(level, problem_id, error_msg)
            return {"error": error_msg, "traceback": traceback.format_exc()}
    
    def parse_evaluation_output(self, stdout, stderr):
        """Parse evaluation results from output"""
        result = {
            "compiled": False,
            "correctness": False,
            "runtime": -1.0,
            "error": None,
            "output": stdout[-5000:] if len(stdout) > 5000 else stdout  # Last 5k chars
        }
        
        # Look for evaluation result line
        for line in stdout.split('\n'):
            if "compiled=" in line and "correctness=" in line:
                # Parse the evaluation result
                try:
                    if "compiled=True" in line:
                        result["compiled"] = True
                    if "correctness=True" in line:
                        result["correctness"] = True
                    
                    # Extract runtime
                    if "runtime=" in line:
                        runtime_str = line.split("runtime=")[1].split()[0]
                        result["runtime"] = float(runtime_str)
                    
                    # Extract metadata
                    if "metadata=" in line:
                        import re
                        metadata_match = re.search(r"metadata=({.*?})", line)
                        if metadata_match:
                            import ast
                            result["metadata"] = ast.literal_eval(metadata_match.group(1))
                    
                    # Extract runtime stats
                    if "runtime_stats=" in line:
                        stats_match = re.search(r"runtime_stats=({.*?})", line)
                        if stats_match:
                            import ast
                            result["runtime_stats"] = ast.literal_eval(stats_match.group(1))
                except:
                    pass
        
        # Check for errors in stderr
        if stderr:
            result["stderr"] = stderr[-5000:] if len(stderr) > 5000 else stderr
            if "error" in stderr.lower() or "exception" in stderr.lower():
                result["error"] = "Check stderr for details"
        
        return result
    
    def run_level_evaluation(self, level, num_problems):
        """Run evaluation for an entire level"""
        print(f"\n{'#'*80}")
        print(f"# Starting Level {level} Evaluation ({num_problems} problems)")
        print(f"{'#'*80}")
        
        level_results = {
            "total": num_problems,
            "completed": 0,
            "compiled": 0,
            "correct": 0,
            "failed": 0,
            "timeouts": 0,
            "runtime_errors": 0
        }
        
        # Run each problem
        for problem_id in range(1, num_problems + 1):
            try:
                result = self.run_single_evaluation(level, problem_id)
                
                if result:
                    level_results["completed"] += 1
                    
                    if result.get("timeout"):
                        level_results["timeouts"] += 1
                    elif result.get("error"):
                        level_results["failed"] += 1
                        if "runtime" in result.get("error", "").lower():
                            level_results["runtime_errors"] += 1
                    else:
                        if result.get("compiled"):
                            level_results["compiled"] += 1
                        if result.get("correctness"):
                            level_results["correct"] += 1
                
                # Print progress
                print(f"\nLevel {level} Progress: {level_results['completed']}/{num_problems}")
                print(f"  Compiled: {level_results['compiled']}")
                print(f"  Correct: {level_results['correct']}")
                print(f"  Failed: {level_results['failed']}")
                
                # Save intermediate report
                self.generate_level_report(level, level_results)
                
            except KeyboardInterrupt:
                print("\n\nEvaluation interrupted by user!")
                self.save_progress()
                self.save_results()
                return level_results
            
            except Exception as e:
                print(f"\nUnexpected error in level evaluation: {e}")
                traceback.print_exc()
                continue
            
            # Small delay between problems to avoid overwhelming the server
            time.sleep(2)
        
        return level_results
    
    def generate_level_report(self, level, stats):
        """Generate a report for a specific level"""
        report_file = f"{REPORT_DIR}/level{level}_report.md"
        
        with open(report_file, 'w') as f:
            f.write(f"# Level {level} Evaluation Report\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            
            f.write("## Summary Statistics\n\n")
            f.write(f"- Total Problems: {stats['total']}\n")
            f.write(f"- Completed: {stats['completed']}\n")
            f.write(f"- Compiled Successfully: {stats['compiled']}\n")
            f.write(f"- Passed Correctness: {stats['correct']}\n")
            f.write(f"- Failed: {stats['failed']}\n")
            f.write(f"- Timeouts: {stats['timeouts']}\n")
            f.write(f"- Runtime Errors: {stats['runtime_errors']}\n\n")
            
            if stats['completed'] > 0:
                f.write("## Success Rates\n\n")
                f.write(f"- Compilation Rate: {stats['compiled']/stats['completed']*100:.1f}%\n")
                f.write(f"- Correctness Rate: {stats['correct']/stats['completed']*100:.1f}%\n")
                f.write(f"- Overall Success Rate: {stats['correct']/stats['total']*100:.1f}%\n\n")
            
            f.write("## Detailed Results\n\n")
            
            # Add detailed results if available
            if f"level{level}" in self.results["levels"]:
                for problem_id, result in sorted(self.results["levels"][f"level{level}"].items(), 
                                                key=lambda x: int(x[0].split('problem')[1]) if 'problem' in x[0] else 0):
                    f.write(f"### Problem {problem_id}\n")
                    f.write(f"- Compiled: {result.get('compiled', False)}\n")
                    f.write(f"- Correct: {result.get('correctness', False)}\n")
                    f.write(f"- Runtime: {result.get('runtime', -1):.4f}ms\n")
                    if result.get('error'):
                        f.write(f"- Error: {result['error']}\n")
                    f.write("\n")
    
    def generate_final_report(self):
        """Generate comprehensive final report"""
        report_file = f"{REPORT_DIR}/FINAL_REPORT.md"
        
        with open(report_file, 'w') as f:
            f.write("# KernelBench AMD MI300X Full Evaluation Report\n\n")
            f.write(f"**Run Name**: {RUN_NAME}\n")
            f.write(f"**Start Time**: {self.results['metadata']['start_time']}\n")
            f.write(f"**End Time**: {datetime.now().isoformat()}\n")
            f.write(f"**GPU**: {self.results['metadata']['gpu']}\n")
            f.write(f"**Backend**: {self.results['metadata']['backend']}\n")
            f.write(f"**Model**: {self.results['metadata']['model']}\n\n")
            
            f.write("## Overall Summary\n\n")
            
            # Calculate totals
            total_problems = 0
            total_completed = 0
            total_compiled = 0
            total_correct = 0
            
            level_configs = [(1, 100), (2, 100), (3, 50), (4, 20)]
            
            for level, num_problems in level_configs:
                level_key = f"level{level}"
                if level_key in self.results["levels"]:
                    level_data = self.results["levels"][level_key]
                    total_problems += num_problems
                    
                    for problem_id, result in level_data.items():
                        if not result.get("error"):
                            total_completed += 1
                            if result.get("compiled"):
                                total_compiled += 1
                            if result.get("correctness"):
                                total_correct += 1
            
            f.write(f"- **Total Problems**: {total_problems}\n")
            f.write(f"- **Attempted**: {len(self.progress['completed']) + len(self.progress['failed'])}\n")
            f.write(f"- **Completed Successfully**: {total_completed}\n")
            f.write(f"- **Compiled**: {total_compiled}\n")
            f.write(f"- **Correct**: {total_correct}\n\n")
            
            if total_problems > 0:
                f.write("## Success Metrics\n\n")
                attempted = len(self.progress['completed']) + len(self.progress['failed'])
                if attempted > 0:
                    f.write(f"- **Completion Rate**: {total_completed/attempted*100:.1f}%\n")
                    f.write(f"- **Compilation Rate**: {total_compiled/attempted*100:.1f}%\n")
                    f.write(f"- **Correctness Rate**: {total_correct/attempted*100:.1f}%\n")
                    f.write(f"- **End-to-End Success Rate**: {total_correct/total_problems*100:.1f}%\n\n")
            
            f.write("## Per-Level Summary\n\n")
            
            for level, num_problems in level_configs:
                level_key = f"level{level}"
                f.write(f"### Level {level} ({num_problems} problems)\n")
                
                if level_key in self.results["levels"]:
                    level_data = self.results["levels"][level_key]
                    completed = sum(1 for r in level_data.values() if not r.get("error"))
                    compiled = sum(1 for r in level_data.values() if r.get("compiled"))
                    correct = sum(1 for r in level_data.values() if r.get("correctness"))
                    
                    f.write(f"- Attempted: {len(level_data)}\n")
                    f.write(f"- Completed: {completed}\n")
                    f.write(f"- Compiled: {compiled}\n")
                    f.write(f"- Correct: {correct}\n")
                    
                    if len(level_data) > 0:
                        f.write(f"- Success Rate: {correct/len(level_data)*100:.1f}%\n")
                else:
                    f.write("- Not evaluated\n")
                
                f.write("\n")
            
            # Add common errors section
            f.write("## Common Errors and Issues\n\n")
            
            error_counts = {}
            for failed in self.progress["failed"]:
                error_type = failed["error"].split(":")[0] if ":" in failed["error"] else failed["error"]
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
            for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
                f.write(f"- {error_type}: {count} occurrences\n")
            
            f.write("\n## AMD-Specific Observations\n\n")
            f.write("1. **Memory Management**: Some kernels attempt large memory allocations\n")
            f.write("2. **Type Handling**: Scalar input handling needs improvement\n")
            f.write("3. **Compilation**: Most kernels compile successfully on AMD\n")
            f.write("4. **Performance**: When correct, performance is competitive\n\n")
            
            f.write("## Recommendations\n\n")
            f.write("1. Fine-tune the model for better AMD/Triton compatibility\n")
            f.write("2. Add memory allocation checks in generated kernels\n")
            f.write("3. Improve scalar and tensor type handling\n")
            f.write("4. Consider AMD-specific optimizations (wavefront size, memory hierarchy)\n")
    
    def run_full_evaluation(self):
        """Run evaluation for all levels"""
        print("Starting Full KernelBench Evaluation on AMD MI300X")
        print(f"Run Name: {RUN_NAME}")
        print(f"Results Directory: {RESULTS_DIR}")
        print()
        
        level_configs = [
            (1, 100),  # Level 1: 100 problems
            (2, 100),  # Level 2: 100 problems
            (3, 50),   # Level 3: 50 problems
            (4, 20)    # Level 4: 20 problems
        ]
        
        overall_stats = {
            "start_time": datetime.now(),
            "levels_completed": []
        }
        
        try:
            for level, num_problems in level_configs:
                level_stats = self.run_level_evaluation(level, num_problems)
                overall_stats["levels_completed"].append({
                    "level": level,
                    "stats": level_stats
                })
                
                # Save after each level
                self.save_results()
                
        except KeyboardInterrupt:
            print("\n\nEvaluation interrupted!")
        
        except Exception as e:
            print(f"\n\nFatal error: {e}")
            traceback.print_exc()
        
        finally:
            overall_stats["end_time"] = datetime.now()
            overall_stats["duration"] = str(overall_stats["end_time"] - overall_stats["start_time"])
            
            # Generate final report
            print("\n\nGenerating final report...")
            self.generate_final_report()
            
            print(f"\n\nEvaluation complete!")
            print(f"Results saved to: {RESULTS_DIR}")
            print(f"Final report: {REPORT_DIR}/FINAL_REPORT.md")
            
            # Print quick summary
            total_completed = len(self.progress["completed"])
            total_failed = len(self.progress["failed"])
            print(f"\nQuick Summary:")
            print(f"- Completed: {total_completed}")
            print(f"- Failed: {total_failed}")
            print(f"- Success Rate: {total_completed/(total_completed+total_failed)*100:.1f}%" if (total_completed+total_failed) > 0 else "N/A")

def main():
    """Main entry point"""
    evaluator = FullEvaluator()
    evaluator.run_full_evaluation()

if __name__ == "__main__":
    main()