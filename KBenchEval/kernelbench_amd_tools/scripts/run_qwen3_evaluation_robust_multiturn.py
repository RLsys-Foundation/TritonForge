#!/usr/bin/env python3
"""
Robust MULTI-TURN evaluation script for Qwen3-8B on AMD MI300X.
Mimics the multi-turn generation logic from SLIME RL training.
Handles memory faults, thinking tags, and continues evaluation after crashes.
"""

import os
import sys
import json
import time
import re
import traceback
import signal
import subprocess
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from openai import OpenAI
import copy

# Add paths
sys.path.insert(0, '/root/TritonForge/KBenchEval/kernelbench_amd_tools/')

from src.eval import eval_kernel_against_ref
from src.dataset import construct_kernelbench_dataset

# Environment setup
os.environ['SGLANG_API_KEY'] = 'local-key'
os.environ['OPENAI_API_KEY'] = 'dummy-key'
os.environ['ROCM_HOME'] = '/opt/rocm'
os.environ['HIP_PLATFORM'] = 'amd'
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx942'
os.environ['PYTHONPATH'] = '/root/TritonForge/KBenchEval/kernelbench_amd_tools/:' + os.environ.get('PYTHONPATH', '')

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

# Multi-turn configuration (matching RL training)
DEFAULT_MAX_TURNS = 3  # Maximum number of turns
DEFAULT_GAMMA = 0.4    # Discount factor for aggregated return

# Reward configuration (matching RL training)
KERNELBENCH_REWARDS = {
    "compilation": 0.1,
    "correctness": 1.0,
    "max_reward": 3.0,  # correctness + max performance bonus
}


def evaluate_kernel_subprocess(ref_code: str, triton_code: str, timeout: int = 60) -> Dict[str, Any]:
    """
    Evaluate kernel in a subprocess to isolate memory faults.
    Returns evaluation result or error information.
    """
    import tempfile
    import base64
    
    # Encode the code strings in base64 to avoid escaping issues
    ref_code_b64 = base64.b64encode(ref_code.encode()).decode()
    triton_code_b64 = base64.b64encode(triton_code.encode()).decode()
    
    # Create a temporary script to run the evaluation
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        eval_script = f"""
import sys
import json
import signal
import os
import base64

# Add paths
sys.path.insert(0, '/root/TritonForge/KBenchEval/kernelbench_amd_tools/')

# Set environment
os.environ['ROCM_HOME'] = '/opt/rocm'
os.environ['HIP_PLATFORM'] = 'amd'
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx942'
os.environ['HSA_ENABLE_COREDUMP'] = '0'
os.environ['AMD_LOG_LEVEL'] = '0'
os.environ['ROCM_DISABLE_CRASH_DUMP'] = '1'

from src.eval import eval_kernel_against_ref

# Timeout handler
def timeout_handler(signum, frame):
    print(json.dumps({{"error": "Evaluation timeout", "compiled": False, "correctness": False}}))
    sys.exit(1)

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm({timeout})

try:
    # Decode the base64 encoded code
    ref_code = base64.b64decode('{ref_code_b64}').decode()
    triton_code = base64.b64decode('{triton_code_b64}').decode()
    
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
    
    result = {{
        "compiled": eval_result.compiled,
        "correctness": eval_result.correctness,
        "runtime": eval_result.runtime if hasattr(eval_result, 'runtime') else None,
        "metadata": eval_result.metadata if hasattr(eval_result, 'metadata') else None
    }}
    
    print(json.dumps(result))
    
except Exception as e:
    result = {{
        "error": str(e)[:500],
        "compiled": False,
        "correctness": False
    }}
    print(json.dumps(result))
"""
        f.write(eval_script)
        script_path = f.name
    
    try:
        # Run evaluation in subprocess
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        # Parse result
        if result.returncode == 0 and result.stdout:
            try:
                return json.loads(result.stdout.strip().split('\n')[-1])
            except:
                return {
                    "error": f"Failed to parse result: {result.stdout[:200]}",
                    "compiled": False,
                    "correctness": False
                }
        elif result.returncode == -signal.SIGSEGV:
            return {
                "error": "Segmentation fault (memory access violation)",
                "compiled": False,
                "correctness": False
            }
        elif result.returncode == -signal.SIGABRT:
            return {
                "error": "Process aborted (likely GPU memory fault)",
                "compiled": False,
                "correctness": False
            }
        else:
            return {
                "error": f"Process exited with code {result.returncode}: {result.stderr[:200]}",
                "compiled": False,
                "correctness": False
            }
            
    except subprocess.TimeoutExpired:
        return {
            "error": f"Evaluation timeout ({timeout}s)",
            "compiled": False,
            "correctness": False
        }
    except Exception as e:
        return {
            "error": f"Subprocess error: {str(e)[:200]}",
            "compiled": False,
            "correctness": False
        }
    finally:
        # Clean up temp file
        try:
            os.unlink(script_path)
        except:
            pass


def strip_thinking_tags(content: str) -> Tuple[str, str]:
    """Strip <think>...</think> tags and return (cleaned_content, thinking_content)."""
    if '<think>' in content and '</think>' in content:
        # Extract thinking content
        think_pattern = r'<think>(.*?)</think>'
        matches = re.findall(think_pattern, content, re.DOTALL)
        thinking_content = '\n'.join(matches) if matches else ""
        
        # Remove thinking tags
        cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        return cleaned, thinking_content
    return content, ""


class Qwen3TritonExtractor:
    """Handles extraction of Triton code from Qwen3-8B responses."""
    
    @staticmethod
    def remove_thinking_tags(response: str) -> str:
        """Remove <think>...</think> tags and return actual response."""
        cleaned, _ = strip_thinking_tags(response)
        return cleaned
    
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


def calculate_aggregated_return(turn_rewards: List[float], gamma: float = DEFAULT_GAMMA) -> float:
    """Calculate aggregated return with discount factor (matching RL training)."""
    if not turn_rewards:
        return 0.0
    
    aggregated_return = 0.0
    for t, reward in enumerate(turn_rewards):
        aggregated_return += (gamma ** t) * reward
    
    return aggregated_return


def calculate_turn_reward(eval_result: Dict[str, Any], baseline_runtime: Optional[float] = None) -> float:
    """Calculate reward for a turn (matching RL training logic)."""
    reward = 0.0
    
    if eval_result.get("compiled"):
        reward = KERNELBENCH_REWARDS["compilation"]
        
        if eval_result.get("correctness"):
            reward = KERNELBENCH_REWARDS["correctness"]
            
            # Add performance reward if runtime is available
            if eval_result.get("runtime") and eval_result["runtime"] > 0 and baseline_runtime:
                speedup = baseline_runtime / eval_result["runtime"]
                # Cap the performance reward at 2.0 for 3x speedup or better
                performance_reward = min(max(speedup - 1.0, 0.0), 2.0)
                reward += performance_reward
    
    return reward


class RobustMultiTurnQwen3Evaluator:
    """Robust MULTI-TURN evaluator for Qwen3-8B with subprocess isolation."""
    
    def __init__(self, run_name=None, use_subprocess=True, eval_timeout=60, 
                 max_turns=DEFAULT_MAX_TURNS, gamma=DEFAULT_GAMMA,
                 use_native_template=True):
        """Initialize multi-turn evaluator for Qwen3-8B."""
        
        if run_name is None:
            run_name = f"qwen3_multiturn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.run_name = run_name
        self.extractor = Qwen3TritonExtractor()
        self.use_subprocess = use_subprocess
        self.eval_timeout = eval_timeout
        self.max_turns = max_turns
        self.gamma = gamma
        self.use_native_template = use_native_template
        
        # Setup directories
        self.results_dir = f"/root/TritonForge/KBenchEval/kernelbench_amd_tools/runs/{run_name}"
        self.report_dir = f"{self.results_dir}/reports"
        self.kernels_dir = f"{self.results_dir}/generated_kernels"
        self.logs_dir = f"{self.results_dir}/logs"
        self.responses_dir = f"{self.results_dir}/responses"
        self.turns_dir = f"{self.results_dir}/turns"  # For multi-turn data
        
        for dir_path in [self.results_dir, self.report_dir, self.kernels_dir, 
                         self.logs_dir, self.responses_dir, self.turns_dir]:
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
        
        # Load baseline timings
        self.baseline_timings = self.load_baseline_timings()
        
        # Results tracking
        self.results = {
            "metadata": {
                "run_name": run_name,
                "start_time": datetime.now().isoformat(),
                "model": "Qwen/Qwen3-8B-fined-tuned",
                "gpu": "AMD MI300X",
                "backend": "triton",
                "use_subprocess": use_subprocess,
                "multi_turn": True,
                "max_turns": max_turns,
                "gamma": gamma,
                "use_native_template": use_native_template
            },
            "summary": {
                "total": 0,
                "multi_turn_success": 0,
                "avg_turns": 0.0,
                "thinking_detected": 0,
                "code_generated": 0,
                "compiled": 0,
                "correct": 0,
                "memory_faults": 0,
                "timeouts": 0,
                "other_errors": 0,
                "early_terminations": 0
            },
            "problems": {}
        }
        
        # Progress tracking
        self.progress_file = f"{self.results_dir}/progress.json"
        self.results_file = f"{self.results_dir}/results.json"
        self.load_progress()
    
    def load_baseline_timings(self) -> Dict[str, Dict[str, float]]:
        """Load baseline timing data for performance comparison."""
        baseline_path = "/workspace/KernelBench/results/timing/H100_atlas/baseline_time_torch_compile_inductor_default.json"
        
        if not os.path.exists(baseline_path):
            print(f"Warning: Baseline timing file not found at {baseline_path}")
            return {}
        
        try:
            with open(baseline_path, 'r') as f:
                data = json.load(f)
                
            # Restructure for easy access
            timings = {}
            for level_key, level_data in data.items():
                level_num = int(level_key.replace("level", ""))
                timings[level_num] = {}
                for problem_file, stats in level_data.items():
                    timings[level_num][problem_file] = stats.get("mean", None)
            
            print(f"Loaded baseline timings for {sum(len(v) for v in timings.values())} problems")
            return timings
        except Exception as e:
            print(f"Error loading baseline timings: {e}")
            return {}
    
    def get_baseline_runtime(self, level: int, problem_name: str) -> Optional[float]:
        """Get baseline runtime for a specific problem."""
        if not self.baseline_timings:
            return None
        
        if level in self.baseline_timings:
            # Try exact match first
            if problem_name in self.baseline_timings[level]:
                return self.baseline_timings[level][problem_name]
            
            # Try with .py extension
            if not problem_name.endswith('.py'):
                problem_name_py = problem_name + '.py'
                if problem_name_py in self.baseline_timings[level]:
                    return self.baseline_timings[level][problem_name_py]
        
        return None
    
    def load_jsonl_templates(self):
        """Load JSONL templates for prompt construction."""
        jsonl_files = {
            1: "/root/TritonForge/SLIME/data/kernel_bench/kernel_bench_triton_level_1_2.jsonl",
            2: "/root/TritonForge/SLIME/data/kernel_bench/kernel_bench_triton_level_2.jsonl"
        }
        
        for level, jsonl_path in jsonl_files.items():
            if not os.path.exists(jsonl_path):
                alt_path = f"/root/TritonForge/SLIME/data/kernel_bench/kernel_bench_triton_level_{level}.jsonl"
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
            self.progress = {"completed": [], "failed": [], "memory_faults": []}
    
    def save_progress(self):
        """Save current progress."""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def save_results(self):
        """Save results."""
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def construct_initial_messages(self, level: int, problem_id: str, ref_code: str) -> list:
        """Construct initial chat messages from JSONL template (matching RL training)."""
        
        # Try to use JSONL template
        if level in self.jsonl_data and problem_id in self.jsonl_data[level]:
            template = self.jsonl_data[level][problem_id]
            # Use the prompt messages directly (already formatted correctly for training)
            return copy.deepcopy(template['prompt'])
        
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
    
    def build_improvement_message(self, turn_idx: int, history_entry: dict) -> dict:
        """Build improvement instruction message for next turn (matching RL training)."""
        improvement_parts = []
        improvement_parts.append("Based on the previous attempt above, generate an improved kernel that:")
        
        eval_result = history_entry["eval_result"]
        
        if not eval_result["compiled"]:
            improvement_parts.append("1. Fixes the compilation errors")
        elif not eval_result["correctness"]:
            improvement_parts.append("1. Fixes the correctness issues")
        else:
            improvement_parts.append("1. Maintains correctness")
        
        improvement_parts.append("2. Improves performance if possible")
        improvement_parts.append("3. Maintains the same functionality as required")
        
        # Add evaluation feedback if available
        if eval_result.get("error_message"):
            improvement_parts.append(f"\nError from previous attempt: {eval_result['error_message'][:200]}")
        
        improvement_parts.append("\nPlease generate the improved kernel code:")
        
        return {
            "role": "user",
            "content": "\n".join(improvement_parts)
        }
    
    def query_model(self, messages: list, max_tokens: int = 8192) -> str:
        """Query Qwen3-8B with proper chat API."""
        
        response = self.client.chat.completions.create(
            model="Qwen/Qwen3-8B",  # Use actual model name
            messages=messages,
            temperature=0.8,  # Match training temperature
            max_tokens=max_tokens,
            stop=["<|im_end|>", "<|endoftext|>"]
        )
        
        return response.choices[0].message.content
    
    def evaluate_problem_multiturn(self, level: int, problem_id: int, problem_name: str, ref_code: str):
        """Evaluate a single problem with multi-turn generation (matching RL training)."""
        
        problem_key = f"level{level}_problem{problem_id}"
        
        print(f"\n{'='*60}")
        print(f"Multi-Turn Evaluation: {problem_key}: {problem_name}")
        print(f"Max turns: {self.max_turns}, Gamma: {self.gamma}")
        print(f"{'='*60}")
        
        # Get baseline runtime for this problem
        baseline_runtime = self.get_baseline_runtime(level, problem_name)
        if baseline_runtime:
            print(f"Baseline runtime: {baseline_runtime:.2f}ms")
        
        # Initialize tracking
        history = []
        turn_rewards = []
        messages = self.construct_initial_messages(level, str(problem_id), ref_code)
        
        # Multi-turn loop (matching RL training logic)
        for turn_idx in range(self.max_turns):
            print(f"\n--- Turn {turn_idx + 1}/{self.max_turns} ---")
            
            # Log current message state
            print(f"Messages: {len(messages)} total, last role: {messages[-1]['role']}")
            
            # Query model
            print("Querying model...", end=" ")
            start_time = time.time()
            
            try:
                assistant_content = self.query_model(messages)
                generation_time = time.time() - start_time
                print(f"({generation_time:.1f}s)")
                
                # Save raw response
                turn_response_file = f"{self.responses_dir}/{problem_key}_turn{turn_idx}.txt"
                with open(turn_response_file, 'w') as f:
                    f.write(assistant_content)
                
            except Exception as e:
                print(f"ERROR: {str(e)}")
                assistant_content = f"Error: Failed to generate response in turn {turn_idx}"
                generation_time = 0
            
            # Add assistant message to conversation
            assistant_message = {
                "role": "assistant",
                "content": assistant_content
            }
            messages.append(assistant_message)
            
            # Check for thinking tags
            has_thinking = '<think>' in assistant_content
            if has_thinking:
                print("✓ Thinking detected")
            
            # Extract and evaluate code
            cleaned_content, thinking_content = strip_thinking_tags(assistant_content)
            kernel_code = self.extractor.extract_triton_code(assistant_content, verbose=False)
            
            eval_result = {
                "compiled": False,
                "correctness": False,
                "runtime": None,
                "error_message": ""
            }
            
            if kernel_code:
                print("✓ Code extracted", end=" ")
                
                # Save kernel code
                kernel_file = f"{self.kernels_dir}/{problem_key}_turn{turn_idx}.py"
                with open(kernel_file, 'w') as f:
                    f.write(kernel_code)
                
                # Evaluate kernel
                print("Evaluating...", end=" ")
                
                if self.use_subprocess:
                    raw_eval_result = evaluate_kernel_subprocess(
                        ref_code,
                        kernel_code,
                        timeout=self.eval_timeout
                    )
                    
                    eval_result["compiled"] = raw_eval_result.get("compiled", False)
                    eval_result["correctness"] = raw_eval_result.get("correctness", False)
                    eval_result["runtime"] = raw_eval_result.get("runtime")
                    eval_result["error_message"] = raw_eval_result.get("error", "")
                    
                    if eval_result["compiled"]:
                        print("COMPILED", end=" ")
                    if eval_result["correctness"]:
                        if eval_result["runtime"]:
                            print(f"✓ CORRECT ({eval_result['runtime']:.2f}ms)", end="")
                        else:
                            print("✓ CORRECT", end="")
                    else:
                        print("✗ INCORRECT", end="")
                    
                    if eval_result["error_message"]:
                        print(f" [{eval_result['error_message'][:50]}...]")
                    else:
                        print()
            else:
                print("✗ No valid code extracted")
                eval_result["error_message"] = "No valid Triton code extracted"
            
            # Calculate turn reward
            turn_reward = calculate_turn_reward(eval_result, baseline_runtime)
            turn_rewards.append(turn_reward)
            
            # Calculate speedup if applicable
            speedup = None
            if eval_result["runtime"] and baseline_runtime:
                speedup = baseline_runtime / eval_result["runtime"]
            
            print(f"Turn reward: {turn_reward:.3f}", end="")
            if speedup:
                print(f" (speedup: {speedup:.2f}x)")
            else:
                print()
            
            # Build history entry
            history_entry = {
                "turn_idx": turn_idx,
                "kernel_code": kernel_code if kernel_code else "",
                "has_thinking": has_thinking,
                "generation_time": generation_time,
                "eval_result": eval_result,
                "reward": turn_reward,
                "speedup": speedup
            }
            
            # Add thinking content if available
            if thinking_content:
                history_entry["thinking_content"] = thinking_content[:500]  # Truncate for storage
            
            history.append(history_entry)
            
            # Save turn data
            turn_data = {
                "turn_idx": turn_idx,
                "messages": copy.deepcopy(messages),
                "history_entry": history_entry,
                "turn_reward": turn_reward,
                "accumulated_rewards": turn_rewards.copy()
            }
            
            turn_file = f"{self.turns_dir}/{problem_key}_turn{turn_idx}.json"
            with open(turn_file, 'w') as f:
                json.dump(turn_data, f, indent=2)
            
            # Early termination conditions (matching RL training)
            if turn_reward >= KERNELBENCH_REWARDS["correctness"] + 1.0:
                print(f"Early termination: achieved good performance (reward: {turn_reward:.3f})")
                self.results["summary"]["early_terminations"] += 1
                break
            
            # Add improvement instruction for next turn (if not the last turn)
            if self.use_native_template and turn_idx < self.max_turns - 1:
                improvement_message = self.build_improvement_message(turn_idx, history_entry)
                messages.append(improvement_message)
                print(f"Added improvement instruction for turn {turn_idx + 2}")
        
        # Calculate aggregated return
        aggregated_return = calculate_aggregated_return(turn_rewards, self.gamma)
        
        # Determine final success metrics
        final_compiled = any(h["eval_result"]["compiled"] for h in history)
        final_correct = any(h["eval_result"]["correctness"] for h in history)
        best_turn = max(range(len(history)), key=lambda i: history[i]["reward"]) if history else None
        
        print(f"\n--- Multi-Turn Summary ---")
        print(f"Turns executed: {len(turn_rewards)}")
        print(f"Turn rewards: {[f'{r:.3f}' for r in turn_rewards]}")
        print(f"Aggregated return: {aggregated_return:.3f}")
        print(f"Best turn: {best_turn + 1 if best_turn is not None else 'N/A'}")
        print(f"Final success: Compiled={final_compiled}, Correct={final_correct}")
        
        # Return comprehensive result
        return {
            "level": level,
            "problem_id": problem_id,
            "problem_name": problem_name,
            "num_turns": len(turn_rewards),
            "turn_rewards": turn_rewards,
            "aggregated_return": aggregated_return,
            "history": history,
            "final_compiled": final_compiled,
            "final_correct": final_correct,
            "best_turn": best_turn,
            "baseline_runtime": baseline_runtime,
            "messages": messages  # Final conversation
        }
    
    def evaluate_problem(self, level: int, problem_id: int):
        """Wrapper to evaluate a single problem with multi-turn generation."""
        
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
        
        try:
            # Run multi-turn evaluation
            result = self.evaluate_problem_multiturn(level, problem_id, problem_name, ref_code)
            
            # Update summary statistics
            self.results["summary"]["total"] += 1
            
            # Count thinking detection across turns
            if any(h.get("has_thinking", False) for h in result["history"]):
                self.results["summary"]["thinking_detected"] += 1
            
            # Count code generation across turns
            if any(h.get("kernel_code") for h in result["history"]):
                self.results["summary"]["code_generated"] += 1
            
            if result["final_compiled"]:
                self.results["summary"]["compiled"] += 1
            
            if result["final_correct"]:
                self.results["summary"]["correct"] += 1
                if result["aggregated_return"] > KERNELBENCH_REWARDS["correctness"]:
                    self.results["summary"]["multi_turn_success"] += 1
            
            # Count errors across turns
            for h in result["history"]:
                error_msg = h["eval_result"].get("error_message", "")
                if "memory" in error_msg.lower() or "segmentation" in error_msg.lower():
                    self.results["summary"]["memory_faults"] += 1
                elif "timeout" in error_msg.lower():
                    self.results["summary"]["timeouts"] += 1
                elif error_msg and not h["eval_result"]["compiled"]:
                    self.results["summary"]["other_errors"] += 1
            
            # Update average turns
            current_total = self.results["summary"]["total"]
            current_avg = self.results["summary"]["avg_turns"]
            new_avg = ((current_avg * (current_total - 1)) + result["num_turns"]) / current_total
            self.results["summary"]["avg_turns"] = new_avg
            
            # Store result
            self.results["problems"][problem_key] = result
            self.progress["completed"].append(problem_key)
            
        except Exception as e:
            print(f"  ERROR: {str(e)[:200]}")
            self.results["summary"]["total"] += 1
            self.results["summary"]["other_errors"] += 1
            self.progress["failed"].append(problem_key)
            
            self.results["problems"][problem_key] = {
                "error": str(e),
                "level": level,
                "problem_id": problem_id
            }
        
        finally:
            self.save_progress()
            self.save_results()
        
        return self.results["problems"].get(problem_key)
    
    def run_evaluation(self, levels: list, max_problems: Optional[int] = None, start_from: Optional[int] = None):
        """Run multi-turn evaluation for specified levels."""
        
        print(f"\n{'='*70}")
        print(f"Robust Multi-Turn Qwen3-8B Triton Code Generation Evaluation")
        print(f"Model: Qwen/Qwen3-8B-fined-tuned on SGLang")
        print(f"GPU: AMD MI300X")
        print(f"Levels: {levels}")
        print(f"Max turns: {self.max_turns}, Gamma: {self.gamma}")
        print(f"Native template: {self.use_native_template}")
        print(f"Subprocess isolation: {self.use_subprocess}")
        print(f"Eval timeout: {self.eval_timeout}s")
        print(f"{'='*70}\n")
        
        for level in levels:
            if level not in self.datasets:
                continue
            
            dataset = self.datasets[level]
            num_problems = min(len(dataset), max_problems) if max_problems else len(dataset)
            
            # Determine starting point
            start_idx = start_from if start_from else 1
            
            print(f"\nLevel {level}: Problems {start_idx}-{num_problems}")
            print("-"*40)
            
            for problem_id in range(start_idx, num_problems + 1):
                result = self.evaluate_problem(level, problem_id)
                
                # Progress update every 5 problems
                if problem_id % 5 == 0 or problem_id == num_problems:
                    self.print_progress()
        
        # Final report
        self.generate_final_report()
    
    def print_progress(self):
        """Print current progress statistics."""
        s = self.results["summary"]
        if s["total"] == 0:
            return
        
        print(f"\n=== Progress Update ===")
        print(f"Total: {s['total']}")
        print(f"Avg turns: {s['avg_turns']:.1f}")
        print(f"Thinking: {s['thinking_detected']}/{s['total']} ({s['thinking_detected']/s['total']*100:.1f}%)")
        print(f"Generated: {s['code_generated']}/{s['total']} ({s['code_generated']/s['total']*100:.1f}%)")
        print(f"Compiled: {s['compiled']}/{s['total']} ({s['compiled']/s['total']*100:.1f}%)")
        print(f"Correct: {s['correct']}/{s['total']} ({s['correct']/s['total']*100:.1f}%)")
        print(f"Multi-turn success: {s['multi_turn_success']}/{s['total']} ({s['multi_turn_success']/s['total']*100:.1f}%)")
        print(f"Early terminations: {s['early_terminations']}")
        print(f"Memory faults: {s['memory_faults']}, Timeouts: {s['timeouts']}")
        print("=" * 23)
    
    def generate_final_report(self):
        """Generate comprehensive multi-turn evaluation report."""
        report_file = f"{self.report_dir}/MULTI_TURN_REPORT.md"
        s = self.results["summary"]
        
        with open(report_file, 'w') as f:
            f.write("# Multi-Turn Qwen3-8B Triton Evaluation Report\n\n")
            f.write(f"**Run**: {self.run_name}\n")
            f.write(f"**Model**: Qwen/Qwen3-8B-fined-tuned\n")
            f.write(f"**Configuration**: max_turns={self.max_turns}, gamma={self.gamma}\n")
            f.write(f"**Start**: {self.results['metadata']['start_time']}\n")
            f.write(f"**End**: {datetime.now().isoformat()}\n\n")
            
            f.write("## Overall Results\n\n")
            f.write(f"- Total Problems: {s['total']}\n")
            f.write(f"- Average Turns: {s['avg_turns']:.2f}\n")
            f.write(f"- Early Terminations: {s['early_terminations']}\n")
            f.write(f"- Thinking Detected: {s['thinking_detected']}\n")
            f.write(f"- Code Generated: {s['code_generated']}\n")
            f.write(f"- Compiled (any turn): {s['compiled']}\n")
            f.write(f"- Correct (any turn): {s['correct']}\n")
            f.write(f"- Multi-turn Success: {s['multi_turn_success']}\n\n")
            
            if s['total'] > 0:
                f.write("## Success Metrics\n\n")
                f.write(f"- Thinking Rate: {s['thinking_detected']/s['total']*100:.1f}%\n")
                f.write(f"- Generation Rate: {s['code_generated']/s['total']*100:.1f}%\n")
                f.write(f"- Compilation Rate: {s['compiled']/s['total']*100:.1f}%\n")
                f.write(f"- Correctness Rate: {s['correct']/s['total']*100:.1f}%\n")
                f.write(f"- Multi-turn Success Rate: {s['multi_turn_success']/s['total']*100:.1f}%\n")
                f.write(f"- Early Termination Rate: {s['early_terminations']/s['total']*100:.1f}%\n\n")
            
            f.write("## Turn Analysis\n\n")
            
            # Analyze turn-by-turn improvements
            turn_improvements = {"1to2": 0, "2to3": 0}
            turn_rewards_by_position = {0: [], 1: [], 2: []}
            
            for prob_key, result in self.results["problems"].items():
                if isinstance(result, dict) and "turn_rewards" in result:
                    for i, reward in enumerate(result["turn_rewards"]):
                        if i < 3:
                            turn_rewards_by_position[i].append(reward)
                    
                    # Check improvements
                    if len(result["turn_rewards"]) >= 2:
                        if result["turn_rewards"][1] > result["turn_rewards"][0]:
                            turn_improvements["1to2"] += 1
                    if len(result["turn_rewards"]) >= 3:
                        if result["turn_rewards"][2] > result["turn_rewards"][1]:
                            turn_improvements["2to3"] += 1
            
            f.write("### Average Reward by Turn\n\n")
            for turn, rewards in turn_rewards_by_position.items():
                if rewards:
                    avg_reward = sum(rewards) / len(rewards)
                    f.write(f"- Turn {turn + 1}: {avg_reward:.3f} (n={len(rewards)})\n")
            
            f.write(f"\n### Turn-to-Turn Improvements\n\n")
            f.write(f"- Turn 1→2 improvements: {turn_improvements['1to2']}\n")
            f.write(f"- Turn 2→3 improvements: {turn_improvements['2to3']}\n\n")
            
            f.write("## Best Performing Problems\n\n")
            
            # Find top problems by aggregated return
            problem_returns = []
            for prob_key, result in self.results["problems"].items():
                if isinstance(result, dict) and "aggregated_return" in result:
                    problem_returns.append((prob_key, result["aggregated_return"], result["num_turns"]))
            
            problem_returns.sort(key=lambda x: x[1], reverse=True)
            
            f.write("| Problem | Aggregated Return | Turns |\n")
            f.write("|---------|------------------|-------|\n")
            for prob_key, agg_return, num_turns in problem_returns[:10]:
                f.write(f"| {prob_key} | {agg_return:.3f} | {num_turns} |\n")
            
            f.write("\n## Key Findings\n\n")
            f.write(f"1. Model uses multi-turn refinement effectively in {s['multi_turn_success']} problems\n")
            f.write(f"2. Average of {s['avg_turns']:.1f} turns needed per problem\n")
            f.write(f"3. {s['early_terminations']} problems achieved good performance early\n")
            f.write(f"4. Thinking tags detected in {s['thinking_detected']} problem attempts\n")
        
        print(f"\n{'='*70}")
        print(f"Multi-Turn Evaluation Complete!")
        print(f"Results saved to: {self.results_dir}")
        print(f"Final report: {report_file}")
        print(f"{'='*70}")


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Robust MULTI-TURN Qwen3-8B Triton evaluation")
    parser.add_argument("--levels", type=str, default="1,2",
                       help="Comma-separated levels (e.g., '1,2')")
    parser.add_argument("--max-problems", type=int, default=None,
                       help="Max problems per level")
    parser.add_argument("--start-from", type=int, default=None,
                       help="Start from problem number (useful for resuming)")
    parser.add_argument("--run-name", type=str, default=None,
                       help="Custom run name")
    parser.add_argument("--no-subprocess", action="store_true",
                       help="Disable subprocess isolation (not recommended)")
    parser.add_argument("--timeout", type=int, default=60,
                       help="Evaluation timeout in seconds (default: 60)")
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS,
                       help=f"Maximum turns per problem (default: {DEFAULT_MAX_TURNS})")
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA,
                       help=f"Discount factor for aggregated return (default: {DEFAULT_GAMMA})")
    parser.add_argument("--no-native-template", action="store_true",
                       help="Disable native template mode (not recommended)")
    
    args = parser.parse_args()
    
    # Parse levels
    levels = [int(l) for l in args.levels.split(",")]
    
    # Set GPU architecture
    from src.utils import set_gpu_arch
    set_gpu_arch(["MI300X", "gfx942"])
    
    # Create multi-turn evaluator
    evaluator = RobustMultiTurnQwen3Evaluator(
        run_name=args.run_name,
        use_subprocess=not args.no_subprocess,
        eval_timeout=args.timeout,
        max_turns=args.max_turns,
        gamma=args.gamma,
        use_native_template=not args.no_native_template
    )
    
    # Run evaluation
    evaluator.run_evaluation(
        levels, 
        max_problems=args.max_problems,
        start_from=args.start_from
    )


if __name__ == "__main__":
    main()