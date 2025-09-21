#!/usr/bin/env python3
"""
Multi-turn SFT data generation pipeline using Claude API for KernelBench
This generates multi-turn conversations for supervised fine-tuning.
"""

import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import copy

import torch
import requests
from anthropic import Anthropic

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from src.eval import eval_kernel_against_ref, KernelExecResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Multi-turn configuration
DEFAULT_MAX_TURNS = 3
DEFAULT_MODEL = "claude-sonnet-4-20250514"  # Claude 3.5 Sonnet Latest

# Evaluation configuration
DEFAULT_NUM_CORRECT_TRIALS = 5
DEFAULT_NUM_PERF_TRIALS = 20  # Reduced for faster evaluation

# Output directory for logs
OUTPUT_DIR = "/root/KernelBench/multi_turn_sft_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_code_from_response(response: str) -> Optional[str]:
    """Extract Python code from Claude's response"""
    # Look for code blocks
    if "```python" in response:
        start = response.find("```python") + 9
        end = response.find("```", start)
        if end != -1:
            return response[start:end].strip()
    elif "```" in response:
        start = response.find("```") + 3
        end = response.find("```", start)
        if end != -1:
            # Check if it might be Python code
            code = response[start:end].strip()
            if "import" in code or "def " in code or "class " in code:
                return code
    
    # If no code blocks, try to extract code heuristically
    lines = response.split('\n')
    code_lines = []
    in_code = False
    
    for line in lines:
        # Simple heuristic: lines that look like Python code
        if line.strip().startswith(('import ', 'from ', 'def ', 'class ', '@')) or in_code:
            in_code = True
            code_lines.append(line)
        elif in_code and line.strip() == '':
            code_lines.append(line)
        elif in_code and not line.startswith(' ') and not line.startswith('\t'):
            # Likely end of code block
            if not any(keyword in line for keyword in ['import', 'def', 'class', 'return']):
                break
    
    if code_lines:
        return '\n'.join(code_lines)
    
    return None


def format_eval_feedback(eval_result: KernelExecResult) -> str:
    """Format evaluation results as feedback text"""
    feedback_parts = []
    
    feedback_parts.append("## Evaluation Results")
    
    if not eval_result.compiled:
        feedback_parts.append("❌ **Compilation Failed**")
        if eval_result.metadata.get("compilation_error"):
            error_msg = eval_result.metadata['compilation_error']
            # Extract the most relevant part of the error
            if "error:" in error_msg.lower():
                # Find the actual error message
                error_lines = error_msg.split('\n')
                for line in error_lines:
                    if 'error:' in line.lower():
                        feedback_parts.append(f"**Error**: {line.strip()}")
                        break
            else:
                # Show first 500 chars
                feedback_parts.append(f"**Error**: {error_msg[:500]}")
        if eval_result.metadata.get("compilation_error_name"):
            feedback_parts.append(f"**Error Type**: {eval_result.metadata['compilation_error_name']}")
    else:
        feedback_parts.append("✅ **Compilation Successful**")
        
        if not eval_result.correctness:
            feedback_parts.append("❌ **Correctness Check Failed**")
            
            # Provide specific feedback about what went wrong
            if eval_result.metadata.get("correctness_issue"):
                issue = eval_result.metadata['correctness_issue']
                feedback_parts.append(f"**Issue**: {issue}")
                
            if eval_result.metadata.get("max_difference"):
                feedback_parts.append(f"**Max Difference**: {eval_result.metadata['max_difference']}")
            if eval_result.metadata.get("avg_difference"):
                feedback_parts.append(f"**Avg Difference**: {eval_result.metadata['avg_difference']}")
                
            if eval_result.metadata.get("runtime_error"):
                error = eval_result.metadata['runtime_error']
                if len(error) > 500:
                    # Extract key part
                    if "RuntimeError:" in error:
                        error = error[error.find("RuntimeError:"):][:500]
                    else:
                        error = error[:500]
                feedback_parts.append(f"**Runtime Error**: {error}")
                
            if eval_result.metadata.get("runtime_error_name"):
                feedback_parts.append(f"**Error Type**: {eval_result.metadata['runtime_error_name']}")
                
            # Add specific guidance
            feedback_parts.append("\n**Note**: The output does not match the expected result from the reference implementation.")
        else:
            feedback_parts.append("✅ **Correctness Check Passed**")
            feedback_parts.append("The kernel produces the same output as the reference implementation!")
            
            if eval_result.runtime > 0:
                feedback_parts.append(f"\n⏱️ **Performance Metrics**:")
                feedback_parts.append(f"**Runtime**: {eval_result.runtime:.3f} ms")
                
                # If we have runtime stats
                if eval_result.runtime_stats:
                    stats = eval_result.runtime_stats
                    feedback_parts.append(f"   - Mean: {stats.get('mean', 0):.3f} ms")
                    feedback_parts.append(f"   - Std: {stats.get('std', 0):.3f} ms")
                    feedback_parts.append(f"   - Min: {stats.get('min', 0):.3f} ms")
                    feedback_parts.append(f"   - Max: {stats.get('max', 0):.3f} ms")
                    
                    # Add hardware info if available
                    if stats.get('hardware'):
                        feedback_parts.append(f"   - Hardware: {stats['hardware']}")
    
    return "\n".join(feedback_parts)


class ClaudeMultiTurnSFTGenerator:
    """Multi-turn SFT data generator using Claude API"""
    
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        max_turns: int = DEFAULT_MAX_TURNS,
        verbose: bool = True
    ):
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.max_turns = max_turns
        self.verbose = verbose
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This evaluator requires GPU.")
        
        self.device = torch.cuda.current_device()
        logger.info(f"Initialized with device: {torch.cuda.get_device_name(self.device)}")
    
    def fix_original_model_name(self, code: str) -> str:
        """Fix the original model code to use 'Model' as class name for KernelBench compatibility"""
        import re
        
        # Find the class name
        class_match = re.search(r'class\s+(\w+)\s*\([^)]*nn\.(Module|LayerNorm|Linear|Conv\w+|[A-Z]\w+)', code)
        
        if class_match:
            original_class_name = class_match.group(1)
            
            if original_class_name != "Model":
                # Replace class definition
                code = re.sub(
                    rf'class\s+{original_class_name}\s*\(',
                    'class Model(',
                    code
                )
                
                # Fix super() calls
                code = re.sub(
                    rf'super\s*\(\s*{original_class_name}\s*,',
                    'super(Model,',
                    code
                )
        
        return code
    
    def fix_original_model_issues(self, code: str) -> str:
        """Fix common issues in original model code for proper initialization and input generation"""
        import re
        
        fixes_applied = []
        
        # Fix 1: get_inputs() should use torch.randint for embedding models
        # Check for embedding-like patterns (EmbeddingBag, Embedding, or uniform initialization with sqrt(1/n))
        is_embedding_model = ('Embedding' in code or 
                             'embedding' in code.lower() or
                             ('weight' in code and 'uniform_' in code and 'sqrt(1' in code))
        
        if is_embedding_model and 'def get_inputs' in code:
            # Check if get_inputs uses torch.rand
            get_inputs_match = re.search(r'def get_inputs\(\):(.*?)(?=def\s|\Z)', code, re.DOTALL)
            if get_inputs_match:
                func_content = get_inputs_match.group(0)
                if 'torch.rand(' in func_content:
                    # Replace torch.rand with torch.randint for embedding models
                    # Extract dimensions
                    rand_calls = re.findall(r'torch\.rand\(\[([^\]]+)\]\)', func_content)
                    for dims in rand_calls:
                        # Use first dimension as max value for embedding indices
                        dims_list = [d.strip() for d in dims.split(',')]
                        if dims_list:
                            max_val = dims_list[0]
                            old = f'torch.rand([{dims}])'
                            new = f'torch.randint(0, {max_val}, [{dims}]).cuda()'
                            code = code.replace(old, new)
                            fixes_applied.append(f"Replaced {old} with {new} for embedding inputs")
        
        # Fix 2: reset_parameters() uniform initialization
        if 'def reset_parameters' in code:
            # Fix nn.init.uniform_ with single parameter (should be range)
            # Pattern: nn.init.uniform_(param, value) -> nn.init.uniform_(param, -value, value)
            pattern = r'nn\.init\.uniform_\(([^,]+),\s*(np\.sqrt\(1\s*/\s*[^\)]+\))\)'
            matches = re.findall(pattern, code)
            for param_name, bound_expr in matches:
                old = f'nn.init.uniform_({param_name}, {bound_expr})'
                new = f'nn.init.uniform_({param_name}, -{bound_expr}, {bound_expr})'
                code = code.replace(old, new)
                fixes_applied.append(f"Fixed uniform initialization bounds for {param_name.strip()}")
            
            # Also fix direct numeric values
            pattern2 = r'nn\.init\.uniform_\(([^,]+),\s*([0-9\.]+)\)'
            matches2 = re.findall(pattern2, code)
            for param_name, value in matches2:
                old = f'nn.init.uniform_({param_name}, {value})'
                new = f'nn.init.uniform_({param_name}, -{value}, {value})'
                code = code.replace(old, new)
                fixes_applied.append(f"Fixed uniform initialization bounds for {param_name.strip()}")
        
        # Fix 3: Ensure CUDA for tensors in get_inputs
        if 'def get_inputs' in code:
            # Add .cuda() to tensor creation if missing
            get_inputs_match = re.search(r'def get_inputs\(\):(.*?)(?=def\s|\Z)', code, re.DOTALL)
            if get_inputs_match:
                func_content = get_inputs_match.group(0)
                if '.cuda()' not in func_content:
                    # Add .cuda() to tensor creations
                    new_func = func_content
                    # Handle torch.rand/randn
                    new_func = re.sub(r'(torch\.rand[n]?\([^\)]+\))(?!\.cuda)', r'\1.cuda()', new_func)
                    # Handle torch.randint (if not already fixed above)
                    new_func = re.sub(r'(torch\.randint\([^\)]+\))(?!\.cuda)', r'\1.cuda()', new_func)
                    # Handle torch.zeros/ones
                    new_func = re.sub(r'(torch\.zeros\([^\)]+\))(?!\.cuda)', r'\1.cuda()', new_func)
                    new_func = re.sub(r'(torch\.ones\([^\)]+\))(?!\.cuda)', r'\1.cuda()', new_func)
                    
                    if new_func != func_content:
                        code = code.replace(func_content, new_func)
                        fixes_applied.append("Added .cuda() to tensor creations in get_inputs()")
        
        if fixes_applied:
            logger.info(f"Applied {len(fixes_applied)} fixes to original model code:")
            for fix in fixes_applied:
                logger.info(f"  - {fix}")
        
        return code
    
    def fix_generated_model_name(self, code: str) -> str:
        """Fix the generated model code to use 'ModelNew' as class name for KernelBench compatibility"""
        import re
        
        # Find any class with 'New' suffix or any model class
        class_matches = re.findall(r'class\s+(\w+)\s*\([^)]*nn\.(Module|LayerNorm|Linear|Conv\w+|[A-Z]\w+)', code)
        
        for match in class_matches:
            class_name = match[0]
            # Skip if it's already ModelNew or is a helper class
            if class_name == "ModelNew" or not any(x in class_name for x in ["New", "Model", "Layer", "Block"]):
                continue
                
            # Replace class definition
            code = re.sub(
                rf'class\s+{class_name}\s*\(',
                'class ModelNew(',
                code
            )
            
            # Fix super() calls  
            code = re.sub(
                rf'super\s*\(\s*{class_name}\s*,',
                'super(ModelNew,',
                code
            )
            
            # Only fix the first main model class
            break
        
        return code
    
    def create_improvement_prompt(self, eval_feedback: str, previous_code: str, turn_idx: int) -> str:
        """Create a prompt asking for improvement based on evaluation feedback"""
        prompt_parts = []
        
        # Include the previous attempt's code
        prompt_parts.append("## Your Previous Attempt")
        prompt_parts.append("You generated the following Triton kernel code:")
        prompt_parts.append("```python")
        prompt_parts.append(previous_code)
        prompt_parts.append("```")
        prompt_parts.append("")
        
        # Include the evaluation results
        prompt_parts.append(eval_feedback)
        prompt_parts.append("")
        
        # Provide specific guidance based on the error
        if "Compilation Failed" in eval_feedback:
            prompt_parts.append("### Issue Analysis")
            prompt_parts.append("The kernel failed to compile. Common issues include:")
            prompt_parts.append("- Syntax errors in Triton kernel definition")
            prompt_parts.append("- Incorrect pointer arithmetic or indexing")
            prompt_parts.append("- Missing or incorrect tensor operations")
            prompt_parts.append("- Type mismatches or undefined variables")
            prompt_parts.append("")
            prompt_parts.append("Please carefully review the code above, fix the compilation errors, and generate a corrected version.")
        elif "Correctness Check Failed" in eval_feedback:
            prompt_parts.append("### Issue Analysis")
            prompt_parts.append("The kernel compiled but produced incorrect results. Common issues include:")
            prompt_parts.append("- Incorrect algorithm implementation")
            prompt_parts.append("- Wrong tensor dimensions or strides")
            prompt_parts.append("- Missing normalization or scaling factors")
            prompt_parts.append("- Incorrect reduction operations")
            prompt_parts.append("- Edge cases not handled properly")
            prompt_parts.append("")
            prompt_parts.append("Please review the implementation logic, ensure it matches the reference PyTorch implementation, and generate a corrected version.")
        else:
            # Both compilation and correctness passed
            prompt_parts.append("### Success!")
            prompt_parts.append("The kernel works correctly! Now let's optimize it further for better performance.")
            prompt_parts.append("")
            prompt_parts.append("### Optimization Opportunities")
            prompt_parts.append("Consider the following techniques:")
            prompt_parts.append("- **Memory Coalescing**: Ensure consecutive threads access consecutive memory locations")
            prompt_parts.append("- **Shared Memory**: Use shared memory to reduce global memory accesses")
            prompt_parts.append("- **Tiling**: Process data in tiles to maximize cache utilization")
            prompt_parts.append("- **Vectorization**: Use vector loads/stores when possible")
            prompt_parts.append("- **Kernel Fusion**: Combine multiple operations into a single kernel")
            prompt_parts.append("- **Occupancy**: Tune block size and register usage for better GPU occupancy")
            prompt_parts.append("")
            prompt_parts.append("Generate an optimized version that improves upon the working implementation above.")
        
        prompt_parts.append("")
        prompt_parts.append(f"### Requirements")
        prompt_parts.append(f"This is attempt {turn_idx + 1} of {self.max_turns}.")
        prompt_parts.append("- Generate a complete, working implementation")
        prompt_parts.append("- Ensure the class is named 'ModelNew'")
        prompt_parts.append("- Include all necessary imports and helper functions")
        prompt_parts.append("- Output only the code in a Python code block")
        prompt_parts.append("")
        prompt_parts.append("Generate your improved implementation:")
        
        return "\n".join(prompt_parts)
    
    def query_claude(self, messages: List[dict], max_tokens: int = 4096) -> str:
        """Query Claude API with retry logic"""
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                # Convert messages to Claude format
                system_message = None
                claude_messages = []
                
                for msg in messages:
                    if msg["role"] == "system":
                        system_message = msg["content"]
                    else:
                        claude_messages.append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })
                
                # Debug: log what we're sending
                logger.debug(f"System message length: {len(system_message) if system_message else 0}")
                logger.debug(f"Number of messages: {len(claude_messages)}")
                if claude_messages:
                    logger.debug(f"First message role: {claude_messages[0]['role']}")
                    logger.debug(f"First message content length: {len(claude_messages[0]['content'])}")
                
                # Validate messages
                if not claude_messages:
                    raise ValueError("No messages to send to Claude API")
                
                # Make API call
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system_message,
                    messages=claude_messages,
                    temperature=0.7,
                )
                
                # Debug logging
                logger.debug(f"Response type: {type(response)}")
                logger.debug(f"Response attributes: {dir(response)}")
                
                # Handle response content - different formats for different library versions
                if hasattr(response, 'content'):
                    logger.debug(f"Response content type: {type(response.content)}")
                    if isinstance(response.content, list) and len(response.content) > 0:
                        # New format
                        content_item = response.content[0]
                        if hasattr(content_item, 'text'):
                            return content_item.text
                        else:
                            logger.debug(f"Content item attributes: {dir(content_item)}")
                            raise ValueError(f"No text attribute in content item: {content_item}")
                    elif isinstance(response.content, str):
                        # Direct string content
                        return response.content
                    else:
                        raise ValueError(f"Unexpected content format: {response.content}")
                # Try older API format
                elif hasattr(response, 'completion'):
                    return response.completion
                elif hasattr(response, 'text'):
                    return response.text
                else:
                    raise ValueError(f"Unexpected response format. Available attributes: {dir(response)}")
                
            except Exception as e:
                logger.warning(f"Claude API error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))
                else:
                    raise
    
    def evaluate_kernel(
        self,
        original_code: str,
        generated_code: str,
        backend: str = "triton"
    ) -> KernelExecResult:
        """Evaluate a generated kernel"""
        try:
            # Run evaluation
            result = eval_kernel_against_ref(
                original_model_src=original_code,
                custom_model_src=generated_code,
                seed_num=42,
                num_correct_trials=DEFAULT_NUM_CORRECT_TRIALS,
                num_perf_trials=DEFAULT_NUM_PERF_TRIALS,
                verbose=self.verbose,
                measure_performance=True,
                device=self.device,
                backend=backend
            )
            
            # Handle None result (usually compilation lock issues)
            if result is None:
                logger.warning("Evaluation returned None (likely compilation lock issue)")
                # Retry once
                logger.info("Retrying evaluation...")
                time.sleep(1)
                result = eval_kernel_against_ref(
                    original_model_src=original_code,
                    custom_model_src=generated_code,
                    seed_num=42,
                    num_correct_trials=DEFAULT_NUM_CORRECT_TRIALS,
                    num_perf_trials=DEFAULT_NUM_PERF_TRIALS,
                    verbose=self.verbose,
                    measure_performance=True,
                    device=self.device,
                    backend=backend
                )
                
                if result is None:
                    # Still None, create a failed result
                    return KernelExecResult(
                        compiled=False,
                        metadata={"evaluation_error": "Compilation lock or file system issue"}
                    )
            
            return result
            
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            # Create a failed result
            return KernelExecResult(
                compiled=False,
                metadata={"evaluation_error": str(e)}
            )
    
    def generate_multi_turn_conversation(
        self,
        item: dict,
        instance_id: str
    ) -> dict:
        """Generate a multi-turn conversation for SFT"""
        
        all_messages = copy.deepcopy(item["messages"])
        
        # Save reference assistant message if exists
        reference_assistant = next((msg for msg in all_messages if msg["role"] == "assistant"), None)
        
        # Remove any existing assistant messages - we'll generate new ones
        messages = [msg for msg in all_messages if msg["role"] != "assistant"]
        
        # Fix the user prompt to specify ModelNew naming
        for msg in messages:
            if msg["role"] == "user":
                # Add explicit instruction about naming
                if "Name your optimized output architecture" in msg["content"]:
                    # Replace the naming instruction to be more explicit
                    msg["content"] = msg["content"].replace(
                        "Name your optimized output architecture LayerNormNew.",
                        "Name your optimized output architecture class exactly 'ModelNew' (not LayerNormNew or any other name)."
                    )
                    msg["content"] = msg["content"].replace(
                        "Name your optimized output architecture UpBlockNew.",
                        "Name your optimized output architecture class exactly 'ModelNew' (not UpBlockNew or any other name)."
                    )
                    # Generic pattern
                    import re
                    msg["content"] = re.sub(
                        r"Name your optimized output architecture (\w+)New\.",
                        "Name your optimized output architecture class exactly 'ModelNew'.",
                        msg["content"]
                    )
                # Also add a note at the end if not already there
                if "ModelNew" not in msg["content"]:
                    msg["content"] += "\n\nIMPORTANT: The optimized model class MUST be named exactly 'ModelNew', not any other name."
        
        # Extract original model code from the prompt and fix it
        original_code = None
        for msg in messages:
            if msg["role"] == "user" and "```python" in msg["content"]:
                original_code = extract_code_from_response(msg["content"])
                if original_code:
                    # Fix the original model class name to 'Model'
                    original_code = self.fix_original_model_name(original_code)
                    # Fix common issues in the original model code
                    original_code = self.fix_original_model_issues(original_code)
                break
        
        if not original_code:
            logger.error(f"Could not extract original model code for {instance_id}")
            return {
                "instance_id": instance_id,
                "messages": messages,
                "error": "Could not extract original model code",
                "num_turns": 0
            }
        
        conversation_messages = copy.deepcopy(messages)
        turn_history = []
        
        for turn_idx in range(self.max_turns):
            logger.info(f"[{instance_id}] Generating turn {turn_idx + 1}/{self.max_turns}")
            
            # Generate response
            try:
                response = self.query_claude(conversation_messages)
                
                # Add assistant response to conversation
                conversation_messages.append({
                    "role": "assistant",
                    "content": response
                })
                
            except Exception as e:
                logger.error(f"[{instance_id}] Generation error at turn {turn_idx}: {e}")
                turn_history.append({
                    "turn": turn_idx + 1,
                    "compiled": False,
                    "correct": False,
                    "error": str(e)
                })
                break
            
            # Extract kernel code
            kernel_code = extract_code_from_response(response)
            if not kernel_code:
                logger.warning(f"[{instance_id}] No code extracted at turn {turn_idx}")
                # Add feedback about missing code
                conversation_messages.append({
                    "role": "user",
                    "content": "No code was found in your response. Please provide the complete kernel implementation."
                })
                turn_history.append({
                    "turn": turn_idx + 1,
                    "compiled": False,
                    "error": "No code extracted"
                })
                continue
            
            # Fix the generated model name to ModelNew for KernelBench compatibility
            kernel_code_fixed = self.fix_generated_model_name(kernel_code)
            
            # Evaluate the generated kernel
            logger.info(f"[{instance_id}] Evaluating kernel for turn {turn_idx + 1}")
            eval_result = self.evaluate_kernel(original_code, kernel_code_fixed)
            
            # Record turn results
            turn_info = {
                "turn": turn_idx + 1,
                "compiled": eval_result.compiled if eval_result else False,
                "correct": eval_result.correctness if eval_result else False,
                "runtime": eval_result.runtime if eval_result and eval_result.runtime > 0 else None,
            }
            
            if eval_result and eval_result.metadata:
                if eval_result.metadata.get("compilation_error"):
                    turn_info["compilation_error"] = eval_result.metadata["compilation_error"][:200]
                if eval_result.metadata.get("runtime_error"):
                    turn_info["runtime_error"] = eval_result.metadata["runtime_error"][:200]
            
            turn_history.append(turn_info)
            
            # Log turn summary
            logger.info(
                f"[{instance_id}] Turn {turn_idx + 1}: "
                f"Compiled:{turn_info['compiled']} "
                f"Correct:{turn_info['correct']} "
                f"Runtime:{turn_info.get('runtime', 'N/A')}"
            )
            
            # Check if we should continue
            if turn_idx < self.max_turns - 1:
                # Check termination conditions
                if eval_result and eval_result.compiled and eval_result.correctness:
                    # Success! But let's ask for one more optimization
                    if turn_idx < self.max_turns - 2:  # Not the second-to-last turn
                        eval_feedback = format_eval_feedback(eval_result)
                        # Pass the original kernel_code (not the fixed one) so the conversation looks natural
                        improvement_prompt = self.create_improvement_prompt(eval_feedback, kernel_code, turn_idx + 1)
                        conversation_messages.append({
                            "role": "user",
                            "content": improvement_prompt
                        })
                    else:
                        # We've achieved correctness, that's good enough for SFT
                        logger.info(f"[{instance_id}] Early termination - achieved correctness")
                        break
                else:
                    # Need to fix issues
                    if eval_result:
                        eval_feedback = format_eval_feedback(eval_result)
                        # Pass the original kernel_code (not the fixed one) so the conversation looks natural
                        improvement_prompt = self.create_improvement_prompt(eval_feedback, kernel_code, turn_idx + 1)
                        conversation_messages.append({
                            "role": "user",
                            "content": improvement_prompt
                        })
                    else:
                        conversation_messages.append({
                            "role": "user",
                            "content": "The evaluation failed. Please try again with a corrected implementation."
                        })
        
        # Create the final SFT data entry
        sft_entry = {
            "instance_id": instance_id,
            "messages": conversation_messages,
            "num_turns": len([t for t in turn_history if "error" not in t]),
            "turn_history": turn_history,
            "final_compiled": turn_history[-1]["compiled"] if turn_history else False,
            "final_correct": turn_history[-1]["correct"] if turn_history else False,
            "timestamp": datetime.now().isoformat()
        }
        
        return sft_entry
    
    def process_batch(
        self,
        items: List[dict],
        start_idx: int = 0
    ) -> List[dict]:
        """Process a batch of items"""
        results = []
        
        for idx, item in enumerate(items):
            instance_id = f"kernelbook_{start_idx + idx}"
            logger.info(f"Processing {instance_id} ({idx + 1}/{len(items)})")
            
            try:
                sft_entry = self.generate_multi_turn_conversation(item, instance_id)
                results.append(sft_entry)
                
                # Log summary
                logger.info(
                    f"Completed {instance_id}: "
                    f"Turns={sft_entry['num_turns']}, "
                    f"Final Compiled={sft_entry['final_compiled']}, "
                    f"Final Correct={sft_entry['final_correct']}"
                )
                
            except Exception as e:
                logger.error(f"Error processing {instance_id}: {e}")
                traceback.print_exc()
                results.append({
                    "instance_id": instance_id,
                    "error": str(e),
                    "messages": item.get("messages", [])
                })
        
        return results


def process_kernelbook_data(
    input_file: str,
    api_key: str,
    output_dir: str = OUTPUT_DIR,
    max_samples: Optional[int] = None,
    batch_size: int = 10,
    verbose: bool = True
):
    """Process KernelBook data to generate multi-turn SFT data"""
    
    # Initialize generator
    generator = ClaudeMultiTurnSFTGenerator(
        api_key=api_key,
        verbose=verbose
    )
    
    # Use output_dir directly if it already contains a run directory name
    # Otherwise create a new timestamped directory
    if "run_" in os.path.basename(output_dir):
        run_output_dir = output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)
    
    # Load data
    with open(input_file, 'r') as f:
        lines = f.readlines()
        
    if max_samples:
        lines = lines[:max_samples]
    
    items = [json.loads(line) for line in lines]
    logger.info(f"Loaded {len(items)} items from {input_file}")
    
    # Process in batches
    all_results = []
    for batch_start in range(0, len(items), batch_size):
        batch_end = min(batch_start + batch_size, len(items))
        batch_items = items[batch_start:batch_end]
        
        logger.info(f"Processing batch {batch_start // batch_size + 1}: items {batch_start}-{batch_end}")
        
        batch_results = generator.process_batch(batch_items, start_idx=batch_start)
        all_results.extend(batch_results)
        
        # Save intermediate results
        intermediate_path = os.path.join(run_output_dir, f"batch_{batch_start:04d}.jsonl")
        with open(intermediate_path, 'w') as f:
            for result in batch_results:
                f.write(json.dumps(result) + '\n')
        logger.info(f"Saved batch results to {intermediate_path}")
    
    # Save all results as JSONL (SFT format) - renamed file
    output_path = os.path.join(run_output_dir, "multi-turn-sft.jsonl")
    with open(output_path, 'w') as f:
        for result in all_results:
            # Save in the same format as input for SFT
            sft_entry = {
                "messages": result["messages"]
            }
            f.write(json.dumps(sft_entry) + '\n')
    
    logger.info(f"Saved SFT data to {output_path}")
    
    # Save as parquet for efficient storage and loading
    try:
        import pandas as pd
        import pyarrow.parquet as pq
        
        # Convert to DataFrame
        df_data = []
        for result in all_results:
            df_data.append({
                "instance_id": result.get("instance_id", ""),
                "messages": json.dumps(result["messages"]),  # Store as JSON string
                "num_turns": result.get("num_turns", 0),
                "final_compiled": result.get("final_compiled", False),
                "final_correct": result.get("final_correct", False),
                "timestamp": result.get("timestamp", ""),
                "turn_history": json.dumps(result.get("turn_history", []))
            })
        
        df = pd.DataFrame(df_data)
        parquet_path = os.path.join(run_output_dir, "multi-turn-sft.parquet")
        df.to_parquet(parquet_path, engine='pyarrow', compression='snappy')
        logger.info(f"Saved parquet file to {parquet_path}")
        
    except ImportError:
        logger.warning("pandas or pyarrow not installed, skipping parquet export")
    except Exception as e:
        logger.error(f"Error saving parquet: {e}")
    
    # Save detailed results with metadata - renamed file
    detailed_path = os.path.join(run_output_dir, "test_output_conversation.jsonl")
    with open(detailed_path, 'w') as f:
        for result in all_results:
            f.write(json.dumps(result) + '\n')
    
    # Generate summary statistics
    successful_results = [r for r in all_results if "error" not in r]
    
    summary = {
        "timestamp": timestamp,
        "total_samples": len(all_results),
        "successful_samples": len(successful_results),
        "failed_samples": len(all_results) - len(successful_results),
        "config": {
            "max_turns": generator.max_turns,
            "model": generator.model,
            "input_file": input_file
        }
    }
    
    if successful_results:
        summary["statistics"] = {
            "avg_turns": sum(r["num_turns"] for r in successful_results) / len(successful_results),
            "max_turns": max(r["num_turns"] for r in successful_results),
            "min_turns": min(r["num_turns"] for r in successful_results),
            "final_compilation_rate": sum(1 for r in successful_results if r.get("final_compiled", False)) / len(successful_results),
            "final_correctness_rate": sum(1 for r in successful_results if r.get("final_correct", False)) / len(successful_results),
        }
        
        # Turn-by-turn statistics
        turn_stats = {}
        for turn_idx in range(generator.max_turns):
            turn_data = []
            for r in successful_results:
                if r.get("turn_history") and len(r["turn_history"]) > turn_idx:
                    turn_data.append(r["turn_history"][turn_idx])
            
            if turn_data:
                turn_stats[f"turn_{turn_idx + 1}"] = {
                    "count": len(turn_data),
                    "compilation_rate": sum(1 for t in turn_data if t.get("compiled", False)) / len(turn_data),
                    "correctness_rate": sum(1 for t in turn_data if t.get("correct", False)) / len(turn_data),
                }
        
        summary["turn_statistics"] = turn_stats
    
    # Save summary
    summary_path = os.path.join(run_output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n=== Processing Complete ===")
    logger.info(f"Results saved to: {run_output_dir}")
    logger.info(f"SFT data: {output_path}")
    logger.info(f"Summary: {summary_path}")
    
    if successful_results:
        logger.info(f"\n=== Summary Statistics ===")
        logger.info(f"Successful: {len(successful_results)}/{len(all_results)}")
        logger.info(f"Average Turns: {summary['statistics']['avg_turns']:.1f}")
        logger.info(f"Final Compilation Rate: {summary['statistics']['final_compilation_rate']:.1%}")
        logger.info(f"Final Correctness Rate: {summary['statistics']['final_correctness_rate']:.1%}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-turn SFT data generation with Claude")
    parser.add_argument("--input", type=str, default="/root/kernel_book/kernelbook_sft_format.jsonl",
                        help="Input JSONL file with KernelBook data")
    parser.add_argument("--api-key", type=str, required=True,
                        help="Anthropic API key")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help="Output directory for results")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples to process")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Batch size for processing")
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS,
                        help="Maximum number of turns")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Claude model to use")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Process data
    process_kernelbook_data(
        input_file=args.input,
        api_key=args.api_key,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        verbose=args.verbose
    )