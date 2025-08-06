import copy
import logging
import time
from functools import partial
from multiprocessing import Process, Queue, Semaphore
from typing import Dict, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm

from slime_plugins.rollout_buffer.generator.base_generator import BaseGenerator
from slime_plugins.rollout_buffer.generator.kernel_generator import (
    DEFAULT_REMOTE_EVAL_SERVER_URL,
    EVAL_CONCURRENCY,
    SAMPLING_PARAMS,
    get_baseline_runtime,
    load_baseline_timings,
    query_llm_with_retry,
    submit_kernel_eval_request,
)
from slime_plugins.rollout_buffer.generator.kernelbench_config import KERNELBENCH_REWARDS
from slime_plugins.rollout_buffer.generator.reward_utils.kernel_utils import extract_last_code

logger = logging.getLogger(__name__)

# Task type for registration with buffer server
TASK_TYPE = "kernelbench_multiturn"

# Multi-turn configuration
DEFAULT_MAX_TURNS = 3  # Maximum number of turns
DEFAULT_GAMMA = 0.4  # Discount factor for aggregated return


def construct_multi_turn_prompt(
    original_prompt: List[dict],
    turn_idx: int,
    history: List[dict],
) -> List[dict]:
    """Construct multi-turn prompt with history of previous attempts.

    Args:
        original_prompt: The original instruction prompt
        turn_idx: Current turn index (0-based)
        history: List of previous turns with kernel code and evaluation results

    Returns:
        Constructed prompt with history context
    """
    if turn_idx == 0:
        # First turn: use original prompt as-is
        return copy.deepcopy(original_prompt)

    # For subsequent turns, prepend history to the original prompt
    messages = copy.deepcopy(original_prompt)

    # Build context from previous attempts
    context_parts = []
    for i, turn_data in enumerate(history):
        context_parts.append(f"## Previous Attempt {i + 1}")

        # Add the generated kernel code
        if "kernel_code" in turn_data:
            context_parts.append("Generated Kernel:")
            context_parts.append("```python")
            context_parts.append(turn_data["kernel_code"])
            context_parts.append("```")

        # Add evaluation results
        if "eval_result" in turn_data:
            eval_result = turn_data["eval_result"]
            context_parts.append("\nEvaluation Results:")
            context_parts.append(f"- Compilation: {'✓ Passed' if eval_result.get('compiled', False) else '✗ Failed'}")
            context_parts.append(
                f"- Correctness: {'✓ Passed' if eval_result.get('correctness', False) else '✗ Failed'}"
            )

            if eval_result.get("runtime", 0) > 0:
                runtime = eval_result["runtime"]
                speedup = eval_result.get("speedup", 0)
                context_parts.append(f"- Runtime: {runtime:.3f}ms")
                if speedup > 0:
                    context_parts.append(f"- Speedup: {speedup:.2f}x")

            # Add any error messages
            if eval_result.get("error_message"):
                context_parts.append(f"- Error: {eval_result['error_message']}")

        context_parts.append("")  # Empty line between attempts

    # Add instruction for improvement
    context_parts.append(f"## Attempt {turn_idx + 1}")
    context_parts.append("Based on the previous attempts above, generate an improved kernel that:")
    context_parts.append("1. Fixes any compilation or correctness errors")
    context_parts.append("2. Improves performance if possible")
    context_parts.append("3. Maintains the same functionality as required")
    context_parts.append("\nGenerate the improved kernel code:")

    # Prepend context to the user message
    context_str = "\n".join(context_parts)

    # Find the user message and prepend context
    for msg in messages:
        if msg["role"] == "user":
            msg["content"] = context_str + "\n\n" + msg["content"]
            break

    return messages


def calculate_aggregated_return(
    turn_rewards: List[float],
    gamma: float = DEFAULT_GAMMA,
) -> float:
    """Calculate aggregated return with discount factor.

    Args:
        turn_rewards: List of rewards for each turn
        gamma: Discount factor (default 0.4)

    Returns:
        Aggregated discounted return
    """
    if not turn_rewards:
        return 0.0

    aggregated_return = 0.0
    for t, reward in enumerate(turn_rewards):
        aggregated_return += (gamma**t) * reward

    return aggregated_return


def rollout_multi_turn_trajectory(
    item: dict,
    client: OpenAI,
    sampling_params: dict,
    remote_eval_server_url: str,
    eval_semaphore: Semaphore,
    backend: str = "triton",
    max_turns: int = DEFAULT_MAX_TURNS,
    gamma: float = DEFAULT_GAMMA,
    baseline_timings: Optional[Dict] = None,
) -> Tuple[List[dict], float, List[float], List[dict]]:
    """Execute multi-turn rollout for kernel generation.

    Args:
        item: Data item with prompt and metadata
        client: OpenAI client for LLM queries
        sampling_params: Sampling parameters for generation
        remote_eval_server_url: URL of evaluation server
        eval_semaphore: Semaphore for evaluation concurrency
        backend: Backend type (triton/cuda)
        max_turns: Maximum number of turns
        gamma: Discount factor for aggregated return
        baseline_timings: Baseline timing data for performance comparison

    Returns:
        Tuple of (final_messages, aggregated_return, turn_rewards, history)
    """
    original_prompt = item["prompt"]
    history = []
    turn_rewards = []
    messages = None

    for turn_idx in range(max_turns):
        # Construct prompt with history
        prompt = construct_multi_turn_prompt(original_prompt, turn_idx, history)

        # Generate response
        try:
            assistant_content = query_llm_with_retry(client, prompt, sampling_params, tools=None)
            assistant_message = {
                "role": "assistant",
                "content": assistant_content,
            }
        except Exception as e:
            logger.error(f"Error in turn {turn_idx}: {e}")
            assistant_message = {
                "role": "assistant",
                "content": f"Error: Failed to generate response in turn {turn_idx}",
            }

        # Build messages for evaluation
        messages = prompt + [assistant_message]

        # Extract kernel code
        kernel_code = extract_last_code(assistant_content)

        # Evaluate the generated kernel
        eval_item = copy.deepcopy(item)
        eval_item["messages"] = messages
        eval_result = submit_kernel_eval_request(
            eval_semaphore,
            remote_eval_server_url,
            eval_item,
            backend=backend,
            baseline_timings=baseline_timings,
        )

        # Extract reward and evaluation details
        turn_reward = eval_result.reward if hasattr(eval_result, "reward") else 0.0
        turn_rewards.append(turn_reward)

        # Build history entry
        history_entry = {
            "turn_idx": turn_idx,
            "kernel_code": kernel_code if kernel_code else "",
            "eval_result": {
                "compiled": eval_result.exec_result.compiled if hasattr(eval_result, "exec_result") else False,
                "correctness": eval_result.exec_result.correctness if hasattr(eval_result, "exec_result") else False,
                "runtime": eval_result.exec_result.runtime if hasattr(eval_result, "exec_result") else 0,
                "speedup": 0,  # Will be calculated if baseline available
                "error_message": eval_result.eval_response if hasattr(eval_result, "eval_response") else "",
            },
            "reward": turn_reward,
        }

        # Calculate speedup if applicable
        if hasattr(eval_result, "exec_result") and eval_result.exec_result.runtime > 0 and baseline_timings:
            extra_info = item.get("extra_info", {})
            level = extra_info.get("level", None)
            problem_name = extra_info.get("problem_name", "")
            problem_id = extra_info.get("problem_id", None)

            if problem_id is not None and problem_name:
                baseline_key = f"{problem_id}_{problem_name}.py"
            elif problem_name:
                baseline_key = problem_name + ".py"
            else:
                baseline_key = None

            if baseline_key:
                baseline_runtime = get_baseline_runtime(level, baseline_key, baseline_timings)
                if baseline_runtime and baseline_runtime > 0:
                    speedup = baseline_runtime / eval_result.exec_result.runtime
                    history_entry["eval_result"]["speedup"] = speedup

        history.append(history_entry)

        # Early termination conditions
        if turn_reward >= KERNELBENCH_REWARDS["correctness"] + 1.0:
            # Got correctness + good performance, no need to continue
            logger.info(f"Early termination at turn {turn_idx}: achieved good performance")
            break

        if turn_idx > 0 and all(r == 0 for r in turn_rewards):
            # Multiple failures, unlikely to improve
            logger.info(f"Early termination at turn {turn_idx}: multiple failures")
            break

    # Calculate aggregated return
    aggregated_return = calculate_aggregated_return(turn_rewards, gamma)

    return messages, aggregated_return, turn_rewards, history


def worker_process_multi_turn(
    task_queue,
    done_queue,
    rollout_func,
    client,
    sampling_params,
    remote_eval_server_url,
    eval_semaphore,
    baseline_timings,
    max_turns,
    gamma,
):
    """Worker process for multi-turn rollout."""
    while True:
        item = task_queue.get()
        if item == "STOP":
            break

        # Execute multi-turn rollout
        messages, aggregated_return, turn_rewards, history = rollout_func(
            item,
            client,
            sampling_params,
            remote_eval_server_url,
            eval_semaphore,
            max_turns=max_turns,
            gamma=gamma,
            baseline_timings=baseline_timings,
        )

        # Prepare output item
        item["messages"] = messages
        item["reward"] = aggregated_return  # Use aggregated return as final reward
        item["turn_rewards"] = turn_rewards
        item["history"] = history
        item["num_turns"] = len(turn_rewards)
        item["rollout_index"] = 1

        # Add execution details from the last turn
        if history:
            last_turn = history[-1]
            execution_details = {
                "final_compiled": last_turn["eval_result"]["compiled"],
                "final_correctness": last_turn["eval_result"]["correctness"],
                "final_runtime": last_turn["eval_result"]["runtime"],
                "final_speedup": last_turn["eval_result"].get("speedup", 0),
                "num_turns": len(history),
                "turn_rewards": turn_rewards,
                "aggregated_return": aggregated_return,
            }
        else:
            execution_details = {
                "num_turns": 0,
                "turn_rewards": [],
                "aggregated_return": 0,
            }

        # Preserve original extra_info
        original_extra_info = item.get("extra_info", {})
        item.update(sampling_params)
        item["timestamp"] = str(time.time())

        output_item = {
            "uid": item.pop("uid"),
            "messages": messages,
            "reward": aggregated_return,
            "instance_id": item.pop("instance_id"),
            "extra_info": {**original_extra_info, **item},
            "execution_details": execution_details,
            "multi_turn_data": {
                "history": history,
                "turn_rewards": turn_rewards,
                "aggregated_return": aggregated_return,
                "gamma": gamma,
                "max_turns": max_turns,
            },
        }
        done_queue.put(output_item)

    done_queue.put("COMPLETE")


class MultiTurnKernelGenerator(BaseGenerator):
    """Multi-turn trajectory generator for KernelBench."""

    def __init__(
        self,
        remote_engine_url,
        remote_buffer_url,
        num_repeat_per_sample=1,
        queue_size=1000000,
        num_process=10,
        task_type=None,  # Will use TASK_TYPE constant if not provided
        max_tokens=4096,
        num_repeats=10,
        skip_instance_ids: Optional[List[str]] = None,
        remote_eval_server_url: str = "http://localhost:18188",
        eval_concurrency: int = 10,
        max_turns: int = DEFAULT_MAX_TURNS,
        gamma: float = DEFAULT_GAMMA,
    ):
        super().__init__(
            remote_engine_url,
            remote_buffer_url,
            num_repeat_per_sample,
            queue_size,
            num_process,
            task_type or TASK_TYPE,  # Use constant if not provided
            max_tokens,
            num_repeats,
            skip_instance_ids,
        )

        if remote_eval_server_url is None:
            remote_eval_server_url = DEFAULT_REMOTE_EVAL_SERVER_URL

        self.remote_eval_server_url = remote_eval_server_url
        self.eval_concurrency = eval_concurrency
        self.eval_semaphore = Semaphore(eval_concurrency)
        self.task_queue, self.done_queue = Queue(maxsize=self.queue_size), Queue(maxsize=self.queue_size)

        # Multi-turn parameters
        self.max_turns = max_turns
        self.gamma = gamma

        # Initialize sampling_params
        self.sampling_params = SAMPLING_PARAMS.copy()
        self.sampling_params["max_tokens"] = max_tokens

        # Load baseline timings
        self.baseline_timings = load_baseline_timings()
        if self.baseline_timings:
            logger.info(f"Loaded baseline timings for {sum(len(v) for v in self.baseline_timings.values())} problems")
        else:
            logger.warning("No baseline timings loaded, performance rewards will not be calculated")

        logger.info(f"Multi-turn kernel generator initialized with max_turns={max_turns}, gamma={gamma}")

    def rollout_one_epoch(self, input_file, rollout_func):
        """Execute one epoch of multi-turn rollout."""
        processes = []
        for _ in range(self.num_process):
            process = Process(
                target=partial(
                    worker_process_multi_turn,
                    self.task_queue,
                    self.done_queue,
                    rollout_func,
                    self.client,
                    self.sampling_params,
                    self.remote_eval_server_url,
                    self.eval_semaphore,
                    self.baseline_timings,
                    self.max_turns,
                    self.gamma,
                ),
            )
            process.start()
            processes.append(process)

        # Read data into queue
        from slime_plugins.rollout_buffer.generator.kernel_generator import read_data_into_queue

        reader_process = Process(
            target=read_data_into_queue,
            args=(
                input_file,
                set(self.skip_instance_ids) if self.skip_instance_ids else set(),
                self.num_repeats,
                self.num_repeat_per_sample,
                self.task_queue,
                self.num_process,
            ),
        )
        reader_process.start()

        # Process results
        progress_bar = tqdm(desc="Multi-turn rollout")
        num_finished = 0
        while num_finished < self.num_process:
            item = self.done_queue.get()
            if item == "COMPLETE":
                num_finished += 1
            else:
                assert "reward" in item, f"reward not in item: {item}"
                assert "instance_id" in item, f"instance_id not in item: {item}"
                self.send_data_to_buffer(item)
                progress_bar.update(1)

                # Log multi-turn statistics
                if "multi_turn_data" in item:
                    mt_data = item["multi_turn_data"]
                    logger.info(
                        f"Instance {item['instance_id']}: "
                        f"turns={len(mt_data['turn_rewards'])}, "
                        f"rewards={mt_data['turn_rewards']}, "
                        f"aggregated={mt_data['aggregated_return']:.3f}"
                    )

        progress_bar.close()

        # Wait for all processes to complete
        for process in processes:
            process.join()
        reader_process.join()

        return "finished"

    def entry(self, input_file, rollout_func, num_epoch=1):
        """Entry point for multi-turn rollout."""
        for epoch in range(num_epoch):
            logger.info(f"Starting epoch {epoch + 1}/{num_epoch}")
            status = self.rollout_one_epoch(input_file, rollout_func)
            logger.info(f"Epoch {epoch + 1} completed with status: {status}")


def run_rollout(data: dict):
    """Run multi-turn rollout with provided configuration (entry point for buffer server)."""
    logger.info(f"Starting multi-turn kernel rollout with data: {data}")

    rollout_func = rollout_multi_turn_trajectory

    logger.info(f"Waiting for 10 seconds for buffer server to start")
    time.sleep(10)

    global SAMPLING_PARAMS
    for k, v in data["sampling_params"].items():
        SAMPLING_PARAMS[k] = v
        logger.info(f"Set {k} to {v}")

    # Extract multi-turn parameters
    max_turns = int(data.get("max_turns", DEFAULT_MAX_TURNS))
    gamma = float(data.get("gamma", DEFAULT_GAMMA))

    generator = MultiTurnKernelGenerator(
        data["remote_engine_url"],
        data["remote_buffer_url"],
        num_repeat_per_sample=int(data["num_repeat_per_sample"]),
        queue_size=1000000,
        max_tokens=int(data["sampling_params"]["max_tokens"]),
        num_process=int(data.get("num_process", 100)),
        task_type=data["task_type"],
        skip_instance_ids=data.get("skip_instance_ids", None),
        remote_eval_server_url=data.get("remote_eval_server_url", DEFAULT_REMOTE_EVAL_SERVER_URL),
        eval_concurrency=int(data.get("eval_concurrency", EVAL_CONCURRENCY)),
        max_turns=max_turns,
        gamma=gamma,
    )

    generator.entry(data["input_file"], rollout_func, int(data.get("num_epoch", 1)))
