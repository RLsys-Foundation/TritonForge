import json
import random
import copy
import uuid
import time
import warnings
import logging
import re
from typing import Optional, List
from functools import partial
# from queue import Queue
import requests
from multiprocessing import Queue, Semaphore, Process
from concurrent.futures import ProcessPoolExecutor
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from slime_plugins.rollout_buffer.generator.reward_utils.kernel_utils import extract_last_code, KernelEvalResult, KernelExecResult
from slime_plugins.rollout_buffer.generator.base_generator import BaseGenerator
from slime_plugins.rollout_buffer.generator.kernelbench_config import KERNELBENCH_REWARDS, KERNELBENCH_VALIDATION
from slime_plugins.rollout_buffer.generator.triton_ops import TRITON_CORE_OPS

TASK_TYPE = "kernelbench"
DEFAULT_REMOTE_EVAL_SERVER_URL = "http://localhost:18188"
EVAL_CONCURRENCY = 2
SAMPLING_PARAMS = {
    "top_p": 1,
}

logger = logging.getLogger(__name__)


def is_valid_reward(reward: float) -> bool:
    """Check if a reward value is valid.

    Args:
        reward: The reward value to check

    Returns:
        bool: True if reward is valid (between 0 and max_reward), False otherwise
    """
    return KERNELBENCH_REWARDS["max_reward"] >= reward >= 0


def validate_submission(code: str) -> bool:
    """Pre-check if submission is valid before evaluation.
    
    Refined validation that allows torch._inductor utilities but ensures real Triton kernels.
    
    Args:
        code: The generated code to validate
        
    Returns:
        bool: True if code passes validation, False otherwise
    """
    if not code:
        return False
    
    # Check if validation is enabled
    if not KERNELBENCH_VALIDATION.get("require_triton_jit", True):
        return True
        
    # Must have @triton.jit decorator
    if '@triton.jit' not in code:
        logger.warning("Submission rejected: Missing @triton.jit decorator")
        return False
    
    # Extract all function definitions with @triton.jit decorator
    triton_kernel_pattern = r'@triton\.jit\s*\n\s*def\s+(\w+)\s*\([^)]*\):[^@]*?(?=\n(?:def|class|@|$))'
    triton_kernels = list(re.finditer(triton_kernel_pattern, code, re.DOTALL | re.MULTILINE))
    
    if not triton_kernels:
        logger.warning("Submission rejected: No Triton kernel functions found")
        return False
    
    # Check if we need to validate Triton operations
    if KERNELBENCH_VALIDATION.get("require_triton_ops", True):
        # For each Triton kernel, check it uses proper Triton operations
        for match in triton_kernels:
            kernel_name = match.group(1)
            kernel_body = match.group(0)
            
            # Use comprehensive list of Triton operations
            has_triton_ops = any(op in kernel_body for op in TRITON_CORE_OPS)
            if not has_triton_ops:
                logger.warning(f"Submission rejected: Kernel '{kernel_name}' doesn't use Triton operations")
                return False
            
            # Only check for forbidden PyTorch operations if configured
            if not KERNELBENCH_VALIDATION.get("allow_torch_in_kernel", False):
                # Check for forbidden PyTorch operations inside kernel
                # Note: We allow torch operations outside kernels for setup/wrapper code
                forbidden_in_kernel = [
                    'torch.', 'nn.', '.cuda()', '.cpu()', 
                    'F.', 'torch.ops', 'aten.', '.backward()'
                ]
                
                for pattern in forbidden_in_kernel:
                    if pattern in kernel_body:
                        logger.warning(f"Submission rejected: Kernel '{kernel_name}' uses forbidden operation '{pattern}'")
                        return False
    
    # Additional check: Must have either a forward method or call function
    if 'def forward(' not in code and 'def call(' not in code:
        logger.warning("Submission rejected: No forward() or call() function found")
        return False
    
    logger.info("Submission passed pre-validation: Contains valid Triton kernel(s)")
    return True


def submit_kernel_eval_request(
    semaphore: Semaphore, 
    eval_server_url: str, 
    item: dict, 
    backend: str = "triton",
    max_retry: int = 3,
) -> KernelEvalResult:
    original_model_src = item["label"]
    messages = item["messages"]
    if messages[-1]["role"] != "assistant":
        raise ValueError(f"last message must be assistant, but got {messages[-1]['role']}")
    
    custom_model_src = extract_last_code(messages[-1]["content"])
    if custom_model_src is None:
        return KernelEvalResult(
            eval_status="failed",
            eval_response="no custom model source found",
            completed_at=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            reward=0.,
            exec_result=KernelExecResult(),
        )
    
    # Pre-validate submission before sending to evaluation
    if not validate_submission(custom_model_src):
        return KernelEvalResult(
            eval_status="rejected",
            eval_response="Submission failed validation: must contain @triton.jit kernel with Triton operations",
            completed_at=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            reward=0.,
            exec_result=KernelExecResult(),
        )
    
    payload = {
        "original_model_src": original_model_src,
        "custom_model_src": custom_model_src,
        "num_correct_trials": 5,
        "num_perf_trials": 100,
        "measure_performance": True,
        "backend": backend,
        "verbose": False,  # Default verbose setting
        "seed": 42,
    }
    res = None
    with semaphore:
        for _ in range(max_retry):
            try:
                response = requests.post(
                    f"{eval_server_url}/eval",
                    json=payload,
                )
                if response.status_code == 200:
                    exec_result = KernelExecResult.model_validate(response.json())
                    reward, response = 0.0, "Current implementation con't pass compile check"
                    if exec_result.compiled:
                        # Use config for rewards
                        reward = KERNELBENCH_REWARDS["compilation"]
                        if exec_result.correctness:
                            reward = KERNELBENCH_REWARDS["correctness"]
                            response = "Current implementation passes correctness check"
                        else:
                            response = "Current implementation passes compile check but fails correctness check"
                    res = KernelEvalResult(
                        eval_status="completed",
                        eval_response=response,
                        completed_at=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                        reward=reward,
                        exec_result=exec_result,
                    )
            except Exception as e:
                logger.error(f"Error submitting kernel eval request: {e}, response: {response.text}")
                return KernelEvalResult(
                    eval_status="failed",
                    eval_response=str(e),
                    completed_at=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    reward=0.,
                    exec_result=KernelExecResult(),
                )

    return res


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=15))
def query_llm_with_retry(
    client: OpenAI,
    messages: List[dict],
    sampling_params: dict,
    tools: Optional[List[dict]] = None,
) -> str:
    response = client.chat.completions.create(
        model="custom",
        messages=messages,
        stream=False,
        seed=random.randint(1, 10000000),
        tools=tools,
        **sampling_params,
    )
    print(f"{response.choices[0]=}")
    return response.choices[0].message.content


def rollout_one_trajectory(
    item: dict, 
    client: OpenAI, 
    sampling_params: dict,
    remote_eval_server_url: str,
    eval_semaphore: Semaphore,
    backend: str = "triton",
    max_retry: int = 3,
) -> List[dict]:
    messages = item["prompt"]
    assistant_message = None
    
    for _ in range(max_retry):
        try:
            assistant_message_content = query_llm_with_retry(client, messages, sampling_params, tools=None)
            assistant_message = {
                "role": "assistant",
                "content": assistant_message_content,
            }
            break  # Success, exit retry loop
        except Exception as e:
            logger.error(f"Error querying LLM: {e}")
            continue

    if assistant_message is None:
        # All retries failed, create a default error message
        assistant_message = {
            "role": "assistant",
            "content": "Error: Failed to generate response after multiple retries."
        }

    messages.append(assistant_message)
    return messages


def worker_process(task_queue, done_queue, rollout_func, reward_func, client, sampling_params, remote_eval_server_url, eval_semaphore):
    while True:
        item = task_queue.get()
        if item == "STOP":
            break
        messages = rollout_func(item, client, sampling_params, remote_eval_server_url, eval_semaphore)
        item["messages"] = messages
        eval_result = reward_func(eval_semaphore, remote_eval_server_url, item)
        reward = eval_result.reward if hasattr(eval_result, 'reward') else 0.0
        item["rollout_index"] = 1
        item["reward"] = reward
        item["extra_info"] = {}
        item.update(sampling_params)
        item["timestamp"] = str(time.time())
        item["round_number"] = len([_ for _ in item["messages"] if _["role"] == "assistant"])

        output_item = {
            "uid": item.pop("uid"),
            "messages": messages,
            "reward": reward,
            "instance_id": item.pop("instance_id"),
            "extra_info": item,
        }
        done_queue.put(output_item)
    
    done_queue.put("COMPLETE")


def read_data_into_queue(
    input_file: str, 
    skip_instance_ids: List[str], 
    num_repeats: int, 
    num_repeat_per_sample: int,
    task_queue: Queue,
    num_process: int,
):
    items = []
    actual_skipped_ids = []

    with open(input_file, "r") as r:
        for line in r:
            item = json.loads(line)
            if skip_instance_ids and item["instance_id"] in skip_instance_ids:
                actual_skipped_ids.append(item["instance_id"])
                continue
            items.append(item)
    
    random.shuffle(items) # shuffle items
    logger.info(f"Read {len(items)} items, skipped {len(actual_skipped_ids)} items")
    
    if skip_instance_ids and len(actual_skipped_ids) < len(skip_instance_ids):
        logger.warning(f"Warning: some instance_ids are skipped, but not all")
        not_skipped_ids = set(skip_instance_ids) ^ set(actual_skipped_ids)
        logger.warning(f"Instance_ids that should be skipped but weren't: {not_skipped_ids}")
        raise ValueError(f"Some instance_ids are skipped, but not all")

    for _ in range(num_repeats):
        for item in items:
            for rollout_index in range(num_repeat_per_sample):
                item_repeat = copy.deepcopy(item)
                if "instance_id" not in item_repeat:
                    raise ValueError(f"instance_id not in item: {item}, the input data must have instance_id")
                
                if "uid" not in item_repeat:
                    item_repeat["uid"] = str(uuid.uuid4())
                
                item_repeat["rollout_index"] = rollout_index
                while task_queue.full():
                    time.sleep(1)
                task_queue.put(item_repeat)
                
    # Put STOP signal for each process
    for _ in range(num_process):
        task_queue.put("STOP")
    logger.info(f"Put {num_process} STOP signals into task_queue")


class KernelGenerator(BaseGenerator):
    """Trajectory generator for KernelBench"""
    
    def __init__(
        self,
        remote_engine_url,
        remote_buffer_url,
        num_repeat_per_sample=1,
        queue_size=1000000,
        num_process=10,
        task_type=TASK_TYPE,
        max_tokens=4096,
        num_repeats=10,
        skip_instance_ids: Optional[List[str]] = None,
        remote_eval_server_url: str = "http://localhost:18188",
        eval_concurrency: int = 10,
    ):
        super().__init__(
            remote_engine_url,
            remote_buffer_url,
            num_repeat_per_sample,
            queue_size,
            num_process,
            task_type,
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
        
        # Initialize sampling_params with global SAMPLING_PARAMS
        # This will be updated in run_rollout function
        self.sampling_params = SAMPLING_PARAMS.copy()
        self.sampling_params["max_tokens"] = max_tokens
    
    def entry(self, input_file, rollout_func, reward_func, num_epoch=1):
        for _ in range(num_epoch):
            status = self.rollout_one_epoch(input_file, rollout_func, reward_func)

    def run(self, input_file, rollout_func, reward_func):
        warnings.warn("This method is deprecated. Please use rollout_one_epoch instead.")
        self.rollout_one_epoch(input_file, rollout_func, reward_func)
    
    def rollout_one_epoch(self, input_file, rollout_func, reward_func):
        processes = []
        for _ in range(self.num_process):
            process = Process(
                target=partial(worker_process, self.task_queue, self.done_queue, rollout_func, reward_func, self.client, self.sampling_params, self.remote_eval_server_url, self.eval_semaphore),
            )
            process.start()
            processes.append(process)
        
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

        progress_bar = tqdm()
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

        progress_bar.close()
        
        # Wait for all processes to complete
        for process in processes:
            process.join()
        reader_process.join()
        
        return "finished"
    

def run_rollout(data: dict):
    logger.info(f"Starting kernel rollout with data: {data}")
    
    rollout_func = rollout_one_trajectory
    reward_func = submit_kernel_eval_request
    
    logger.info(f"Waiting for 10 seconds for buffer server to start")
    time.sleep(10)
    global SAMPLING_PARAMS
    for k, v in data["sampling_params"].items():
        SAMPLING_PARAMS[k] = v
        logger.info(f"Set {k} to {v}", type(v))
    
    generator = KernelGenerator(
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
    )
    
    generator.entry(data["input_file"], rollout_func, reward_func, int(data.get("num_epoch", 1)))
