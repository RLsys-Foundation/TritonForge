import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import uvicorn
import os
import sys

# Add project root to path to allow importing from src
# The script is in KernelBench/scripts, so we go up two levels
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.eval import eval_kernel_against_ref, KernelExecResult

app = FastAPI(
    title="KernelBench Evaluation Server",
    description="A simple server to evaluate custom CUDA kernels, processing one request at a time.",
)

# Lock to ensure only one evaluation runs at a time
eval_lock = asyncio.Lock()


class EvalRequest(BaseModel):
    original_model_src: str
    custom_model_src: str
    seed_num: int = 42
    num_correct_trials: int = 5
    num_perf_trials: int = 100
    verbose: bool = False
    measure_performance: bool = True


@app.post("/eval", response_model=KernelExecResult)
async def evaluate_kernel(request: EvalRequest):
    """
    Accepts a kernel evaluation request and processes it.
    This endpoint is protected by a lock to ensure that only one evaluation
    is processed at a time, preventing resource contention on the GPU.
    Subsequent requests will wait until the current evaluation is complete.
    """
    if not torch.cuda.is_available():
        raise HTTPException(
            status_code=503, detail="CUDA is not available on the server."
        )

    device = torch.device("cuda")

    async with eval_lock:
        try:
            # FastAPI runs sync functions in a threadpool automatically.
            result = eval_kernel_against_ref(
                original_model_src=request.original_model_src,
                custom_model_src=request.custom_model_src,
                seed_num=request.seed_num,
                num_correct_trials=request.num_correct_trials,
                num_perf_trials=request.num_perf_trials,
                verbose=request.verbose,
                measure_performance=request.measure_performance,
                device=device,
            )
            return result
        except Exception as e:
            import traceback

            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"An unexpected error occurred during evaluation: {e}",
            )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=18188)
