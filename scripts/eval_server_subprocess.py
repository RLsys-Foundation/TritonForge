#!/usr/bin/env python3
"""
Subprocess-isolated evaluation server for KernelBench
Each evaluation runs in a separate process to prevent CUDA context corruption

Default timeout: 600 seconds (10 minutes) to account for:
- Process spawn overhead
- CUDA context initialization per process
- Triton kernel compilation (no cross-process caching)
- Complex kernel evaluation
"""

import asyncio
import subprocess
import json
import tempfile
import os
import sys
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import multiprocessing as mp
import pickle
import traceback

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# Set CUDA architecture for H100 compatibility at startup
if torch.cuda.is_available():
    device_capability = torch.cuda.get_device_capability()
    major, minor = device_capability
    
    # Set appropriate CUDA architecture
    if major == 9 and minor == 0:  # H100
        os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
        print(f"[Server] Detected H100 GPU, setting TORCH_CUDA_ARCH_LIST=9.0")
    elif major == 8:  # Ampere
        if minor == 6:
            os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
        print(f"[Server] Detected Ampere GPU, setting TORCH_CUDA_ARCH_LIST={major}.{minor}")
    elif major == 7:  # Turing/Volta
        if minor == 5:
            os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0"
        print(f"[Server] Detected Turing/Volta GPU, setting TORCH_CUDA_ARCH_LIST={major}.{minor}")
    else:
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        print(f"[Server] Detected GPU with compute capability {major}.{minor}")

app = FastAPI(
    title="KernelBench Subprocess Isolation Server",
    description="Evaluation server with process isolation to prevent CUDA context corruption"
)

# Configuration
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"[Server] Detected {NUM_GPUS} CUDA devices")

# GPU allocation tracking
available_devices = list(range(NUM_GPUS))
device_lock = asyncio.Lock()
gpu_semaphore = asyncio.Semaphore(NUM_GPUS) if NUM_GPUS > 0 else None


class EvalRequest(BaseModel):
    original_model_src: str
    custom_model_src: str
    seed_num: int = 42
    num_correct_trials: int = 5
    num_perf_trials: int = 100
    verbose: bool = False
    measure_performance: bool = True
    preferred_device: Optional[int] = None
    backend: str = "cuda"


def _check_gpu_health(device_id: int) -> bool:
    """Check GPU health in isolated process (module-level for pickling)"""
    import os
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    try:
        device = torch.device("cuda:0")
        torch.cuda.synchronize(device)
        return True
    except:
        return False


def _get_gpu_info(device_id: int) -> Dict[str, Any]:
    """Get GPU info in isolated process (module-level for pickling)"""
    import torch
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    try:
        props = torch.cuda.get_device_properties(0)
        return {
            "device_id": device_id,
            "name": props.name,
            "available": True,
            "memory_allocated": torch.cuda.memory_allocated(0),
            "memory_cached": torch.cuda.memory_reserved(0),
            "compute_capability": (props.major, props.minor)
        }
    except Exception as e:
        return {
            "device_id": device_id,
            "error": str(e),
            "available": False
        }


def run_isolated_evaluation(request_dict: Dict[str, Any], device_id: int) -> Dict[str, Any]:
    """
    Run evaluation in an isolated subprocess
    This function is executed in a separate process
    """
    import os
    import sys
    import torch
    
    # Set up environment for kernel compilation with DSA
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    # Set Triton cache directory to avoid conflicts
    os.environ["TRITON_CACHE_DIR"] = f"/tmp/triton_cache_gpu_{device_id}"
    
    # Important: Do NOT clear cache on every evaluation (inefficient)
    # DSA will be compiled into kernels automatically with env vars set
    
    # Add project root to path
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, PROJECT_ROOT)
    
    try:
        from src.eval import eval_kernel_against_ref, KernelExecResult
        
        # Convert dict back to proper types
        result = eval_kernel_against_ref(
            original_model_src=request_dict["original_model_src"],
            custom_model_src=request_dict["custom_model_src"],
            seed_num=request_dict["seed_num"],
            num_correct_trials=request_dict["num_correct_trials"],
            num_perf_trials=request_dict["num_perf_trials"],
            verbose=request_dict["verbose"],
            measure_performance=request_dict["measure_performance"],
            device=0,  # Always device 0 since we set CUDA_VISIBLE_DEVICES
            backend=request_dict["backend"],
        )

        # Handle None result (e.g., from SyntaxError)
        if result is None:
            return {
                "success": False,
                "error": "Evaluation returned None (likely due to SyntaxError in code)",
                "category": "syntax_error",
                "details": "The kernel code contains syntax errors or failed to compile",
                "traceback": ""
        }
        
        # Convert result to dict for serialization
        return {
            "success": True,
            "result": result.dict() if hasattr(result, 'dict') else result.__dict__
        }
        
    except Exception as e:
        # Capture detailed error information
        error_info = {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc()
        }
        
        # Add specific error categorization
        error_str = str(e)
        if "out of resource: shared memory" in error_str:
            error_info["category"] = "shared_memory_exceeded"
            error_info["details"] = "Triton kernel requires more shared memory than available"
        elif "illegal memory access" in error_str:
            error_info["category"] = "illegal_memory_access"
            error_info["details"] = "Kernel accessed memory outside allocated bounds"
        elif "Unknown CUDA arch" in error_str:
            error_info["category"] = "unsupported_architecture"
            error_info["details"] = "GPU architecture not supported for this kernel"
        else:
            error_info["category"] = "unknown"
            
        return error_info


async def acquire_gpu_device(preferred_device: Optional[int] = None) -> int:
    """Acquire a GPU device for evaluation"""
    async with device_lock:
        if not available_devices:
            raise HTTPException(status_code=503, detail="No GPU devices available")
        
        if preferred_device is not None and preferred_device in available_devices:
            device_id = preferred_device
            available_devices.remove(device_id)
        else:
            device_id = available_devices.pop(0)
        
        print(f"[Server] Assigned GPU device {device_id}")
        return device_id


async def release_gpu_device(device_id: int):
    """Release a GPU device back to the pool"""
    async with device_lock:
        if device_id not in available_devices:
            available_devices.append(device_id)
            available_devices.sort()
            print(f"[Server] Released GPU device {device_id}")


@app.post("/eval")
async def evaluate_kernel(request: EvalRequest):
    """
    Evaluate kernel in isolated subprocess to prevent CUDA corruption
    """
    if not gpu_semaphore:
        raise HTTPException(status_code=503, detail="No GPUs available on this system")
    
    async with gpu_semaphore:
        device_id = await acquire_gpu_device(request.preferred_device)
        
        try:
            print(f"[Server] Starting isolated evaluation on GPU {device_id} with {request.backend} backend")
            if request.verbose:
                print(f"[Server] Kernel snippet: {request.custom_model_src[:200]}...")
            
            # Convert request to dict for subprocess
            request_dict = request.model_dump() if hasattr(request, 'model_dump') else request.dict()
            
            # Run evaluation in subprocess using multiprocessing with timeout
            try:
                with mp.Pool(processes=1) as pool:
                    # Use apply_async to enable timeout
                    async_result = pool.apply_async(run_isolated_evaluation, (request_dict, device_id))
                    # Wait for result with extended timeout for process isolation overhead
                    # Process isolation adds: subprocess spawn + CUDA init + Triton compilation
                    result_dict = async_result.get(timeout=600)
            except mp.TimeoutError:
                raise HTTPException(
                    status_code=504,
                    detail=f"Evaluation timed out (600 seconds) on GPU device {device_id} with {request.backend} backend. The kernel may be stuck or taking too long to compile."
                )
            
            if result_dict["success"]:
                # Convert dict back to KernelExecResult
                from src.eval import KernelExecResult
                return KernelExecResult(**result_dict["result"])
            else:
                # Handle error from subprocess
                error_msg = f"Evaluation failed on GPU {device_id}: {result_dict['error']}"
                
                if result_dict["category"] == "shared_memory_exceeded":
                    error_msg = f"Triton kernel exceeded shared memory limit on GPU {device_id}. {result_dict['details']}"
                elif result_dict["category"] == "illegal_memory_access":
                    error_msg = f"CUDA illegal memory access on GPU {device_id}. The evaluation was isolated and the GPU remains healthy. {result_dict['details']}"
                elif result_dict["category"] == "unsupported_architecture":
                    error_msg = f"Unsupported GPU architecture on device {device_id}. {result_dict['details']}"
                elif result_dict["category"] == "syntax_error":
                    error_msg = f"Syntax error in kernel code: {result_dict['details']}"
                
                # Log detailed error for debugging
                if request.verbose:
                    print(f"[Server] Detailed error:\n{result_dict['traceback']}")
                
                raise HTTPException(status_code=500, detail=error_msg)
                
        except HTTPException:
            raise
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error during evaluation setup: {str(e)}"
            )
        finally:
            await release_gpu_device(device_id)


@app.get("/")
async def root():
    """Root endpoint providing service information"""
    return {
        "service": "KernelBench Subprocess Isolation Server",
        "status": "running",
        "cuda_available": torch.cuda.is_available(),
        "num_gpus": NUM_GPUS,
        "backends": ["cuda", "triton"],
        "features": [
            "Process isolation for each evaluation",
            "CUDA DSA enabled during compilation",
            "GPU corruption prevention",
            "Automatic recovery from failures",
            "No cross-contamination between evaluations"
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    gpu_info = []
    
    if torch.cuda.is_available():
        for i in range(NUM_GPUS):
            try:
                # Quick check in subprocess to avoid corruption
                with mp.Pool(processes=1) as pool:
                    is_healthy = pool.apply(_check_gpu_health, (i,))
                
                gpu_info.append({
                    "device_id": i,
                    "available": i in available_devices,
                    "status": "healthy" if is_healthy else "error"
                })
            except Exception as e:
                gpu_info.append({
                    "device_id": i,
                    "available": False,
                    "status": "error",
                    "error": str(e)
                })
    
    # Calculate busy devices
    busy_count = NUM_GPUS - len(available_devices)
    
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "supported_backends": ["cuda", "triton"],
        "total_gpu_devices": NUM_GPUS,
        "available_gpu_devices": len(available_devices),
        "busy_gpu_devices": busy_count,
        "available_device_ids": available_devices.copy(),
        "cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST", "not set"),
        "devices": gpu_info,
        "isolation": "subprocess",
        "cuda_dsa": "enabled"
    }


@app.post("/reset_gpu/{device_id}")
async def reset_gpu(device_id: int):
    """Reset a specific GPU (no-op with subprocess isolation)"""
    if device_id < 0 or device_id >= NUM_GPUS:
        raise HTTPException(status_code=400, detail=f"Invalid device ID: {device_id}")
    
    return {
        "status": "GPU reset not needed with subprocess isolation",
        "device_id": device_id,
        "message": "Each evaluation runs in a fresh process"
    }


@app.get("/gpu_status")
async def gpu_status():
    """Get detailed GPU status information (compatible with simple_eval_server)"""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    async with device_lock:
        gpu_info = []
        for i in range(NUM_GPUS):
            try:
                # Get device info in subprocess to avoid corruption
                with mp.Pool(processes=1) as pool:
                    device_info = pool.apply(_get_gpu_info, (i,))
                    device_info["available"] = i in available_devices
                    gpu_info.append(device_info)
                    
            except Exception as e:
                gpu_info.append({
                    "device_id": i,
                    "error": str(e),
                    "available": False
                })
    
    return {
        "total_devices": NUM_GPUS,
        "devices": gpu_info,
        "semaphore_permits": gpu_semaphore._value if gpu_semaphore else 0
    }


@app.post("/cleanup")
async def manual_cleanup():
    """Manual cleanup endpoint (no-op with subprocess isolation)"""
    return {
        "status": "cleanup not needed with subprocess isolation",
        "message": "Each evaluation runs in a fresh process which cleans up automatically"
    }


@app.get("/backend_info")
async def backend_info():
    """Get information about supported backends"""
    return {
        "supported_backends": ["cuda", "triton"],
        "default_backend": "cuda",
        "backend_descriptions": {
            "cuda": "Custom CUDA kernels compiled with PyTorch's C++ extension system",
            "triton": "Custom Triton kernels using OpenAI's Triton compiler"
        },
        "cuda_available": torch.cuda.is_available(),
        "triton_available": True,
        "process_isolation": True,
        "cuda_dsa_enabled": True
    }


@app.post("/reset_devices")
async def reset_devices():
    """Reset device availability (for debugging)"""
    async with device_lock:
        global available_devices
        available_devices = list(range(NUM_GPUS))
        return {
            "status": "device pool reset",
            "available_devices": available_devices.copy()
        }


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    # Check for GPUs
    if NUM_GPUS == 0:
        print("[Server] WARNING: No GPUs detected. Server will run but cannot evaluate kernels.")
    else:
        print(f"[Server] Starting subprocess isolation server with {NUM_GPUS} GPUs")
        print("[Server] Each evaluation will run in an isolated process")
        print("[Server] CUDA DSA will be enabled for all kernel compilations")
    
    # Start server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=18188,
        log_level="info",
        access_log=True
    )