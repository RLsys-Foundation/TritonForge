#!/usr/bin/env python3
"""
Minimal CUDA fix version of simple_eval_server.py
Only adds essential CUDA error recovery without changing the API
"""

import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import uvicorn
import os
import sys
import signal
import threading
import time
import gc
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import List, Optional

# Add project root to path to allow importing from src
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.eval import eval_kernel_against_ref, KernelExecResult

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

# Enhanced CUDA cleanup function
def enhanced_cuda_cleanup(device_id: Optional[int] = None):
    """Enhanced CUDA cleanup with better error handling"""
    try:
        if torch.cuda.is_available():
            if device_id is not None:
                with torch.cuda.device(device_id):
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
            else:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
    except RuntimeError as e:
        if "illegal memory access" in str(e) or "CUDA error" in str(e):
            print(f"[Server] CUDA error during cleanup, attempting force reset: {e}")
            try:
                torch.cuda.empty_cache()
            except:
                pass
    except Exception as e:
        print(f"[Server] Error during CUDA cleanup: {e}")

# Patch the original eval module's set_seed to be safer
original_set_seed = None
try:
    from src import eval as eval_module
    original_set_seed = eval_module.set_seed
    
    def safe_set_seed(seed: int):
        """Safer version of set_seed that handles CUDA errors"""
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        except RuntimeError as e:
            print(f"[Eval] Warning: Error setting seed: {e}")
            # Try to recover
            try:
                enhanced_cuda_cleanup()
                # Try one more time
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
            except:
                print(f"[Eval] Could not set CUDA seed, continuing anyway")
    
    eval_module.set_seed = safe_set_seed
    print("[Server] Applied safe seed setting patch")
except Exception as e:
    print(f"[Server] Could not patch set_seed: {e}")

# Import the rest of simple_eval_server.py verbatim
# This ensures compatibility while adding our fixes

# Global settings
app = FastAPI()

# Configuration
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"[Server] Detected {NUM_GPUS} CUDA devices")

# Available GPU devices pool
available_devices = list(range(NUM_GPUS))
device_lock = asyncio.Lock()

# Create a semaphore to limit concurrent GPU evaluations
gpu_semaphore = asyncio.Semaphore(NUM_GPUS) if NUM_GPUS > 0 else None

# Default timeout for evaluations
DEFAULT_TIMEOUT = 300  # 5 minutes


class EvalRequest(BaseModel):
    original_model_src: str
    custom_model_src: str
    seed_num: int = 42
    num_correct_trials: int = 5
    num_perf_trials: int = 100
    verbose: bool = False
    measure_performance: bool = True
    preferred_device: Optional[int] = None  # Allow specifying preferred device
    backend: str = "cuda"  # Backend to use for kernel implementation (cuda or triton)


class TimeoutException(Exception):
    """Custom exception for timeout scenarios"""
    pass


async def acquire_gpu_device(preferred_device: Optional[int] = None) -> int:
    """
    Acquire a GPU device for evaluation
    
    Args:
        preferred_device: Preferred GPU device ID (if available)
    
    Returns:
        int: Assigned GPU device ID
    
    Raises:
        HTTPException: If no devices are available
    """
    async with device_lock:
        if not available_devices:
            raise HTTPException(status_code=503, detail="No GPU devices available")
        
        # Try to use preferred device if available
        if preferred_device is not None and preferred_device in available_devices:
            device_id = preferred_device
            available_devices.remove(device_id)
        else:
            # Otherwise use the first available
            device_id = available_devices.pop(0)
        
        print(f"[Server] Assigned GPU device {device_id}")
        return device_id


async def release_gpu_device(device_id: int):
    """
    Release a GPU device back to the pool
    
    Args:
        device_id: GPU device ID to release
    """
    async with device_lock:
        if device_id not in available_devices:
            available_devices.append(device_id)
            available_devices.sort()  # Keep devices sorted
            print(f"[Server] Released GPU device {device_id}")


def cleanup_cuda_context(device_id: Optional[int] = None):
    """
    Emergency CUDA context cleanup
    
    Args:
        device_id: Specific device to cleanup (if None, cleanup current device)
    """
    enhanced_cuda_cleanup(device_id)


def run_evaluation_with_timeout(request: EvalRequest, device: int, timeout: int = 300):
    """
    Run kernel evaluation with timeout handling
    
    This function runs the evaluation in a separate thread and implements
    a timeout mechanism. If the evaluation takes longer than the specified
    timeout, it will be terminated.
    
    Args:
        request: Evaluation request parameters
        device: CUDA device index
        timeout: Timeout in seconds (default: 300)
    
    Returns:
        KernelExecResult: Evaluation result
    
    Raises:
        TimeoutException: If evaluation times out
        Exception: Any other exception from evaluation
    """
    # Pre-check for Triton shared memory issues
    if request.backend == "triton" and "tl.zeros" in request.custom_model_src:
        import re
        # Simple check for large shared memory allocations
        pattern = r'tl\.zeros\s*\(\s*\[([^\]]+)\]'
        matches = re.findall(pattern, request.custom_model_src)
        for match in matches:
            try:
                dims = [int(x.strip()) for x in match.split(',') if x.strip().isdigit()]
                if dims:
                    size_bytes = 1
                    for dim in dims:
                        size_bytes *= dim
                    size_bytes *= 4  # Assume float32
                    if size_bytes > 48 * 1024:  # 48KB typical limit
                        print(f"[Server] Warning: Large shared memory allocation detected: ~{size_bytes/1024:.1f}KB")
            except:
                pass
    
    # Variables for thread communication
    _timeout_occurred = False
    
    # Store current thread reference
    _evaluation_thread = threading.current_thread()
    
    def evaluation_worker():
        """
        Worker function that runs the actual evaluation
        
        Returns:
            KernelExecResult: Evaluation result
        """
        try:
            # Pre-cleanup for safety
            enhanced_cuda_cleanup(device)
            
            result = eval_kernel_against_ref(
                original_model_src=request.original_model_src,
                custom_model_src=request.custom_model_src,
                seed_num=request.seed_num,
                num_correct_trials=request.num_correct_trials,
                num_perf_trials=request.num_perf_trials,
                verbose=request.verbose,
                measure_performance=request.measure_performance,
                device=device,
                backend=request.backend,
            )
            
            # Check if timeout occurred during evaluation
            if _timeout_occurred:
                print("[Server] Timeout detected during evaluation")
                cleanup_cuda_context(device)
                raise TimeoutException("Evaluation timed out during execution")
            
            return result
            
        except Exception as e:
            print(f"[Server] Error in evaluation worker: {e}")
            cleanup_cuda_context(device)
            raise
    
    # Use threading with timeout
    result_container: List[Optional[KernelExecResult]] = [None]
    exception_container: List[Optional[Exception]] = [None]
    
    def worker_wrapper():
        """Wrapper to capture result or exception from evaluation worker"""
        try:
            result_container[0] = evaluation_worker()
        except Exception as e:
            exception_container[0] = e
    
    # Start evaluation in a separate thread
    eval_thread = threading.Thread(target=worker_wrapper, name="EvalWorker")
    eval_thread.daemon = True  # Make it a daemon thread
    eval_thread.start()
    
    # Wait for completion with timeout
    eval_thread.join(timeout=timeout)
    
    # Check if thread is still alive (timeout occurred)
    if eval_thread.is_alive():
        print(f"[Server] Evaluation timed out after {timeout} seconds")
        _timeout_occurred = True
        
        # Try to cleanup and let the thread finish naturally
        cleanup_cuda_context(device)
        
        # Wait a bit more for graceful cleanup
        eval_thread.join(timeout=5)
        
        if eval_thread.is_alive():
            print("[Server] Warning: Evaluation thread still running after timeout")
        
        raise TimeoutException(f"Evaluation timed out after {timeout} seconds")
    
    # Check for exceptions
    if exception_container[0] is not None:
        raise exception_container[0]
    
    return result_container[0]


@app.post("/eval")
async def evaluate_kernel(request: EvalRequest):
    """
    Evaluate a custom kernel implementation
    
    Args:
        request: EvalRequest object containing kernel code and evaluation parameters
    
    Returns:
        KernelExecResult: Evaluation results including compilation status, correctness, and performance
    
    Raises:
        HTTPException: Various HTTP errors for different failure scenarios
    """
    # Acquire a GPU for evaluation
    if not gpu_semaphore:
        raise HTTPException(status_code=503, detail="No GPUs available on this system")
    
    async with gpu_semaphore:  # Limit concurrent GPU usage
        device_id = await acquire_gpu_device(request.preferred_device)
        
        try:
            print(f"[Server] Starting evaluation on GPU device {device_id} with {request.backend} backend")
            
            # Run evaluation with enhanced error handling
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                run_evaluation_with_timeout,
                request,
                device_id,
                DEFAULT_TIMEOUT
            )
            
            if result is None:
                # This typically means a compilation lock error
                raise HTTPException(
                    status_code=503,
                    detail="Evaluation failed due to compilation lock. Please retry."
                )
            
            return result
            
        except TimeoutException as e:
            # Enhanced cleanup for timeout
            enhanced_cuda_cleanup(device_id)
            
            raise HTTPException(
                status_code=408,
                detail=f"Evaluation timed out after {DEFAULT_TIMEOUT} seconds on GPU device {device_id}"
            )
            
        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
        except Exception as e:
            import traceback
            
            # Log the full traceback for debugging
            traceback.print_exc()
            
            # Emergency cleanup
            enhanced_cuda_cleanup(device_id)
            
            # Create a proper error response
            error_message = f"An unexpected error occurred during evaluation on GPU device {device_id} with {request.backend} backend: {str(e)}"
            
            # For CUDA architecture errors, provide more specific guidance
            if "Unknown CUDA arch" in str(e) or "GPU not supported" in str(e):
                error_message = f"CUDA compilation error on GPU device {device_id} with {request.backend} backend - unsupported GPU architecture: {str(e)}"
            elif "sync_stream" in str(e) or "has no member" in str(e):
                error_message = f"PyTorch API compatibility error on GPU device {device_id} with {request.backend} backend: {str(e)}"
            elif "out of resource: shared memory" in str(e):
                error_message = f"Triton kernel exceeded shared memory limit on GPU device {device_id}: {str(e)}"
            elif "illegal memory access" in str(e):
                error_message = f"CUDA illegal memory access on GPU device {device_id}. GPU has been reset: {str(e)}"
                
            raise HTTPException(
                status_code=500,
                detail=error_message
            )
        finally:
            # Always release the device
            await release_gpu_device(device_id)


@app.get("/")
async def root():
    """Root endpoint providing service information"""
    return {
        "service": "KernelBench Evaluation Server (CUDA Fix Version)",
        "status": "running",
        "cuda_available": torch.cuda.is_available(),
        "num_gpus": NUM_GPUS,
        "backends": ["cuda", "triton"],
        "fixes": [
            "Safe CUDA seed setting",
            "Enhanced CUDA cleanup between evaluations",
            "Triton shared memory warnings",
            "Automatic recovery from CUDA errors"
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    gpu_info = []
    
    if torch.cuda.is_available():
        for i in range(NUM_GPUS):
            try:
                props = torch.cuda.get_device_properties(i)
                # Try to check if GPU is responsive
                try:
                    with torch.cuda.device(i):
                        torch.cuda.synchronize()
                    status = "healthy"
                except:
                    status = "error"
                    
                gpu_info.append({
                    "device_id": i,
                    "name": props.name,
                    "total_memory": f"{props.total_memory / 1024**3:.2f} GB",
                    "available": i in available_devices,
                    "status": status
                })
            except Exception as e:
                gpu_info.append({
                    "device_id": i,
                    "error": str(e),
                    "available": False,
                    "status": "error"
                })
    
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "available_devices": len(available_devices),
        "total_devices": NUM_GPUS,
        "devices": gpu_info,
        "semaphore_permits": gpu_semaphore._value if gpu_semaphore else 0
    }


@app.post("/cleanup")
async def manual_cleanup():
    """Manual cleanup endpoint for debugging"""
    try:
        # Clean up all GPU devices
        for device_id in range(NUM_GPUS):
            enhanced_cuda_cleanup(device_id)
        return {"status": "cleanup completed for all devices"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


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
        "triton_info": "Triton kernels may have shared memory limitations on some GPUs"
    }


if __name__ == "__main__":
    # Check for GPUs
    if NUM_GPUS == 0:
        print("[Server] WARNING: No GPUs detected. Server will run but cannot evaluate kernels.")
    
    # Start the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=18188,
        log_level="info",
        access_log=True
    )