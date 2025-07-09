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
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import List, Optional

# Add project root to path to allow importing from src
# The script is in KernelBench/scripts, so we go up two levels
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

# Initialize GPU management
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"[Server] Detected {NUM_GPUS} CUDA devices")

# Create semaphore for GPU resource management
# Each permit corresponds to one GPU device
gpu_semaphore = asyncio.Semaphore(NUM_GPUS) if NUM_GPUS > 0 else None

# Track available GPU devices
available_devices = list(range(NUM_GPUS)) if NUM_GPUS > 0 else []
device_lock = asyncio.Lock()  # Lock for managing device assignment

app = FastAPI(
    title="KernelBench Evaluation Server",
    description=f"A server to evaluate custom CUDA and Triton kernels with {NUM_GPUS} GPU(s) support.",
)

# Global variables for timeout handling
_timeout_occurred = False
_evaluation_thread = None

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
        HTTPException: If no GPU devices are available
    """
    if not available_devices:
        raise HTTPException(
            status_code=503, 
            detail="No GPU devices available"
        )
    
    async with device_lock:
        # Try to use preferred device if specified and available
        if preferred_device is not None and preferred_device in available_devices:
            device_id = preferred_device
            available_devices.remove(device_id)
            print(f"[Server] Assigned preferred GPU device {device_id}")
            return device_id
        
        # Otherwise, assign the first available device
        if available_devices:
            device_id = available_devices.pop(0)
            print(f"[Server] Assigned GPU device {device_id}")
            return device_id
        else:
            raise HTTPException(
                status_code=503,
                detail="No GPU devices available at the moment"
            )

async def release_gpu_device(device_id: int):
    """
    Release a GPU device back to the available pool
    
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
    try:
        if torch.cuda.is_available():
            if device_id is not None:
                with torch.cuda.device(device_id):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            else:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            import gc
            gc.collect()
            print(f"[Server] Emergency CUDA cleanup completed for device {device_id}")
    except Exception as e:
        print(f"[Server] Error during emergency cleanup: {e}")

def timeout_handler(signum, frame):
    """
    Signal handler for SIGALRM timeout
    
    Args:
        signum: Signal number (SIGALRM)
        frame: Current stack frame
    
    Raises:
        TimeoutException: Always raises to interrupt execution
    """
    global _timeout_occurred
    _timeout_occurred = True
    print("[Server] Timeout signal received, setting timeout flag")
    
    # Try to interrupt the evaluation thread if it's running
    if _evaluation_thread and _evaluation_thread.is_alive():
        print("[Server] Attempting to interrupt evaluation thread")
        # Note: We can't forcefully kill the thread, but we can set the flag
        # The evaluation function should check this flag periodically
    
    raise TimeoutException("Evaluation timed out")

def run_evaluation_with_timeout(request: EvalRequest, device: int, timeout: int = 300):
    """
    Run evaluation with threading-based timeout protection
    
    This approach uses threading.Thread.join(timeout) to implement timeout.
    It's more portable than signal-based timeout and works in any thread context.
    
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
    global _timeout_occurred, _evaluation_thread
    
    # Reset timeout flag
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

def run_evaluation_with_signal_timeout(request: EvalRequest, device: int, timeout: int = 300):
    """
    Alternative implementation using signal.alarm() for timeout (only works in main thread)
    
    This approach uses UNIX signals for timeout handling. It's more precise than
    threading-based timeout but only works in the main thread and on UNIX systems.
    
    Args:
        request: Evaluation request parameters
        device: CUDA device index
        timeout: Timeout in seconds (default: 300)
    
    Returns:
        KernelExecResult: Evaluation result
    
    Raises:
        TimeoutException: If evaluation times out
        Exception: Any other exception from evaluation
    
    Note:
        This function only works when called from the main thread on UNIX systems.
        Use run_evaluation_with_timeout() for better portability.
    """
    global _timeout_occurred
    
    # Reset timeout flag
    _timeout_occurred = False
    
    # Set up signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    
    try:
        # Set alarm for timeout
        signal.alarm(timeout)
        
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
        
        # Cancel alarm if evaluation completed successfully
        signal.alarm(0)
        return result
        
    except TimeoutException:
        # Timeout occurred
        print(f"[Server] Evaluation timed out after {timeout} seconds")
        cleanup_cuda_context(device)
        raise
    except Exception as e:
        # Other exceptions
        signal.alarm(0)  # Cancel alarm
        print(f"[Server] Error in evaluation: {e}")
        cleanup_cuda_context(device)
        raise
    finally:
        # Restore original signal handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

def run_evaluation_hybrid_timeout(request: EvalRequest, device: int, timeout: int = 300, use_signal: bool = False):
    """
    Hybrid timeout implementation that can use either signal or threading approach
    
    Args:
        request: Evaluation request parameters
        device: CUDA device index
        timeout: Timeout in seconds (default: 300)
        use_signal: Whether to use signal-based timeout (default: False)
    
    Returns:
        KernelExecResult: Evaluation result
    
    Raises:
        TimeoutException: If evaluation times out
        Exception: Any other exception from evaluation
    """
    if use_signal and threading.current_thread() is threading.main_thread():
        # Use signal-based timeout if requested and we're in the main thread
        print("[Server] Using signal-based timeout")
        return run_evaluation_with_signal_timeout(request, device, timeout)
    else:
        # Use threading-based timeout as fallback
        print("[Server] Using threading-based timeout")
        return run_evaluation_with_timeout(request, device, timeout)

@app.post("/eval", response_model=KernelExecResult)
async def evaluate_kernel(request: EvalRequest):
    """
    Accepts a kernel evaluation request and processes it using available GPUs.
    This endpoint uses a semaphore to control concurrent access to GPU resources,
    allowing multiple evaluations to run simultaneously on different GPUs.
    Each request gets assigned a specific GPU device ID based on availability.
    
    Supports both CUDA and Triton backends for kernel evaluation.
    """
    if not torch.cuda.is_available():
        raise HTTPException(
            status_code=503, detail="CUDA is not available on the server."
        )
    
    if NUM_GPUS == 0:
        raise HTTPException(
            status_code=503, detail="No CUDA devices detected on the server."
        )

    # Acquire semaphore permit (blocks if all GPUs are busy)
    async with gpu_semaphore:
        # Acquire a specific GPU device
        device_id = await acquire_gpu_device(request.preferred_device)
        
        try:
            print(f"[Server] Starting evaluation on GPU device {device_id} with {request.backend} backend")
            
            # Run evaluation with timeout using threading approach
            # Note: We use threading approach because signal-based timeout
            # only works in the main thread, but FastAPI runs in asyncio event loop
            loop = asyncio.get_event_loop()
            
            try:
                result = await loop.run_in_executor(
                    None,  # Use default executor
                    run_evaluation_with_timeout,
                    request,
                    device_id,
                    300  # 5 minute timeout
                )
            except TimeoutException:
                raise HTTPException(
                    status_code=504,
                    detail=f"Evaluation timed out (300 seconds) on GPU device {device_id} with {request.backend} backend. The kernel may be stuck or taking too long to compile."
                )
            
            # Handle None result (compilation lock error)
            if result is None:
                raise HTTPException(
                    status_code=503, 
                    detail=f"Compilation lock error occurred on GPU device {device_id} with {request.backend} backend. Please retry."
                )
            
            print(f"[Server] Completed evaluation on GPU device {device_id} with {request.backend} backend")
            return result
            
        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
        except Exception as e:
            import traceback
            
            # Log the full traceback for debugging
            traceback.print_exc()
            
            # Emergency cleanup
            cleanup_cuda_context(device_id)
            
            # Create a proper error response
            error_message = f"An unexpected error occurred during evaluation on GPU device {device_id} with {request.backend} backend: {str(e)}"
            
            # For CUDA architecture errors, provide more specific guidance
            if "Unknown CUDA arch" in str(e) or "GPU not supported" in str(e):
                error_message = f"CUDA compilation error on GPU device {device_id} with {request.backend} backend - unsupported GPU architecture: {str(e)}"
            elif "sync_stream" in str(e) or "has no member" in str(e):
                error_message = f"PyTorch API compatibility error on GPU device {device_id} with {request.backend} backend: {str(e)}"
                
            raise HTTPException(
                status_code=500,
                detail=error_message
            )
        finally:
            # Always release the GPU device back to the pool
            await release_gpu_device(device_id)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    async with device_lock:
        available_count = len(available_devices)
        busy_count = NUM_GPUS - available_count
    
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "supported_backends": ["cuda", "triton"],
        "total_gpu_devices": NUM_GPUS,
        "available_gpu_devices": available_count,
        "busy_gpu_devices": busy_count,
        "available_device_ids": available_devices.copy(),
        "cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST", "not set")
    }

@app.get("/gpu_status")
async def gpu_status():
    """Get detailed GPU status information"""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    async with device_lock:
        gpu_info = []
        for i in range(NUM_GPUS):
            device_info = {
                "device_id": i,
                "name": torch.cuda.get_device_name(i),
                "available": i in available_devices,
                "memory_allocated": torch.cuda.memory_allocated(i),
                "memory_cached": torch.cuda.memory_reserved(i),
                "compute_capability": torch.cuda.get_device_capability(i)
            }
            gpu_info.append(device_info)
    
    return {
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
            cleanup_cuda_context(device_id)
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
        "triton_available": True,  # Triton is available if PyTorch is available
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
    uvicorn.run(app, host="0.0.0.0", port=18188)