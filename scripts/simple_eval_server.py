import asyncio
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import torch
import uvicorn
import os
import sys
import threading
import time
from typing import List, Optional
from contextlib import asynccontextmanager

# Add project root to path to allow importing from src
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.eval import eval_kernel_against_ref, KernelExecResult


class CUDAConfig:
    """CUDA configuration management"""
    
    @staticmethod
    def setup_cuda_arch():
        """Set up CUDA architecture based on detected GPU"""
        if not torch.cuda.is_available():
            return
        
        device_capability = torch.cuda.get_device_capability()
        major, minor = device_capability
        
        # Simplified CUDA architecture mapping
        arch_map = {
            (9, 0): "9.0",  # H100
            (8, 6): "8.6",  # Ampere RTX 30xx
            (8, 0): "8.0",  # Ampere A100
            (7, 5): "7.5",  # Turing RTX 20xx
            (7, 0): "7.0",  # Volta V100
        }
        
        arch = arch_map.get((major, minor), f"{major}.{minor}")
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch
        print(f"[Server] Detected GPU compute capability {major}.{minor}, setting TORCH_CUDA_ARCH_LIST={arch}")


class GPUDeviceManager:
    """GPU device management with automatic resource allocation"""
    
    def __init__(self):
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.available_devices = list(range(self.num_gpus))
        self.device_lock = asyncio.Lock()
        self.semaphore = asyncio.Semaphore(self.num_gpus) if self.num_gpus > 0 else None
        print(f"[Server] Initialized GPU manager with {self.num_gpus} devices")
    
    async def acquire_device(self, preferred_device: Optional[int] = None) -> int:
        """Acquire a GPU device for evaluation"""
        if not self.available_devices:
            raise HTTPException(status_code=503, detail="No GPU devices available")
        
        async with self.device_lock:
            # Try preferred device first
            if preferred_device is not None and preferred_device in self.available_devices:
                device_id = preferred_device
                self.available_devices.remove(device_id)
                print(f"[Server] Assigned preferred GPU device {device_id}")
                return device_id
            
            # Otherwise assign first available
            if self.available_devices:
                device_id = self.available_devices.pop(0)
                print(f"[Server] Assigned GPU device {device_id}")
                return device_id
            else:
                raise HTTPException(status_code=503, detail="No GPU devices available")
    
    async def release_device(self, device_id: int):
        """Release a GPU device back to the pool"""
        async with self.device_lock:
            if device_id not in self.available_devices:
                self.available_devices.append(device_id)
                self.available_devices.sort()
                print(f"[Server] Released GPU device {device_id}")
    
    def cleanup_device(self, device_id: Optional[int] = None):
        """Clean up CUDA context for a device"""
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
                print(f"[Server] CUDA cleanup completed for device {device_id}")
        except Exception as e:
            print(f"[Server] Error during cleanup: {e}")
    
    @asynccontextmanager
    async def get_device(self, preferred_device: Optional[int] = None):
        """Context manager for automatic device acquisition and release"""
        if not self.semaphore:
            raise HTTPException(status_code=503, detail="No GPU devices available")
        
        async with self.semaphore:
            device_id = await self.acquire_device(preferred_device)
            try:
                yield device_id
            except Exception as e:
                self.cleanup_device(device_id)
                raise
            finally:
                await self.release_device(device_id)
    
    async def get_status(self, detailed: bool = False):
        """Get GPU status information"""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        async with self.device_lock:
            available_count = len(self.available_devices)
            busy_count = self.num_gpus - available_count
            
            if not detailed:
                return {
                    "total_devices": self.num_gpus,
                    "available_devices": available_count,
                    "busy_devices": busy_count,
                    "available_device_ids": self.available_devices.copy()
                }
            
            gpu_info = []
            for i in range(self.num_gpus):
                device_info = {
                    "device_id": i,
                    "name": torch.cuda.get_device_name(i),
                    "available": i in self.available_devices,
                    "memory_allocated": torch.cuda.memory_allocated(i),
                    "memory_cached": torch.cuda.memory_reserved(i),
                    "compute_capability": torch.cuda.get_device_capability(i)
                }
                gpu_info.append(device_info)
            
            return {
                "total_devices": self.num_gpus,
                "available_devices": available_count,
                "busy_devices": busy_count,
                "devices": gpu_info,
                "semaphore_permits": self.semaphore._value if self.semaphore else 0
            }
    
    async def reset_devices(self):
        """Reset device availability (for debugging)"""
        async with self.device_lock:
            self.available_devices = list(range(self.num_gpus))
            return {
                "status": "device pool reset",
                "available_devices": self.available_devices.copy()
            }


class TimeoutException(Exception):
    """Custom exception for timeout scenarios"""
    pass


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


# Initialize components
CUDAConfig.setup_cuda_arch()
gpu_manager = GPUDeviceManager()

app = FastAPI(
    title="KernelBench Evaluation Server",
    description=f"A server to evaluate custom CUDA and Triton kernels with {gpu_manager.num_gpus} GPU(s) support.",
)


# Dependencies
def check_cuda_available():
    """Dependency to check CUDA availability"""
    if not torch.cuda.is_available():
        raise HTTPException(status_code=503, detail="CUDA is not available on the server")
    if gpu_manager.num_gpus == 0:
        raise HTTPException(status_code=503, detail="No CUDA devices detected on the server")
    return True


def run_evaluation_with_timeout(request: EvalRequest, device: int, timeout: int = 300):
    """Run evaluation with timeout protection"""
    result_container: List[Optional[KernelExecResult]] = [None]
    exception_container: List[Optional[Exception]] = [None]
    
    def evaluation_worker():
        """Worker function that runs the actual evaluation"""
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
            result_container[0] = result
        except Exception as e:
            exception_container[0] = e
    
    # Start evaluation in a separate thread
    eval_thread = threading.Thread(target=evaluation_worker, name="EvalWorker", daemon=True)
    eval_thread.start()
    
    # Wait for completion with timeout
    eval_thread.join(timeout=timeout)
    
    if eval_thread.is_alive():
        print(f"[Server] Evaluation timed out after {timeout} seconds")
        gpu_manager.cleanup_device(device)
        eval_thread.join(timeout=5)  # Wait a bit more for cleanup
        
        if eval_thread.is_alive():
            print("[Server] Warning: Evaluation thread still running after timeout")
        
        raise TimeoutException(f"Evaluation timed out after {timeout} seconds")
    
    if exception_container[0] is not None:
        raise exception_container[0]
    
    return result_container[0]


@app.post("/eval", response_model=KernelExecResult)
async def evaluate_kernel(request: EvalRequest, _: bool = Depends(check_cuda_available)):
    """Evaluate a kernel with automatic GPU resource management"""
    
    async with gpu_manager.get_device(request.preferred_device) as device_id:
        print(f"[Server] Starting evaluation on GPU device {device_id} with {request.backend} backend")
        
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                run_evaluation_with_timeout,
                request,
                device_id,
                300  # 5 minute timeout
            )
            
            if result is None:
                raise HTTPException(
                    status_code=503,
                    detail=f"Compilation lock error occurred on GPU device {device_id} with {request.backend} backend. Please retry."
                )
            
            print(f"[Server] Completed evaluation on GPU device {device_id} with {request.backend} backend")
            return result
            
        except TimeoutException:
            raise HTTPException(
                status_code=504,
                detail=f"Evaluation timed out (300 seconds) on GPU device {device_id} with {request.backend} backend"
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            
            # Handle specific error types
            if "Unknown CUDA arch" in str(e) or "GPU not supported" in str(e):
                error_msg = f"CUDA compilation error - unsupported GPU architecture: {str(e)}"
            elif "sync_stream" in str(e) or "has no member" in str(e):
                error_msg = f"PyTorch API compatibility error: {str(e)}"
            else:
                error_msg = f"Unexpected error during evaluation: {str(e)}"
            
            raise HTTPException(status_code=500, detail=error_msg)


@app.get("/status")
async def get_status(detailed: bool = False):
    """Get comprehensive server status information"""
    gpu_status = await gpu_manager.get_status(detailed=detailed)
    
    base_status = {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "supported_backends": ["cuda", "triton"],
        "backend_descriptions": {
            "cuda": "Custom CUDA kernels compiled with PyTorch's C++ extension system",
            "triton": "Custom Triton kernels using OpenAI's Triton compiler"
        },
        "cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST", "not set"),
        "gpu_info": gpu_status
    }
    
    return base_status


@app.post("/cleanup")
async def manual_cleanup():
    """Manual cleanup endpoint for debugging"""
    try:
        for device_id in range(gpu_manager.num_gpus):
            gpu_manager.cleanup_device(device_id)
        return {"status": "cleanup completed for all devices"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


@app.post("/reset_devices")
async def reset_devices():
    """Reset device availability (for debugging)"""
    return await gpu_manager.reset_devices()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=18188)