#!/usr/bin/env python3
"""
Robust subprocess-isolated evaluation server for KernelBench
Enhanced with better memory fault handling and crash recovery
Each evaluation runs in a separate process with base64 encoding to prevent escaping issues
Supports both NVIDIA CUDA and AMD ROCm/HIP backends

Default timeout: 600 seconds (10 minutes) to account for:
- Process spawn overhead
- GPU context initialization per process
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
import signal
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import multiprocessing as mp
import pickle
import traceback
import resource

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# Disable core dumps at system level
try:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
except:
    pass  # May not have permission in some environments

# AMD MI300X support: Set environment variables
if 'HIP_VISIBLE_DEVICES' in os.environ or os.path.exists('/opt/rocm'):
    # AMD GPU environment setup
    os.environ['ROCM_HOME'] = os.environ.get('ROCM_HOME', '/opt/rocm')
    os.environ['HIP_PLATFORM'] = 'amd'
    os.environ['PYTORCH_ROCM_ARCH'] = os.environ.get('PYTORCH_ROCM_ARCH', 'gfx942')
    
    # Disable GPU core dumps for stability
    os.environ['HSA_ENABLE_COREDUMP'] = '0'
    os.environ['AMD_LOG_LEVEL'] = '0'
    os.environ['ROCM_DISABLE_CRASH_DUMP'] = '1'
    os.environ['HIP_ENABLE_COREDUMP'] = '0'
    
    # Import and set GPU architecture for AMD
    from src.utils import set_gpu_arch
    set_gpu_arch(["MI300X", "gfx942"])
    
    print(f"[Server] AMD GPU mode enabled - ROCm/HIP backend")
    print(f"[Server] PYTORCH_ROCM_ARCH={os.environ.get('PYTORCH_ROCM_ARCH')}")
    IS_AMD_GPU = True
else:
    IS_AMD_GPU = False
    # NVIDIA GPU setup (original code)
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
    title="KernelBench Robust Subprocess Isolation Server",
    description="Evaluation server with enhanced process isolation and memory fault handling (CUDA/ROCm)"
)

# Configuration - detect GPUs (works for both CUDA and ROCm)
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
gpu_type = "ROCm/HIP" if IS_AMD_GPU else "CUDA"
print(f"[Server] Detected {NUM_GPUS} {gpu_type} devices")

# GPU allocation tracking
available_devices = list(range(NUM_GPUS))
device_lock = asyncio.Lock()
gpu_semaphore = asyncio.Semaphore(NUM_GPUS) if NUM_GPUS > 0 else None

# Track memory fault counts per device
device_fault_counts = {i: 0 for i in range(NUM_GPUS)}
MAX_FAULTS_PER_DEVICE = 10  # Reset device after this many faults


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


def run_isolated_evaluation_script(ref_code_b64: str, triton_code_b64: str, 
                                  request_dict: Dict[str, Any], device_id: int, 
                                  timeout: int = 600) -> Dict[str, Any]:
    """
    Run evaluation in a completely isolated subprocess script
    Uses base64 encoding to avoid escaping issues
    """
    # Create a temporary script to run the evaluation
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        eval_script = f"""
import sys
import json
import signal
import os
import base64
import traceback
import resource

# Disable core dumps
try:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
except:
    pass

# Add paths
sys.path.insert(0, '{PROJECT_ROOT}')

# Set environment
if os.path.exists('/opt/rocm'):
    # AMD GPU settings
    os.environ["HIP_VISIBLE_DEVICES"] = "{device_id}"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{device_id}"
    os.environ['ROCM_HOME'] = '/opt/rocm'
    os.environ['HIP_PLATFORM'] = 'amd'
    os.environ['PYTORCH_ROCM_ARCH'] = 'gfx942'
    os.environ['HSA_ENABLE_COREDUMP'] = '0'
    os.environ['AMD_LOG_LEVEL'] = '0'
    os.environ['ROCM_DISABLE_CRASH_DUMP'] = '1'
    os.environ['HIP_ENABLE_COREDUMP'] = '0'
    os.environ["HIP_LAUNCH_BLOCKING"] = "1"
else:
    # NVIDIA GPU settings
    os.environ["CUDA_VISIBLE_DEVICES"] = "{device_id}"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Set Triton cache directory
os.environ["TRITON_CACHE_DIR"] = f"/tmp/triton_cache_gpu_{device_id}"

# Import and set GPU architecture for AMD
if os.path.exists('/opt/rocm'):
    from src.utils import set_gpu_arch
    set_gpu_arch(["MI300X", "gfx942"])

# Timeout handler
def timeout_handler(signum, frame):
    print(json.dumps({{
        "success": False,
        "error": "Evaluation timeout ({timeout}s)",
        "category": "timeout"
    }}))
    sys.exit(1)

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm({timeout})

try:
    from src.eval import eval_kernel_against_ref, KernelExecResult
    
    # Decode the base64 encoded code
    original_model_src = base64.b64decode('{ref_code_b64}').decode()
    custom_model_src = base64.b64decode('{triton_code_b64}').decode()
    
    # Run evaluation
    result = eval_kernel_against_ref(
        original_model_src=original_model_src,
        custom_model_src=custom_model_src,
        seed_num={request_dict.get('seed_num', 42)},
        num_correct_trials={request_dict.get('num_correct_trials', 5)},
        num_perf_trials={request_dict.get('num_perf_trials', 100)},
        verbose={request_dict.get('verbose', False)},
        measure_performance={request_dict.get('measure_performance', True)},
        device=0,  # Always device 0 since we set CUDA_VISIBLE_DEVICES
        backend="{request_dict.get('backend', 'cuda')}",
    )
    
    # Handle None result
    if result is None:
        print(json.dumps({{
            "success": False,
            "error": "Evaluation returned None (likely due to SyntaxError)",
            "category": "syntax_error"
        }}))
    else:
        # Convert result to dict
        result_dict = result.dict() if hasattr(result, 'dict') else result.__dict__
        print(json.dumps({{
            "success": True,
            "result": result_dict
        }}))
        
except Exception as e:
    error_str = str(e)
    
    # Categorize error
    if "out of resource: shared memory" in error_str:
        category = "shared_memory_exceeded"
    elif "illegal memory access" in error_str:
        category = "illegal_memory_access"
    elif "Memory access fault" in error_str:
        category = "memory_fault"
    elif "Segmentation fault" in error_str:
        category = "segmentation_fault"
    elif "Unknown CUDA arch" in error_str:
        category = "unsupported_architecture"
    elif "unsupported AST node type" in error_str:
        category = "triton_compilation_error"
    elif "Did you forget to add @triton.jit" in error_str:
        category = "triton_jit_error"
    else:
        category = "unknown"
    
    print(json.dumps({{
        "success": False,
        "error": error_str[:500],
        "category": category,
        "traceback": traceback.format_exc()[:2000]
    }}))
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
        if result.stdout:
            try:
                # Get the last JSON output (in case there are print statements)
                lines = result.stdout.strip().split('\n')
                for line in reversed(lines):
                    if line.strip().startswith('{'):
                        return json.loads(line)
            except json.JSONDecodeError:
                pass
        
        # Check return codes for specific errors
        if result.returncode == -signal.SIGSEGV:
            return {
                "success": False,
                "error": "Segmentation fault (memory access violation)",
                "category": "segmentation_fault"
            }
        elif result.returncode == -signal.SIGABRT:
            return {
                "success": False,
                "error": "Process aborted (likely GPU memory fault)",
                "category": "memory_fault"
            }
        elif result.returncode != 0:
            stderr_msg = result.stderr[:500] if result.stderr else "Unknown error"
            return {
                "success": False,
                "error": f"Process exited with code {result.returncode}: {stderr_msg}",
                "category": "process_error"
            }
        else:
            # Process succeeded but couldn't parse output
            return {
                "success": False,
                "error": f"Failed to parse result: {result.stdout[:200]}",
                "category": "parse_error"
            }
            
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"Evaluation timeout ({timeout}s)",
            "category": "timeout"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Subprocess error: {str(e)[:200]}",
            "category": "subprocess_error"
        }
    finally:
        # Clean up temp file
        try:
            os.unlink(script_path)
        except:
            pass


def _check_gpu_health(device_id: int) -> bool:
    """Check GPU health in isolated process"""
    import os
    import torch
    
    # Set device visibility based on platform
    if 'HIP_VISIBLE_DEVICES' in os.environ or os.path.exists('/opt/rocm'):
        os.environ["HIP_VISIBLE_DEVICES"] = str(device_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    
    try:
        device = torch.device("cuda:0")
        torch.cuda.synchronize(device)
        # Try to allocate a small tensor
        test_tensor = torch.zeros(100, device=device)
        del test_tensor
        torch.cuda.empty_cache()
        return True
    except:
        return False


def _get_gpu_info(device_id: int) -> Dict[str, Any]:
    """Get GPU info in isolated process"""
    import torch
    import os
    
    # Set device visibility based on platform
    if 'HIP_VISIBLE_DEVICES' in os.environ or os.path.exists('/opt/rocm'):
        os.environ["HIP_VISIBLE_DEVICES"] = str(device_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    else:
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


async def acquire_gpu_device(preferred_device: Optional[int] = None) -> int:
    """Acquire a GPU device for evaluation"""
    async with device_lock:
        if not available_devices:
            raise HTTPException(status_code=503, detail="No GPU devices available")
        
        # Try to get device with lowest fault count
        if preferred_device is not None and preferred_device in available_devices:
            device_id = preferred_device
        else:
            # Sort by fault count and pick the one with least faults
            available_sorted = sorted(available_devices, key=lambda x: device_fault_counts.get(x, 0))
            device_id = available_sorted[0]
        
        available_devices.remove(device_id)
        print(f"[Server] Assigned GPU device {device_id} (fault count: {device_fault_counts[device_id]})")
        return device_id


async def release_gpu_device(device_id: int, had_fault: bool = False):
    """Release a GPU device back to the pool"""
    async with device_lock:
        if had_fault:
            device_fault_counts[device_id] = device_fault_counts.get(device_id, 0) + 1
            print(f"[Server] GPU {device_id} had a fault (total: {device_fault_counts[device_id]})")
            
            # If too many faults, try to reset the device
            if device_fault_counts[device_id] >= MAX_FAULTS_PER_DEVICE:
                print(f"[Server] GPU {device_id} exceeded fault limit, attempting reset...")
                device_fault_counts[device_id] = 0
                # Note: Actual GPU reset would require nvidia-smi or rocm-smi commands
        
        if device_id not in available_devices:
            available_devices.append(device_id)
            available_devices.sort()
            print(f"[Server] Released GPU device {device_id}")


@app.post("/eval")
async def evaluate_kernel(request: EvalRequest):
    """
    Evaluate kernel in isolated subprocess to prevent CUDA/ROCm corruption
    Enhanced with better error handling and memory fault recovery
    """
    if not gpu_semaphore:
        raise HTTPException(status_code=503, detail="No GPUs available on this system")
    
    async with gpu_semaphore:
        device_id = await acquire_gpu_device(request.preferred_device)
        had_fault = False
        
        try:
            print(f"[Server] Starting isolated evaluation on GPU {device_id} with {request.backend} backend")
            if request.verbose:
                print(f"[Server] Kernel snippet: {request.custom_model_src[:200]}...")
            
            # Encode the code strings in base64 to avoid escaping issues
            ref_code_b64 = base64.b64encode(request.original_model_src.encode()).decode()
            triton_code_b64 = base64.b64encode(request.custom_model_src.encode()).decode()
            
            # Convert request to dict
            request_dict = request.model_dump() if hasattr(request, 'model_dump') else request.dict()
            
            # Run evaluation in completely isolated subprocess
            result_dict = run_isolated_evaluation_script(
                ref_code_b64, triton_code_b64, request_dict, device_id, timeout=600
            )
            
            if result_dict["success"]:
                # Convert dict back to KernelExecResult
                from src.eval import KernelExecResult
                return KernelExecResult(**result_dict["result"])
            else:
                # Handle error from subprocess
                error_category = result_dict.get("category", "unknown")
                error_msg = result_dict.get("error", "Unknown error")
                
                # Check if this was a memory fault
                if error_category in ["memory_fault", "segmentation_fault", "illegal_memory_access"]:
                    had_fault = True
                    error_msg = f"GPU memory fault on device {device_id}. The evaluation was isolated and the server remains stable. Error: {error_msg}"
                    print(f"[Server] Memory fault detected on GPU {device_id}: {error_category}")
                elif error_category == "shared_memory_exceeded":
                    error_msg = f"Triton kernel exceeded shared memory limit on GPU {device_id}."
                elif error_category == "timeout":
                    error_msg = f"Evaluation timed out on GPU {device_id} after 600 seconds."
                elif error_category == "triton_compilation_error":
                    error_msg = f"Triton compilation error: {error_msg}"
                elif error_category == "triton_jit_error":
                    error_msg = f"Triton JIT error: {error_msg}"
                elif error_category == "syntax_error":
                    error_msg = f"Syntax error in kernel code"
                
                # Log detailed error for debugging
                if request.verbose and "traceback" in result_dict:
                    print(f"[Server] Detailed error:\n{result_dict['traceback']}")
                
                # Return error with appropriate status code
                status_code = 500
                if error_category == "timeout":
                    status_code = 504
                elif error_category == "syntax_error":
                    status_code = 400
                
                raise HTTPException(status_code=status_code, detail=error_msg)
                
        except HTTPException:
            raise
        except Exception as e:
            had_fault = True  # Treat unexpected errors as potential faults
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error during evaluation: {str(e)}"
            )
        finally:
            await release_gpu_device(device_id, had_fault)


@app.get("/")
async def root():
    """Root endpoint providing service information"""
    gpu_platform = "ROCm/HIP" if IS_AMD_GPU else "CUDA"
    return {
        "service": "KernelBench Robust Subprocess Isolation Server",
        "status": "running",
        "gpu_platform": gpu_platform,
        "cuda_available": torch.cuda.is_available(),
        "num_gpus": NUM_GPUS,
        "backends": ["cuda", "triton"],
        "features": [
            "Enhanced process isolation for each evaluation",
            f"{gpu_platform} GPU support",
            "Memory fault detection and recovery",
            "GPU corruption prevention",
            "Automatic recovery from crashes",
            "Base64 encoding for code safety",
            "Fault tracking per GPU device",
            "AMD MI300X optimized" if IS_AMD_GPU else "NVIDIA GPU optimized"
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint with fault statistics"""
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
                    "status": "healthy" if is_healthy else "error",
                    "fault_count": device_fault_counts.get(i, 0)
                })
            except Exception as e:
                gpu_info.append({
                    "device_id": i,
                    "available": False,
                    "status": "error",
                    "error": str(e),
                    "fault_count": device_fault_counts.get(i, 0)
                })
    
    # Calculate busy devices
    busy_count = NUM_GPUS - len(available_devices)
    
    gpu_platform = "ROCm/HIP" if IS_AMD_GPU else "CUDA"
    arch_info = os.environ.get("PYTORCH_ROCM_ARCH", "not set") if IS_AMD_GPU else os.environ.get("TORCH_CUDA_ARCH_LIST", "not set")
    
    # Calculate total faults
    total_faults = sum(device_fault_counts.values())
    
    return {
        "status": "healthy",
        "gpu_platform": gpu_platform,
        "cuda_available": torch.cuda.is_available(),
        "supported_backends": ["cuda", "triton"],
        "total_gpu_devices": NUM_GPUS,
        "available_gpu_devices": len(available_devices),
        "busy_gpu_devices": busy_count,
        "available_device_ids": available_devices.copy(),
        "gpu_arch": arch_info,
        "devices": gpu_info,
        "isolation": "enhanced_subprocess",
        "total_memory_faults": total_faults,
        "max_faults_per_device": MAX_FAULTS_PER_DEVICE,
        "gpu_settings": {
            "rocm_home": os.environ.get("ROCM_HOME") if IS_AMD_GPU else None,
            "hip_platform": os.environ.get("HIP_PLATFORM") if IS_AMD_GPU else None,
            "cuda_dsa": "enabled" if not IS_AMD_GPU else None,
            "core_dumps": "disabled"
        }
    }


@app.post("/reset_gpu/{device_id}")
async def reset_gpu(device_id: int):
    """Reset fault counter for a specific GPU"""
    if device_id < 0 or device_id >= NUM_GPUS:
        raise HTTPException(status_code=400, detail=f"Invalid device ID: {device_id}")
    
    async with device_lock:
        old_count = device_fault_counts.get(device_id, 0)
        device_fault_counts[device_id] = 0
    
    return {
        "status": "fault counter reset",
        "device_id": device_id,
        "old_fault_count": old_count,
        "new_fault_count": 0
    }


@app.get("/gpu_status")
async def gpu_status():
    """Get detailed GPU status information"""
    if not torch.cuda.is_available():
        return {"error": "CUDA/ROCm not available"}
    
    async with device_lock:
        gpu_info = []
        for i in range(NUM_GPUS):
            try:
                # Get device info in subprocess to avoid corruption
                with mp.Pool(processes=1) as pool:
                    device_info = pool.apply(_get_gpu_info, (i,))
                    device_info["available"] = i in available_devices
                    device_info["fault_count"] = device_fault_counts.get(i, 0)
                    gpu_info.append(device_info)
                    
            except Exception as e:
                gpu_info.append({
                    "device_id": i,
                    "error": str(e),
                    "available": False,
                    "fault_count": device_fault_counts.get(i, 0)
                })
    
    return {
        "total_devices": NUM_GPUS,
        "devices": gpu_info,
        "semaphore_permits": gpu_semaphore._value if gpu_semaphore else 0,
        "total_faults": sum(device_fault_counts.values())
    }


@app.post("/cleanup")
async def manual_cleanup():
    """Reset all fault counters"""
    async with device_lock:
        global device_fault_counts
        old_counts = device_fault_counts.copy()
        device_fault_counts = {i: 0 for i in range(NUM_GPUS)}
    
    return {
        "status": "fault counters reset",
        "old_fault_counts": old_counts,
        "message": "All GPU fault counters have been reset"
    }


@app.get("/backend_info")
async def backend_info():
    """Get information about supported backends"""
    gpu_platform = "ROCm/HIP" if IS_AMD_GPU else "CUDA"
    return {
        "supported_backends": ["cuda", "triton"],
        "default_backend": "triton" if IS_AMD_GPU else "cuda",
        "gpu_platform": gpu_platform,
        "backend_descriptions": {
            "cuda": f"Custom {gpu_platform} kernels compiled with PyTorch's C++ extension system",
            "triton": "Custom Triton kernels using OpenAI's Triton compiler"
        },
        "cuda_available": torch.cuda.is_available(),
        "triton_available": True,
        "process_isolation": "enhanced",
        "memory_fault_handling": True,
        "gpu_architecture": os.environ.get("PYTORCH_ROCM_ARCH") if IS_AMD_GPU else os.environ.get("TORCH_CUDA_ARCH_LIST"),
        "cuda_dsa_enabled": not IS_AMD_GPU
    }


@app.post("/reset_devices")
async def reset_devices():
    """Reset device availability and fault counters"""
    async with device_lock:
        global available_devices, device_fault_counts
        available_devices = list(range(NUM_GPUS))
        device_fault_counts = {i: 0 for i in range(NUM_GPUS)}
        return {
            "status": "device pool and fault counters reset",
            "available_devices": available_devices.copy(),
            "fault_counts": device_fault_counts.copy()
        }


@app.get("/fault_statistics")
async def fault_statistics():
    """Get detailed fault statistics"""
    async with device_lock:
        stats = {
            "per_device_faults": device_fault_counts.copy(),
            "total_faults": sum(device_fault_counts.values()),
            "max_faults_threshold": MAX_FAULTS_PER_DEVICE,
            "devices_near_limit": [
                device_id for device_id, count in device_fault_counts.items() 
                if count >= MAX_FAULTS_PER_DEVICE * 0.8
            ]
        }
    return stats


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    # Check for GPUs
    if NUM_GPUS == 0:
        print("[Server] WARNING: No GPUs detected. Server will run but cannot evaluate kernels.")
    else:
        gpu_platform = "ROCm/HIP" if IS_AMD_GPU else "CUDA"
        gpu_arch = os.environ.get("PYTORCH_ROCM_ARCH") if IS_AMD_GPU else os.environ.get("TORCH_CUDA_ARCH_LIST", "auto")
        
        print(f"[Server] Starting ROBUST subprocess isolation server")
        print(f"[Server] GPU Platform: {gpu_platform}")
        print(f"[Server] Number of GPUs: {NUM_GPUS}")
        print(f"[Server] GPU Architecture: {gpu_arch}")
        print(f"[Server] Enhanced memory fault handling enabled")
        print(f"[Server] Core dumps disabled for stability")
        print(f"[Server] Each evaluation will run in a completely isolated process")
        
        if IS_AMD_GPU:
            print(f"[Server] AMD MI300X optimizations enabled")
            print(f"[Server] ROCm/HIP environment configured")
        else:
            print(f"[Server] CUDA DSA enabled for kernel debugging")
    
    # Start server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=18188,
        log_level="info",
        access_log=True
    )