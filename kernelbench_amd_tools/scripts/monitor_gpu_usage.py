#!/usr/bin/env python3
"""
Monitor GPU usage across all AMD GPUs
Shows which GPUs are in use and their memory usage
"""

import subprocess
import time
import os

def get_gpu_usage():
    """Get GPU usage information using rocm-smi"""
    try:
        # Get GPU utilization
        result = subprocess.run(
            ['rocm-smi', '--showuse'],
            capture_output=True,
            text=True
        )
        util_output = result.stdout
        
        # Get memory usage
        result = subprocess.run(
            ['rocm-smi', '--showmeminfo', 'vram'],
            capture_output=True,
            text=True
        )
        mem_output = result.stdout
        
        return util_output, mem_output
    except Exception as e:
        return None, None

def parse_gpu_info(util_output, mem_output):
    """Parse GPU information from rocm-smi output"""
    gpu_info = {}
    
    # Simple parsing - this may need adjustment based on exact rocm-smi output format
    if util_output:
        lines = util_output.strip().split('\n')
        for line in lines:
            if 'GPU[' in line and '%' in line:
                # Extract GPU ID and usage
                parts = line.split()
                for part in parts:
                    if 'GPU[' in part:
                        gpu_id = part.strip('GPU[]')
                        # Find usage percentage
                        for p in parts:
                            if '%' in p:
                                usage = p.strip('%')
                                gpu_info[gpu_id] = {'usage': usage}
                                break
    
    return gpu_info

def monitor_gpus(interval=5):
    """Monitor GPU usage continuously"""
    print("AMD GPU Usage Monitor")
    print("Press Ctrl+C to stop")
    print("=" * 80)
    print()
    print("Note: GPU 0 is hosting SGLang server, evaluations use GPU 1")
    print()
    
    try:
        while True:
            # Clear screen
            os.system('clear' if os.name == 'posix' else 'cls')
            
            print("AMD GPU Usage Monitor")
            print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 80)
            print()
            
            util_output, mem_output = get_gpu_usage()
            
            if util_output:
                print("GPU Utilization:")
                print(util_output)
            
            if mem_output:
                print("\nMemory Usage:")
                print(mem_output)
            
            print("\n" + "=" * 80)
            print("GPU 0: SGLang Server (facebook/KernelLLM)")
            print("GPU 1: KernelBench Evaluation")
            print("GPU 2-7: Available")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

def main():
    import sys
    
    interval = 5  # Default 5 seconds
    if len(sys.argv) > 1:
        try:
            interval = int(sys.argv[1])
        except:
            pass
    
    monitor_gpus(interval)

if __name__ == "__main__":
    main()