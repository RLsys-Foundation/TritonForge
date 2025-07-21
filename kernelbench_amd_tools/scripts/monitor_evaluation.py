#!/usr/bin/env python3
"""
Real-time monitoring script for the batch evaluation
Shows progress, success rates, and recent errors
"""

import json
import time
import os
from datetime import datetime
from pathlib import Path

def load_json_file(filepath):
    """Safely load a JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except:
        return None

def monitor_progress(run_dir):
    """Monitor evaluation progress in real-time"""
    progress_file = f"{run_dir}/progress.json"
    results_file = f"{run_dir}/results.json"
    
    if not os.path.exists(progress_file):
        print(f"Progress file not found: {progress_file}")
        return
    
    print(f"Monitoring evaluation in: {run_dir}")
    print("Press Ctrl+C to stop monitoring\n")
    
    last_update = None
    
    try:
        while True:
            # Load current progress
            progress = load_json_file(progress_file)
            results = load_json_file(results_file)
            
            if progress:
                # Clear screen (works on most terminals)
                print("\033[2J\033[H")
                
                print("=" * 80)
                print(f"KernelBench AMD MI300X Evaluation Monitor")
                print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("=" * 80)
                
                # Basic stats
                completed = len(progress.get("completed", []))
                failed = len(progress.get("failed", []))
                total = completed + failed
                
                print(f"\nProgress Overview:")
                print(f"  Completed: {completed}")
                print(f"  Failed: {failed}")
                print(f"  Total Attempted: {total}")
                
                if total > 0:
                    print(f"  Success Rate: {completed/total*100:.1f}%")
                
                # Current task
                current = progress.get("current")
                if current:
                    print(f"\n  Currently Processing: {current}")
                
                # Level breakdown
                if results and "levels" in results:
                    print(f"\nLevel Breakdown:")
                    for level in sorted(results["levels"].keys()):
                        level_data = results["levels"][level]
                        level_total = len(level_data)
                        level_compiled = sum(1 for r in level_data.values() if r.get("compiled"))
                        level_correct = sum(1 for r in level_data.values() if r.get("correctness"))
                        
                        print(f"  {level}: {level_total} attempted, {level_compiled} compiled, {level_correct} correct")
                        if level_total > 0:
                            print(f"         Success rate: {level_correct/level_total*100:.1f}%")
                
                # Recent failures
                if progress.get("failed"):
                    print(f"\nRecent Failures (last 5):")
                    for failure in progress["failed"][-5:]:
                        problem = failure.get("problem", "Unknown")
                        error = failure.get("error", "Unknown error")
                        error_short = error[:60] + "..." if len(error) > 60 else error
                        print(f"  - {problem}: {error_short}")
                
                # Performance stats
                if results and "levels" in results:
                    successful_times = []
                    for level_data in results["levels"].values():
                        for result in level_data.values():
                            if result.get("correctness") and result.get("runtime") > 0:
                                successful_times.append(result["runtime"])
                    
                    if successful_times:
                        print(f"\nPerformance Stats (successful kernels):")
                        print(f"  Average runtime: {sum(successful_times)/len(successful_times):.3f}ms")
                        print(f"  Min runtime: {min(successful_times):.3f}ms")
                        print(f"  Max runtime: {max(successful_times):.3f}ms")
                
                print("\n" + "=" * 80)
            
            time.sleep(5)  # Update every 5 seconds
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

def find_latest_run():
    """Find the most recent evaluation run"""
    runs_dir = "/workspace/KernelBench/runs"
    if not os.path.exists(runs_dir):
        return None
    
    # Find directories starting with "amd_mi300x_full_eval_"
    run_dirs = [d for d in os.listdir(runs_dir) if d.startswith("amd_mi300x_full_eval_")]
    if not run_dirs:
        return None
    
    # Sort by timestamp (in the directory name)
    run_dirs.sort()
    return os.path.join(runs_dir, run_dirs[-1])

def main():
    import sys
    
    if len(sys.argv) > 1:
        run_dir = sys.argv[1]
    else:
        # Try to find the latest run
        run_dir = find_latest_run()
        if not run_dir:
            print("No evaluation runs found. Please specify a run directory.")
            return
    
    monitor_progress(run_dir)

if __name__ == "__main__":
    main()