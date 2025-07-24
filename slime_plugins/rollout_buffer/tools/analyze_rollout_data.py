#!/usr/bin/env python3
"""
Analyze rollout data files to view execution details, performance metrics, and debugging information
"""

import argparse
import glob
import json
import os
from datetime import datetime
from typing import Any, Dict, List

from tabulate import tabulate


def load_rollout_data(filepath: str) -> List[Dict[str, Any]]:
    """Load rollout data from JSON file"""
    with open(filepath, "r") as f:
        return json.load(f)


def analyze_single_file(filepath: str, verbose: bool = False):
    """Analyze a single rollout data file"""
    print(f"\n{'='*80}")
    print(f"Analyzing: {filepath}")
    print(f"Modified: {datetime.fromtimestamp(os.path.getmtime(filepath))}")

    data = load_rollout_data(filepath)
    print(f"Total entries: {len(data)}")

    # Collect statistics
    stats = {
        "total": len(data),
        "with_exec_details": 0,
        "compiled": 0,
        "correct": 0,
        "with_speedup": 0,
        "rewards": [],
        "speedups": [],
        "runtimes": [],
        "baseline_runtimes": [],
        "by_level": {},
        "by_problem": {},
    }

    for entry in data:
        reward = entry.get("reward", 0.0)
        stats["rewards"].append(reward)

        # Extract problem info
        extra_info = entry.get("extra_info", {})
        level = extra_info.get("level", "unknown")
        problem_id = extra_info.get("problem_id", "unknown")
        problem_name = extra_info.get("problem_name", "unknown")
        problem_key = f"L{level}_P{problem_id}_{problem_name}"

        if level not in stats["by_level"]:
            stats["by_level"][level] = {"total": 0, "compiled": 0, "correct": 0, "speedups": []}
        stats["by_level"][level]["total"] += 1

        if problem_key not in stats["by_problem"]:
            stats["by_problem"][problem_key] = {"total": 0, "compiled": 0, "correct": 0, "speedups": []}
        stats["by_problem"][problem_key]["total"] += 1

        # Analyze execution details
        if "execution_details" in entry:
            stats["with_exec_details"] += 1
            exec_details = entry["execution_details"]

            if exec_details.get("compiled", False):
                stats["compiled"] += 1
                stats["by_level"][level]["compiled"] += 1
                stats["by_problem"][problem_key]["compiled"] += 1

            if exec_details.get("correctness", False):
                stats["correct"] += 1
                stats["by_level"][level]["correct"] += 1
                stats["by_problem"][problem_key]["correct"] += 1

            if "speedup" in exec_details and exec_details["speedup"] > 0:
                stats["with_speedup"] += 1
                speedup = exec_details["speedup"]
                stats["speedups"].append(speedup)
                stats["by_level"][level]["speedups"].append(speedup)
                stats["by_problem"][problem_key]["speedups"].append(speedup)

            if exec_details.get("runtime", -1) > 0:
                stats["runtimes"].append(exec_details["runtime"])

            if exec_details.get("baseline_runtime", -1) > 0:
                stats["baseline_runtimes"].append(exec_details["baseline_runtime"])

    # Print summary statistics
    print(f"\n{'='*40} SUMMARY {'='*40}")

    # Overall stats
    print("\nOverall Statistics:")
    print(
        f"  Entries with execution details: {stats['with_exec_details']}/{stats['total']} ({stats['with_exec_details']/stats['total']*100:.1f}%)"
    )
    print(
        f"  Compiled successfully: {stats['compiled']}/{stats['total']} ({stats['compiled']/stats['total']*100:.1f}%)"
    )
    print(f"  Passed correctness: {stats['correct']}/{stats['total']} ({stats['correct']/stats['total']*100:.1f}%)")
    print(
        f"  Has performance data: {stats['with_speedup']}/{stats['total']} ({stats['with_speedup']/stats['total']*100:.1f}%)"
    )

    # Reward stats
    if stats["rewards"]:
        avg_reward = sum(stats["rewards"]) / len(stats["rewards"])
        max_reward = max(stats["rewards"])
        min_reward = min(stats["rewards"])
        print(f"\nReward Statistics:")
        print(f"  Average: {avg_reward:.3f}")
        print(f"  Min: {min_reward:.3f}, Max: {max_reward:.3f}")

    # Performance stats
    if stats["speedups"]:
        avg_speedup = sum(stats["speedups"]) / len(stats["speedups"])
        max_speedup = max(stats["speedups"])
        min_speedup = min(stats["speedups"])
        print(f"\nPerformance Statistics (n={len(stats['speedups'])}):")
        print(f"  Average speedup: {avg_speedup:.2f}x")
        print(f"  Min speedup: {min_speedup:.2f}x, Max speedup: {max_speedup:.2f}x")

        # Speedup distribution
        speedup_ranges = [(0, 1), (1, 1.5), (1.5, 2), (2, 3), (3, float("inf"))]
        print("\n  Speedup Distribution:")
        for low, high in speedup_ranges:
            count = sum(1 for s in stats["speedups"] if low <= s < high)
            pct = count / len(stats["speedups"]) * 100
            if high == float("inf"):
                print(f"    {low:.1f}x+: {count} ({pct:.1f}%)")
            else:
                print(f"    {low:.1f}x-{high:.1f}x: {count} ({pct:.1f}%)")

    # Stats by level
    if stats["by_level"]:
        print(f"\nStatistics by Level:")
        level_data = []
        for level in sorted(stats["by_level"].keys()):
            level_stats = stats["by_level"][level]
            speedups = level_stats["speedups"]
            avg_speedup = sum(speedups) / len(speedups) if speedups else 0
            level_data.append(
                [
                    f"Level {level}",
                    level_stats["total"],
                    f"{level_stats['compiled']}/{level_stats['total']}",
                    f"{level_stats['correct']}/{level_stats['total']}",
                    f"{avg_speedup:.2f}x" if speedups else "N/A",
                ]
            )
        print(tabulate(level_data, headers=["Level", "Total", "Compiled", "Correct", "Avg Speedup"], tablefmt="grid"))

    # Top performing problems
    if stats["by_problem"]:
        print(f"\nTop 5 Problems by Average Speedup:")
        problem_speedups = []
        for problem, pstats in stats["by_problem"].items():
            if pstats["speedups"]:
                avg_speedup = sum(pstats["speedups"]) / len(pstats["speedups"])
                problem_speedups.append((problem, avg_speedup, len(pstats["speedups"])))

        problem_speedups.sort(key=lambda x: x[1], reverse=True)
        for i, (problem, avg_speedup, count) in enumerate(problem_speedups[:5]):
            print(f"  {i+1}. {problem}: {avg_speedup:.2f}x (n={count})")

    # Sample entries
    if verbose:
        print(f"\n{'='*40} SAMPLE ENTRIES {'='*40}")
        # Show best performing entry
        best_entry = None
        best_speedup = 0
        for entry in data:
            if "execution_details" in entry:
                speedup = entry["execution_details"].get("speedup", 0)
                if speedup > best_speedup:
                    best_speedup = speedup
                    best_entry = entry

        if best_entry:
            print(f"\nBest Performing Entry (Speedup: {best_speedup:.2f}x):")
            print_entry_details(best_entry)

        # Show a failed entry
        for entry in data:
            if "execution_details" in entry:
                exec_details = entry["execution_details"]
                if exec_details.get("compiled", False) and not exec_details.get("correctness", False):
                    print(f"\nExample Failed Entry (Compiled but Incorrect):")
                    print_entry_details(entry)
                    break


def print_entry_details(entry: Dict[str, Any]):
    """Print detailed information about a single entry"""
    print(f"  Instance ID: {entry.get('instance_id', 'N/A')}")
    print(f"  Reward: {entry.get('reward', 0.0):.3f}")

    if "extra_info" in entry:
        extra = entry["extra_info"]
        print(
            f"  Problem: Level {extra.get('level', '?')}, "
            f"ID {extra.get('problem_id', '?')}, "
            f"{extra.get('problem_name', 'unknown')}"
        )

    if "execution_details" in entry:
        exec_details = entry["execution_details"]
        print(f"  Execution Details:")
        print(f"    - Compiled: {exec_details.get('compiled', 'N/A')}")
        print(f"    - Correctness: {exec_details.get('correctness', 'N/A')}")
        print(f"    - Runtime: {exec_details.get('runtime', 'N/A')} ms")
        print(f"    - Baseline: {exec_details.get('baseline_runtime', 'N/A')} ms")
        print(f"    - Speedup: {exec_details.get('speedup', 'N/A')}x")
        print(f"    - Performance Reward: {exec_details.get('performance_reward', 'N/A')}")
        if "eval_response" in exec_details:
            print(f"    - Response: {exec_details['eval_response'][:100]}...")


def main():
    parser = argparse.ArgumentParser(description="Analyze rollout data files")
    parser.add_argument("files", nargs="*", help="Rollout data files to analyze (default: latest)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed sample entries")
    parser.add_argument("-a", "--all", action="store_true", help="Analyze all rollout files")
    parser.add_argument("-n", "--latest", type=int, default=1, help="Analyze N latest files")

    args = parser.parse_args()

    rollout_dir = "/workspace/slime/slime_plugins/rollout_buffer/rollout_data"

    if args.files:
        # Analyze specified files
        files_to_analyze = args.files
    elif args.all:
        # Analyze all files
        files_to_analyze = sorted(glob.glob(os.path.join(rollout_dir, "rollout_data_*.json")))
    else:
        # Analyze latest N files
        all_files = sorted(
            glob.glob(os.path.join(rollout_dir, "rollout_data_*.json")), key=os.path.getmtime, reverse=True
        )
        files_to_analyze = all_files[: args.latest]

    if not files_to_analyze:
        print("No rollout data files found!")
        return

    print(f"Analyzing {len(files_to_analyze)} file(s)...")

    for filepath in files_to_analyze:
        try:
            analyze_single_file(filepath, verbose=args.verbose)
        except Exception as e:
            print(f"Error analyzing {filepath}: {e}")

    print(f"\n{'='*80}")
    print("Analysis complete!")


if __name__ == "__main__":
    main()
