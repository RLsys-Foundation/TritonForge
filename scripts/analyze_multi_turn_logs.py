#!/usr/bin/env python3
"""
Analyze multi-turn training logs to understand training progression and effects.

Usage:
    python analyze_multi_turn_logs.py [--log-dir /path/to/logs]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_json_file(filepath: str) -> dict:
    """Load a JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def analyze_trajectory(trajectory_file: str) -> dict:
    """Analyze a single trajectory file."""
    data = load_json_file(trajectory_file)

    analysis = {
        "instance_id": data.get("instance_id", "unknown"),
        "num_turns": data.get("num_turns", 0),
        "turn_rewards": data.get("turn_rewards", []),
        "aggregated_return": data.get("aggregated_return", 0),
        "final_correctness": data.get("execution_details", {}).get("final_correctness", False),
        "final_speedup": data.get("execution_details", {}).get("final_speedup", 0),
        "final_runtime": data.get("execution_details", {}).get("final_runtime", 0),
    }

    # Analyze improvement across turns
    if data.get("history"):
        history = data["history"]
        analysis["compilation_progress"] = [h["eval_result"]["compiled"] for h in history]
        analysis["correctness_progress"] = [h["eval_result"]["correctness"] for h in history]
        analysis["runtime_progress"] = [h["eval_result"]["runtime"] for h in history]
        analysis["speedup_progress"] = [h["eval_result"].get("speedup", 0) for h in history]

        # Check if there was improvement
        if len(history) > 1:
            first_reward = history[0]["reward"]
            last_reward = history[-1]["reward"]
            analysis["reward_improvement"] = last_reward - first_reward
            analysis["improved"] = last_reward > first_reward
        else:
            analysis["reward_improvement"] = 0
            analysis["improved"] = False

    return analysis


def analyze_turn_data(turn_file: str) -> dict:
    """Analyze a single turn file."""
    data = load_json_file(turn_file)
    turn_data = data.get("turn_data", {})

    return {
        "instance_id": data.get("instance_id", "unknown"),
        "turn_idx": data.get("turn_idx", 0),
        "compiled": turn_data.get("eval_result", {}).get("compiled", False),
        "correct": turn_data.get("eval_result", {}).get("correctness", False),
        "runtime": turn_data.get("eval_result", {}).get("runtime", 0),
        "speedup": turn_data.get("eval_result", {}).get("speedup", 0),
        "reward": turn_data.get("reward", 0),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze multi-turn training logs")
    parser.add_argument(
        "--log-dir", type=str, default="/workspace/slime/multi_turn_logs", help="Directory containing multi-turn logs"
    )
    parser.add_argument(
        "--output", type=str, default="multi_turn_analysis.json", help="Output file for analysis results"
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"Log directory {log_dir} does not exist!")
        return

    # Collect all log files
    trajectory_files = list(log_dir.glob("trajectory_*.json"))
    turn_files = list(log_dir.glob("turn_*.json"))

    print(f"Found {len(trajectory_files)} trajectory files and {len(turn_files)} turn files")

    # Analyze trajectories
    trajectory_analyses = []
    for traj_file in trajectory_files:
        try:
            analysis = analyze_trajectory(str(traj_file))
            trajectory_analyses.append(analysis)
        except Exception as e:
            print(f"Error analyzing {traj_file}: {e}")

    # Analyze turns
    turn_analyses = defaultdict(list)
    for turn_file in turn_files:
        try:
            analysis = analyze_turn_data(str(turn_file))
            turn_analyses[analysis["instance_id"]].append(analysis)
        except Exception as e:
            print(f"Error analyzing {turn_file}: {e}")

    # Compute statistics
    if trajectory_analyses:
        num_trajectories = len(trajectory_analyses)
        avg_turns = sum(t["num_turns"] for t in trajectory_analyses) / num_trajectories
        avg_return = sum(t["aggregated_return"] for t in trajectory_analyses) / num_trajectories

        # Success rates
        final_correct = sum(1 for t in trajectory_analyses if t["final_correctness"])
        improved_count = sum(1 for t in trajectory_analyses if t.get("improved", False))

        # Performance statistics
        speedups = [t["final_speedup"] for t in trajectory_analyses if t["final_speedup"] > 0]
        avg_speedup = sum(speedups) / len(speedups) if speedups else 0

        # Turn-by-turn statistics
        turn_rewards_by_idx = defaultdict(list)
        for t in trajectory_analyses:
            for idx, reward in enumerate(t["turn_rewards"]):
                turn_rewards_by_idx[idx].append(reward)

        avg_rewards_by_turn = {}
        for idx, rewards in turn_rewards_by_idx.items():
            avg_rewards_by_turn[f"turn_{idx}"] = sum(rewards) / len(rewards)

        # Print summary
        print("\n" + "=" * 60)
        print("MULTI-TURN TRAINING ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Total trajectories analyzed: {num_trajectories}")
        print(f"Average turns per trajectory: {avg_turns:.2f}")
        print(f"Average aggregated return: {avg_return:.4f}")
        print(
            f"Final correctness rate: {final_correct}/{num_trajectories} ({100*final_correct/num_trajectories:.1f}%)"
        )
        print(
            f"Trajectories that improved: {improved_count}/{num_trajectories} ({100*improved_count/num_trajectories:.1f}%)"
        )
        print(f"Average speedup (when correct): {avg_speedup:.2f}x")
        print("\nAverage reward by turn:")
        for turn, avg_reward in sorted(avg_rewards_by_turn.items()):
            print(f"  {turn}: {avg_reward:.4f}")

        # Detailed trajectory breakdown
        print("\n" + "=" * 60)
        print("TRAJECTORY DETAILS")
        print("=" * 60)
        for t in sorted(trajectory_analyses, key=lambda x: x["aggregated_return"], reverse=True)[:10]:
            print(f"\n{t['instance_id']}:")
            print(f"  Turns: {t['num_turns']}")
            print(f"  Turn rewards: {[f'{r:.3f}' for r in t['turn_rewards']]}")
            print(f"  Aggregated return: {t['aggregated_return']:.4f}")
            print(f"  Final correct: {t['final_correctness']}")
            print(f"  Final speedup: {t['final_speedup']:.2f}x")
            if "correctness_progress" in t:
                print(f"  Correctness progression: {t['correctness_progress']}")

        # Save detailed results
        results = {
            "summary": {
                "num_trajectories": num_trajectories,
                "avg_turns": avg_turns,
                "avg_return": avg_return,
                "final_correctness_rate": final_correct / num_trajectories,
                "improvement_rate": improved_count / num_trajectories,
                "avg_speedup": avg_speedup,
                "avg_rewards_by_turn": avg_rewards_by_turn,
            },
            "trajectories": trajectory_analyses,
            "turns": dict(turn_analyses),
        }

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed analysis saved to {args.output}")

    else:
        print("No trajectory files found to analyze!")


if __name__ == "__main__":
    main()
