#!/usr/bin/env python3
"""
Merge KernelBench Triton data from levels 1-4 into a single comprehensive dataset.
This ensures balanced data distribution across all difficulty levels.
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Any
import random
import argparse

def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load a JSONL file and return list of entries"""
    entries = []
    with open(filepath, 'r') as f:
        for line in f:
            entries.append(json.loads(line.strip()))
    return entries


def save_jsonl(data: List[Dict[str, Any]], filepath: str):
    """Save data to a JSONL file"""
    with open(filepath, 'w') as f:
        for entry in data:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')


def analyze_data(data: List[Dict[str, Any]], level: int) -> Dict[str, Any]:
    """Analyze data from a specific level"""
    problem_names = set()
    instance_ids = set()
    
    for entry in data:
        instance_ids.add(entry.get('instance_id', ''))
        extra_info = entry.get('extra_info', {})
        problem_names.add(extra_info.get('problem_name', ''))
    
    return {
        'level': level,
        'total_entries': len(data),
        'unique_problems': len(problem_names),
        'instance_ids': sorted(list(instance_ids)),
        'problem_names': sorted(list(problem_names))
    }


def merge_triton_levels(shuffle: bool = True, seed: int = 42) -> List[Dict[str, Any]]:
    """Merge all Triton levels into a single dataset"""
    all_data = []
    stats = {}
    
    print("Merging KernelBench Triton data from levels 1-4...")
    print("="*60)
    
    for level in range(1, 5):
        filepath = f"kernel_bench_triton_level_{level}.jsonl"
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found, skipping level {level}")
            continue
        
        data = load_jsonl(filepath)
        level_stats = analyze_data(data, level)
        stats[f'level_{level}'] = level_stats
        
        # Add all data
        all_data.extend(data)
        
        print(f"Level {level}: {level_stats['total_entries']} entries, "
              f"{level_stats['unique_problems']} unique problems")
    
    # Shuffle if requested (important for training)
    if shuffle:
        random.seed(seed)
        random.shuffle(all_data)
    
    return all_data, stats


def verify_merged_data(merged_data: List[Dict[str, Any]]) -> bool:
    """Verify the integrity of merged data"""
    print("\nVerifying merged data integrity...")
    
    # Check for required fields
    required_fields = ['prompt', 'label', 'instance_id', 'extra_info']
    missing_fields = defaultdict(int)
    
    # Count by level
    level_counts = defaultdict(int)
    problem_counts = defaultdict(int)
    
    for i, entry in enumerate(merged_data):
        # Check required fields
        for field in required_fields:
            if field not in entry:
                missing_fields[field] += 1
        
        # Count by level
        extra_info = entry.get('extra_info', {})
        level = extra_info.get('level', 0)
        level_counts[level] += 1
        
        # Count by problem
        problem_name = extra_info.get('problem_name', 'unknown')
        problem_counts[problem_name] += 1
        
        # Verify prompt structure
        prompt = entry.get('prompt', [])
        if not isinstance(prompt, list) or len(prompt) < 2:
            print(f"Warning: Entry {i} has invalid prompt structure")
    
    # Report findings
    print(f"Total entries: {len(merged_data)}")
    print(f"\nDistribution by level:")
    for level in sorted(level_counts.keys()):
        count = level_counts[level]
        percentage = (count / len(merged_data)) * 100
        print(f"  Level {level}: {count} entries ({percentage:.1f}%)")
    
    if missing_fields:
        print(f"\nWarning: Found entries with missing fields:")
        for field, count in missing_fields.items():
            print(f"  {field}: {count} entries missing")
        return False
    
    print("\n✓ All entries have required fields")
    print(f"✓ Found {len(problem_counts)} unique problems across all levels")
    
    return True


def create_data_distribution_report(stats: Dict[str, Any], merged_data: List[Dict[str, Any]]):
    """Create a detailed report of the data distribution"""
    report_path = "kernel_bench_triton_all_levels_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("KernelBench Triton All Levels Data Distribution Report\n")
        f.write("="*60 + "\n\n")
        
        # Summary
        f.write(f"Total merged entries: {len(merged_data)}\n")
        f.write(f"Data file: kernel_bench_triton_all_levels.jsonl\n\n")
        
        # Level distribution
        f.write("Distribution by Level:\n")
        level_counts = defaultdict(int)
        for entry in merged_data:
            level = entry.get('extra_info', {}).get('level', 0)
            level_counts[level] += 1
        
        for level in sorted(level_counts.keys()):
            count = level_counts[level]
            percentage = (count / len(merged_data)) * 100
            f.write(f"  Level {level}: {count} entries ({percentage:.1f}%)\n")
        
        # Original stats
        f.write("\nOriginal File Statistics:\n")
        for level_key, level_stats in sorted(stats.items()):
            f.write(f"\n{level_key}:\n")
            f.write(f"  Total entries: {level_stats['total_entries']}\n")
            f.write(f"  Unique problems: {level_stats['unique_problems']}\n")
        
        # Problem distribution
        f.write("\nProblem Distribution:\n")
        problem_by_level = defaultdict(lambda: defaultdict(int))
        for entry in merged_data:
            extra_info = entry.get('extra_info', {})
            level = extra_info.get('level', 0)
            problem_name = extra_info.get('problem_name', 'unknown')
            problem_by_level[level][problem_name] += 1
        
        for level in sorted(problem_by_level.keys()):
            f.write(f"\nLevel {level} ({len(problem_by_level[level])} unique problems):\n")
            for problem, count in sorted(problem_by_level[level].items())[:5]:  # Show top 5
                f.write(f"  - {problem}: {count} entries\n")
            if len(problem_by_level[level]) > 5:
                f.write(f"  ... and {len(problem_by_level[level]) - 5} more problems\n")
    
    print(f"\n📊 Data distribution report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Merge KernelBench Triton levels 1-4')
    parser.add_argument('--no-shuffle', action='store_true', help='Do not shuffle the merged data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for shuffling')
    parser.add_argument('--output', type=str, default='kernel_bench_triton_all_levels.jsonl',
                        help='Output filename for merged data')
    
    args = parser.parse_args()
    
    # Change to data directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Merge data
    merged_data, stats = merge_triton_levels(shuffle=not args.no_shuffle, seed=args.seed)
    
    # Verify integrity
    if not verify_merged_data(merged_data):
        print("⚠️  Warning: Data integrity check found issues!")
    
    # Save merged data
    save_jsonl(merged_data, args.output)
    print(f"\n✅ Merged data saved to: {args.output}")
    print(f"   Total entries: {len(merged_data)}")
    
    # Create distribution report
    create_data_distribution_report(stats, merged_data)
    
    # Print usage instructions
    print("\n📝 To use the merged data in training, update your script:")
    print(f"   export PROMPT_DATA=/workspace/slime/data/kernel_bench/{args.output}")

if __name__ == "__main__":
    main()