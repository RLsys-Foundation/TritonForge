#!/usr/bin/env python3
"""
Comprehensive verification of merged KernelBench data integrity
"""

import json
import os
from collections import defaultdict
import hashlib

def load_jsonl(filepath):
    entries = []
    with open(filepath, 'r') as f:
        for line in f:
            entries.append(json.loads(line.strip()))
    return entries

def verify_data_structure(data, filename):
    """Verify that all entries have the correct structure"""
    print(f"\nVerifying {filename}...")
    print("="*60)
    
    errors = []
    warnings = []
    
    # Check each entry
    for i, entry in enumerate(data):
        # Required top-level fields
        required_fields = ['prompt', 'label', 'instance_id', 'extra_info']
        for field in required_fields:
            if field not in entry:
                errors.append(f"Entry {i}: Missing required field '{field}'")
        
        # Verify prompt structure
        if 'prompt' in entry:
            prompt = entry['prompt']
            if not isinstance(prompt, list):
                errors.append(f"Entry {i}: 'prompt' should be a list")
            elif len(prompt) < 2:
                errors.append(f"Entry {i}: 'prompt' should have at least 2 messages")
            else:
                # Check message structure
                for j, msg in enumerate(prompt):
                    if not isinstance(msg, dict):
                        errors.append(f"Entry {i}, message {j}: Should be a dict")
                    elif 'role' not in msg or 'content' not in msg:
                        errors.append(f"Entry {i}, message {j}: Missing 'role' or 'content'")
        
        # Verify extra_info structure
        if 'extra_info' in entry:
            extra_info = entry['extra_info']
            required_extra = ['level', 'problem_id', 'problem_name', 'data_source', 'task_type']
            for field in required_extra:
                if field not in extra_info:
                    warnings.append(f"Entry {i}: Missing '{field}' in extra_info")
        
        # Verify label is a string
        if 'label' in entry and not isinstance(entry['label'], str):
            errors.append(f"Entry {i}: 'label' should be a string")
    
    # Summary
    print(f"Total entries: {len(data)}")
    print(f"Errors found: {len(errors)}")
    print(f"Warnings found: {len(warnings)}")
    
    if errors:
        print("\nERRORS:")
        for error in errors[:10]:  # Show first 10
            print(f"  ❌ {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    
    if warnings:
        print("\nWARNINGS:")
        for warning in warnings[:10]:  # Show first 10
            print(f"  ⚠️  {warning}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more warnings")
    
    return len(errors) == 0

def compare_with_original(merged_data):
    """Compare merged data with original files to ensure no data loss"""
    print("\nComparing with original files...")
    print("="*60)
    
    # Load original data
    original_entries = {}
    original_count = 0
    
    for level in range(1, 5):
        filepath = f"kernel_bench_triton_level_{level}.jsonl"
        if os.path.exists(filepath):
            data = load_jsonl(filepath)
            for entry in data:
                instance_id = entry.get('instance_id', '')
                original_entries[instance_id] = entry
                original_count += 1
    
    # Check merged data
    merged_ids = set()
    for entry in merged_data:
        instance_id = entry.get('instance_id', '')
        merged_ids.add(instance_id)
    
    # Compare
    original_ids = set(original_entries.keys())
    missing_ids = original_ids - merged_ids
    extra_ids = merged_ids - original_ids
    
    print(f"Original total entries: {original_count}")
    print(f"Merged total entries: {len(merged_data)}")
    print(f"Unique instance IDs in original: {len(original_ids)}")
    print(f"Unique instance IDs in merged: {len(merged_ids)}")
    
    if missing_ids:
        print(f"\n❌ Missing {len(missing_ids)} entries from merged data:")
        for id in list(missing_ids)[:5]:
            print(f"  - {id}")
    
    if extra_ids:
        print(f"\n❌ Extra {len(extra_ids)} entries in merged data:")
        for id in list(extra_ids)[:5]:
            print(f"  - {id}")
    
    if not missing_ids and not extra_ids:
        print("\n✅ All entries accounted for!")
    
    # Verify content integrity by sampling
    print("\nVerifying content integrity (sampling)...")
    sample_size = min(10, len(merged_ids))
    sampled_ids = list(merged_ids)[:sample_size]
    
    content_matches = 0
    for instance_id in sampled_ids:
        # Find in merged
        merged_entry = None
        for entry in merged_data:
            if entry.get('instance_id') == instance_id:
                merged_entry = entry
                break
        
        if instance_id in original_entries and merged_entry:
            original = original_entries[instance_id]
            # Compare key fields
            if (original.get('label') == merged_entry.get('label') and
                original.get('prompt') == merged_entry.get('prompt')):
                content_matches += 1
    
    print(f"Content integrity check: {content_matches}/{sample_size} samples match")
    
    return len(missing_ids) == 0 and len(extra_ids) == 0

def analyze_distribution(data):
    """Analyze the distribution of problems across levels"""
    print("\nAnalyzing data distribution...")
    print("="*60)
    
    level_dist = defaultdict(int)
    problem_dist = defaultdict(int)
    level_problem_dist = defaultdict(lambda: defaultdict(int))
    
    for entry in data:
        extra_info = entry.get('extra_info', {})
        level = extra_info.get('level', 0)
        problem_name = extra_info.get('problem_name', 'unknown')
        
        level_dist[level] += 1
        problem_dist[problem_name] += 1
        level_problem_dist[level][problem_name] += 1
    
    # Print distribution
    print("Level distribution:")
    total = len(data)
    for level in sorted(level_dist.keys()):
        count = level_dist[level]
        pct = (count / total) * 100
        print(f"  Level {level}: {count} entries ({pct:.1f}%)")
    
    print(f"\nTotal unique problems: {len(problem_dist)}")
    
    # Problems per level
    print("\nUnique problems per level:")
    for level in sorted(level_problem_dist.keys()):
        unique_problems = len(level_problem_dist[level])
        print(f"  Level {level}: {unique_problems} unique problems")
    
    # Check for duplicates
    duplicates = [(name, count) for name, count in problem_dist.items() if count > 1]
    if duplicates:
        print(f"\n⚠️  Found {len(duplicates)} problems with multiple entries:")
        for name, count in duplicates[:5]:
            print(f"  - {name}: {count} entries")

def check_compatibility():
    """Check that the merged data is compatible with the training pipeline"""
    print("\nChecking training pipeline compatibility...")
    print("="*60)
    
    # Check file exists
    merged_file = "kernel_bench_triton_all_levels.jsonl"
    if not os.path.exists(merged_file):
        print("❌ Merged file not found!")
        return False
    
    # Check file size
    file_size = os.path.getsize(merged_file)
    file_size_mb = file_size / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")
    
    if file_size_mb > 100:
        print("⚠️  Warning: File size is large, may affect loading time")
    
    # Check that script has been updated
    script_path = "/workspace/slime/scripts/agent-example-kbench-kernelllm-8B.sh"
    if os.path.exists(script_path):
        with open(script_path, 'r') as f:
            content = f.read()
            if "kernel_bench_triton_all_levels.jsonl" in content:
                print("✅ Training script has been updated to use merged data")
            else:
                print("❌ Training script still references old data file!")
    
    return True

def main():
    print("KernelBench Merged Data Verification")
    print("="*80)
    
    # Load merged data
    merged_file = "kernel_bench_triton_all_levels.jsonl"
    if not os.path.exists(merged_file):
        print(f"Error: {merged_file} not found!")
        return
    
    merged_data = load_jsonl(merged_file)
    
    # Run all checks
    structure_ok = verify_data_structure(merged_data, merged_file)
    integrity_ok = compare_with_original(merged_data)
    analyze_distribution(merged_data)
    compatibility_ok = check_compatibility()
    
    # Final summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY:")
    print(f"  Data structure: {'✅ PASS' if structure_ok else '❌ FAIL'}")
    print(f"  Data integrity: {'✅ PASS' if integrity_ok else '❌ FAIL'}")
    print(f"  Compatibility: {'✅ PASS' if compatibility_ok else '❌ FAIL'}")
    
    if structure_ok and integrity_ok and compatibility_ok:
        print("\n🎉 All checks passed! The merged data is ready for use.")
    else:
        print("\n⚠️  Some issues found. Please review the errors above.")

if __name__ == "__main__":
    main()