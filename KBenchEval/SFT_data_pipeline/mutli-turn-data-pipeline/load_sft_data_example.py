#!/usr/bin/env python3
"""
Example script showing how to load and use the generated multi-turn SFT data
"""

import json
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path


def load_jsonl_data(file_path):
    """Load multi-turn SFT data from JSONL file"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_parquet_data(file_path):
    """Load multi-turn SFT data from Parquet file"""
    df = pd.read_parquet(file_path)
    
    # Parse JSON strings back to objects
    df['messages'] = df['messages'].apply(json.loads)
    df['turn_history'] = df['turn_history'].apply(json.loads)
    
    return df


def analyze_sft_data(data):
    """Analyze the SFT data and print statistics"""
    if isinstance(data, pd.DataFrame):
        # Parquet data
        print(f"Total conversations: {len(data)}")
        print(f"Average turns: {data['num_turns'].mean():.2f}")
        print(f"Compilation rate: {data['final_compiled'].mean():.1%}")
        print(f"Correctness rate: {data['final_correct'].mean():.1%}")
        
        # Show distribution of turns
        print("\nTurn distribution:")
        print(data['num_turns'].value_counts().sort_index())
        
    else:
        # JSONL data
        print(f"Total conversations: {len(data)}")
        
        # Calculate statistics from messages
        turn_counts = []
        for item in data:
            messages = item['messages']
            # Count assistant messages as turns
            turns = sum(1 for msg in messages if msg['role'] == 'assistant')
            turn_counts.append(turns)
        
        if turn_counts:
            print(f"Average turns: {sum(turn_counts)/len(turn_counts):.2f}")
            print(f"Max turns: {max(turn_counts)}")
            print(f"Min turns: {min(turn_counts)}")


def format_conversation_for_training(conversation):
    """Format a conversation for training"""
    messages = conversation['messages'] if isinstance(conversation, dict) else json.loads(conversation['messages'])
    
    formatted = []
    for msg in messages:
        role = msg['role']
        content = msg['content']
        
        # Truncate long content for display
        if len(content) > 200:
            content = content[:200] + "..."
        
        formatted.append(f"[{role.upper()}]: {content}")
    
    return "\n".join(formatted)


def main():
    # Example paths - update these to your actual output paths
    base_dir = Path("/root/KernelBench/multi_turn_sft_outputs")
    
    # Find the most recent run
    run_dirs = sorted(base_dir.glob("run_*/run_*"))
    
    if not run_dirs:
        print("No output directories found. Please run generate_multiturn_sft.sh first.")
        return
    
    latest_dir = run_dirs[-1]
    print(f"Loading data from: {latest_dir}")
    print("=" * 50)
    
    # Load JSONL data
    jsonl_path = latest_dir / "multi-turn-sft.jsonl"
    if jsonl_path.exists():
        print("\nLoading JSONL data...")
        jsonl_data = load_jsonl_data(jsonl_path)
        print(f"Loaded {len(jsonl_data)} conversations from JSONL")
        
        # Analyze JSONL data
        print("\nJSONL Data Analysis:")
        analyze_sft_data(jsonl_data)
        
        # Show first conversation
        if jsonl_data:
            print("\nFirst conversation preview:")
            print("-" * 40)
            print(format_conversation_for_training(jsonl_data[0]))
            print("-" * 40)
    
    # Load Parquet data if available
    parquet_path = latest_dir / "multi-turn-sft.parquet"
    if parquet_path.exists():
        print("\nLoading Parquet data...")
        parquet_data = load_parquet_data(parquet_path)
        print(f"Loaded {len(parquet_data)} conversations from Parquet")
        
        # Analyze Parquet data
        print("\nParquet Data Analysis:")
        analyze_sft_data(parquet_data)
        
        # Show data types and memory usage
        print("\nDataFrame info:")
        print(f"Memory usage: {parquet_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Show first row
        if len(parquet_data) > 0:
            first_row = parquet_data.iloc[0]
            print(f"\nFirst conversation details:")
            print(f"  Instance ID: {first_row['instance_id']}")
            print(f"  Turns: {first_row['num_turns']}")
            print(f"  Final compiled: {first_row['final_compiled']}")
            print(f"  Final correct: {first_row['final_correct']}")
    
    # Load detailed conversation data
    detailed_path = latest_dir / "test_output_conversation.jsonl"
    if detailed_path.exists():
        print("\nDetailed conversation data available at:")
        print(f"  {detailed_path}")
        
        # Count lines
        with open(detailed_path) as f:
            num_detailed = sum(1 for _ in f)
        print(f"  Contains {num_detailed} detailed conversations with metadata")


if __name__ == "__main__":
    main()