import os
import json
import re
import argparse

PROMPT_TEMPLATE = """\
You are a helpful assistant that can generate code to solve a problem.
You write custom Triton kernels to replace the pytorch operators in the given architecture to get speedups.

You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom Triton kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.

    
Here's an example to show you the syntax of inline embedding custom Triton kernels in torch: The example given architecture is: 

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return a + b

def get_inputs():
    # randomly generate input tensors based on the model architecture
    a = torch.randn(1, 128).cuda()
    b = torch.randn(1, 128).cuda()
    return [a, b]

def get_init_inputs():
    # randomly generate tensors required for initialization based on the model architecture
    return []
``` 

The example new arch with custom Triton kernels looks like this: 
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,  # Pointer to first input
    y_ptr,  # Pointer to second input
    out_ptr,  # Pointer to output
    n_elements,  # Total number of elements in input/output
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    # Perform the elementwise addition
    out = x + y
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_add(x: torch.Tensor, y: torch.Tensor):
    \"\"\"
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    \"\"\"
    assert x.is_cuda and y.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    y = y.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        # Instead of "return a + b", call our Triton-based addition
        return triton_add(a, b)

def get_inputs():
    # randomly generate input tensors based on the model architecture
    a = torch.randn(4096).cuda()
    b = torch.randn(4096).cuda()
    return [a, b]

def get_init_inputs():
    # randomly generate tensors required for initialization based on the model architecture
    return []
``` 

You are given the following architecture: 
```python
{content}
```

Optimize the architecture named Model with custom Triton kernels! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code!\
"""



def scan_kernel_bench_data_triton(base_dir=None, output_dir=None, levels=None):
    """
    Scans the KernelBench directory for Python files in level_1 to level_4,
    extracts extra_info, and generates Triton-specific training data in JSON lines format.

    Args:
        base_dir (str): Base directory containing KernelBench data. If None, uses default.
        output_dir (str): Output directory for generated JSONL files. If None, uses default.
        levels (list): List of level names to process. If None, processes all levels.

    The script iterates through specified level directories, and for each .py file,
    it extracts the level number from the directory path, the problem ID and name
    from the filename, and reads the file's content.

    The output is a series of JSON objects, one per file, written to JSONL files.
    Each JSON object has the following structure:
    {
        "prompt": [...],
        "label": "<file_content>",
        "extra_info": {
            "level": <level_number>,
            "problem_id": "<problem_id>",
            "problem_name": "<problem_name>",
            "data_source": "kernel_bench_triton",
            "task_type": "kernelbench"
        },
        "instance_id": "kernel_bench_triton_lv<level>_<problem_id>"
    }
    """
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(__file__), '..', 'KernelBench')
    
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    if levels is None:
        levels = ['level1', 'level2', 'level3', 'level4']

    for level_dir_name in levels:
        level_path = os.path.join(base_dir, level_dir_name)
        if not os.path.isdir(level_path):
            print(f"Warning: Directory {level_path} does not exist, skipping...")
            continue

        level_num_match = re.search(r'\d+', level_dir_name)
        if not level_num_match:
            print(f"Warning: Could not extract level number from {level_dir_name}, skipping...")
            continue
        level = int(level_num_match.group())

        records = []
        for filename in os.listdir(level_path):
            if filename.endswith('.py'):
                file_path = os.path.join(level_path, filename)
                extra_info = os.path.splitext(filename)[0]
                problem_id, problem_name = extra_info.split('_')[0], "_".join(extra_info.split('_')[1:])
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                record = {
                    'extra_info': {
                        "level": level,
                        "problem_id": problem_id,
                        "problem_name": problem_name,
                        "data_source": "kernel_bench_triton",
                        "task_type": "kernelbench"
                    },
                    'content': content
                }
                records.append(record)
        
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"kernel_bench_triton_level_{level}.jsonl")
        with open(output_file, 'w') as f:
            for record in records:
                _record = {
                    "prompt": [
                        {
                            "role": "system",
                            "content": "You are an expert in writing Triton kernels for efficient GPU programming."
                        },
                        {
                            "role": "user",
                            "content": PROMPT_TEMPLATE.format(content=record['content'])
                        }
                    ],
                    "label": record['content'],
                    "extra_info": record['extra_info'],
                    "instance_id": f"kernel_bench_triton_lv{level}_{record['extra_info']['problem_id']}",
                }
                f.write(json.dumps(_record, ensure_ascii=False) + '\n')
        
        print(f"Generated {output_file} with {len(records)} records")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Scan KernelBench data and convert to Triton training set format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default settings
  python scan_kernel_bench_data_to_train_set_triton.py
  
  # Specify custom directories
  python scan_kernel_bench_data_to_train_set_triton.py --base-dir /path/to/KernelBench --output-dir /path/to/output
  
  # Process only specific levels
  python scan_kernel_bench_data_to_train_set_triton.py --levels level1 level3
        """
    )
    
    parser.add_argument(
        '--base_dir', 
        type=str,
        default=None,
        help='Base directory containing KernelBench data (default: ../KernelBench relative to script location)'
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str,
        default=None,
        help='Output directory for generated JSONL files (default: ../data relative to script location)'
    )
    
    parser.add_argument(
        '--levels', 
        nargs='+',
        default=None,
        choices=['level1', 'level2', 'level3', 'level4'],
        help='Specific levels to process (default: all levels)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        print(f"Base directory: {args.base_dir}")
        print(f"Output directory: {args.output_dir}")
        print(f"Levels to process: {args.levels}")
    
    scan_kernel_bench_data_triton(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        levels=args.levels
    )


if __name__ == '__main__':
    main() 