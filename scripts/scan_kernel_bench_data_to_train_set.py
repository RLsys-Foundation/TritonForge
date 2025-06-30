import os
import json
import re

PROMPT_TEMPLATE = """\
You are a helpful assistant that can generate code to solve a problem.
You write custom CUDA kernels to replace the pytorch operators in the given architecture to get speedups. 

You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom CUDA kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.

    
Here's an example to show you the syntax of inline embedding custom CUDA operators in torch: The example given architecture is: 

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

The example new arch with custom CUDA kernels looks like this: 
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for element-wise addition
elementwise_add_source = \"\"\"
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int size) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {{
        out[idx] = a[idx] + b[idx];
    }}
}}

torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b) {{
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elementwise_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}}
\"\"\"

elementwise_add_cpp_source = "torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b);"

# Compile the inline CUDA code for element-wise addition
elementwise_add = load_inline(
    name='elementwise_add',
    cpp_sources=elementwise_add_cpp_source,
    cuda_sources=elementwise_add_source,
    functions=['elementwise_add_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.elementwise_add = elementwise_add

    def forward(self, a, b):
        return self.elementwise_add.elementwise_add_cuda(a, b)

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

Optimize the architecture named Model with custom CUDA operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code!\
"""



def scan_kernel_bench_data():
    """
    Scans the KernelBench directory for Python files in level_1 to level_4,
    extracts metadata, and prints the data in JSON lines format.

    The script iterates through specified level directories, and for each .py file,
    it extracts the level number from the directory path, the problem ID and name
    from the filename, and reads the file's content.

    The output is a series of JSON objects, one per file, written to stdout.
    Each JSON object has the following structure:
    {
        "level": <level_number>,
        "extra_info": "<problem_id>_<problem_name>",
        "content": "<file_content>"
    }
    """
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'KernelBench')
    levels = ['level1', 'level2', 'level3', 'level4']

    for level_dir_name in levels:
        level_path = os.path.join(base_dir, level_dir_name)
        if not os.path.isdir(level_path):
            continue

        level_num_match = re.search(r'\d+', level_dir_name)
        if not level_num_match:
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
                        "problem_name": problem_name
                    },
                    'content': content
                }
                records.append(record)
                #print(json.dumps(record))
        
        output_dir = "/workspace/data/"
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"kernel_bench_level_{level}.jsonl"), 'w') as f:
            for record in records:
                _record = {
                    "prompt": [
                        {
                            "role": "system",
                            "content": "You are an expert in writing CUDA Kernels for efficient GPU programming."
                        },
                        {
                            "role": "user",
                            "content": PROMPT_TEMPLATE.format(content=record['content'])
                        }
                    ],
                    "label": record['content'],
                    "extra_info": record['extra_info']
                }
                f.write(json.dumps(_record, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    scan_kernel_bench_data()
