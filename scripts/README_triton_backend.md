# Triton 后端实现说明

## 概述

本文档介绍了为 KernelBench 项目实现的 Triton 后端功能，包括数据格式转换脚本和内核生成器的扩展。

## 新增功能

### 1. Triton 数据格式转换脚本

**文件**: `KernelBench/scripts/scan_kernel_bench_data_to_train_set_triton.py`

基于现有的 CUDA 版本 (`scan_kernel_bench_data_to_train_set.py`)，创建了专门用于 Triton 后端的数据格式转换脚本。

**主要差异**:
- 系统消息: "You are an expert in writing Triton kernels for efficient GPU programming."
- 数据源标识: `kernel_bench_triton`
- 实例 ID 前缀: `kernel_bench_triton_lv{level}_{problem_id}`
- 示例代码: 使用 Triton 内核而非 CUDA 内核

**使用方法**:
```bash
# 处理所有级别
python scan_kernel_bench_data_to_train_set_triton.py

# 处理特定级别
python scan_kernel_bench_data_to_train_set_triton.py --levels level1 level3

# 指定自定义目录
python scan_kernel_bench_data_to_train_set_triton.py --base-dir /path/to/KernelBench --output-dir /path/to/output
```

### 2. Triton 内核生成器

**文件**: `slime/slime_plugins/rollout_buffer/generator/kernel_generator.py`

扩展了现有的内核生成器，增加了 Triton 后端支持。

**新增函数**:
- `generate_mock_triton_kernel()`: 生成 Triton 内核的模拟实现
- 修改 `kernel_rollout_func()`: 根据 backend 参数选择合适的内核生成器

**Triton 内核特点**:
- 使用 `@triton.jit` 装饰器
- 导入 `triton.language as tl`
- 使用 Triton 特有的 API (`tl.load`, `tl.store`, `tl.program_id` 等)
- 支持自动错误处理和回退机制

## 技术架构

### 数据流程

1. **数据准备**: 使用 `scan_kernel_bench_data_to_train_set_triton.py` 生成训练数据
2. **内核生成**: `KernelGenerator` 根据 `backend` 参数选择生成器
3. **代码生成**: 
   - LLM 优先: 使用大语言模型生成内核代码
   - 回退机制: 如果 LLM 失败，使用 mock 生成器
4. **评估**: 提交到评估服务器进行性能和正确性验证

### 后端选择逻辑

```python
# 在 kernel_rollout_func 中
if backend.lower() == "triton":
    generated_code = generate_mock_triton_kernel(original_code, problem_info)
else:
    generated_code = generate_mock_cuda_kernel(original_code, problem_info)
```

## 代码示例

### Triton 内核模板

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def optimized_relu_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    n_elements,  # Number of elements to process
    BLOCK_SIZE: tl.constexpr,
):
    # Get the current program ID
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create mask to handle out-of-bounds elements
    mask = offsets < n_elements
    
    # Load input values
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU operation: max(0, x)
    output_vals = tl.maximum(0.0, input_vals)
    
    # Store results
    tl.store(output_ptr + offsets, output_vals, mask=mask)

def triton_relu(input_tensor: torch.Tensor) -> torch.Tensor:
    # Triton kernel wrapper function
    # ... (implementation details)
    
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_triton_kernel = True
        
    def forward(self, x):
        if self.use_triton_kernel and x.is_cuda:
            try:
                return triton_relu(x)
            except Exception as e:
                print(f"Warning: Triton kernel failed, falling back: {e}")
                return F.relu(x)
        else:
            return F.relu(x)
```

## 使用指南

### 1. 生成 Triton 训练数据

```bash
cd KernelBench/scripts
python scan_kernel_bench_data_to_train_set_triton.py --verbose
```

输出文件: `data/kernel_bench_triton_level_{1,2,3,4}.jsonl`

### 2. 运行 Triton 后端生成器

```python
from kernel_generator import run_rollout

config = {
    "input_file": "/path/to/kernel_bench_triton_level_1.jsonl",
    "remote_engine_url": "http://localhost:4206",
    "remote_buffer_url": "http://localhost:8889",
    "eval_server_url": "http://localhost:18188",
    "backend": "triton",  # 指定使用 Triton 后端
    "use_simple_eval": True,
    "num_process": 5,
    "num_repeat_per_sample": 1,
    "num_repeats": 1,
    "task_type": "kernel",
}

run_rollout(config)
```

### 3. 验证功能

```bash
cd slime/slime_plugins/rollout_buffer/generator
python test_triton_generator.py
```

## 性能对比

### 数据生成统计

| 级别 | 问题数量 | 输出文件 |
|------|----------|----------|
| Level 1 | 100 | `kernel_bench_triton_level_1.jsonl` |
| Level 2 | 100 | `kernel_bench_triton_level_2.jsonl` |
| Level 3 | 50 | `kernel_bench_triton_level_3.jsonl` |
| Level 4 | 20 | `kernel_bench_triton_level_4.jsonl` |

### 主要差异总结

| 特性 | CUDA 版本 | Triton 版本 |
|------|-----------|-------------|
| 导入语句 | `torch.utils.cpp_extension.load_inline` | `triton`, `triton.language` |
| 内核定义 | `__global__ void kernel()` | `@triton.jit def kernel()` |
| 内存访问 | 直接指针操作 | `tl.load()`, `tl.store()` |
| 编译方式 | 运行时 C++ 编译 | Triton JIT 编译 |
| 网格启动 | `kernel<<<blocks, threads>>>()` | `kernel[grid](args)` |
| 数据源标识 | `kernel_bench_cuda` | `kernel_bench_triton` |

## 测试验证

运行 `test_triton_generator.py` 验证:
- ✅ Triton 内核生成功能
- ✅ CUDA 内核生成功能（对比）
- ✅ 数据格式验证
- ✅ 后端选择逻辑

## 注意事项

1. **依赖要求**: 确保安装了 Triton 库
2. **GPU 支持**: Triton 内核需要 CUDA 设备
3. **回退机制**: 如果 Triton 内核失败，会自动回退到标准 PyTorch 实现
4. **并发控制**: 使用信号量控制评估服务器的并发请求数量

## 未来改进

1. **动态内核生成**: 根据不同的算子类型生成更专业的 Triton 内核
2. **性能优化**: 调整 BLOCK_SIZE 和其他参数以获得最佳性能
3. **错误处理**: 增强错误处理和诊断信息
4. **模板扩展**: 添加更多常见算子的 Triton 内核模板 