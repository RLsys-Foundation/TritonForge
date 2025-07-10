# 子进程评估服务器指南

## 问题背景

原始的 `simple_eval_server.py` 存在以下问题：

1. **复杂的错误处理**：当 CUDA 出现 "illegal memory access" 时，清理逻辑会导致错误的级联传播
2. **CUDA 上下文污染**：错误的 CUDA 内核会污染整个进程的 CUDA 上下文
3. **设备状态混乱**：复杂的设备管理逻辑在出错时容易导致设备被错误标记为不可用
4. **难以恢复**：一旦出现 CUDA 错误，整个服务器可能需要重启

## 子进程方案优势

### 1. 完全隔离
- 每个评估运行在独立的子进程中
- 子进程有自己的 CUDA 上下文
- 一个子进程崩溃不会影响主进程和其他评估

### 2. 自动清理
- 子进程退出时，操作系统自动清理所有资源
- 无需复杂的 CUDA 清理逻辑
- 避免了 `graceful_eval_cleanup` 中的问题

### 3. 简化错误处理
- 只需要处理子进程的返回值
- 不需要复杂的 CUDA 错误恢复逻辑
- 错误分类更加清晰

### 4. 更好的稳定性
- 主进程始终保持健康状态
- GPU 设备不会被错误标记为不可用
- 可以并发处理多个评估请求

## 使用方法

### 启动服务器

```bash
# 使用子进程隔离版本
python KernelBench/scripts/eval_server_subprocess.py
```

### 主要功能特性

**`eval_server_subprocess.py`** 提供了完整的生产级功能：

1. **自动 CUDA 架构检测**：支持 H100、Ampere、Turing/Volta 等各种 GPU
2. **完整的环境设置**：自动配置 CUDA DSA、Triton 缓存等
3. **GPU 健康检查**：独立进程监控 GPU 状态
4. **丰富的 API 端点**：提供 9 个不同的管理和调试端点
5. **详细的错误分类**：精确识别各种错误类型

### API 接口

```python
import requests

# 评估请求
response = requests.post("http://localhost:18188/eval", json={
    "original_model_src": "...",
    "custom_model_src": "...",
    "backend": "triton",
    "verbose": True
})

# 服务器状态
status = requests.get("http://localhost:18188/health")

# GPU 详细信息
gpu_info = requests.get("http://localhost:18188/gpu_status")

# 后端信息
backend_info = requests.get("http://localhost:18188/backend_info")
```

### 错误处理改进

提供了详细的错误分类和处理：

- `illegal_memory_access`: CUDA 内存错误（已隔离，GPU 仍然健康）
- `shared_memory_exceeded`: 共享内存超限，提供具体的内存使用建议
- `compilation_error`: 编译错误（如 `triton_helpers` 未定义）
- `unsupported_architecture`: 不支持的 GPU 架构，自动检测并提供解决方案

## 核心改进

### 1. 自动 CUDA 架构检测
```python
# 自动检测并设置合适的 CUDA 架构
if major == 9 and minor == 0:  # H100
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
elif major == 8:  # Ampere
    # 自动设置 8.6 或 8.0
```

### 2. 完整的环境配置
```python
# 在子进程中自动设置
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TRITON_CACHE_DIR"] = f"/tmp/triton_cache_gpu_{device_id}"
```

### 3. 独立进程健康检查
```python
# 在独立进程中检查 GPU 健康状态
def _check_gpu_health(device_id: int) -> bool:
    # 避免主进程 CUDA 上下文污染
```

## 性能考虑

### 优势
- **完全隔离**：每个评估都在独立环境中运行
- **智能缓存**：Triton 缓存按 GPU 分离，避免冲突
- **自动优化**：根据 GPU 类型自动配置最佳参数
- **并发支持**：支持多个并发评估请求

### 权衡
- **启动开销**：子进程启动需要额外时间（约 1-2 秒）
- **内存使用**：每个子进程需要独立的内存空间
- **缓存管理**：Triton 缓存不能跨进程共享

## 配置建议

### 1. 环境变量（自动设置）
```bash
# 服务器会自动设置这些环境变量
export TORCH_USE_CUDA_DSA=1
export CUDA_LAUNCH_BLOCKING=1
export TRITON_CACHE_DIR=/tmp/triton_cache_gpu_X
```

### 2. 多进程设置
```python
# 在服务器启动时自动设置
mp.set_start_method('spawn', force=True)
```

### 3. 超时设置
```python
# 针对子进程开销优化的超时时间
timeout = 600  # 10 分钟
```

## 对比分析

| 特性 | 原始版本 | 子进程版本 |
|------|----------|------------|
| 错误隔离 | ❌ | ✅ |
| 自动清理 | ❌ | ✅ |
| CUDA 架构检测 | ❌ | ✅ |
| GPU 健康监控 | ❌ | ✅ |
| 复杂度 | 高 | 中 |
| 稳定性 | 低 | 高 |
| 启动开销 | 低 | 中 |
| 内存使用 | 低 | 中 |
| 并发支持 | 有限 | 良好 |
| 生产就绪 | ❌ | ✅ |

## 故障排除

### 常见问题

1. **子进程超时**
   - 检查 GPU 资源是否充足
   - 确认内核代码没有死循环
   - 查看详细的错误日志

2. **架构不兼容**
   - 服务器会自动检测并设置 CUDA 架构
   - 查看启动日志确认架构设置

3. **Triton 缓存冲突**
   - 每个 GPU 有独立的缓存目录
   - 可以通过 `/cleanup` 端点清理

### 调试技巧

```python
# 启用详细日志
request = EvalRequest(
    verbose=True,
    # ... 其他参数
)

# 检查服务器完整状态
health = requests.get("http://localhost:18188/health")
gpu_status = requests.get("http://localhost:18188/gpu_status")
backend_info = requests.get("http://localhost:18188/backend_info")
```

## 总结

`eval_server_subprocess.py` 是一个**生产就绪**的子进程评估服务器，彻底解决了原始服务器的所有问题：

1. **完全隔离**：每个评估在独立进程中运行
2. **自动配置**：智能检测 GPU 类型并优化配置
3. **健壮性**：comprehensive error handling and recovery
4. **易于维护**：清晰的 API 和详细的状态报告

对于生产环境，这是**推荐的解决方案**，它提供了最佳的稳定性和可靠性。 