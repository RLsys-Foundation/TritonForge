# Rollout Buffer 使用文档

## 概述

Rollout Buffer 是 Slime 框架中用于智能体轨迹生成的独立组件，其主要功能是使用 Slime 训练启动的 LLM OpenAI Server 进行智能体轨迹的生成。

### 设计理念

我们将 Rollout Buffer 独立出来的主要原因包括：

1. **框架解耦**：不同 Agent 任务所依赖的 Agent Framework 和工具都不相同，很可能会复用第三方的 Agent Framework
2. **灵活扩展**：如果将所有组件都封装到 Slime 内部会导致架构混乱，不利于扩展和维护
3. **职责分离**：Rollout Buffer 只负责通过调用 Slime 中启动的 Server 生成对应的轨迹，具体使用什么框架没有任何限制
4. **完全解耦**：轨迹生成逻辑和 Slime 训练逻辑完全解耦，支持引入各种复杂的 Agent Framework

### 工作流程

```
Slime Training Process ←─── HTTP API ───→ Rollout Buffer
        ↓                                      ↓
   LLM Server ←─────── HTTP Requests ─────── Agent Framework
        ↓                                      ↓
   Model Response ──────────────────────→ Trajectory Generation
```

对于每一个不同的 Agent 任务，都应该对应一个独立的 Generator 类，负责生成该类任务的轨迹。Rollout Buffer 会自动读取并加载不同类型的 Generator。

## 快速开始

### 基本使用流程

1. **复制模板**：将 `base_generator.py` 作为模板进行复制
2. **修改任务类型**：将 `TASK_TYPE` 修改为您的任务名称（不能与其他 Generator 重复）
3. **实现核心函数**：实现 `run_rollout()` 函数
4. **可选定制**：根据需要重写五个可选函数
5. **启动训练**：按照 [Agent Training 文档](./agent_training.md) 中的启动流程启动 Agent 训练

### 文件结构规范

Generator 文件必须以 `_generator.py` 结尾，并放置在 `generator/` 目录下：

```
generator/
├── base_generator.py      # Math 任务实现（默认模板）
└── your_task_generator.py # 您的自定义任务
```

## 核心组件

### 必需组件

每个 Generator 文件必须包含以下组件：

#### 1. `TASK_TYPE` 常量
定义任务类型的唯一标识符：
```python
TASK_TYPE = "your_task_name"
```

#### 2. `run_rollout()` 函数
核心数据生成逻辑的入口函数：
```python
def run_rollout(data: dict):
    # 实现您的轨迹生成逻辑
    pass
```

### 可选组件

除了必需组件外，Rollout Buffer 还提供了五个可自定义的函数来满足不同任务的特殊需求。如果不提供自定义实现，系统将使用默认实现（位于 `slime_plugins/rollout_buffer/generator/utils/default_func.py`）：

1. **`normalize_group_data()`**：奖励归一化函数
2. **`pad_group_data()`**：数据填充策略函数
3. **`is_valid_group()`**：组数据有效性验证函数
4. **`get_group_data_meta_info()`**：元信息统计函数
5. **`filter_item()`**：单个数据项过滤函数

## 参数配置

### Generator 核心参数

`run_rollout(data: dict)` 函数接收的主要参数如下（传入的 `data` 需要与 Slime 中发送的参数保持一致）：

| 参数名 | 类型 | 描述 |
|--------|------|------|
| `remote_engine_url` | string | 推理引擎服务地址，通常为 Slime 中的 SGLang Router 地址 |
| `remote_buffer_url` | string | Rollout Buffer 服务地址，通常为 Master 节点的某个端口（默认 8889） |
| `input_file` | string | 输入数据文件路径 |
| `task_type` | string | 任务类型标识符，定义在每个 `_generator.py` 文件中 |
| `num_repeat_per_sample` | int | 每个样本重复生成次数（Group Size） |
| `num_epoch` | int | 数据集遍历轮次（默认为 10） |
| `sampling_params` | dict | 模型采样参数（包含 max_tokens、temperature 等） |
| `num_process` | int | 并行进程数 |
| `skip_instance_ids` | list | 要跳过的实例 ID 列表，用于续训时跳过之前已处理的实例 |

### Buffer 控制参数

Buffer 的行为由以下关键参数控制，这些参数直接影响数据的收集、验证和输出策略：

#### 核心控制参数

| 参数名 | 默认值 | 描述 |
|--------|--------|------|
| `group_size` | - | 每组的目标数据数量，通常等于 `num_repeat_per_sample` |
| `min_valid_group_size_ratio` | 1.0 | 组被认为"有效"的最小数据比例（100%） |
| `min_valid_item_size_ratio` | 0.7 | 过滤后组内有效数据的最小比例（70%） |

**重要说明**：
- `group_size`：所有数据最终会被填充到这个大小，直接影响训练时每个实例的采样数量
- `min_valid_group_size_ratio`：建议设为 1.0，无效数据也可以写入，通过后续步骤过滤（如赋予极端 Reward）
- `min_valid_item_size_ratio`：过滤后组内有效数据的最小比例，应大于 0.5，用于过滤质量过差的组

#### 超时控制参数

| 参数名 | 默认值 | 描述 |
|--------|--------|------|
| `group_timeout_seconds` | 300 | 组超时时间（5分钟），防止部分组长时间卡住 |
| `min_timeout_group_size_ratio` | 0.7 | 超时组的最小数据比例阈值（70%） |

#### 系统资源参数

| 参数名 | 默认值 | 描述 |
|--------|--------|------|
| `max_buffer_size` | 1,000,000,000 | Buffer 最大容量（10亿），防止内存溢出 |

## 数据处理流程

### 完整处理流程

当从 Rollout Buffer 中获取一批训练数据时，五个可选函数按照以下固定顺序执行：

```
buffer.read(batch_size) 调用
    ↓
1. 📊 get_group_data_meta_info()
   └── 收集统计信息（进度、奖励分布等）
    ↓
2. ✅ is_valid_group()
   └── 判断每个组是否完成且有效
    ↓
3. 🔍 filter_item()
   └── 对有效组中的每个数据项进行过滤
    ↓
4. ⚖️ normalize_group_data()
   └── 对过滤后的组数据进行奖励归一化
    ↓
5. 📦 pad_group_data()
   └── 将归一化后的数据填充至目标 group_size
    ↓
📤 返回处理完成的批次数据
```

### 处理步骤详解

#### 第1步：元信息统计 - `get_group_data_meta_info()`

**功能**：收集当前 Buffer 中所有原始组数据的统计信息
- **输入**：Buffer 中所有原始组数据（包含无效组和无效轨迹）
- **输出**：包含统计信息的字典，用于日志记录和监控，比如可以记录平均奖励等信息

#### 第2步：组有效性验证 - `is_valid_group()`

**功能**：确定哪些组可以用于训练
- **输入**：每个组的完整数据 `(instance_id, group_data)`
- **输出**：`(is_valid, is_finished)` 元组
- **逻辑关系**：`有效组 ⊆ 已完成组 ⊆ 所有组`，其中已完成组中的实例将会在续训时被跳过，有效组中的符合要求的组将会被用于训练模型

#### 第3步：单项数据过滤 - `filter_item()`

**功能**：对有效组内的每个数据项进行精细化过滤
- **输入**：组内的单个数据项
- **输出**：布尔值，决定该项是否保留，因为写入 Rollout Buffer 的数据可能存在无效项，需要将其过滤

#### 第4步：奖励归一化 - `normalize_group_data()`

**功能**：对组内奖励值进行标准化处理
- **注意**：如果在此处进行归一化，需要在 Slime 中禁用奖励归一化，这里默认实现归一化的方式是只对于有效的数据 item 进行归一化并进行缩放
- **其他**：原始奖励值会保存到 `raw_reward` 字段，方便进行日志记录

#### 第5步：数据填充 - `pad_group_data()`

**功能**：将数据填充至标准的 `group_size`
- **策略**：通过奖励缩放保持总奖励一致性
- **输出**：固定大小的组数据，可直接用于训练
- **注意**：返回的数据数量**必须**要是 Group Size 的整数倍

### 重要机制说明

#### 数据存储策略
- **全量存储**：无论轨迹生成是否成功，都应将所有数据存入 Buffer
- **后续过滤**：通过过滤机制筛选出有用的 Group 和 Item
- **失败处理**：为失败的轨迹分配特殊的 Reward 值便于识别

#### 超时清理机制
- **自动清理**：每次执行 `get_rollout_data` 时检查时间戳
- **判断逻辑**：超时组根据有效数据数量决定取出或丢弃
- **防止积累**：有效防止数据在 Buffer 中过度积累

## 实现示例

### 基础实现模板

以 Math 任务为例，展示完整的 Generator 实现：

```python
TASK_TYPE = "math"

def run_rollout(data: dict):

    print(f"Starting math rollout with data: {data}")

    rollout_func = query_single_turn
    reward_func = get_rule_based_math_reward

    print(f"Waiting for 10 seconds for buffer server to start")
    time.sleep(10)
    global SAMPLING_PARAMS
    for k, v in data["sampling_params"].items():
        SAMPLING_PARAMS[k] = v
        print(f"Set {k} to {v}", type(v))

    generator = BaseGenerator(
        data["remote_engine_url"],
        data["remote_buffer_url"],
        num_repeat_per_sample=int(data["num_repeat_per_sample"]),
        queue_size=1000000,
        max_tokens=int(data["sampling_params"]["max_tokens"]),
        num_process=int(data.get("num_process", 100)),
        task_type=data["task_type"],
        skip_instance_ids=data.get("skip_instance_ids", None),
    )

    generator.entry(data["input_file"], rollout_func, reward_func, int(data.get("num_epoch", 1)))


```

### 轨迹生成函数示例

```python

def query_single_turn(client, messages, sampling_params, tools=None):
    base_payload = {
        "messages": messages,
        **sampling_params,
        "model": "custom",
        "stream": False,
        "seed": random.randint(1, 10000000),
        "tools": tools,
    }

    text = None
    accumulated_tokens = 0

    for attempt in range(6):
        try:
            # Create a fresh payload for each attempt
            current_payload = copy.deepcopy(base_payload)

            if text is not None:
                # Update messages with current progress
                current_messages = copy.deepcopy(messages)
                current_messages.append({"role": "assistant", "content": text})
                current_payload["messages"] = current_messages

                # Adjust max_tokens based on accumulated tokens
                if "max_tokens" in sampling_params:
                    current_payload["max_tokens"] = max(0, sampling_params["max_tokens"] - accumulated_tokens)

                # Add continue flag for partial rollouts
                current_payload["extra_body"] = {"continue_final_message": True}
            if current_payload["max_tokens"] == 0:
                break
            response = client.chat.completions.create(**current_payload)

            if len(response.choices) > 0:
                if response.choices[0].finish_reason == "abort":
                    print(
                        f"query failed, reason: {response.choices[0].finish_reason}, currently generated: {response.usage.completion_tokens}"
                    )

                    accumulated_tokens += response.usage.completion_tokens

                    if text is None:
                        text = response.choices[0].message.content
                    else:
                        text += response.choices[0].message.content

                    sleep(10)
                    continue
                if text is None:
                    text = response.choices[0].message.content
                elif response.choices[0].message.content is not None:
                    text += response.choices[0].message.content
                break
            else:
                print(f"Error in query, status code: {response.status_code}")
                continue
        except Exception as e:
            print(f"query failed in single turn, error: {e}")
            continue

    # Update final messages
    if len(messages) > 0 and messages[-1]["role"] == "assistant":
        messages = messages[:-1]
    messages.append({"role": "assistant", "content": text})

    return messages

```

## 常见问题

### Q: 如何处理生成失败的数据？
A: 将失败数据也存入 Buffer，但分配特殊的 Reward 值（如 -1），通过后续过滤机制处理。

### Q: 如何调试数据质量问题？
A: 利用 `get_group_data_meta_info()` 函数收集详细统计信息，监控奖励分布和数据质量。

### Q: 超时机制如何工作？
A: 当组的最后一次数据生成时间超过 `group_timeout_seconds` 时，系统会根据 `min_timeout_group_size_ratio` 决定是否使用该组数据。

### Q: 如何实现续训？
A: Slime 将通过 `skip_instance_ids` 参数传递已处理的实例 ID 列表，Generator 会自动跳过这些实例。所有已完成的组都会自动的被跳过。