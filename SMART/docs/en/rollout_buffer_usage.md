# Rollout Buffer Usage Documentation

## Overview

Rollout Buffer is an independent component in the Slime framework for agent trajectory generation, with the main function of using the LLM OpenAI Server launched by Slime training to generate agent trajectories.

### Design Philosophy

The main reasons we made Rollout Buffer independent include:

1. **Framework Decoupling**: Different Agent tasks depend on different Agent Frameworks and tools, and are likely to reuse third-party Agent Frameworks
2. **Flexible Extension**: If all components are encapsulated within Slime, it would lead to architectural chaos and be unfavorable for extension and maintenance
3. **Responsibility Separation**: Rollout Buffer is only responsible for generating corresponding trajectories by calling the Server launched in Slime, with no restrictions on what framework is used
4. **Complete Decoupling**: Trajectory generation logic and Slime training logic are completely decoupled, supporting the introduction of various complex Agent Frameworks

### Workflow

```
Slime Training Process ←─── HTTP API ───→ Rollout Buffer
        ↓                                      ↓
   LLM Server ←─────── HTTP Requests ─────── Agent Framework
        ↓                                      ↓
   Model Response ──────────────────────→ Trajectory Generation
```

For each different Agent task, there should be a corresponding independent Generator class, responsible for generating trajectories for that type of task. Rollout Buffer automatically reads and loads different types of Generators.

## Quick Start

### Basic Usage Process

1. **Copy Template**: Copy `base_generator.py` as a template
2. **Modify Task Type**: Change `TASK_TYPE` to your task name (cannot duplicate with other Generators)
3. **Implement Core Function**: Implement the `run_rollout()` function
4. **Optional Customization**: Rewrite five optional functions as needed
5. **Start Training**: Follow the startup process in [Agent Training Documentation](./agent_training.md) to start Agent training

### File Structure Standards

Generator files must end with `_generator.py` and be placed in the `generator/` directory:

```
generator/
├── base_generator.py      # Math task implementation (default template)
└── your_task_generator.py # Your custom task
```

## Core Components

### Required Components

Each Generator file must contain the following components:

#### 1. `TASK_TYPE` Constant
Define the unique identifier for the task type:
```python
TASK_TYPE = "your_task_name"
```

#### 2. `run_rollout()` Function
Entry function for core data generation logic:
```python
def run_rollout(data: dict):
    # Implement your trajectory generation logic
    pass
```

### Optional Components

In addition to required components, Rollout Buffer also provides five customizable functions to meet special needs of different tasks. If no custom implementation is provided, the system will use default implementations (located in `slime_plugins/rollout_buffer/generator/utils/default_func.py`):

1. **`normalize_group_data()`**: Reward normalization function
2. **`pad_group_data()`**: Data padding strategy function
3. **`is_valid_group()`**: Group data validity verification function
4. **`get_group_data_meta_info()`**: Meta-information statistics function
5. **`filter_item()`**: Individual data item filtering function

## Parameter Configuration

### Generator Core Parameters

The main parameters received by the `run_rollout(data: dict)` function are as follows (the incoming `data` needs to be consistent with parameters sent from Slime):

| Parameter Name | Type | Description |
|----------------|------|-------------|
| `remote_engine_url` | string | Inference engine service address, usually the SGLang Router address in Slime |
| `remote_buffer_url` | string | Rollout Buffer service address, usually a port on the Master node (default 8889) |
| `input_file` | string | Input data file path |
| `task_type` | string | Task type identifier, defined in each `_generator.py` file |
| `num_repeat_per_sample` | int | Number of repeated generations per sample (Group Size) |
| `num_epoch` | int | Number of dataset traversal rounds (default 10) |
| `sampling_params` | dict | Model sampling parameters (including max_tokens, temperature, etc.) |
| `num_process` | int | Number of parallel processes |
| `skip_instance_ids` | list | List of instance IDs to skip, used for resume training to skip previously processed instances |

### Buffer Control Parameters

Buffer behavior is controlled by the following key parameters, which directly affect data collection, validation, and output strategies:

#### Core Control Parameters

| Parameter Name | Default Value | Description |
|----------------|---------------|-------------|
| `group_size` | - | Target number of data items per group, usually equal to `num_repeat_per_sample` |
| `min_valid_group_size_ratio` | 1.0 | Minimum data ratio for a group to be considered "valid" (100%) |
| `min_valid_item_size_ratio` | 0.7 | Minimum ratio of valid data within a group after filtering (70%) |

**Important Notes**:
- `group_size`: All data will eventually be padded to this size, directly affecting the number of samples per instance during training
- `min_valid_group_size_ratio`: Recommended to set to 1.0, invalid data can also be written and filtered through subsequent steps (such as assigning extreme Rewards)
- `min_valid_item_size_ratio`: Minimum ratio of valid data within a group after filtering, should be greater than 0.5, used to filter groups with poor quality

#### Timeout Control Parameters

| Parameter Name | Default Value | Description |
|----------------|---------------|-------------|
| `group_timeout_seconds` | 300 | Group timeout time (5 minutes), prevents some groups from being stuck for a long time |
| `min_timeout_group_size_ratio` | 0.7 | Minimum data ratio threshold for timeout groups (70%) |

#### System Resource Parameters

| Parameter Name | Default Value | Description |
|----------------|---------------|-------------|
| `max_buffer_size` | 1,000,000,000 | Maximum Buffer capacity (1 billion), prevents memory overflow |

## Data Processing Flow

### Complete Processing Flow

When retrieving a batch of training data from Rollout Buffer, five optional functions execute in the following fixed order:

```
buffer.read(batch_size) call
    ↓
1. 📊 get_group_data_meta_info()
   └── Collect statistics (progress, reward distribution, etc.)
    ↓
2. ✅ is_valid_group()
   └── Determine if each group is complete and valid
    ↓
3. 🔍 filter_item()
   └── Filter each data item in valid groups
    ↓
4. ⚖️ normalize_group_data()
   └── Perform reward normalization on filtered group data
    ↓
5. 📦 pad_group_data()
   └── Pad normalized data to target group_size
    ↓
📤 Return processed batch data
```

### Processing Step Details

#### Step 1: Meta-information Statistics - `get_group_data_meta_info()`

**Function**: Collect statistical information for all raw group data in the current Buffer
- **Input**: All raw group data in Buffer (including invalid groups and invalid trajectories)
- **Output**: Dictionary containing statistical information, used for logging and monitoring, such as recording average rewards and other information

#### Step 2: Group Validity Verification - `is_valid_group()`

**Function**: Determine which groups can be used for training
- **Input**: Complete data for each group `(instance_id, group_data)`
- **Output**: `(is_valid, is_finished)` tuple
- **Logical Relationship**: `Valid Groups ⊆ Finished Groups ⊆ All Groups`, where instances in finished groups will be skipped during resume training, and qualified groups in valid groups will be used for model training

#### Step 3: Individual Data Filtering - `filter_item()`

**Function**: Perform fine-grained filtering on each data item within valid groups
- **Input**: Individual data items within groups
- **Output**: Boolean value determining whether the item should be retained, as data written to Rollout Buffer may contain invalid items that need to be filtered

#### Step 4: Reward Normalization - `normalize_group_data()`

**Function**: Perform standardized processing on group reward values
- **Note**: If normalization is performed here, reward normalization needs to be disabled in Slime. The default implementation normalizes only valid data items and performs scaling
- **Other**: Original reward values are saved to the `raw_reward` field for convenient logging

#### Step 5: Data Padding - `pad_group_data()`

**Function**: Pad data to standard `group_size`
- **Strategy**: Maintain total reward consistency through reward scaling
- **Output**: Fixed-size group data that can be directly used for training
- **Note**: The returned data count **must** be a multiple of Group Size

### Important Mechanism Descriptions

#### Data Storage Strategy
- **Full Storage**: All data should be stored in Buffer regardless of whether trajectory generation is successful
- **Subsequent Filtering**: Filter out useful Groups and Items through filtering mechanisms
- **Failure Handling**: Assign special Reward values to failed trajectories for easy identification

#### Timeout Cleanup Mechanism
- **Automatic Cleanup**: Check timestamps each time `get_rollout_data` is executed
- **Decision Logic**: Timeout groups decide whether to retrieve or discard based on valid data count
- **Prevent Accumulation**: Effectively prevent excessive data accumulation in Buffer

## Implementation Examples

### Basic Implementation Template

Using Math task as an example, showing complete Generator implementation:

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

### Trajectory Generation Function Example

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

## Frequently Asked Questions

### Q: How to handle failed generation data?
A: Store failed data in Buffer as well, but assign special Reward values (such as -1), and handle through subsequent filtering mechanisms.

### Q: How to debug data quality issues?
A: Use the `get_group_data_meta_info()` function to collect detailed statistical information and monitor reward distribution and data quality.

### Q: How does the timeout mechanism work?
A: When a group's last data generation time exceeds `group_timeout_seconds`, the system will decide whether to use that group's data based on `min_timeout_group_size_ratio`.

### Q: How to implement resume training?
A: Slime will pass a list of processed instance IDs through the `skip_instance_ids` parameter, and the Generator will automatically skip these instances. All completed groups will be automatically skipped. 