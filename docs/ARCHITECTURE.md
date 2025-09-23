# TritonForge Architecture Overview

## 🏗️ Overall System Architecture

TritonForge is a **server-based SFT + RL post-training framework** that operates across both AMD and NVIDIA platforms, featuring complete disaggregation of training, inference, and evaluation components.

```mermaid
graph TB
    subgraph DL["Data Layer"]
        KB["KernelBook Dataset<br/>18.2k PyTorch-to-Triton pairs"]
        KBench["KernelBench Dataset<br/>200+ problems L1-L4"]

        subgraph Pipeline["SFT Data Pipeline"]
            DP1[Multi-Turn Data Generator]
            DP2[Thinking Tags Injector]
            DP3["Length & Quality Filter"]
            DP1 --> DP2
            DP2 --> DP3
        end

        KB --> DP1
        KBench --> DP1
        DP3 --> ProcessedData["Processed Training Data<br/>~17k filtered samples"]
    end

    subgraph TI["Training Infrastructure"]
        subgraph SLIME["SLIME Framework (Server-Based)"]
            Router["SLIME Router<br/>Orchestration Layer"]

            subgraph ActorG["Actor Group (Training)"]
                MegatronServer["Megatron-LM Server<br/>Distributed Training<br/>TP=2, CP=4, PP=1"]
                GRPO["GRPO Actor<br/>Group Relative Policy Optimization"]
                WeightSync[Weight Synchronizer]
            end

            subgraph RolloutG["Rollout Group (Generation)"]
                SGLangServer["SGLang Server Pool<br/>High-Performance Inference"]
                RolloutBuffer["Async Rollout Buffer<br/>Experience Collection"]

                subgraph Gens[Generators]
                    SingleTurnGen[Single-Turn Generator]
                    MultiTurnGen["Multi-Turn Generator<br/>Max 3 iterations"]
                end
            end

            Router --> MegatronServer
            Router --> SGLangServer
            Router --> RolloutBuffer
            WeightSync -.->|Weight Updates| SGLangServer
        end
    end

    subgraph EI["Evaluation Infrastructure"]
        subgraph EvalServers["Disaggregated Eval Servers"]
            EvalServer1["KernelBench Eval Server<br/>Port 18188"]
            EvalServer2["Remote Eval Server<br/>Compilation & Correctness"]
            EvalServer3["Performance Eval Server<br/>Speedup Metrics"]
        end

        subgraph Backends["Platform-Specific Backends"]
            NVBackend["NVIDIA Backend<br/>CUDA 12.6+<br/>Triton Compiler"]
            AMDBackend["AMD Backend<br/>ROCm 6.3+<br/>HIP Translation Layer"]
        end

        MultiTurnGen -->|Eval Request| EvalServer1
        SingleTurnGen -->|Eval Request| EvalServer1
        EvalServer1 --> EvalServer2
        EvalServer2 --> EvalServer3
        EvalServer3 --> NVBackend
        EvalServer3 --> AMDBackend
    end

    subgraph ME["Model Evolution"]
        BaseModel["Qwen3-8B<br/>Base Model"]
        SFTModel["Qwen3-8B-Kernelbook-SFT<br/>Fine-tuned Model"]
        RLModel["TritonForge-8B<br/>RL-Optimized Model"]

        BaseModel -->|SFT Training| SFTModel
        SFTModel -->|RL Training| RLModel
    end

    ProcessedData --> MegatronServer
    MegatronServer --> SFTModel

    subgraph RS["Reward System"]
        CompileReward["Compilation Success<br/>+0.3 base"]
        CorrectReward["Functional Correctness<br/>+0.5 if correct"]
        SpeedupReward["Performance Speedup<br/>+0.2 × log(speedup)"]
        DiscountFactor["γ = 0.4<br/>Multi-turn discount"]

        CompileReward --> RewardAgg[Reward Aggregator]
        CorrectReward --> RewardAgg
        SpeedupReward --> RewardAgg
        DiscountFactor --> RewardAgg
    end

    NVBackend --> RewardAgg
    AMDBackend --> RewardAgg
    RewardAgg --> GRPO

    style Router fill:#f9f,stroke:#333,stroke-width:4px
    style SFTModel fill:#9f9,stroke:#333,stroke-width:2px
    style RLModel fill:#99f,stroke:#333,stroke-width:4px
```

## 🔄 Training Pipeline Flow

### Stage 1: SFT Data Generation Pipeline

```mermaid
flowchart LR
    subgraph RDS["Raw Data Sources"]
        KB["KernelBook<br/>18.2k samples"]
        KBench["KernelBench<br/>L1-L4 problems"]
    end

    subgraph DA["Data Augmentation"]
        MT[Multi-Turn Converter]
        TT["Thinking Tags<br/>⟨thinking⟩...⟨/thinking⟩"]
        Conv[Conversation Format]

        KB --> MT
        KBench --> MT
        MT --> TT
        TT --> Conv
    end

    subgraph QC["Quality Control"]
        LenFilter["Length Filter<br/>Min: 100 tokens<br/>Max: 8192 tokens"]
        QualFilter["Quality Filter<br/>Remove invalid code"]
        Dedup[Deduplication]

        Conv --> LenFilter
        LenFilter --> QualFilter
        QualFilter --> Dedup
    end

    subgraph OF["Output Formats"]
        JSONL["JSONL Format<br/>kernel_bench_triton_all_levels.jsonl"]
        Megatron["Megatron Format<br/>For distributed training"]

        Dedup --> JSONL
        Dedup --> Megatron
    end

    JSONL --> FinalData[17k Filtered Samples]
    Megatron --> FinalData
```

### Stage 2: Server-Based Training Architecture

```mermaid
flowchart TB
    subgraph ATL["Async Training Loop"]
        Init[Initialize Ray Cluster]
        CreatePG["Create Placement Groups<br/>Actor GPUs | Rollout GPUs"]
        StartServers["Start All Servers<br/>Megatron | SGLang | Eval"]

        Init --> CreatePG
        CreatePG --> StartServers

        subgraph TIter["Training Iteration"]
            GenExp["Generate Experience<br/>Async Rollout"]
            CollectData[Collect to Buffer]
            TrainStep[GRPO Training Step]
            UpdateWeights[Update Weights]

            GenExp --> CollectData
            CollectData --> TrainStep
            TrainStep --> UpdateWeights
            UpdateWeights -->|Every N steps| GenExp
        end

        StartServers --> GenExp
    end

    subgraph SC["Server Communication"]
        HTTP["HTTP/REST APIs<br/>Between servers"]
        Ray["Ray Remote Calls<br/>Actor coordination"]
        Queue["Message Queues<br/>Async buffering"]

        HTTP -.-> GenExp
        Ray -.-> CollectData
        Queue -.-> TrainStep
    end
```

## 🚀 Multi-Turn Refinement Process

```mermaid
sequenceDiagram
    participant User
    participant Router as SLIME Router
    participant Gen as Multi-Turn Generator
    participant SGLang as SGLang Server
    participant Eval as Eval Server
    participant Backend as GPU Backend

    User->>Router: Start RL Training
    Router->>Gen: Initialize Multi-Turn Rollout

    loop For each problem (up to 3 turns)
        Gen->>SGLang: Generate Kernel (Turn N)
        SGLang-->>Gen: Triton Code
        Gen->>Eval: Evaluate Kernel
        Eval->>Backend: Compile & Execute

        alt Compilation Success
            Backend-->>Eval: Runtime & Correctness
            Eval-->>Gen: Reward + Metrics

            alt If Correct & Fast
                Gen-->>Gen: Mark Complete
            else If Issues Found
                Gen-->>Gen: Prepare Feedback
                Note over Gen: Include error messages<br/>and performance data
                Gen->>SGLang: Generate Improved Kernel
            end
        else Compilation Failed
            Backend-->>Eval: Error Message
            Eval-->>Gen: Negative Reward
            Gen-->>Gen: Prepare Error Feedback
        end
    end

    Gen->>Router: Return Trajectory
    Router->>Router: Aggregate Returns (γ=0.4)
    Router-->>User: Training Metrics
```

## 🖥️ Platform-Specific Implementations

### NVIDIA H100 Configuration

```yaml
Hardware:
  GPU: H100 80GB
  CUDA: 12.6.1
  Triton: 3.0+

Training Config:
  Tensor Parallel: 2
  Context Parallel: 4
  Pipeline Parallel: 1
  Batch Size: 32
  Learning Rate: 1e-5

Evaluation:
  Backend: CUDA + Triton JIT
  Profiling: NSight Systems
  Metrics: FLOPS, Memory Bandwidth
```

### AMD MI300X Configuration

```yaml
Hardware:
  GPU: MI300X 192GB
  ROCm: 6.3.4
  HIP: 6.3+

Training Config:
  Tensor Parallel: 2
  Context Parallel: 4
  Pipeline Parallel: 1
  Batch Size: 32
  Learning Rate: 1e-5

Evaluation:
  Backend: ROCm + HIP Translation
  Profiling: rocprof
  Metrics: FLOPS, Memory Bandwidth

Special Handling:
  - Subprocess isolation for memory faults
  - HIP_PLATFORM=amd environment
  - PYTORCH_ROCM_ARCH=gfx942
```

## 🔧 Key Components Details

### SLIME Router
- **Purpose**: Orchestrates communication between all components
- **Features**:
  - Async message passing
  - Load balancing across servers
  - Fault tolerance and retry logic
  - Weight synchronization management

### Rollout Buffer
- **Purpose**: Asynchronously collects generation experiences
- **Features**:
  - Multi-process data collection
  - Experience replay buffer
  - Priority sampling support
  - Trajectory aggregation

### Evaluation Servers
- **Purpose**: Isolated evaluation environments
- **Features**:
  - Sandboxed execution
  - Resource limit enforcement
  - Performance profiling
  - Error recovery mechanisms

### Reward System
- **GRPO (Group Relative Policy Optimization)**:
  - Uses group-relative returns instead of traditional advantage estimation
  - Better suited for multi-turn refinement scenarios
  - Reduces variance in policy gradient estimation

- **Multi-Turn Aggregation**:
  ```python
  total_return = sum(reward_t * (gamma ** t) for t, reward_t in enumerate(rewards))
  ```

- **Reward Components**:
  - Compilation: 0.3 base reward
  - Correctness: 0.5 if functionally correct
  - Speedup: 0.2 × log(speedup_ratio)

## 📊 Data Flow Summary

1. **Input**: PyTorch operations from KernelBook/KernelBench
2. **SFT Pipeline**: Multi-turn augmentation → Thinking tags → Filtering → 17k samples
3. **SFT Training**: Distributed training via Megatron-LM → Fine-tuned model
4. **RL Pipeline**: Multi-turn generation → Evaluation → Reward → GRPO updates
5. **Output**: Optimized TritonForge model capable of generating efficient GPU kernels

## 🚦 System Status Indicators

- **Green**: Component operational
- **Yellow**: Component under load
- **Red**: Component failed/recovering
- **Dotted Lines**: Async communication
- **Solid Lines**: Sync communication

## 🔄 Continuous Improvement Loop

```mermaid
graph LR
    Generate[Generate Kernel] --> Evaluate[Evaluate Performance]
    Evaluate --> Feedback[Compilation + Runtime Feedback]
    Feedback --> Improve[Multi-Turn Refinement]
    Improve --> Generate

    Evaluate --> Reward[Calculate Reward]
    Reward --> Train[GRPO Training]
    Train --> Model[Update Model]
    Model --> Generate
```

This architecture enables:
- ✅ Cross-platform GPU kernel generation (NVIDIA & AMD)
- ✅ Server-based disaggregated training at scale
- ✅ Multi-turn iterative kernel refinement
- ✅ Comprehensive evaluation with compilation and performance metrics
- ✅ Asynchronous training for improved efficiency
- ✅ Fault tolerance and recovery mechanisms