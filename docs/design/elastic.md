# Elastic LLM Training Design (Training with MS-Swift)

## PyTorch LLM Scenario Overview

This document introduces the elastic training design solution for large language model (LLM) training with PyTorch + DeepSpeed, focusing on checkpoint recovery, data re-partitioning, and optimizer state recovery in elastic scaling scenarios.

Note: The "training elasticity" here is not the TensorFlow Parameter Server style of elasticity. We do not scale a central parameter server or push/pull model parameters dynamically; instead, we coordinate PyTorch worker group membership changes via rendezvous, then restore model and optimizer state from checkpoints.

## PyTorch LLM Elasticity Philosophy

### LLM Training Concepts

- **job**: The entire training task
- **node**: A physical instance or a container
- **worker**: A worker instance created according to WorkerSpec, with its lifecycle managed by ElasticAgent. It can be understood as a training process
- **workergroup**: A group of worker instances
- **ElasticAgent**: A TorchElastic control plane component, an independent process used to manage the lifecycle of workers within the current node, respond to member join/exit events, monitor worker health status, and other lifecycle activities

## DeepSpeed Universal Checkpoint for PyTorch LLMs

### Parallelism Techniques

| Abbreviation | Description       |
| ------------ |  ----------------- |
| **DP**       | Data Parallel     |
| **TP**       |  Tensor Parallel   |
| **PP**       |  Pipeline Parallel |
| **SP**       |  Sequence Parallel |

Each parallelism method generates distributed checkpoints (checkpoint files) when saving the model. For example, each GPU saves its own weight slice or optimizer state. The problem is: these distributed checkpoint files have different structures and cannot be directly loaded from one another.

You trained a checkpoint on 8 GPUs (TP=8) and want to continue training with 4 GPUs (TP=4) in the next experiment. → Traditional approach requires "merge then split", which is very cumbersome, requiring specialized scripts for different partitioning strategies.

### Universal Checkpoint (hereinafter referred to as UCP)

```
┌────────────────────┐
│  Source Checkpoint │ (DP / TP / PP / SP)
└──────────┬─────────┘
           │
           ▼
   ┌──────────────────┐
   │  UCP Atomic Files│  ← Each parameter stored independently (universal format)
   └──────────┬───────┘
              │
              ▼
┌────────────────────────┐
│  Target Checkpoint     │ (New configuration: different GPU count or parallelism mode)
└────────────────────────┘
```

![UCP Architecture Diagram](../figures/universal-checkpoint.png)

UCP introduces an intermediate layer concept:

**Atomic checkpoint** = Independent, fine-grained storage file for each parameter + optimizer state. This means: instead of saving distributed fragments per GPU, directly save a complete copy of "each model parameter", while recording optimizer states (such as Adam's momentum, variance).

## Core Issues in PyTorch LLM Elasticity

DLRover + DeepSpeed elastic training needs to solve the following three core issues:

1. **Implement ElasticAgent suitable for DeepSpeed**
2. **How to re-partition data before and after scaling**
3. **How to recover optimizer states and model parameters before and after scaling**

The relationship between DLRover and Swift is shown in the following diagram:

![DLRover-Swift Architecture Diagram](../figures/dlrover-swift.png)

## Solution for PyTorch LLM Elastic Training

### 1. PyTorch LLM ElasticAgent for DeepSpeed (DLRover side)

**ElasticAgent**: A TorchElastic control plane component, an independent process used to manage the lifecycle of workers within the current node, respond to member join/exit events, monitor worker health status, and other lifecycle activities. Each training node runs an independent Agent process responsible for starting and managing local Worker processes. It assigns each process's WORLD_SIZE, RANK, and other information, and continuously monitors Worker health status. When a Worker fails or crashes, the Elastic Agent terminates all current Workers and restarts all processes according to the new node membership; when a new node joins, it also triggers the Agent to restart new Workers.

From the previous introduction, ElasticAgent is a control plane component that controls the entire training process's worker lifecycle, member join and exit. The logic we need to implement is: when detecting new workers joining (or exiting), complete checkpoint saving work (wrapping up before scaling), and prepare for the next round of training resumption (training recovery after scaling). We need to add this wrapping-up logic to both the worker side and ElasticAgent.

#### 1.1 Worker-side Logic

ElasticAgent.invoke_run() method detects member changes → sends previous-round-completed=False to master node (only group_rank=0 sends) → executes ucp（save universal chechpoint） process → sends previous-round-completed=True to master node.

#### 1.2 Master-side Logic

Use the previous-round-completed flag to regulate worker grouping rounds. If the flag is True, the next round of rendezvous can start.

### 2. PyTorch LLM Elastic Trainer (Swift side)

#### 2.1 Implement ElasticSampler

Need to exclude already processed data and reallocate data according to new member count.

- Swift's current data loader already has this capability. It only needs to specify samples to skip and new batch_size when resuming.

#### 2.2 Handling batchsize and lr after member count changes

By adjusting gradient accumulation to keep total batchsize unchanged, so learning rate doesn't need modification.

#### 2.3 How to recover trainer optimizer states after member count changes

When members change, the worker with group_rank=0 will use universal checkpoint (ds_to_universal.py) to save the checkpoint to the specified path. When restarting training, add the `--universal-checkpoint` parameter and load the checkpoint from the universal checkpoint saved path.

ms-swift side implementation: https://github.com/modelscope/ms-swift/pull/6955
