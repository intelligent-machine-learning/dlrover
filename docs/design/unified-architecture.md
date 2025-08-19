## Background

As machine learning workloads become increasingly complex, there is a growing need to support unified training paradigms
that combine elastic training with other processing strategies. Traditional elastic training frameworks are optimized
for scaling model training dynamically, but often lack the flexibility to integrate with heterogeneous tasks such as
data preprocessing, reinforcement learning environments, or custom compute-intensive operations.

Unified training addresses these challenges by enabling the orchestration of diverse workloads within a unified
framework. This approach allows for seamless coordination between elastic model training and auxiliary tasks, such as
large-scale data transformation, environment simulation, or multi-stage pipelines. By managing the lifecycle and
resource allocation for both elastic and non-elastic components, unified training frameworks can maximize resource
utilization, improve throughput, and simplify the development of complex machine learning systems.

Typical scenarios include:

- **Reinforcement Learning (RL) Environments:** Multiple training tasks may need to run in parallel, with environment
  simulation requiring significant CPU resources and coordination with model training.
- **Data Fusion and Preprocessing:** Integrating data processing with model training, where various preprocessing or
  transformation tasks must be performed alongside or prior to training, often involving different computation patterns
  and resource requirements.

By supporting these unified scenarios, the framework enables users to build more flexible, efficient, and scalable
machine learning pipelines that can adapt to a wide range of real-world requirements.

This document outlines the design and architecture of a unified training framework that integrates elastic training with
other processing paradigms, providing a comprehensive solution for complex machine learning workloads.

> This is also an architecture design for [Unified MPMD Control](./unified-mpmd-control-proposal.md)

## Core Design

### Architecture Overview

![General Architecture Diagram](../figures/unified/architecture.excalidraw.svg)

The unified training framework integrates both elastic and non-elastic training paradigms to support diverse workloads.
In this architecture, the `Controller/PrimeMaster` is responsible for orchestrating the entire unified training process,
while the
`Backend`(including `ElasticMaster` and `ElasticWorker`) components focus on elastic training tasks.
> The `ElasticMaster` and `ElasticWorker` represent the original elastic training architecture, while the `PrimeMaster`
> is the newly introduced component.

The architecture is designed to be modular and extensible, allowing for the integration of various training strategies
and workloads. The key components include:

#### Controller/PrimeMaster

- Oversees the lifecycle management of unified training jobs.
- Schedules and monitors various training/processing workloads.
- Maintains a registry of all actors and their assigned roles.
- Stores and manages the global state of the training process.

#### Backend

One backend consists of `Worker` components and, optionally, `SubMaster` components, which are responsible for executing
specific workloads.

##### Worker

The actual processing nodes executing the `WorkLoad`. Workers could run independently or collectively.

##### SubMaster

- Optional, but recommended for scenarios with complex orchestration needs.
- Manages a group of `Worker` instances, providing additional orchestration and management capabilities.
- Handles job-specific logic, such as dynamic resource allocation and workload balancing.

#### ElasticBackend (example for Backend)

##### ElasticMaster

- Manages one elastic training process with lots of `ElasticWorker` instances.
- Performs node health checks for the elastic training.
- Provides rendezvous services for worker coordination.
- Monitors training progress and collects relevant metrics.

##### ElasticWorker (specialized Worker for elastic training)

- The worker nodes that execute elastic training workloads.
- Created by the `PrimeMaster` and managed by the `ElasticMaster`.

### Sequence Diagram

From the perspective of `PrimeMaster`, both `SubMaster` and `Worker` are treated as `Actor` instances, implemented using
`ray.Actor`.

```mermaid
%% @formatter:off
sequenceDiagram
    autonumber
    participant Driver
    box Prime Master
        participant PrimeMaster
        participant PrimeManager
        participant Scheduler
    end

    Driver ->> PrimeMaster: create(config)
    PrimeMaster ->> PrimeManager: create(config)
    note over PrimeManager: INIT

    Driver ->> PrimeMaster: start()
    PrimeMaster ->> PrimeManager: prepare()
    activate PrimeManager
    PrimeManager ->> Scheduler: allocate_placement_group()
    PrimeManager ->> Scheduler: create_actors()
    create participant Workers
    Scheduler ->> Workers: create()
    note over Workers: INIT
    Scheduler ->> Workers: status() [Ping]
    Workers -->> PrimeMaster: get_nodes_by_role()
    loop Wait all actors Ready
        PrimeManager -->> Workers: status()
    end
    note over Workers: READY
    PrimeManager ->> Workers: check_child() [SubMaster]
    note over PrimeManager: READY
  
    PrimeMaster ->> PrimeManager: start()
    PrimeManager ->> Workers: start() [Trainer/SubMaster]
    note over Workers: RUNNING
    note over PrimeManager: RUNNING
    PrimeMaster -->> Driver: 
    note right of Driver: Running

    loop Monitor() while RUNNING
        PrimeManager ->> Workers: status()
    end
    note over Workers: FINISH/FAILED

    note over PrimeManager: STOPPING
    deactivate PrimeManager
    PrimeManager ->> PrimeManager: stop()
    PrimeManager ->> Workers: stop()
    note over PrimeManager: STOPPED
%% @formatter:on
```

### Extension Points

The unified training framework is designed with extensibility in mind, enabling users to tailor its functionality to
accommodate diverse workloads. Key extension points include:

- **Worker Customization:**  
  Users can implement custom `WorkerLoad` classes to handle specialized tasks such as data preprocessing, reinforcement
  learning, or other domain-specific operations. This flexibility allows seamless integration of varied processing
  strategies within the unified training pipeline.

- **Custom Backend(Advanced):**  
  Users may define their own `SubMaster` and corresponding `Worker` implementations to orchestrate specific training
  workflows. This supports fine-grained control over elastic workload management and enables backend customization for
  specialized use cases.

The `Worker` and `SubMaster` components are orchestrated by the `PrimeManager` via stages and lifecycle hooks, which
govern the
initialization, execution, and monitoring of each workload. This hook-based design empowers users to extend the
framework with new processing paradigms—without altering the core architecture.

### Common Lifecycle Hooks

- `__init__`: Initializes the node with the provided configuration.
- `status`: Retrieves the current runtime status of the node.
- `check_child`: Monitors the status of child nodes (applicable to `SubMaster`).
- `start`: Launches the node’s processing logic (e.g., training loop, monitoring routine).

### Core Lifecycle Stages

- `INIT`: The initial stage of the actor lifecycle.
- `READY`: The actor has completed its setup and is ready to handle RPC requests.
- `RUNNING`: Indicates that the task is actively running; set after the `start` hook is invoked.
- `FINISH` / `FAILED`: Terminal stages that signify the completion or failure of the task.

```mermaid
stateDiagram-v2
    [*] --> INIT: __init__
    INIT --> READY: _setup, _self_check, ...
    READY --> RUNNING: check_workers, rendezvous, ...
    RUNNING --> FINISH: 
    RUNNING --> FAILED: task error
FINISH --> [*]
FAILED --> [*]
```

### Stability

#### Pre-Check

- Each worker performs a self-check before entering the READY state to ensure it can handle assigned tasks.
- The SubMaster (e.g., `ElasticMaster`) runs `check_workers` to verify that all workers are prepared before starting the
  training process. For example, the `ElasticMaster` conducts a rendezvous and communication check prior to launching
  elastic training.

#### Fault Tolerance

The unified training framework is designed with robust fault tolerance to maintain reliability throughout the training
lifecycle. Key mechanisms include:

- **Node Health Monitoring:** All nodes undergo regular health checks to promptly detect failures. If a node becomes
  unresponsive, the framework reallocates its tasks to healthy nodes.
- **Dynamic Node Management:** Nodes can be added or removed dynamically based on workload demands. Upon node failure,
  the `PrimeManager` reassigns tasks to available nodes, ensuring uninterrupted operation.
- **Transactional State Management:** The global training state is managed transactionally by the `PrimeManager`,
  enabling recovery from failures without loss of progress. The `PrimeMaster` can fail over to a backup if necessary.
  Nodes are stateless and can recover by reloading their state from the `PrimeManager`.

#### Failover

The framework provides comprehensive failover strategies to ensure operational continuity:

- If an `ElasticWorker` fails during `node_check`, the `ElasticMaster` restarts all abnormal workers and retries
  `node_check` until all workers are ready.
- If an `ElasticWorker` fails during elastic training, the `ElasticMaster` stops all running workers, performs
  `node_check`, and restarts the training process.

## Driving Patterns

There are three different driving patterns for the unified training framework:

- SubMaster Driven: The `PrimeManager` drives the `SubMaster` and its workers, which are specialized for elastic
  training.
- Worker Self-Loop: The `Worker` nodes operate in a self-driven loop, continuously pulling data, processing it, and
  writing results to a `DataChannel`.
- Trainer Driven: The `Trainer` orchestrates the training process, coordinating interactions among various roles such as
  `Actor`, `Critic`, and `Rollout` in reinforcement learning scenarios.

### Elastic Training (SubMaster Driven)

Elastic training is a core feature of the DLRover framework, enabling dynamic scaling of training resources based on
workload demands. Within the unified training architecture, elastic training is seamlessly integrated and managed by the
`PrimeMaster`, which oversees the orchestration and lifecycle management of all elastic workloads.

The `PrimeMaster` coordinates the creation, preparation, and execution of elastic training jobs, delegating the
management of elastic-specific processes to the `ElasticMaster` (also referred to as `SubMaster`). The `ElasticMaster`
is responsible for:

- Performing regular node health checks to ensure the reliability of the training cluster.
- Providing rendezvous services to facilitate coordination and communication among elastic workers.
- Monitoring training progress and collecting relevant metrics for adaptive scaling and fault tolerance.

The sequence diagram below illustrates the flow of elastic training within the unified framework:
> Communication between the `ElasticMaster` and `ElasticWorker` remains consistent with previous designs. This document
> focuses primarily on the unified architecture and simplifies the communication details.

```mermaid
%% @formatter:off
sequenceDiagram
    autonumber
    actor Driver
    box Prime Master
        participant PrimeMaster
        participant PrimeManager
        participant Scheduler
    end

    Driver ->> PrimeMaster: create(config)
    PrimeMaster ->> PrimeManager: create(config)
    note over PrimeManager: INIT
    Driver ->> PrimeMaster: start()
    PrimeMaster ->> PrimeManager: prepare()
    PrimeManager ->> Scheduler: allocate_placement_group()
    PrimeManager ->> Scheduler: create_nodes()

    create participant ElasticMaster
    Scheduler ->> ElasticMaster: create()
    create participant ElasticWorkers
    Scheduler ->> ElasticWorkers: create()

    Scheduler ->> ElasticMaster: status() [Ping]
    Scheduler ->> ElasticWorkers: status() [Ping]
    ElasticMaster -->> PrimeMaster: get_nodes_by_role()
    ElasticMaster ->> ElasticMaster: self_check()
    ElasticWorkers ->> ElasticWorkers: self_check()
    loop Wait all actors Ready
        PrimeManager -->> ElasticMaster: status()
        PrimeManager -->> ElasticWorkers: status()
        note over ElasticMaster, ElasticWorkers: READY
    end


    PrimeManager ->> ElasticMaster: check_child() [SubMaster]
    ElasticMaster ->> ElasticWorkers: do_node_check()
    note over PrimeManager: READY
    
    PrimeMaster ->> PrimeManager: start()
    PrimeManager ->> ElasticMaster: start() [SubMaster]
    ElasticMaster ->> ElasticWorkers: start_elastic_job() [Run User Script]
    note over ElasticWorkers,ElasticMaster: RUNNING
    activate ElasticWorkers
    activate ElasticMaster
    note over PrimeManager: RUNNING

    PrimeMaster -->> Driver: 
    note right of Driver: Running


    note over PrimeMaster, ElasticWorkers: === Started ===


    loop RUNNING
        ElasticMaster ->> ElasticWorkers: status()
        ElasticWorkers -->> ElasticMaster: status() == RUNNING
        PrimeManager ->> ElasticMaster: status()
        PrimeManager ->> ElasticWorkers: status()
    end


    ElasticWorkers -->> ElasticMaster: rendezvous()

    activate PrimeManager
    loop while RUNNING
        PrimeManager ->> PrimeManager: monitor()
        PrimeManager ->> ElasticMaster: status()
    end


    note over PrimeMaster, ElasticWorkers: === Finish ===

    note over ElasticWorkers: FINISH
    deactivate ElasticWorkers
    ElasticMaster -->> ElasticWorkers: status()
    note over ElasticMaster: FINISH
    PrimeManager -->> ElasticWorkers: status()
    PrimeManager -->> ElasticMaster: status()
    note over PrimeManager: STOPPING
    PrimeManager ->> Scheduler: cleanup()
    Scheduler ->> ElasticMaster: stop()
    Scheduler ->> ElasticWorkers: stop()
    note over PrimeManager: STOPPED
    deactivate PrimeManager
%% @formatter:on
```

### Distributed Data Processing (Workers Self-Loop)

In distributed data processing scenarios, each `Worker` node operates in a self-driven loop. Upon starting, the worker
continuously pulls data from a `DataChannel`, processes the data using a user-defined `process` method, and writes the
results to another `DataChannel`. This loop persists until a stop signal is received, enabling efficient and scalable
parallel data processing across multiple workers.

Source -> Tokenizer[Self-Loop] -> DataChannel -> Sampler[Self-Loop] -> DataChannel -> Trainer[Self-Loop] -> Model Output

```mermaid
flowchart TD
    subgraph Tokenizer
        B1[Pull Data]
        B2[Process Data]
        B3[Write to DataChannel]
        B1 --> B2 --> B3 --> B1
    end
    subgraph Sampler
        D1[Pull Data]
        D2[Process Data]
        D3[Write to DataChannel]
        D1 --> D2 --> D3 --> D1
    end
    subgraph Training
        F1[Pull Data]
        F2[Process Data]
        F1 --> F2 --> F1
    end

    Source[(Source)]
    DataChannel1[(DataChannel1)]
    DataChannel2[(DataChannel2)]
    Source --> B1
    B3 --> DataChannel1
    DataChannel1 --> D1
    D3 --> DataChannel2
    DataChannel2 --> F1
    Start((Start)) --> Tokenizer
    Start --> Sampler
    Start --> Training
```

### Reinforcement Learning (Trainer Driven)

In reinforcement learning (RL) scenarios, the unified framework supports multiple specialized roles such as `Actor`,
`Critic`, and `Rollout`. The `Actor` and `Critic` are typically managed as elastic training tasks, while `Rollout` nodes
handle inference or environment simulation. The overall training process is orchestrated by a dedicated `Trainer` or by
the `Actor` itself, which coordinates the interactions among all roles. This design enables flexible scaling and
efficient resource allocation for complex RL workflows, supporting scenarios like distributed policy optimization,
multi-agent training, and large-scale environment simulation.

```mermaid
flowchart TD
    Start((Start))
    Start --> Trainer

    subgraph Trainer
        T1[Collect Experience Data]
        T2[Calculate Value]
        T3[Invoke Actor Training]
        T4[Invoke Critic Training]
        T5[Update Actor Weights]
        Buffer[Experience Buffer]
        T1 --> Buffer --> T2 --> T3 --> T5 --> T1
        Buffer --> T4
    end

    subgraph Rollout
        R1[Rollout Environment Simulation]
        R2[Collect Experience Data]
        R1 --> R2
    end

    R2 -.-> T1

    subgraph Actor
        actor_in(Consume Experience Data)
        actor_fit[Actor Training]
        actor_weight[Actor Weight]
        actor_in --> actor_fit --> actor_weight
    end

    subgraph Critic
        critic_in(Consume Experience Data)
        critic_fit[Critic Training]
        critic(Evaluate Value Function)
        critic_in --> critic_fit -.-> critic
    end

    T3 -.-> actor_in
    T2 -.-> critic
    T4 -.-> critic_in
    actor_weight -.-> T5 -.-> R1
    critic -.-> T3
```

## FAQ

### What is the difference between `Trainer` , `PrimeMaster` and `SubMaster`?

- The `Trainer` is the main orchestrator for the entire training process, coordinating interactions among various roles such as
`Actor`, `Critic`, and `Rollout`. It manages the overall training logic, including data collection, model updates, and
performance monitoring.
- The `PrimeMaster` is a higher-level orchestrator that manages the lifecycle of the entire unified training job, but it does not
directly handle the training logic. Instead, it oversees the `SubMaster` and its workers, which are specialized for elastic
training tasks. The `PrimeMaster` is responsible for job scheduling, resource allocation, and global state management.
- The `SubMaster`, on the other hand, is a specialized component that manages one role, providing
additional orchestration and management capabilities, like rendezvousing and fault tolerance.
