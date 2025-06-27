## Background

As machine learning workloads become increasingly complex, there is a growing need to support hybrid training paradigms that combine elastic training with other processing strategies. Traditional elastic training frameworks are optimized for scaling model training dynamically, but often lack the flexibility to integrate with heterogeneous tasks such as data preprocessing, reinforcement learning environments, or custom compute-intensive operations.

Hybrid training addresses these challenges by enabling the orchestration of diverse workloads within a unified framework. This approach allows for seamless coordination between elastic model training and auxiliary tasks, such as large-scale data transformation, environment simulation, or multi-stage pipelines. By managing the lifecycle and resource allocation for both elastic and non-elastic components, hybrid training frameworks can maximize resource utilization, improve throughput, and simplify the development of complex machine learning systems.

Typical scenarios include:

- **Reinforcement Learning (RL) Environments:** Multiple training tasks may need to run in parallel, with environment simulation requiring significant CPU resources and coordination with model training.
- **Data Fusion and Preprocessing:** Integrating data processing with model training, where various preprocessing or transformation tasks must be performed alongside or prior to training, often involving different computation patterns and resource requirements.

By supporting these hybrid scenarios, the framework enables users to build more flexible, efficient, and scalable machine learning pipelines that can adapt to a wide range of real-world requirements.

This document outlines the design and architecture of a hybrid training framework that integrates elastic training with other processing paradigms, providing a comprehensive solution for complex machine learning workloads.

> This is also an architecture design for [Unified MPMD Control](./unified-mpmd-control-proposal.md)

## Core Design

### Architecture Overview

![General Architecture Diagram](../figures/hybrid/architecture.excalidraw.svg)

The hybrid training framework integrates both elastic and non-elastic training paradigms to support diverse workloads. In this architecture, the `PrimeMaster` is responsible for orchestrating the entire hybrid training process, while the `ElasticMaster` and `ElasticWorker` components focus on elastic training tasks.
> The `ElasticMaster` and `ElasticWorker` represent the original elastic training architecture, while the `PrimeMaster` is the newly introduced component.

The architecture is designed to be modular and extensible, allowing for the integration of various training strategies and workloads. The key components include:

**PrimeMaster:**

- Oversees the lifecycle management of hybrid training jobs.
- Schedules and monitors various training/processing workloads.
- Maintains a registry of all nodes and their assigned roles.
- Stores and manages the global state of the training process.

**ElasticMaster (one example of SubMasters):**

- Manages one elastic training process with lots of `ElasticWorker` instances.
- Performs node health checks for the elastic training.
- Provides rendezvous services for worker coordination.
- Monitors training progress and collects relevant metrics.

**ElasticWorker (specialized Worker for elastic training):**

- The worker nodes that execute elastic training workloads.
- Created by the `PrimeMaster` and managed by the `ElasticMaster`.

**Worker:**

- Represents general processing nodes. Could execute various workloads, such as data preprocessing, training trainer, or custom compute operations.

### Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    participant Driver
    box Prime Master
        participant PrimeMaster
        participant PrimeManager
        participant Scheduler
    end

    Driver ->> PrimeMaster : create(config)
    PrimeMaster ->> PrimeManager : create(config)
    Driver ->> PrimeMaster : start()
    PrimeMaster ->> PrimeManager : prepare()
    activate PrimeManager
    PrimeManager ->> PrimeManager : Placement.allocate()
    PrimeManager ->> Scheduler : create_nodes()
    create participant Workers # Rename
    Scheduler ->> Workers : create()
    Scheduler ->> Workers : status() [Ping]
    Workers -->> PrimeMaster : get_nodes_by_role()
    PrimeManager ->> PrimeManager : Nodes Check
    PrimeManager ->> Workers : self_check()
    PrimeManager ->> Workers : check_child() [SubMaster]
    PrimeMaster ->> PrimeManager : start()
    PrimeManager ->> Workers : start() [Trainer/SubMaster]
    PrimeMaster -->> Driver : 
    note right of Driver : Running

    loop RUNNING
        PrimeManager ->> PrimeManager : monitor()
        PrimeManager ->> Workers : status()
    end

    deactivate PrimeManager
    PrimeManager ->> PrimeManager : stop()
    PrimeManager ->> Workers : stop()
```

### Entension Points

The hybrid training framework is designed to be extensible, allowing users to customize and extend its functionality to suit various workloads. The key extension points include:

- **Worker Customization:** Users can implement custom `WorkerLoad` classes to handle specific workloads, such as data preprocessing or reinforcement learning tasks. This allows for flexible integration of different processing strategies within the hybrid training framework.
- **Custom SubMaster:** Custom `SubMaster` implementations can be created to manage specific training processes, enabling tailored orchestration of elastic workloads.

The `Worker/SubMaster` is driven by `PrimeManager` through lifecycle hooks, allowing it to manage the lifecycle of each workload. This design enables users to easily extend the framework to support new processing paradigms without modifying the core architecture.

Common Hooks include:

- `__init__`: Initialize the node with configuration.
- `status`: Get the running status of the node.
- `self_check`: Perform self-checks to ensure the node is healthy.
- `check_child`: Check the status of child nodes (for SubMaster).
- `start`: Start the node's processing. (e.g., start training loop or monitoring)

### 稳定性

#### Pre check

#### Fault Tolerance

The hybrid training framework incorporates fault tolerance mechanisms to ensure robustness and reliability during training. Key features include:

- **Node Health Checks:** Regular health checks are performed on all nodes to detect failures early. If a node becomes unresponsive, the framework can reallocate tasks to healthy nodes.
- **Dynamic Node Management:** The framework can dynamically add or remove nodes based on workload requirements. If a node fails, the `PrimeManager` can reassign its tasks to other available nodes, ensuring continuous operation.
- **Transactional State Management:** The global state of the training process is managed in a transactional manner by `PrimeManager`, allowing for recovery from failures without losing progress. `PrimeMaster` could failover to another `PrimeMaster` if needed. `Node` is designed to be stateless, could recover by reloading its state from `PrimeManager`.

## Usage Patterns

There are three different driving patterns for the hybrid training framework:

- SubMaster Driven: The `PrimeManager` drives the `SubMaster` and its workers, which are specialized for elastic training.
- Worker Self-Loop: The `Worker` nodes operate in a self-driven loop, continuously pulling data, processing it, and writing results to a `DataChannel`.
- Trainer Driven: The `Trainer` orchestrates the training process, coordinating interactions among various roles such as `Actor`, `Critic`, and `Rollout` in reinforcement learning scenarios.

### Elastic Training (SubMaster Driven)

Elastic training is a core feature of the DLRover framework, enabling dynamic scaling of training resources based on workload demands. Within the hybrid training architecture, elastic training is seamlessly integrated and managed by the `PrimeMaster`, which oversees the orchestration and lifecycle management of all elastic workloads.

The `PrimeMaster` coordinates the creation, preparation, and execution of elastic training jobs, delegating the management of elastic-specific processes to the `ElasticMaster` (also referred to as `SubMaster`). The `ElasticMaster` is responsible for:

- Performing regular node health checks to ensure the reliability of the training cluster.
- Providing rendezvous services to facilitate coordination and communication among elastic workers.
- Monitoring training progress and collecting relevant metrics for adaptive scaling and fault tolerance.

The sequence diagram below illustrates the flow of elastic training within the hybrid framework:
> Communication between the `ElasticMaster` and `ElasticWorker` remains consistent with previous designs. This document focuses primarily on the hybrid architecture and simplifies the communication details.

```mermaid
sequenceDiagram
    autonumber
    actor Driver
    box Prime Master
        participant PrimeMaster
        participant PrimeManager
        participant Scheduler
    end

    Driver ->> PrimeMaster : create(config)
    PrimeMaster ->> PrimeManager : create(config)
    Driver ->> PrimeMaster : start()
    PrimeMaster ->> PrimeManager : prepare()
    PrimeManager ->> PrimeManager : Placement.allocate()
    PrimeManager ->> Scheduler : create_nodes()

    create participant ElasticMaster
    Scheduler ->> ElasticMaster : create()
    create participant ElasticWorkers
    Scheduler ->> ElasticWorkers : create()

    Scheduler ->> ElasticMaster : status() [Ping]
    Scheduler ->> ElasticWorkers : status() [Ping]
    ElasticMaster -->> PrimeMaster : get_nodes_by_role()
    PrimeManager ->> PrimeManager : Nodes Check
    PrimeManager ->> ElasticMaster : self_check()
    PrimeManager ->> ElasticWorkers : self_check()
    PrimeManager ->> ElasticMaster : check_child() [SubMaster]
    ElasticMaster ->> ElasticWorkers : do_node_check()
    PrimeMaster ->> PrimeManager : start()
    PrimeManager ->> ElasticMaster : start() [SubMaster]
    ElasticMaster ->> ElasticWorkers : start_training() [Run User Script]

    activate ElasticWorkers
    activate ElasticMaster

    PrimeMaster -->> Driver : 
    note right of Driver : Running
    ElasticWorkers -->> ElasticMaster : rendezvous()

    activate PrimeManager
    loop while RUNNING
        PrimeManager ->> PrimeManager : monitor()
        PrimeManager ->> ElasticMaster : status()
    end
    ElasticWorkers -->> ElasticMaster : finish_job()
    deactivate ElasticWorkers
    ElasticMaster -->> PrimeManager : status() == Finish
    deactivate ElasticMaster

    PrimeManager ->> PrimeManager : stop()
    deactivate PrimeManager
    PrimeManager ->> Scheduler : cleanup()
    Scheduler ->> ElasticMaster : stop()
    Scheduler ->> ElasticWorkers : stop()
```

### Distributed Data Processing (Workers Self-Loop)

In distributed data processing scenarios, each `Worker` node operates in a self-driven loop. Upon starting, the worker continuously pulls data from a `DataChannel`, processes the data using a user-defined `process` method, and writes the results to another `DataChannel`. This loop persists until a stop signal is received, enabling efficient and scalable parallel data processing across multiple workers.

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

In reinforcement learning (RL) scenarios, the hybrid framework supports multiple specialized roles such as `Actor`, `Critic`, and `Rollout`. The `Actor` and `Critic` are typically managed as elastic training tasks, while `Rollout` nodes handle inference or environment simulation. The overall training process is orchestrated by a dedicated `Trainer` or by the `Actor` itself, which coordinates the interactions among all roles. This design enables flexible scaling and efficient resource allocation for complex RL workflows, supporting scenarios like distributed policy optimization, multi-agent training, and large-scale environment simulation.

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

## Implementation Design

### Lifecycle Management

```mermaid
sequenceDiagram
    autonumber

    participant PrimeMaster
    participant ElasticMaster
    participant ElasticWorker
    participant ElasticWorkerRunner

    # Start

    PrimeMaster ->> ElasticMaster : start()
    ElasticMaster ->> ElasticWorker : start_elastic_training()
    ElasticWorker ->> ElasticWorkerRunner : run()
    activate ElasticWorkerRunner #running
    note over ElasticWorkerRunner : Running
    activate ElasticWorker #wait run()
    note over ElasticWorker : Thread(wait run())

    ElasticWorker -->> ElasticMaster : status() == RUNNING
    activate ElasticMaster #monitoring
    note over ElasticMaster : Thread(Monitor Workers)
    
    ElasticWorker -->> PrimeMaster : status() == RUNNING
    ElasticMaster -->> PrimeMaster : status() == RUNNING
    activate PrimeMaster #monitoring
    note over PrimeMaster : Thread(Monitor Actors)

    # Monitoring

    loop RUNNING
        ElasticMaster ->> ElasticWorker : status()
    end
    loop RUNNING
        PrimeMaster ->> ElasticMaster : status()
        PrimeMaster ->> ElasticWorker : status()
    end

    # End

    ElasticWorkerRunner -->> ElasticWorker : end run()
    deactivate ElasticWorkerRunner
    ElasticWorker -->> ElasticMaster : status() == FINISH
    deactivate ElasticWorker
    ElasticMaster -->> PrimeMaster : status() == FINISH
    deactivate ElasticMaster

    PrimeMaster ->> PrimeMaster : stop()
    deactivate PrimeMaster


```
