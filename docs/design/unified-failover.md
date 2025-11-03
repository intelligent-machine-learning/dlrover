# Unified Failover Design & Scenarios

## Overview

This document describes the failover (fault tolerance) mechanisms in DLRover 
Unified, a distributed training system. It covers the core logic, component interactions, 
and key scenarios for handling failures at different levels (worker, submaster, 
node, and PrimeMaster). The structure is as follows:

- **Core Failover Logic:** Main decision flow for handling actor failures.
- **Worker Failover:** How different types of workers (with/without SubMaster) and SubMaster recover from failures.
- **PrimeMaster Failover:** Self-healing and state management of the global controller.
- **Node Relaunch:** Multi-component process for replacing faulty nodes and restarting all affected actors.

Each section includes diagrams and concise explanations to help understand the 
system's robustness and recovery strategies.

---

## Core Failover Logic: deal_with_actor_restarting

When a worker (actor) fails in DLRover Unified, the system starts the failover 
process from Ray reconstructing the worker and the worker reporting a RESTARTED 
event. The manager then determines the recovery strategy based on the failure context:

- If the failure is due to node relaunch (the node is in removed_nodes), reset the per-node failure count and do not treat it as a failure.
- Otherwise, update the corresponding failure record.
- If the per-node failure count exceeds its configured limit, trigger node relaunch to replace the faulty node (then return to failover handling).
- Otherwise, proceed to failover handling (such as restarting the job or recovering running state). Role-specific handling (e.g., SubMaster, role-level failover, restart count limit) is managed in subsequent steps.

```mermaid
flowchart TD
    A[Worker Failure] --> B[Ray Reconstructs Worker]
    B --> C[Worker Reports RESTARTED]
    C --> D{Is Influenced by Node Relaunch?}
    D -- Yes --> E[Reset per-node failure count]
    E --> K[Proceed to Failover Handling]
    D -- No --> F[Update Failure Info]
    F --> G{Should Trigger Node Relaunch?}
    G -- Yes --> H[Trigger Node Relaunch]
    H --> K
    G -- No --> K[Proceed to Failover Handling]
```

This logic ensures node-level limits are enforced, and failover is triggered 
only when necessary. Role-specific recovery and restart count limits are handled 
in later steps.

---

## Worker Failover

DLRover Unified handles worker failures via automatic restart or job termination 
when restart limits are reached.

### 1. ElasticWorker (with SubMaster)

If the failed worker belongs to a role with SubMaster (ElasticMaster), 
role-level failover is supported. The failover handling is:

- ElasticWorker detects failure and reports RESTARTED to PrimeMaster.
- PrimeMaster requests SubMaster to perform role-level failover and restart the affected actors.
- PrimeMaster increments restart counts and checks if any actor exceeds the restart limit; if so, the job is stopped to prevent infinite retries.
- If within limits, PrimeMaster restarts all workers and waits for them to be ready, ignoring intermediate restart reports.
- Once all workers are ready, SubMaster starts the elastic job and notifies PrimeMaster that role-level failover is complete.
- The process supports deduplication and multiple recovery attempts, ensuring consistency and high availability.

```mermaid
sequenceDiagram
    participant EW as ElasticWorker
    participant SM as SubMaster
    participant PM as PrimeMaster
    activate EW
    EW->>PM: Report restarted (failure)
    PM->>SM: Request Role-level Failover
    activate SM
    SM->>PM: restart_actors(actors)
        activate PM
        note over PM: Increment restart count, check limit<br> (If any exceed limit, stop job)
        PM->>EW: restart all workers
        EW->>PM: Report restarted (kill)
        note over PM: Ignore report, as it's restarting
        PM->>EW: Setup & Wait Ready
        EW-->>PM: Ready
        PM-->>SM: finish restart_actors
        deactivate PM
    SM->>EW: start_elastic_job
    EW-->>SM: Job Started
    deactivate SM
    SM-->>PM: Role-level Failover Done
    deactivate EW
```

This process ensures that role-level failover is coordinated by SubMaster and 
controlled by PrimeMaster, with restart limits enforced to prevent infinite 
retries. SubMaster always restores its own state before actor recovery.

### 2. Other Worker (no SubMaster)

If the failed worker does not belong to a role with SubMaster, role-level failover is not supported:

- Worker detects failure and reports RESTARTED to PrimeMaster.
- PrimeMaster initiates job-level restart: rolls back stage, cancels monitoring task, and calls `restart_actors` for all workers.
- Reports from actors during non-RUNNING stage are ignored.
- After all actors are ready, PrimeMaster re-enters RUNNING stage and restarts job monitoring.

```mermaid
sequenceDiagram
    participant W as Worker
    participant PM as PrimeMaster
    W->>PM: Report restarted (failure)
    PM->>PM: Rollback stage, cancel monitoring
    PM->>W: restart_actors (all workers)
    W->>PM: Report restarted (kill)
    note over PM: Ignore report, as not RUNNING
    PM->>W: Setup & Wait Ready
    W-->>PM: Ready
    PM->>PM: (re)start job to RUNNING stage
```

This process ensures that job-level failover is managed by PrimeMaster, with all s
tate transitions and actor restarts coordinated for global consistency.

### 3. SubMaster Failover

When SubMaster (such as: ElasticMaster) fails, the system ensures recovery and 
role consistency. SubMaster is uniformly designed as a stateless object, with 
all its internal state derived from the PrimeMaster's state management, 
allowing for direct retries:

```mermaid
flowchart TD
    C[SubMaster Reports RESTARTED] --> D[PrimeMaster Handles Event]
    D --> E[Re-setup Actors & Recover Running]
```

**Test Coverage:**

- `test_manager_handle_actor_restart`, `test_failover_training`, `test_comm_fault`, `test_request_stop_cases`

---

## PrimeMaster Failover

PrimeMaster is the unique, self-healing, and core stateful component in DLRover 
Unified. Its state save/load mechanism is critical for global job consistency 
and recovery:

- PrimeMaster maintains the authoritative job state and orchestrates all failover logic.
- On any state change, PrimeMaster saves its current state to persistent storage, ensuring that both itself and all workers are in a globally consistent state.
- When a failure occurs, PrimeMaster is reconstructed and loads the last saved state from persistent storage. This separation of Save and Load guarantees that job progress is never lost.
- After loading, PrimeMaster performs self-recover. Only if the restored stage is RUNNING (the long-lived stage), failover proceeds and job monitoring resumes. For other short-lived stages, or if any exception occurs during recovery, PrimeMaster safely terminates the job to avoid inconsistency.

This design ensures distributed jobs can always recover from failures in a 
consistent and reliable manner, with all exceptions and abnormal branches leading 
to a safe stop.

```mermaid
flowchart TD
    subgraph save[Save State]
        A[PrimeMaster State Change] --> B[Save State]
    end
    X[PrimeMaster Reconstruct] --> Y[Load State] 
    save -.-> Y
    Y --> Z[Self-Recover]
    Z --> D{Check stage running?}
    D -- Yes --> E[Resume Job]
    D -- No --> F[PrimeMaster Stops Job]
```

**Test Coverage:**

- `test_manager_save_load`, `test_manager_failover`, `test_some_misc_cases`

---

## Node Relaunch

Node relaunch is a coordinated process managed by PrimeManager to maintain job 
reliability when a node's per-node failure count exceeds its configured limit. 
All actors on the same node—including Other Actor—are affected and restarted 
together. The process involves:

- Actor reports failure (RESTARTED) to PrimeManager.
- PrimeManager records the failure, checks per-node failure count, and updates removed_nodes if relaunch is needed.
- PrimeManager requests Extension to relaunch the node; Extension notifies all affected actors (including Other Actor).
- Extension returns relaunched nodes to PrimeManager, which updates node_restart_count and removed_nodes.
- After relaunch, all actors on the node report RESTARTED again; PrimeManager resets their per-node failure count.
- PrimeManager proceeds to failover handling for all affected actors, with deduplication to avoid repeated recovery.

```mermaid
sequenceDiagram
    box Node
        participant ACT as Actor
        participant OACT as Other Actor
    end
    participant PM as PrimeManager
    participant EXT as Extension
    ACT->>PM: Report failure (RESTARTED)
    Note over PM: Record failure, check per-node failure count > limit
    PM->>PM: update removed_nodes
    PM->>EXT: request relaunch_nodes_impl([NodeInfo])
    EXT-->>OACT: Also affected by relaunch
    EXT-->>PM: Return relaunched nodes
    PM->>PM: Update node_restart_count, removed_nodes
    ACT->>PM: Report RESTARTED (after relaunch)
    Note over PM: Reset per-node failure count
    OACT->>PM: Report RESTARTED (after relaunch)
    Note over PM: Reset per-node failure count
    Note over PM: Proceed to Failover Handling <br> (Failover Handling Should support deduplication)
```

This ensures that faulty nodes are replaced and all actors on the node are 
consistently restarted, minimizing disruption and maintaining global state 
consistency.
