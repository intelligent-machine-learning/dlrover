# 03B. Multi-Role Training Example: verl

This guide demonstrates how to express and submit a multi-role training job using the verl example. The walkthrough is based on `examples/unified/rl/verl/main.py` and related files, and is designed to complement the OpenRLHF tutorial.

**Paradigm Note:**
This verl-example demonstrates the Proxy/WorkerGroup paradigm, where the trainer interacts with workers via group proxies rather than explicit remote_call RPCs. This is an alternative to the remote_call (RPC) paradigm used in the OpenRLHF example.

## Overview

Multi-role training allows you to orchestrate several specialized workloads—such as data generators, trainers, and evaluators—each with its own resource requirements and entry point. By composing these roles into a unified distributed configuration, you can flexibly scale and manage complex reinforcement learning workflows.

In the verl example, the main components are:

- **Submitter**: Parses command-line arguments, assembles workload descriptors, constructs the job configuration, and launches the control plane.
- **Trainer**: Manages the main training loop, aggregates experiences, performs optimization steps, and handles checkpointing.
- **Worker Roles**: Implements specialized tasks such as data generation, environment simulation, or evaluation.

## Directory Structure

- `main.py`: Entrypoint for submitting the multi-role job.
- `workers.py`: Implements worker roles (e.g., actor, critic, reward model).
- `util.py`: Shared utilities for worker and group management.
- `run.sh`: Script to run `main.py` with appropriate arguments.

You can combine DLRover with other algorithm implementations using the same paradigm.

## Workload Declaration and Submission (`main.py`)

The core logic for declaring and submitting workloads is found in `main.py`. The typical workflow involves:

1. **Job Builder Initialization**: Create a job builder and configure it using parsed arguments.
2. **Role Definition**: Define each role (trainer, worker, etc.), set resource requirements, and specify entry points.
3. **Job Submission**: Build the job and submit it to the control plane.

Example (simplified):

```python
# examples/unified/rl/verl/main.py
@hydra.main(
    config_path="pkg://verl/trainer/config",
    config_name="ppo_trainer",
    version_base=None,
)
def main(config):
    # 1. Ensure the config is resolved, raise early if not
    OmegaConf.resolve(config)

    nodes = config.trainer.nnodes
    gpus = config.trainer.n_gpus_per_node

    # 2. Initialize the job builder
    builder = (
        DLJobBuilder()
        .global_env(
            {
                # "VLLM_USE_V1": "1",
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                "VLLM_LOGGING_LEVEL": "DEBUG",
            }
        )
        .node_num(nodes)
        .device_per_node(gpus)
        .config(config)
    )
    # 3. Define roles and their resource requirements
    # train() for workloads with Rendezvous, while run() for other workloads.
    builder.role("actor_rollout").train("workers.ActorWorker").total(gpus)
    builder.role("critic").train("workers.CriticWorker").total(gpus)
    if config.reward_model.enable:
        builder.role("rm").train("workers.RMWorker").total(gpus)
    if (
        config.algorithm.use_kl_in_reward
        or config.actor_rollout_ref.actor.use_kl_loss
    ):
        builder.role("ref").train("workers.ActorWorker").total(gpus)
    builder.role("trainer").run("workers.Trainer").resource(cpu=4)
    # 4. Share gpu
    builder.with_collocation_all("trainer")

    # 5. Build the job
    job = builder.build()
    for workload in job.workloads.values():
        if workload.backend == "elastic":
            # workload.comm_pre_check = False
            # veRL will setup process group itself, DLRover provide envs
            workload.comm_auto_setup_process_group = False
    pprint(job)

    # 6. Submit the job, launch training
    job.submit(job_name="dlrover-verl-ppo")
```

## Trainer & Runtime Communication

After submission, DLRover launches all roles and invokes their entry points. Communication between the trainer and workers is managed through the `MyWorkerGroup` abstraction, which combines DLRover's `RoleGroup` and verl's `WorkerGroup` to provide distributed RPC and lifecycle management.

Each worker role (such as `ActorWorker`, `CriticWorker`, `RMWorker`) inherits from `BaseWorker`. This base class automatically exports all relevant RPC methods from the underlying verl worker core (using `export_verl_worker_rpc`), and provides a standard `job_end` RPC and `run` method for lifecycle control.

The trainer sets up the workflow by instantiating a `MyWorkerGroup` for each role and initializing them in parallel. For example, during initialization, the trainer calls `init_model` on all worker groups asynchronously, waits for their completion, and then creates the rollout group. This approach enables efficient parallel setup and resource management.

Role definitions:

```python
class ActorWorker(BaseWorker): ...
class CriticWorker(BaseWorker): ...
class RMWorker(BaseWorker): ...
class Trainer: ...
```

Trainer initialization (simplified):

```python
# In workers.py, Trainer.prepare()
trainer.critic_wg = MyWorkerGroup("critic", verl_workers.CriticWorker)
trainer.ref_policy_wg = MyWorkerGroup("ref", verl_workers.ActorRolloutRefWorker)
trainer.rm_wg = MyWorkerGroup("rm", verl_workers.RewardModelWorker)
trainer.actor_rollout_wg = MyWorkerGroup("actor_rollout", verl_workers.AsyncActorRolloutRefWorker)

# Initialize workers in parallel (use RoleGroup.call, non-blocking)
async_init = [
    trainer.critic_wg.call("init_model"),
    trainer.ref_policy_wg.call("init_model"),
    trainer.rm_wg.call("init_model"),
]
[it.result() for it in async_init]

# Initialize rollout group (use WorkerGroup proxy, blocking)
trainer.actor_rollout_wg.init_model()
```

During training, the trainer interacts with workers using group calls such as `call`, `call_rank0`, or other custom methods. This abstraction allows the trainer to broadcast commands, collect results, and synchronize state across all distributed roles efficiently.

At the end of training, the trainer signals all workers to terminate by invoking the exported `job_end` method:

```python
notify_job_end("critic", "ref", "rm", "actor_rollout")
```

This design enables flexible orchestration and robust communication between distributed components, leveraging the strengths of both DLRover and verl abstractions.

## Customization and Extension

You can extend the verl example by adding new roles, customizing resource requirements, or implementing additional RPC methods. The pattern remains consistent: define roles, implement their logic, and orchestrate communication via RPC.

## Summary

The verl example illustrates how to use DLRover to submit and manage multi-role training jobs. By following this pattern, you can build scalable, distributed RL workflows tailored to your specific requirements.
