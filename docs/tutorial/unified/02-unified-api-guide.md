# 02. Unified API Guide [Experimental]

This section focuses on the DLJobBuilder and submission patterns: how to
construct job configurations programmatically and submit them to the
DLRover control plane.

## Key flow

1. Create a DLJobBuilder instance.
2. Configure job-level settings (user configuration, node/device counts, device type).
3. Register workloads/entrypoints and resources.
4. Build the job object and submit.

## Minimal example

- Single role:

```python
from omegaconf import DictConfig
from dlrover.python.unified.api.builder import DLJobBuilder

args = {"data_dir": "data/nanogpt/", "batch_size": 16}

job = (
    DLJobBuilder()
    .config(DictConfig(args.__dict__))
    .node_num(2)
    .device_per_node(2)
    .device_type("GPU" if torch.cuda.is_available() else "CPU")
    .train("examples.unified.elastic.nanogpt.train.run")
    .nnodes(2)
    .nproc_per_node(2)
    .end() #optional, for type hinting
    .build()
)

job.submit(job_name="nanogpt")
```
- Single role(via CLI command): You can also initialize a single-role job by directly parsing a dlrover-run or
torchrun command string. This automatically configures nnodes, nproc_per_node, 
and the training entrypoint.
```python
from dlrover.python.unified.api.builder import DLJobBuilder

# Conveniently convert a CLI command into a Ray job
cmd = f"dlrover-run --nnodes=1 --nproc_per_node=1 {Your_dlrover_root_dir}/dlrover/python/unified/tests/integration_test/dummy_run.py --test 0"

job = DLJobBuilder().by_dlrover_run_cmd(cmd).build()

job.submit("test_cmd_api", master_cpu=1, master_memory=128)
```
### Advanced examples

- Multiple roles(outline):

```python
job = (
    DLJobBuilder()
    .node_num(worker_node_num)  # total machine number
    .device_per_node(device_per_worker_node)  # device number per machine
    .device_type("GPU")  # device type
    .config({})  # global training variables setting
    .global_env({"DLROVER_LOG_LEVEL": log_level})  # global environment variables setting
    .role(xxx).run(xxx)  # any workloads
      .resource(xxx)  # resource unit for this role group
      .total(xxx)  # total workloads number for this role group
      ...
    .role(xxx).run(xxx)  # any workloads
      ...
    .workload("role","entrypoint")  # any workloads in another way
      ...
    ...
    .build()
)
```

- Use RL as a example:

```python
job = (
    DLJobBuilder()
    .config(DictConfig(args))
    .role("trainer").train("module.trainer.run").total(2).resource(gpu=1).end()
    .role("rollout").run("module.rollout.run").total(4).resource(cpu=4).end()
    .workload("reward", "module.reward.run").total(1).resource(cpu=2).end()
    .build()
)
job.submit(job_name="rl_job")
```

(Use the exact `workload()` method names from the installed builder
version.)

## Common builder methods

- node_num(n) / device_per_node(n): how many nodes and devices per node.
- device_type("GPU"|"CPU"): preferred accelerator type.
- config(DictConfig or dict): user configuration available at runtime.
- role(str): defines the role name for multi-role jobs.
- run(entrypoint): define a non-training workload with entrypoint, and return a sub builder.
- workload(role, entrypoint): single method combine role + run
- train(entrypoint): define a training workload with entrypoint (module path + function or command with python file), and return a sub builder.
- by_dlrover_run_cmd(command_str): Parses a dlrover-run or torchrun command to set up a single-role training job.

### Workload / Role patterns

- Training workload: use `train()` to define the main training entrypoint.
- Non-training workload: use `run()` to define auxiliary entrypoints
  (data loader, evaluator, etc).
- For each workload(any kind), the following can be configured(major parameters):
  - total: Total number of the workload.
  - resource: Resource unit for each workload. Format in dict, supported resource: 'cpu','memory','disk','gpu','disk','user_defined'.
  - envs: Environment variables for each workload. Format in dict.
  - For more parameters, please refer to class: 'BaseWorkloadDesc'.

### Collocations

Use the following builder method to control the affinity and anti-affinity between workloads:
- with_collocation: Logical grouping (affinity) of workloads is determined by combining different roles.
- per_group(workload pattern): Specify the number of the current workloads within each logical group.


### Best practices

- Ensure entrypoint module paths are importable on workers (PYTHONPATH).
- Avoid heavy work at import-time; perform setup in `run()`/`__init__`.
- Use `DictConfig` for structured configs to ease overrides in CI/tests.
- Debug locally with small `nnodes` / `nproc_per_node` values.
- Pin critical dependency versions in CI to avoid runtime mismatches.

## Submission

Use `job.submit()` submits the job; runtime semantics (blocking vs non-blocking) depend on environment.

### Submission parameters

User can use the following environment variables to configure job submitting.

| Config                             | Environment Variable                                 | Default      | Note                                                                                      |
|------------------------------------|------------------------------------------------------| ------------ |-------------------------------------------------------------------------------------------|
| job_name                           | `DLROVER_UNIFIED_JOB_NAME`                           | dlrover-xxxx | Name of the job                                                                           |
| master_cpu                         | `DLROVER_UNIFIED_MASTER_CPU`                         | 2            | Number of CPU cores for the master node                                                   |
| master_mem                         | `DLROVER_UNIFIED_MASTER_MEM`                         | 4096 (in MB) | Amount of memory for the master node                                                      |
| master_create_timeout              | `DLROVER_UNIFIED_MASTER_CREATE_TIMEOUT`              | 600 (in s)   | Timeout for creating master node                                                          |
| node_max_restart                   | `DLROVER_UNIFIED_NODE_MAX_RESTART`                   | 10           | Maximum number of restarts for each node                                                  |
| job_max_restart                    | `DLROVER_UNIFIED_JOB_MAX_RESTART`                    | 10           | Maximum number of job restarts                                                            |
| master_max_restart                 | `DLROVER_UNIFIED_MASTER_MAX_RESTART`                 | 10           | Maximum number of master restarts                                                         |
| master_isolation_schedule_resource | `DLROVER_UNIFIED_MASTER_ISOLATION_SCHEDULE_RESOURCE` | ""           | The master actor's scheduling will use this resource(key:1) if the resource is configured |
| worker_isolation_schedule_resource | `DLROVER_UNIFIED_WORKER_ISOLATION_SCHEDULE_RESOURCE` | ""           | The worker actor's scheduling will use this resource(key:1) if the resource is configured |


See `dlrover.python.unified.common.config.JobConfig` for all options.

#### Isolation scheduling configuration

Users can achieve isolated (targeted) scheduling of master or worker actors by 
specifying custom resources for cluster nodes when creating the cluster and 
providing the corresponding settings when submitting jobs:
```txt
master_isolation_schedule_resource  # for master
worker_isolation_schedule_resource  # for worker
```


For example: if user wants all the master actors be scheduled on head node, 
and all the worker actors be scheduled on worker nodes.

1. Set {"MASTER_RESOURCE":999} for ray head node and {"WORKER_RESOURCE":999} 
   for other ray worker nodes
    ```txt
    Noticed: Please configure the resource value to be sufficiently large, it is 
    recommended to set the value > 999.
    ```
2. Configure the following submitting params:
    ```python
    master_isolation_schedule_resource="MASTER_RESOURCE"
    worker_isolation_schedule_resource="WORKER_RESOURCE"
    ```
    or using envs(before submission):
    ```python
    DLROVER_UNIFIED_MASTER_ISOLATION_SCHEDULE_RESOURCE="MASTER_RESOURCE"
    DLROVER_UNIFIED_WORKER_ISOLATION_SCHEDULE_RESOURCE="WORKER_RESOURCE"
    ```

