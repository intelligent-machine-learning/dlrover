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

### Advanced examples (outline)

- Multiple roles (pseudo):

```python
job = (
    DLJobBuilder()
    .config(DictConfig(args))
    .role("trainer").train("module.trainer.run").total(2).resource(gpu=1).end()
    .role("rollout").run("module.rollout.run").total(4).resource(cpu=4).end()
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
- train(entrypoint): define a training workload with entrypoint (module path + function), and return a sub builder.
- run(entrypoint): define a non-training workload with entrypoint, and return a sub builder.

### Workload / role patterns

- Training workload: use `train()` to define the main training entrypoint.
- Non-training workload: use `run()` to define auxiliary entrypoints
  (data loader, evaluator, etc).

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
