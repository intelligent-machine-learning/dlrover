# 02. Unified API Guide [Experimental]

This section focuses on the DLJobBuilder and submission patterns: how to
construct job configurations programmatically and submit them to the
DLRover control plane.

## Key flow

1. Create a DLJobBuilder instance.
2. Configure job-level settings (type, node/device counts, device type).
3. Register workloads/entrypoints and resources.
4. Provide user configuration (dict or OmegaConf).
5. Build the job object and submit.

## Minimal example

```python
from omegaconf import DictConfig
from dlrover.python.unified.api.builder import DLJobBuilder

args = {"data_dir": "data/nanogpt/", "batch_size": 16}

job = (
    DLJobBuilder()
    .dl_type("SFT")
    .node_num(2)
    .device_per_node(1)
    .device_type("GPU")
    .config(DictConfig(args))
    .dlrover_run(
        "examples.unified.elastic.nanogpt.train.run",
        nnodes=2,
        nproc_per_node=1,
    )
    .build()
)

job.submit(job_name="nanogpt")
```

## Common builder methods

- dl_type(type_str): high-level job category (SFT, PRE, RL, etc.).
- node_num(n) / device_per_node(n): how many nodes and devices per node.
- device_type("GPU"|"CPU"): preferred accelerator type.
- config(DictConfig or dict): user configuration available at runtime.
- dlrover_run(entrypoint, nnodes=?, nproc_per_node=?, **kwargs): sets the
  worker entrypoint (module path + function) and process counts.
- workload(...) / resource helpers: define multiple roles with per-role
  counts and resources (if supported by builder version).

## Workload / role patterns

- Single-role job: use `dlrover_run()` to point to the single entrypoint.
- Multi-role job: declare workloads/roles with entrypoints and resource
  specs (trainer, rollout, evaluator). Use `workload()` or the workload
  tuple format depending on builder API availability.

## Submission patterns

- Synchronous vs asynchronous: `job.submit()` submits the job; runtime
  semantics (blocking vs non-blocking) depend on environment.
- Provide `job_name` for easier tracking of logs, checkpoints and state.
- Programmatic monitoring: use Runtime SDK to query job status and control
  running jobs.

## Best practices before submit

- Ensure entrypoint module paths are importable on workers (PYTHONPATH).
- Avoid heavy work at import-time; perform setup in `run()`/`__init__`.
- Use `DictConfig` for structured configs to ease overrides in CI/tests.
- Debug locally with small `nnodes` / `nproc_per_node` values.
- Pin critical dependency versions in CI to avoid runtime mismatches.

## Troubleshooting

- Import errors on workers: check PYTHONPATH and packaging.
- Rendezvous or NCCL failures: verify network, NCCL envs and consistent
  world sizes.
- Resource allocation failures: verify scheduler/back-end logs for why
  requested resources were not satisfied.

## Advanced examples (outline)

- Multiple roles (pseudo):

```python
job = (
    DLJobBuilder()
    .dl_type("RL")
    .config(DictConfig(args))
    .workload("trainer", total=2, entry_point="module.trainer.run", resource={"gpu":1})
    .workload("rollout", total=4, entry_point="module.rollout.run", resource={"cpu":4})
    .build()
)
job.submit(job_name="rl_job")
```

(Use the exact `workload()` method names from the installed builder
version.)

## See also

- Runtime SDK: `04-runtime-sdk.md` for monitoring, RPC helpers and
  runtime utilities.
