# Unified API Guide[Experimental]

## Instruction

Users rely on the following two steps to express any type of Deep Learning job
through DLRover:

- Step1. Implement the user workload.

- Step2. Use API to express and submit job.

    ```python
    from dlrover.python.unified.api.builder import DLJobBuilder
    
    dl_job = (
        DLJobBuilder()
        .dl_type("SFT")
        .node_num(1)
        .device_per_node(4)
        .config({"k1": "v1"})
        .trainer("xxx", "xxx")
        ...
        .build())
    
    dl_job.submit()
    ```

### Workload Implementation

Implementing a workload means providing the user code that the worker
processes will execute at runtime. A workload must expose a deterministic
entrypoint and accept configuration provided at submission time.

Key points:

- Entrypoint types:
  - Function entrypoint: a top-level function that the worker calls, e.g.
    `def run():` or `def run(args):`.
  - Class entrypoint: a class with a `run()` method. The runtime constructs
    the instance and calls `run()`.

- Configuration:
  - Prefer `omegaconf.DictConfig` or a plain dict for user configuration.
  - Access runtime metadata via `current_worker()` when needed.

- RPC exposure (optional):
  - Annotate functions or methods with `@rpc` to expose them as remote
    endpoints for monitoring or control.

- Module path:
  - The entrypoint is referenced by a module path used in `dlrover_run()`,
    e.g. `examples.unified.elastic.nanogpt.train.run` (module + function).

Minimal examples

- Function entrypoint example:

```python
# examples/unified/my_train.py
from dlrover.python.unified.api.runtime.worker import current_worker

def run():
    args = current_worker().job_info.user_config
    # prepare data, model, optimizer using args
    train(args)
```

- Class entrypoint example with RPC exposure:

```python
# examples/unified/my_service.py
from dlrover.python.unified.api.runtime import rpc
from dlrover.python.unified.api.runtime.worker import current_worker

class TrainerService:
    def __init__(self):
        self.state = {}

    @rpc(export=True)
    def get_status(self):
        return self.state

    def run(self):
        args = current_worker().job_info.user_config
        # main training loop
        while not done():
            step()
```

Checklist before submission

- Entrypoint importable by module path.
- Deterministic initialization: avoid reading random external state on
  import-time (use `if __name__ == "__main__"` or perform setup in
  `run()`/`__init__`).
- Use `current_worker()` for runtime context instead of environment-only
  variables when possible.
- Make sure long-running loops check for graceful shutdown signals.

Tips

- Keep the entrypoint small: delegate heavy logic into helper modules.
- Favor configuration-driven code so tests can override behavior easily.
- When using data loaders that rely on process-local state (e.g. Ray
  DataLoader), perform necessary runtime patches early in `run()`.

### Job Submitting Basic API

Regardless of the scenario, users need to use the following APIs to define and
submit job.  

- DLJobBuilder Usage

    ```python
    dlrover.python.unified.api.builder::DLJobBuilder
    ```

- Configuration: use following to setup unified deep learning job

  - Job Level

      | Method Name          | Mandatory | Type              | Format and Default                     | Description                                |
      |----------------------|-----------|-------------------|----------------------------------------|--------------------------------------------|
      | dl_type              | yes       | str               | "PRE","SFT","RL","MULTIMODAL" / SFT    | deep learning type                         |
      | node_num             | yes       | int               | int greater than 0 / 0                 | how many nodes                             |
      | device_per_node      | yes       | int               | int greater than 0 / 0                 | how many devices per node                  |
      | config               | yes       | dict or OmegaConf | None                                   | training configuration                     |
      | global_env           | no        | dict              | None                                   | global envs, can be override by roles' env |
      | workload             | yes       | tuple             | (${module_name}, ${class_name}) / None | setup workload info                        |
      | with_collocation     | no        | *str              | (${role_0}, ${role_1} ...) / n/a       | roles need to be collcated                 |
      | with_collocation_all | no        | n/a               | not enabled by default                 | collocate all roles                        |

  - Workload Level

      | Method Name                     | Mandatory  | Type | Format and Default     | Description                                                |
      |---------------------------------|------------|------|------------------------|------------------------------------------------------------|
      | total                           | yes        | int  | int greater than 0 / 0 | instances number for current(role) workload                |
      | per_group                       | yes        | int  | int greater than 0 / 1 | per group instances number for current(role) workload      |
      | env                             | no         | dict | None                   | envs for current(role) workload                            |
      | resource                        | no         | dict | None                   | resource for current(role) workload                        |
      | disable_ray_auto_visible_device | no         | n/a  | enabled by default     | whether to disable Ray's device visibility auto-assignment |

- Build: build the job

```python
dl_job = (
    DLJobBuilder()
    ...
    .build())
```

- Submit: submit the job

```python
dl_job.submit()
```

### Process Instruction with Different Types of DeepLearning

The upcoming relevant documentation will provide specific practical
introductions tailored to different deep learning scenarios.  

- how to implement a simple training(PRE-TRAIN / SFT): TODO
- How to implement a hybrid training: TODO
- How to implement a Reinforcement Learning: [doc](./unified_rl_integration_guide.md).
- How to implement a Online Decision Learning: TODO
