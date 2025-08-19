# 04. Runtime Reference

Some have introduced in previous sections, and some have not used in examples yet.

## Workload entrypoints

Unified jobs support two kinds of entrypoints. The entrypoint is the
primary object the runtime constructs and runs; it defines job lifecycle
and is where most user code starts.

- Function entrypoint

  A single, main-like function is used as the job entrypoint. The
  function must accept no arguments and runs in a blocking manner until
  the job completes. Use this for simple scripts or synchronous jobs.

  Example:

  ```python
  from dlrover.python.unified.api.runtime.worker import current_worker

  def run():
      args = current_worker().job_info.user_config
      # prepare data, model, optimizer using args
      train(args)
  ```

- Class entrypoint
  
  Advanced entrypoint, mainly for complex jobs with multiple roles. During `start`, the framework will synchronously construct the class (constructor must be no-arg), and auto export all `@rpc` methods in this class. After construction, the framework calls the instance `run()` method which contains the main execution loop.

  Example:

  ```python
  from dlrover.python.unified.api.runtime import rpc
  from dlrover.python.unified.api.runtime.worker import current_worker

  class TrainerService:
      def __init__(self):
          self.state = {}

      @rpc()#Auto export
      def get_status(self):
          return self.state

      def run(self):
          args = current_worker().job_info.user_config
          # main training loop
          while not done():
              step()
  ```

### How to choose

Recommend function entrypoint for SFT or simple workload, no need handle RPC call.
If you need shared state between multiple `@rpc` methods, you can use class entrypoint.
Whatever you choose, all `@rpc` in top-level will also be exposed as RPC endpoints automatically. And you can use all Runtime API in any time.

## Workload Types

Currently, two workload types are supported:

- **Elastic Workload**: Designed for elastic training scenarios, this type leverages `ElasticMaster` and `ElasticWorker` to enable dynamic node management. It performs node health checks, GPU setup, and rendezvous coordination before executing the entrypoint.
  - It will set environment variables including `LOCAL_RANK`, `RANK`, `LOCAL_WORLD_SIZE`, `WORLD_SIZE`, and `NODE_RANK` to help you write distributed code.
  - Also, by default, it will execute `torch.distributed.init_process_group` automatically, set `torch.cuda.set_device`, and `destroy_process_group` when the job is done.
  - If you need customize the distributed setup, you can set `comm_auto_setup_process_group` to `False` in the workload description, and it will only set `MASTER_ADDR` and `MASTER_PORT`.

- **Simple Workload**: The default type, suitable for `Trainer` or inference roles. It simply runs the entrypoint function or class without additional orchestration.

## Get Runtime Information: current_worker()

As entrypoint is a no-argument function, or class you define, we provide `current_worker()` to get runtime information.

The runtime injects execution context so the function does not require any parameters. Typical returned fields include job id, role, rank, and any other context the scheduler provides.

Example usage:

```python
info = current_worker()
print(info.job_info, info.actor_info)
```

Use `current_worker()` when your code needs to adapt behavior based on
its runtime identity, for example to perform leader-only actions or to
log role-specific metrics.

## RPC utilities and exported RPC helpers

RPC is a central tool for runtime control, evaluation, and
cross-worker coordination. This module exposes a set of RPC helpers and
conveniences that simplify registering and calling RPC endpoints.

Automatic exposure

- The framework automatically exposes all top-level functions in the
  module decorated with `@rpc` as remote-callable endpoints.
- It also exposes methods on the entry class annotated with `@rpc`
  after the class is instantiated by the runtime.

Key exports related to RPC (import from
`dlrover.python.unified.api.runtime`):

- `rpc` — decorator to mark functions/methods as RPC endpoints.
- `RoleGroup` — structured multi-role abstraction for group RPC and
  broadcast/targeted calls.
- `RoleActor` — per-role actor abstraction (where applicable).
- `UserRpcProxy` — client proxy to invoke RPCs from user code.
- `create_rpc_proxy` — helper to build proxies for remote targets.
- `export_rpc_method`, `export_rpc_instance` — helpers to register
  methods or instances for RPC exposure.
- `FutureSequence` — utility to aggregate or stream RPC futures.

### Function interface

Top-level functions can be exported as RPC handlers. The runtime will
auto-register decorated functions so callers can invoke them by name.

Example:

```python
# provider
@rpc(export=True)
def some_method():
    return "test1"

# caller
# remote call by name via proxy/rolegroup
```

### RPC proxy (export_rpc_instance / create_rpc_proxy)

The module supports exporting object instances for RPC and creating a
proxy on the caller side. This is useful for stateful services such as
queue owners.

Example:

```python
class SimpleClass:
    @rpc()
    def hello(self, name: str) -> str:
        return f"Hello, {name}!"

# provider
export_rpc_instance("simple_class", SimpleClass())

# caller
proxy = create_rpc_proxy("actor", "simple_class", SimpleClass)
proxy.hello("World")
```

### RoleGroup and FutureSequence

`RoleGroup` groups actors by role and provides `call`, `call_rank0`, and
`call_batch` helpers. `FutureSequence` is a wrapper over multiple futures
that supports lazy result retrieval and partial blocking.

Example:

```python
rg = RoleGroup(name="agents", roles=["actor", "learner"])
# broadcast
rg.call("sync_weights", payload)
# batched calls
future_seq = rg.call_batch(actor_forward, size, sequences)
```

### Remote call pattern (recommended)

For medium-to-large projects we recommend centralizing remote-call
signatures into a `remote_call.py` module. This provides a single
interface for callers and providers, and decouples signatures from
implementation placement.

Pattern benefits:

- Centralized signatures (can be typed in .pyi)
- Callers import plain functions from `remote_call` without knowing
  RPC details
- Providers decorate/implement the functions and register them with
  the runtime

Minimal example:

```python
# remote_call.py (interface)
def vllm_wakeup() -> None: ...

def vllm_generate(prompt_token_ids, params) -> FutureSequence: ...

# provider
class VLLMActor:
    @rpc(remote_call.vllm_wakeup)
    def wake_up(self):
        self.llm.wake_up()

    @rpc(remote_call.vllm_generate)
    def generate(self, prompt_token_ids, params):
        ...

# caller
from remote_call import vllm_generate
result = vllm_generate(prompt_token_ids, params)
```

This style improves maintainability and allows type-checking the
RPC surface.

## Data pipeline tools

We provide two additional runtime tools to help build data pipelines:

### DataQueue

A distributed queue implementing a multi-producer / multi-consumer
model. It decouples producers and consumers and can be used to control
throughput using size limits.

Interface sketch:

```python
class DataQueue(Generic[T]):
    """
    Distributed data queue interface.
    """

    def __init__(self, name: str, is_master: bool = False, size: int = 1000): ...

    def qsize(self) -> int: ...

    def put(self, *obj: T) -> None: ...

    def put_async(self, *obj: T) -> Future[None]: ...

    def get(self, batch_size: int) -> List[T]: ...

    def get_nowait(self, max_size: int = 1) -> List[T]: ...
```

Usage:

```python
# owner (usually rank 0 producer)
queue_master = DataQueue[str]("test_queue", size=10, is_master=True)
queue_master.put("data1", "data2")

# client (producer or consumer)
queue_client = DataQueue[str]("test_queue")
queue_client.get(1)
```

Notes:

- Queue requires an owner node to store metadata.
- Not suitable for scenarios that need strict ordering across all
  producers.

### Ray data

All workload are running in Ray actors, so you can use Ray Data to build
data pipeline.

Reference to <https://docs.ray.io/en/latest/data/index.html>

### RayDataLoaderIter

Ray Data or `RayDataLoaderIter` provides a pattern to process and
consume large distributed datasets. It supports data source abstraction,
map/filter/batch transforms, and deferred execution.

Example using Ray Data and DataLoader offload:

```python
# high-level pattern
iterator = RayDataLoaderIter(dataloader, num_actors=4)
for batch in iterator:
    train_step(batch)
```

Ray Data examples and patterns are useful when preprocessing or
batching can be parallelized across actors.
