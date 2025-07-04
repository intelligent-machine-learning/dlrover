# Some notes for understanding the code

## Modules Architecture

- dlrover.python.unified
  - common: store data structures/protocols shared by all modules.
  - controller: PrimeMaster
  - backend
    - elastic: ElasticMaster
      - worker: ElasticWorker

### Module Design Patterns

Each module is designed to follow a specific pattern, which includes:

- Master: the domain boundary, holding reference to Manager, all outside interactions go through it.
- Manager: the CORE internal class, singleton, holding references to all internal components, implementing the main business logic. It coordinates between different components.
- Api: the interface for outside to interact with the system. It mainly provides the RemoteStub interface for the Master to call.
- Components: the internal components that implement the business logic. They are usually not exposed to outside, and are held by the Manager.

Code Style Guidelines:

- Favor Composition Over Inheritance
  - If need reuse, decompose logic into pure functions or components.
  - Pro: clearer boundaries, easier to test, and less coupling.
- Prefer Interfaces to Abstract Classes.
  - When classes share similarities, define them using `Protocol` or `Interface`, and keep fields and methods flat.
  - Pro: more flexible, easier to mock or test, and introduces less coupling.
- Expose fields directly, instead of using getters/setters.
  - Fields are scoped and often readonly. Direct access is preferred unless values are computed.
  - Pro: more readable, less boilerplate code, easier to track usage.
- Split Modules Rather Than Branch Logic.
  - Minimize branching; instead, separate distinct behaviors into dedicated modules.
  - Pro: clearer separation of concerns, easier to understand and maintain.

Visibility:

- All submodules are internal to parent module, unless meaningfully exposed to outside.
- All fields(without `_`) are public readonly(inside module), and should not reassigned.

## Controller(PrimeMaster)

- api.py: PrimeMasterRemote, exposed to outside
- config.py: JobConfig, the input config for training.
- master.py: PrimeMaster, the main actor class. As the entrypoint and Rpc layer.
- manager.py: PrimeManager, the Core of this module. It manages the training process, and coordinates between different components.
- schedule:
  - scheduler.py: Scheduler, scheduling actors(SubMaster and Workers).
  - graph.py: ExecutionGraph, the core state for scheduling.
  - placement.py: Placement, the placement logic for scheduling.

## Elastic Backend (one example of Backend)

- master.py: ElasticMaster
  1. Implements `ActorBase`, implementing lifecycle including `_setup`, `status`, `self_check`, `start`, `shutdown`.
  2. As a SubMaster, manages workers, and implements lifecycle `check_workers`
  3. Custom RPCs interface for `ElasticWorker` to use.
- worker
  - worker.py: ElasticWorker
    1. Implements `ActorBase`, implementing lifecycle including `_setup`, `status`, `self_check`, `start`, `shutdown`.
    2. Custom RPCs for `ElasticMaster` to use, including `run_node_check`, `start_elastic_job`.
  - runner.py: ElasticWorkerRunner, the main logic for running the worker, including `run`.
    1. Private usage by `ElasticWorker`, currently it runs the elastic agent. And user code will be run as subprocesses. `1:1` relationship with `ElasticWorker`.
    2. In future, it may be training process, directly running the user code. `1:n` relationship with `ElasticWorker`.

Note: The custom RPCs between `PrimeMaster` and `ElasticMaster` are internal, feel free to modify if needed.
