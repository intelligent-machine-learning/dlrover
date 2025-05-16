# Unified API Guide[Experimental]

## Background
DLRover provides a unified control plane operation tailored for different type 
of training, aimed at enhancing runtime stability and performance.
For more details, refer to the: [Proposal doc](../design/unified-mpmd-control-proposal). 

## Instruction
### Process Instruction
#### Step 1: Implement the Workload

Extend the abstraction class 'BaseWorkload' provided by DLRover to implement 
different roles in reinforcement learning according to the 
following: [SDK doc](#baseworkload).

Users can selectively implement multiple roles based on specific algorithms. 
(The actor is mandatory and cannot be omitted.)

> Notice: The following code is for demonstration purposes only. Except for 
> the required abstract classes that must be inherited, everything else is 
> user-defined.  
```python
import ray
from dlrover.python.unified.trainer.workload import BaseWorkload


@ray.remote
class UserActorModel(BaseWorkload):
    
    def __init__(self, master_handle, config):
        super().__init__(master_handle, config)
        
        # define your actor property
        ...
    
    def init_model(self):
        # preparation before training
        ...
    
    def update_actor(self):
        # policy updating
        ...
    
     # other implement
    ...
```

#### Step 2: Implement the Trainer

Extend the abstraction class 'BaseTrainer' provided by DLRover according to 
the following: [SDK doc](#basetrainer).

> Notice: The following code is for demonstration purposes only. Except for 
> the required abstract classes that must be inherited, everything else is 
> user-defined.  

```python
from dlrover.python.unified.trainer.trainer import BaseTrainer


class UserTrainer(BaseTrainer):

    def __init__(self, actor_handles, actor_classes, config):
        super().__init__(actor_handles, actor_classes, config)
        
        # define your trainer property
        self.field0 = xxx
        self.field1 = xxx
        ...  
    
    def init(self):
        # define process logic for training preparation, 
        # such as: tokenizer, dataloader
        ...
    
    def fit(self):
        # define the main process logic for training
        # operate the multi workloads of different roles defined above
        ...
    
    # other implement
    ...
```

Notice: 
1. Do not perform initialization work in `__init__`; only definitions.  
2. Perform all initialization work in the `init` method.    


#### Step 3: Use API for Submit

Use the common [API](#Job Submitting API) to submit a ray 
job to run the DL training defined by user: 

```python
from dlrover.python.unified.api.api import DLJobBuilder


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


### SDK Instruction

#### BaseWorkload
- Abstraction Class
```python
dlrover.python.unified.trainer.workload::BaseWorkload
```

- Core Property

| Property Name     | Type        | Description                               |
|-------------------|-------------|-------------------------------------------|
| master_handle     | ActorHandle | DLMaster's actor handle                   |
| config            | DictConfig  | configuration for training(use OmegaConf) |
| name              | str         | the unique name of current workload       |
| role              | str         | role of current workload                  |
| rank              | int         | ~                                         |
| word_size         | int         | ~                                         |
| local_rank        | int         | ~                                         |
| local_world_size  | int         | ~                                         |
| torch_master_addr | int         | master address for torch rendezvous       |
| torch_master_port | int         | master port for torch rendezvous          |


- Core Method

| Method Name            | Is Abstract | Input Type | Output Type | Description                                                                                                 |
|------------------------|-------------|------------|-------------|-------------------------------------------------------------------------------------------------------------|
| is_actor_role          | no          | n/a        | bool        | is actor role                                                                                               |
| get_device_collocation | no          | n/a        | str         | get current device collocation group if current role is in a device collocation group. e.g. 'ACTOR,ROLLOUT' |

- Trainer-Invocation Usage

  To enable flexible method invocation between the trainer and workloads, the 
  decorator `@trainer_invocation` has been implemented. Methods decorated with 
  `@trainer_invocation` support the following features during remote invocation.   

  - Usage

| Attribute Name | Type      | Configuration and Default Value                                   | Description                                                                                                                                                                                                                          |
|----------------|-----------|-------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| blocking       | bool      | Default: True                                                     | Whether the method call waits for the result synchronously.<br/> True: Waits for the result, returns the result of the remote method.<br/>False: Does not wait for the result, directly returns the ref object of the remote method. |
| is_async       | bool      | Default: False                                                    | If `blocking=True`, determines whether to use `ray.wait` in non-blocking mode.<br/>True: Uses `ray.wait` to wait.<br/>False: Uses `ray.get` to wait.                                                                                 |
| timeout        | int       | Natural number greater than 0<br/> Unit: seconds<br/> Default: 10 | When `blocking=True` and `is_async=True`, this specifies the timeout for `ray.wait`.                                                                                                                                                 |
| target         | str       | Default: ALL                                                      | Specifies the target for remote execution.<br/>ALL: Targets all workloads.<br/>RANK0: Targets only workloads with `rank=0`.                                                                                                          |
| auto_shard     | bool      | Default: True                                                     | Determines whether to automatically perform parameter sharding during remote execution. If enabled, the input parameters are automatically split to match the number of workload targets.                                            |
| pre_func       | function  | Default: None                                                     | A pre-execution method for the remote method. The return value of this method serves as an argument to the remote method.                                                                                                            |
| post_func      | function  | Default: None                                                     | A post-execution method for the remote method. The return value of the remote method is input to this method, and the final result is the return value of this method.                                                               |


  - Example
      
      e.g. The actor has a method `a`, which can be invoked completely 
           asynchronously without waiting for the result:  
      ```python
      @trainer_invocation(blocking=False)
      def a(xxx):
            xxx
      ```
      
      e.g. The actor has a method `b`, which requires using the `wait` 
           non-blocking mode to wait for the result, with a timeout of 30 
           seconds:  
      ```python
      @trainer_invocation(is_async=True,timeout=30)
      def a(xxx):
            xxx
      ```
      
      e.g. When using method `b`, it also requires a pre-filtering method:  
      ```python
      def filter_func(input: int):
            if input < -1:
                    return 0
            return input
    
      @trainer_invocation(is_async=True,timeout=30，pre_function=filter_func)
      def a(input: int):
            return input + 1
      ```  


#### BaseTrainer
- Abstraction class
```python
dlrover.python.unified.trainer.trainer::BaseTrainer
```

- Core Property

| Property Name | Type                         | Description                                                                       |
|---------------|------------------------------|-----------------------------------------------------------------------------------|
| config        | DictConfig                   | configuration for training(use OmegaConf)                                         |
| actor_handles | Dict[str, List[ActorHandle]] | get all the actor handles with dict format，key: role，value: actor handles in list |


- Core Method

| Method Name            | Is Abstract | Input Type | Output Type | Description                                         |
|------------------------|-------------|------------|------------|-----------------------------------------------------|
| init                   | yes         | n/a        | n/a        | by user implementation: preparation before training |
| fit                    | yes         | n/a        | n/a        | by user implementation: core logic for training     |
| get_role_groups        | no          | n/a        | List[str]  | get all the role groups(existed)                    |
| get_workload_resource  | no          | str        | Dict       | get the resource used by role                       |


- Role-Group Usage

    Considering that calling a specific method of a certain type of workload under 
    the trainer is the most common usage scenario, a runtime binding for the 
    `RoleGroup` object has been implemented for convenience. Specifically, after 
    the trainer object is constructed, `RoleGroup` objects are dynamically bound 
    to the trainer based on the types of workloads present for the current task. 
    Users can directly call remote methods of all workloads of a target type 
    through these `RoleGroup` objects.  

    - Usage
        
        - Format: self.${ROLE_GROUP}.${remote_method}

        - For example:
          - actor -> RG_ACTOR
          - R0 -> RG_RO
          - udf_1 -> RG_UDF_1

    - Example

        e.g. There are 4 actors with the method implementation: `update`. 
             Invoke the `update` method of all 4 actors simultaneously under 
             the trainer.  
        ```python
        self.RG_ACTOR.update(parameter_x)
        ```
      
        e.g. There are 2 rollouts with the method implementation: `generate`. 
             Invoke the `generate` method of all 2 rollouts simultaneously 
             under the trainer.  
        ```python
        self.RG_ROLLOUT.generate(parameter_0, parameter_1)
        ```

#### Job Submitting API

- DLJobBuilder Usage
```python
dlrover.python.unified.api.api::DLJobBuilder
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
      
      | Method Name                    | Mandatory  | Type | Format and Default     | Description                                               |
      |--------------------------------|------------|------|------------------------|-----------------------------------------------------------|
      | total                          | yes        | int  | int greater than 0 / 0 | instances number for current(role) workload               |
      | per_node                       | yes        | int  | int greater than 0 / 0 | per node instances number for current(role) workload      |
      | env                            | no         | dict | None                   | envs for current(role) workload                           |
      | enable_ray_auto_visible_device | no         | n/a  | not enabled by default | whether to enable Ray's device visibility auto-assignment |
  

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
