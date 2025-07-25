# Reinforcement Learning Integration Guide[Experimental]

## Background
This article primarily provides an extended guidance of the RL scenario. 
For the background and other general sections, please refer to: 
[doc](./unified_api_guide.md). 

To be noticed:
```
DLRover itself is not an algorithm framework, and therefore does not natively 
support any specific RL algorithm implementation. As a result, a complete RL 
computation requires users to leverage the fundamental capabilities provided 
by DLRover to implement the corresponding algorithm and ensure its execution.  
```

## Instruction
### Process Instruction

- use 'BaseRLWorker' to implement the core computation(training or inference or something else)
- use 'BaseRLTrainer' to implement RL arithmetic to managed the processing among the workloads
- use 'RLJobBuilder' extended from 'DLJobBuilder' to express the RL job

### SDK Instruction

#### BaseRLWorkload
- Abstraction Class
```python
dlrover.python.unified.backend.rl.worker::BaseRLWorker
```

Property

| Property Name     | Type               | Description                               |
|-------------------|--------------------|-------------------------------------------|
| config            | DictConfig or Dict | configuration for training(use OmegaConf) |
| name              | str                | the unique name of current workload       |
| role              | str                | role of current workload                  |
| rank              | int                | ~                                         |
| word_size         | int                | ~                                         |
| local_rank        | int                | ~                                         |
| local_world_size  | int                | ~                                         |
| torch_master_addr | int                | master address for torch rendezvous       |
| torch_master_port | int                | master port for torch rendezvous          |


- Method

| Method Name                            | Is Abstract   | Input Type | Output Type | Description                                                                                                 |
|----------------------------------------|---------------|------------|-------------|-------------------------------------------------------------------------------------------------------------|
| is_actor_role                          | no            | n/a        | bool        | is actor role                                                                                               |
| is_reference_role                      | no            | n/a        | bool        | is reference role                                                                                           |
| is_rollout_role                        | no            | n/a        | bool        | is rollout role                                                                                             |
| is_reward_role                         | no            | n/a        | bool        | is reward role                                                                                              |
| is_critic_role                         | no            | n/a        | bool        | is critic role                                                                                              |
| is_actor_or_rollout_device_collocation | no            | n/a        | str         | is actor and rollout in deivce collocation                                                                  |
| get_device_collocation                 | no            | n/a        | str         | get current device collocation group if current role is in a device collocation group. e.g. 'ACTOR,ROLLOUT' |


#### BaseRLTrainer
- Abstraction class
```python
dlrover.python.unified.backend.rl.trainer::BaseRLTrainer
```

- Property

| Property Name | Type               | Description                                                                       |
|---------------|--------------------|-----------------------------------------------------------------------------------|
| config        | DictConfig or Dict | configuration for training(use OmegaConf)                                         |
| actors        | List[ActorHandle]  | get all the actor handles for actor                                               |
| references    | List[ActorHandle]  | get all the actor handles for reference                                           |
| rollouts      | List[ActorHandle]  | get all the actor handles for rollout                                             |
| rewards       | List[ActorHandle]  | get all the actor handles for reward                                              |
| critics       | List[ActorHandle]  | get all the actor handles for critic                                              |


- Method

| Method Name            | Is Abstract | Input Type | Output Type  | Description                                         |
|------------------------|-------------|------------|--------------|-----------------------------------------------------|
| init                   | yes         | n/a        | n/a          | by user implementation: preparation before training |
| fit                    | yes         | n/a        | n/a          | by user implementation: core logic for training     |
| get_role_groups        | no          | n/a        | List[str]    | get all the role groups(existed)                    |
| get_workload_resource  | no          | str        | Dict         | get the resource used by role                       |

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

- Role-Group Usage

    Considering that calling a specific method of a certain type of workload under 
    the trainer is the most common usage scenario, a runtime binding for the 
    `RoleGroup` object has been implemented for convenience. Specifically, after 
    the trainer object is constructed, `RoleGroup` objects are dynamically bound 
    to the trainer based on the types of workloads present for the current task. 
    Users can directly call remote methods of all workloads of a target type 
    through these `RoleGroup` objects.  

    - Usage
        
        - Format: self.{ROLE_GROUP}.{remote_method}

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

- RLJobBuilder Usage
```python
dlrover.python.unified.api.rl::RLJobBuilder
```
- Extended Configuration

  - Job Level
    
      | Method Name | Mandatory  | Type  | Format and Default                      | Description                   |
      |-------------|------------|-------|-----------------------------------------|-------------------------------|
      | actor       | yes        | tuple | (${module_name}, ${class_name}) / None  | setup actor workload info     |
      | rollout     | no         | tuple | (${module_name}, ${class_name}) / None  | setup rollout workload info   |
      | reference   | no         | tuple | (${module_name}, ${class_name}) / None  | setup reference workload info |
      | reward      | no         | tuple | (${module_name}, ${class_name}) / None  | setup reward workload info    |
      | critic      | no         | tuple | (${module_name}, ${class_name}) / None  | setup critic workload info    |
      
    

## Example
The following examples primarily provides examples for adapting to
different open-source frameworks.

> To be noticed:
All implementations below is based on a history version of each framework and
not guarantee compatibility with the latest code.

- With OpenRLHF PPO(Deepspeed):
```
example codes in examples/unified/rl/openrlhf
```

- With Verl PPO(Megatron):
```
example codes in examples/unified/rl/verl
```
