# Reinforcement Learning Integration Guide

## Background
DLRover provides a unified control plane operation tailored for large-scale RL 
training, aimed at enhancing runtime stability and performance. To enable rapid 
iteration and achieve loose coupling with the "algorithm" implementation, 
DLRover decouples the control plane from the RL implementation and abstracts 
the algorithm implementation. This abstraction allows adaptation to various RL 
implementations across different algorithm and model architectures. 
For more details, refer to the: [design document](../design/reinforcement-learning-overview.md#pluggable-rl-workload). 

## Instruction
### Process Instruction
#### Step 1: Implement the Workload

Extend the abstraction class 'BaseWorkload' provided by DLRover to implement 
different roles in reinforcement learning according to the 
following: [SDK documentation](#baseworkload).

DLRover provides 5 roles for rl using:
- ACTOR: Makes decisions by selecting actions based on the policy.
- ROLLOUT: Simulates the environment and collects trajectories of actions and rewards. 
- REWARD: Computes the reward signal used for optimizing the policy.
- REFERENCE: Provides baseline or comparison metrics for training stability.
- CRITIC: Evaluates the value function to improve policy updates.

Users can selectively implement multiple roles based on specific algorithms. 
(The actor is mandatory and cannot be omitted.)

> Notice: The following code is for demonstration purposes only. Except for 
> the required abstract classes that must be inherited, everything else is 
> user-defined.  
```python
import ray
from dlrover.python.rl.trainer.workload import BaseWorkload


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
the following: [SDK documentation](#basetrainer).

> Notice: The following code is for demonstration purposes only. Except for 
> the required abstract classes that must be inherited, everything else is 
> user-defined.  

```python
from dlrover.python.rl.trainer.trainer import BaseTrainer


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


### SDK Instruction

#### BaseWorkload
- Abstraction Class
```python
dlrover.python.rl.trainer.trainer::BaseWorkload
```

- Core Property

| Property Name     | Type        | Description                               |
|-------------------|-------------|-------------------------------------------|
| master_handle     | ActorHandle | RLMaster's actor handle                   |
| config            | DictConfig  | configuration for training(use OmegaConf) |
| name              | str         | the unique name of current workload       |
| role              | RLRoleType  | role of current workload                  |
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
| is_reference_role      | no          | n/a        | bool        | is reference role                                                                                           |
| is_rollout_role        | no          | n/a        | bool        | is rollout role                                                                                             |
| is_reward_role         | no          | n/a        | bool        | is reward role                                                                                              |
| is_critic_role         | no          | n/a        | bool        | is critic role                                                                                              |
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
dlrover.python.rl.trainer.trainer::BaseTrainer
```

- Core Property

| Property Name | Type                                | Description                                                                       |
|---------------|-------------------------------------|-----------------------------------------------------------------------------------|
| config        | DictConfig                          | configuration for training(use OmegaConf)                                         |
| actor_handles | Dict[RLRoleType, List[ActorHandle]] | get all the actor handles with dict format，key: role，value: actor handles in list |
| actors        | List[ActorHandle]                   | get all the actor handles for actor                                               |
| references    | List[ActorHandle]                   | get all the actor handles for reference                                           |
| rollouts      | List[ActorHandle]                   | get all the actor handles for rollout                                             |
| rewards       | List[ActorHandle]                   | get all the actor handles for reward                                              |
| critics       | List[ActorHandle]                   | get all the actor handles for critic                                              |


- Core Method

| Method Name            | Is Abstract | Input Type | Output Type | Description                                         |
|------------------------|-------------|------------|------------|-----------------------------------------------------|
| init                   | yes         | n/a        | n/a        | by user implementation: preparation before training |
| fit                    | yes         | n/a        | n/a        | by user implementation: core logic for training     |
| get_role_groups        | no          | n/a        | List[str]  | get all the role groups(existed)                    |
| get_workload_resource  | no          | RLRoleType | Dict       | get the resource used by role                       |


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

        - Supported role group:
          - actor -> RG_ACTOR
          - reference -> RG_REFERENCE
          - rollout -> RG_ROLLOUT
          - reward -> RG_REWARD
          - critic -> RG_CRITIC

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


## Example
The following examples primarily provides examples for adapting to
different open-source frameworks.

> To be noticed:
All implementations below is based on a history version of each framework and
not guarantee compatibility with the latest code.

- With OpenRLHF PPO(Deepspeed):
```
example codes in dlrover/python/rl/trainer/example/openrlhf
```

- With Verl PPO(Megatron):
```
example codes in dlrover/python/rl/trainer/example/verl
```
