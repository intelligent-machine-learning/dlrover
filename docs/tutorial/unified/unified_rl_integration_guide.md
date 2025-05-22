# Reinforcement Learning Integration Guide[Experimental]

## Background
This article primarily provides an extended guidance of the RL scenario. 
For the background and other general sections, please refer to: 
[doc](./unified_api_guide.md). 

## Instruction
### Process Instruction

The main part is the same as [doc](./unified_api_guide.md#process-instruction),
with the differences as follows:

- use 'BaseRLWorkload extended from 'BaseWorkload'
- use 'BaseRLTrainer' extended from 'BaseTrainer'
- use 'RLJobBuilder' extended from 'DLJobBuilder'

### SDK Instruction

#### BaseRLWorkload
- Abstraction Class
```python
dlrover.python.unified.trainer.rl_workload::BaseRLWorkload
```

- Extended Property

| Property Name     | Type        | Description                               |
|-------------------|-------------|-------------------------------------------|


- Extended Method

| Method Name            | Is Abstract | Input Type | Output Type | Description                                |
|------------------------|-------------|------------|-------------|--------------------------------------------|
| is_actor_role          | no          | n/a        | bool        | is actor role                              |
| is_reference_role      | no          | n/a        | bool        | is reference role                          |
| is_rollout_role        | no          | n/a        | bool        | is rollout role                            |
| is_reward_role         | no          | n/a        | bool        | is reward role                             |
| is_critic_role         | no          | n/a        | bool        | is critic role                             |
| is_actor_or_rollout_device_collocation | no          | n/a        | str         | is actor and rollout in deivce collocation |


#### BaseRLTrainer
- Abstraction class
```python
dlrover.python.unified.trainer.rl_trainer::BaseRLTrainer
```

- Extended Property

| Property Name | Type                                | Description                                                                       |
|---------------|-------------------------------------|-----------------------------------------------------------------------------------|
| actors        | List[ActorHandle]                   | get all the actor handles for actor                                               |
| references    | List[ActorHandle]                   | get all the actor handles for reference                                           |
| rollouts      | List[ActorHandle]                   | get all the actor handles for rollout                                             |
| rewards       | List[ActorHandle]                   | get all the actor handles for reward                                              |
| critics       | List[ActorHandle]                   | get all the actor handles for critic                                              |


- Extended Method

| Method Name            | Is Abstract | Input Type | Output Type | Description             |
|------------------------|-------------|------------|-------------|-------------------------|


#### Job Submitting API

- RLJobBuilder Usage
```python
dlrover.python.unified.api.api::RLJobBuilder
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
example codes in dlrover/python/unified/trainer/example/rl/openrlhf
```

- With Verl PPO(Megatron):
```
example codes in dlrover/python/unified/trainer/example/rl/verl
```
