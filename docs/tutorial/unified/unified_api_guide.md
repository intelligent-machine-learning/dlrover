# Unified API Guide[Experimental]

## Background

DLRover provides a unified control plane operation tailored for different type
of training, aimed at enhancing runtime stability and performance.
For more details, refer to the: [Proposal doc](../design/unified-mpmd-control-proposal).

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

For different types of scenarios, the content that users need to implement
varies alot. So please refer to the [next chapter](#process-instruction-with-different-types-of-deeplearning)
for each scenario for details.

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
