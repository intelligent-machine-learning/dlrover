# Proactive Diagnosis to Enhance Training Fault Tolerance

## Background

If you have already used DLRover or read any related introductions, you would 
know that one important role of DLRover is to enhance the availability of 
training tasks, meaning it can automatically handle faults and resume training 
when encountering occasional exceptions. In the initial design, fault tolerance 
during training was performed passively, meaning that only when a runtime 
process or container failed and exited would the DLRover Master be notified and 
attempt the corresponding fault-tolerant actions. However, as production scale 
gradually increased, it became apparent that relying solely on passive fault 
tolerance could not cover all exception scenarios. Therefore, the first 
positive fault tolerance mechanism: 'the heartbeat mechanism' was implemented. 
But considering that in distributed training, the training agent side also 
employs a multi-process implementation, the aforementioned single heartbeat 
mechanism cannot cover various exception scenarios that occur during actual 
training operations (it can only cover some common cases).

Therefore, a comprehensive, scalable framework is needed to gradually improve 
the diagnosis and handling of corresponding exception scenarios.

## Target

- Be capable of proactive probing.
- Requires flexible and scalable implementation.
- Be able to reuse the current fault tolerance mechanisms.

## Design

The proactive diagnostic framework is mainly composed of the following roles:
- Diagnosis Manger
- Diagnosis Agent
- Inference Operator(Observer & Resolver)

These components are distributed across the Master and Worker Nodes, 
working together to achieve the main functionality of proactive diagnostics. 

<img src="../figures/" alt="" width="1000">

Below is a more detailed introduction to the relevant concepts.

### Diagnosis Manger
The diagnosis-manager is implemented and runs in the Master node, with the main 
responsibilities of driving pre-execution checks and runtime diagnostic logic(
across all the workers). 
Additionally, it interacts with other core roles of the master, such as the 
job-manager, to assist in the lifecycle management of training.

<img src="../figures/" alt="" width="1000">

### Diagnosis Agent
The diagnosis-agent is implemented and runs on the worker node, with the main 
responsibilities of driving data collection on the training worker side, 
executing some basic diagnostic logic, and reporting necessary information to 
Master.

<img src="../figures/" alt="" width="1000">

### Inference Operator


### Metric Collection


#### Common Metric


#### XPU_TIMER Metric


## Implementation

Next, we will primarily introduce some of the existing implementations based on 
this diagnostic framework. 

This part of the work will have a long cycle and work in progress for now...

### Training Hang Detection(Basic)

If we start a daemon subprocess of the GPU training process, the daemon
subprocess will exit if the training process fails. In this case, the
parameters in CPU memory of the daemon subprocess will be cleaned. So, the elastic agent
in the main process can allocate the shared memory with the training
process. The training process saves the state dict to the shared memory and
the agent save them into the storage.

### Training Hang Fault Tolerance(Advanced)

WIP
