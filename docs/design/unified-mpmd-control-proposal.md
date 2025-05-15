# A Unified MPMD Controlling System by DLRover(Proposal)

## Background

Refer to the [overview](./dlrover-overview.md), DLRover has already been 
introduced, with its definition being: "An Automatic Distributed Deep Learning 
System." In our previous discussions and practices, we primarily focused on the 
PRE_TRAIN and SFT scenarios (single SPMD). Among the three major scenarios of 
deep learning (supervised learning, unsupervised learning, and reinforcement 
learning), to date, reinforcement learning remains unsupported. Therefore, 
this article will take the implementation in the RL (Reinforcement Learning) 
scenario as a guide, and further extend to DLRover's solution for more general 
and complex AI computation topologies. This general solution is expected to 
support and cover all deep learning scenarios in the future.


## What's the Difference

To give users' a better understanding of the following discussion, we will 
first introduce how RL training computation differs from the training
computation in the existing implementations (Pre-Train and SFT).

In short: 
> it has evolved from a single computation group to a 
> multi-role computation graph.
> 
> aka. SPMD -> MPMD

The upper part of the diagram mainly illustrates the implementation of DLRover 
in the current supervised/unsupervised learning scenarios: DLRover primarily 
takes on the management tasks of the control plane, managing a group of SPMD 
(training workers). DLRover does not handle any management of the data 
interaction between training processes.

The lower part of the diagram focuses on the design approach of DLRover for 
reinforcement learning scenarios (taking the relatively complex PPO setup as 
an example):  
- Evolved from managing a single SPMD group to managing multiple groups.  
- The control plane is further refined into a Master and Trainer Controller.  
- Data plane interactions between different SPMD groups are also incorporated 
into management by the Trainer Controller.    

<img src="../figures/unified/rl_arc_diff.png" alt="Diff for RL">


## Objective
Through the comparison above, we can easily see that computation in RL 
scenario is significantly more complex. But does an even more complex 
scenario implementation exist? 

Therefore, here we introduce the core topic of this article:  
> We aim to create a fully generalized automated system capable of representing 
> AI computation scenarios of any complexity across different computing devices. 
> Furthermore, based on this foundation, we seek to enhance its stability and 
> performance, just like the practical experience of DLRover for now.  

## Key Concept
Before diving into the specific design and implementation, it is necessary to 
introduce several key concepts. These key points are considered essential for 
achieving the aforementioned objective.  

### A General Computation Graph
Firstly, because the computation representation is highly generalized, we need 
to define a brand-new computation graph to represent the actual computational 
topology. Whether it is in the form of SPMD or MPMD, whether using CPUs or 
GPUs, and whether based on model architectures like FSDP or Megatron, all can 
be expressed through this computation graph.  

### Loosely Coupled with Algorithm and Model
Considering the rapid evolution of algorithms and model architectures, a 
generalized runtime framework should not be tightly coupled with these 
"business-like" components. Strong coupling would lead to the abstraction 
becoming concreted and bound to specific implementations. Such dependencies
would make the framework implementation extremely challenging to maintain. 

However, this raises a major point of contention—discussions around: 
flexibility. 

Commonly, there is the following trade-off: 
- high-coupling designs can optimize performance for specific scenarios 
  (e.g. AI algorithms tailored for model optimization) but sacrifice generality.

- low-coupling designs (e.g. modular AI systems) increase flexibility but 
  often fall short in optimizing performance for specific scenarios. 

Therefore, controlling the boundaries of this design is especially important.  

## Core Design
### Overall Architecture
The diagram below represents the overall architecture of DLRover's solution 
where DLRover is positioned at the center.  

<img src="../figures/unified/top_arc.png" alt="ARC" width="22604">

Based on the diagram above, there are several core design points to explain:

1. **DLRover's Role**: DLRover itself is not an algorithm framework (as 
mentioned earlier). Therefore, its role in the entire architecture remains 
primarily focused on the control plane management.
   - API and Driver: resolve input parameter and drive the computation job
   - Master: the main daemon process as an ray actor
     - JobManager: job lifecycle management   
       - scheduler: actor and group management
       - executor: to interact with the runtime computation
     - DiagnosisManager: runtime positive job diagnosis for stability
     - FailoverCoordinator: handle failure and do failover
     - JobContext: the job runtime context

2. **Computation Graph**: To drive the data plane, we need a computation graph 
that can fully represent the entire computation logic. Further details can be 
found [below](#computation-graph).

3. **Pluggable Workload**: To support various algorithms and models while 
meeting the controllable requirements of the control plane, DLRover provides 
an independent abstract definition for user implementation. Further details can 
be found [below](#pluggable-workload).

4. **Ray as the Distributed Backend**:  
   - Since complex AI computation involve multiple roles (e.g., Actor/Rw/Ref/Critic in RL), 
   asynchronous computation control between roles is key to improving performance.  
   - For different production scenarios and resource specifications, affinity 
   or anti-affinity scheduling for different types of roles is crucial for complex 
   workflow orchestration. Ray easily enables these implementations through 
   APIs such as placement groups.  
   - By leveraging Ray's built-in actor fault tolerance combined with DLRover's
   current practices in stability management, the overall process reliability 
   can be enhanced in a more cost-effective and flexible manner. 


### Computation Graph
Before diving into the aforementioned general computation graph, 
let’s first revisit the development of distributed computing. In the earliest 
stage, computation was single-node, where a single machine could handle all 
calculations and implementations. This is illustrated as follows: 
 
<div align="center">
    <img src="../figures/unified/single_cal.png" alt="single_cal" width="200">
</div>

With the growth of data scale, single-machine computation could no longer meet 
the demands, which led to the birth of distributed computing. However, since 
the data is split while the result format still needs to remain consistent 
with the original form, implementing distributed computing is not as simple as 
directly splitting single-node computations across multiple nodes. It instead 
involves splitting the data, performing computation, and merging the results 
(referencing the concept of Map-Reduce). This resulted in the basic 
computational structure as shown below:

<div align="center">
    <img src="../figures/unified/dist_cal.png" alt="dist_cal" width="400">
</div>

Building upon this foundation, we have seen many widely
recognized computation engines, from early Hadoop to later Spark and Flink 
(data + stream parallelism). With the arrival of the AI era, distributed 
computing has undergone further evolution, expanding the dimensions of 
parallel computation even further, although we won’t delve into those details 
here. 

Let's return to the topic of our objective. In order to better manage a general
computational process, we use a DAG (Directed Acyclic Graph) to represent 
the entire computation graph. Vertices represent various instances of 
computation roles, while edges represent the interaction relationships between 
these instances. 

The computation graph here has several key differences compared to the 
distributed computing described earlier:
1) Each node is no longer the implementation of a single operator but a 
   collection of multiple operators.  
2) The collaboration between operators is not necessarily fully 
   upstream-downstream synchronous but rather asynchronous.    

<div align="center">
    <img src="../figures/unified/unified_dist_cal.png" alt="unified_dist_cal" width="500">
</div>

Through the relationship between vertices and edges, DLRover, as the 
control-plane manager, can gain a clearer understanding of the interaction 
logic between all working roles, allowing for greater potential to improve 
overall stability and performance effectively. 
For example, if an anomaly occurs, the relationship topology allows us to 
identify the affected upstream and downstream components immediately, enabling 
more effective fault-tolerant operations.  


### SubDag Based Scheduling
For general training scenarios, the scheduling of all workloads is conducted as a 
whole, meaning that both training and inference are scheduled uniformly based 
on roles. However, to accommodate larger scales and different types of models, 
there are undoubtedly optimization strategies based on model parallelism. 
Therefore, in terms of scheduling, **DLRover** will represent the actual 
computation graph's topology in finer granularity, namely at the 
sub-dag(Directed Acyclic Graph)level. By utilizing the optimized sub-dag and 
considering the current cluster resources, **DLRover** implements more 
fine-grained scheduling, as illustrated in the diagram below:

- A general execution graph representation: as shown in the "Preview Execution" 
layer below.  
- An optimized execution graph representation based on sub-dag: as shown in 
the "Optimized Execution" layer below.    

<img src="../figures/unified/sub_dag.png" alt="Sub DAG Based">


### Pluggable Workload
To achieve loose coupling, we need to abstract a general definition of AI 
computational workloads and then apply it to the aforementioned computation 
graph. Users only need to implement their own algorithm logic based on the 
fixed abstraction and express the computational process in a generalized way. 
This can actually be summarized into two core steps: 

1. Implementing workloads with training logic code(including one or more 
   operators(function)).  
2. Expressing the process (the flow between vertex).   

<div align="center">
    <img src="../figures/unified/pluggable_workload.png" alt="Pluggable Workload" width="400">
</div>


### Flexible Scheduling
For some hybrid computing scenario, involves both training and 
inference. Under varying resource configurations and model architectures, the 
deployment strategies for training and inference workloads have to meet 
high-performance computing requirements. For example, common approaches 
include the **collocation way** (sharing the same device) and the 
**separation way** (using different devices). Moreover, some roles may be 
further divided internally, such as the "PD Separation Deployment" strategy on 
the inference side. Consequently, workload placement and GPU resource 
assignment require finer-grained management.

To support the aforementioned flexible combinations, DLRover offers the 
following three affinity (or anti-affinity) scheduling strategy and each of 
them can be combined with each other:  

<img src="../figures/unified/collocation.png" alt="Collocation">


### Fault Tolerance

#### Persisted Runtime Context
Before discussing fault tolerance, a necessary prerequisite must be 
established: a persistable runtime context. This context stores all essential 
and critical information required during job execution, such as the physical 
execution graph, runtime worker key information records, training key 
information records, and core job states, among others. All core control plane 
operations of the DLRover Master are centered around this runtime context, 
including the fault tolerance operations that will be introduced shortly.

Additionally, this context must be persistable and recoverable. The DLRover 
Master operates as a resident process throughout the lifecycle of the 
computation process, but it is also susceptible to unexpected issues. If the 
Master encounters an exception and restarts, it needs to recover the 
corresponding context from the persistence system to continue safeguarding the 
entire computation process.  


#### Failover
A failover implementation essentially build two actions into an automated 
process: 
- fault detection
- restarting

Depending on the differences in computation scenarios and computational 
complexity, failover can be categorized into the following levels based on 
its scope of implementation:

- CAT1: Job-Level Failover
- CAT2: Node-Level Failover
- CAT3: Process-Level Failover

Based on actual circumstances, higher-level failover may be downgraded to a 
lower-level failover. For example, CAT3 downgraded to CAT2.  

##### Job-Level Failover
At this level, failover is comprehensive and global. This failover strategy is 
designed for the most complex scenarios where in the event of an anomaly, 
it is impossible to coordinate and resolve the state of computation workloads. 
In such cases, the fault-tolerance mechanism at this level needs to be applied. 
This failover also serves as the last line of defense for fault tolerance.  

<div align="center">
    <img src="../figures/unified/job_failover.png" alt="Job Failover" width="500">
</div>

##### Node-Level Failover
At this level, failover affects individual machine nodes and involves multiple 
processes on the corresponding node. This failover strategy is narrower in 
scope compared to Job-level failover.  

<div align="center">
    <img src="../figures/unified/node_failover.png" alt="Node Failover" width="500">
</div>

##### Process-Level Failover
Further, the scope of failover is narrowed down to the process level.

<div align="center">
    <img src="../figures/unified/process_failover.png" alt="Process Failover" width="500">
</div>


## Supported Scenarios
This unified solution mentioned above can theoretically support any AI 
computation scenario. Here, we will provide some explanations and introductions 
to certain anticipated known scenarios. 


### Pre-Train/SFT Case
The simplest computation topology, as it consists of only a single set of SPMD 
computing units. This is also the primary core scenario currently supported 
by DLRover:  

<div align="center">
    <img src="../figures/unified/sft_dag.png" alt="Pre-Train/SFT" width="300">
</div>


### Online Decision Learning Case
Let's take a look at the Online Decision Learning scenario. Since it involves 
near real-time training, data continuously flows in from upstream sources, 
undergoes data preprocessing (e.g., sample processing, feature extraction, etc.), 
and is then fed into the training workloads:  

<div align="center">
    <img src="../figures/unified/odl_dag.png" alt="Online Learning" width="400">
</div>


### Multimodal Case
The multimodal scenario is also a typical multi-role computation topology. 
In addition to data preprocessing, the training process is also divided into 
multiple computational parts:

<div align="center">
    <img src="../figures/unified/multimodal_dag.png" alt="Multimodal" width="400">
</div>


### Reinforcement Learning Case
Reinforcement learning has significant differences in implementation depending 
on the specific algorithms. Here, taking PPO as an example, an illustrative 
computation graph is provided:  

<div align="center">
    <img src="../figures/unified/rl_ppo_dag.png" alt="PPO_DAG" width="400">
</div>

However, because the current computation in RL is strongly coupled with the 
algorithm and model architecture, we are still studying on how to provide an 
abstract, fully generalized representation of the computation graph 
independent of the algorithm and framework. 


## Conclusion
This proposal primarily outlines the aforementioned [objectives](#objective) 
and provides a brief explanation around several key points, and lists 
some practical and applicable scenarios. Further detailed designs and 
implementation plans will be updated in other documents and included under 
the appendix below.  


## Appendix
TODO
