# EasyDL Design

EasyDL is an auto-configuration system for parameter server based training jobs.
With EasyDL system, users need not provide any resource configuration for their 
deep learning training jobs. Instead, the EasyDL can pick up the appropriate resource 
configuration for each job smartly and continue to optimize those jobs during their runtime.

## Background

In parameter server based jobs, all relevant parameters of the training model 
are distributed on the parameter server nodes. Each worker node takes partial training data
as input and compute the new parameters. After that, the worker node sends the
updated parameters to the parameter server node which is keeping the parameters. 

However, the model developers (users) have to learn more rather than model training 
algorithms when they are using those jobs to train their models. In order to 
run a training job, those users have to specify the required resources for their 
this job. Then the kubernetes cluster can allocate the required resources and 
start the job. Unfortunately, we found it is quite ineffective way to ask the users
to take care the resource configuration. At first, the users are usually the 
experts on model training but not training jobs and kubernetes cluster. It is 
not an easy task for them to have the optimal configuration at the first place.
Secondly, a training job's resources requirement may vary during its runtime.
A static resource configuration usually can not be the optimal one all the time.
We have observe jobs failure, poor performance and inefficient resources since
the users fail to provide the optimal resource configuration to their jobs.

## Target

We hope to design and implement a system which can free the users from resource
configuration completely and focus on the model training itself. Without any
input (on resource configuration), the EasyDL can still provide the optimal
resource plan for each training job, Meanwhile, the EasyDL can optimize the 
performance of training jobs further through resource adjustment when the jobs
are running.

## Design

EasyDL consists of three main components: Brain, Elastic Trainer and Operator.

### Elastic Trainer

For each training job, there is an elastic trainer to manage the job during 
the job's whole life cycle. The elastic trainer is to:

1. collect and persistent job's runtime information. 
2. provide elasticity support to the training job. 

When job is running, the elastic training keeps collecting the important runtime
information of the job (e.g., CPU and memory usage). Those information is persistent
to a storage and will be used in further optimization of this job.

Unlike some other system which provides similar auto-configuration, elasticity 
is the key basic function which support the auto-configuration in EasyDL. EasyDL
does not pursue to have the best resource plan at the first place. Instead,
EasyDL is designed to adjust the resources smoothly. 

#### Parameter Server Elasticity

When to scale up/down parameter servers, the elastic trainer will checkpointing
all parameters on the parameter servers. After the trainer adjusts the number
of parameter servers, it re-distribute the parameters to new parameter servers
and inform workers the location of those parameters (i.e., on which parameter server).

#### Worker Elasticity

The elastic trainer splits the training data into multiple *shards*. Each worker
will process a shard at a time. Since each worker is not assigned fixed training
data. The scale up/down of workers does not influence the computation on other
workers. The new worker just simply picks up a shard for computation.

#### Fault Tolerance

Parameter servers and workers can fail at any time. Thus the trainer will checkpoint
the parameters periodically. When a parameter server failed, the trainer starts
another parameter server and resume the checkpointing. For worker failure, 
the trainer just starts a worker and let the work picks up a shard for computation.

### Brain

The Brain in EasyDL is to provide the optimal resource plans for each job. It 
includes three components.

#### Administor

When a training job is created, the corresponding administor is also created 
in the brain. This administor will administer the job during the job's whole
lifetime. When to initializ the job or observe a performance issue in the job,
the administor will create an optimize event for a new resource plan.

#### Optimize Processor

The optimize processor is to process the optimize events created by the administors.

Since EasyDL waives the input from the users, optimize processor has to determine
the appropriate optimize algorithms for the training jobs. For example, we should
use different algorithms to process unbalance workload on PS and insufficient PS numbers.
Then we can have the optimal resource plans.

#### Algorithm Executor

After the optimize processor decides the algorithm for the job, the algorithm 
executor executes the algorithm and generate the resource plan.

### Operator

When a training job is submitted, the EasyDL operator creates the elastic trainer
for the training job. Furthermore, when the elastic trainer is to create or deletes
Pods, the operator is to execute the actual Pod creation and deletion.  
