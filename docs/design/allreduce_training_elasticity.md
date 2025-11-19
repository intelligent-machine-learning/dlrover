# ALLReduce Elasticity Training

## Elastic DDP Training

In a Distributed Data Parallel (DDP) training setup, each worker 
maintains a complete copy of the model parameters. When DLRover 
initiates an elastic operation for DDP training, it performs a full 
restart: the current training is stopped, and the training process is 
restarted with the new number of workers.

During the training process, the job master periodically requests a 
new resource configuration from the Brain service. When the Brain 
service returns an updated configuration, the master handles the 
elasticity operation by stopping the current training and applying 
the changes.

There are two possible scenarios for the resource change:

1. Change in Worker Count: The master only adds or removes the 
specified number of workers and then restarts the training on the 
updated set of workers.

2. Change in Per-Worker Resources (e.g., CPU, Memory, GPU): 
The master must terminate all existing workers, create a new set of 
workers with the updated resource specifications, and then restart 
the training.

On the Brain side, the service currently reads the updated job 
resource configuration from the ConfigMap. In the future, the Brain 
service will apply advanced algorithms to dynamically determine the 
optimal resources for the training job.

<div align="center">
<img src="../figures/ddp_elasticity.jpg" alt="Editor" width="500">
</div>