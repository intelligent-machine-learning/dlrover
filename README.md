# EasyDL: An Automatic Distributed Deep Learning Service

EasyDL is a distributued deep learning service to support elastic,
fault-tolerance and automatic resource configuration on the cloud.
Using EasyDL, users don't need to configure any resources to submit
DL training jobs on a Kubernetes (k8s) cluster. 

EasyDL consists three components:
- ElasticTrainer: A framework to use EasyDL in training.
- ElasticOperator: A k8s controller to manage training Pods.
- Brain: An optimization service to generate resource
configurations for training jobs.


## Why EasyDL ?

### Automatic Resource Configuration
EasyDL can automatically configure the resources to start a training job
and monitor the performance of a training job and dynamically adjust
the resources to improve the training performance.

### Fault-tolerance
EasyDL can recover failed parameter servers and workers and resume the training.
Some failed nodes do not interrupt the training and hurts the convergence
accuracy.

### Elaticity
EasyDL can scale up/down the resources(CPU, memory and number) of workers
and PS during training. Each node can have its own resource configuration
to improve training performance and resource utilization.


## Quick Start
