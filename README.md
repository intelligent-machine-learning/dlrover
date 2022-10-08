# EasyDL: An Automatic Distributed Deep Learning System

EasyDL, as it says, is making deep learning models' training easy. It helps model developers focus on model algorithm itself, without taking care of any engineering stuff, say, hardware acceleration, distribute running, etc. It provides static and dynamic nodes' configuration automatically, before and during a model training job running on k8s. Detail features as,

- Fault-Tolerance.
- Static and Dynamic resource configuration.
- Automatic distributed model training.

EasyDL consists three components:

- ElasticTrainer: A framework to use EasyDL in training.
- ElasticOperator: A k8s controller to manage training nodes.
- Brain: An optimization service to generate resources plans, distributed running plans, etc.

## Why EasyDL

### Fault-tolerance

EasyDL can recover failed parameter servers and workers and resume the training.
Some failed nodes do not interrupt the training and hurt the convergence
accuracy.

### Static and Dynamic Resource Configuration

EasyDL can automatically configure the resources to start a training job
and monitor the performance of a training job and dynamically adjust
the resources to improve the training performance.

### Automatic distributed model training

(To be added)

## Quick Start
