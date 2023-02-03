# DLRover: An Automatic Distributed Deep Learning System

[![Build](https://github.com/intelligent-machine-learning/easydl/actions/workflows/main.yml/badge.svg)](https://github.com/intelligent-machine-learning/easydl/actions/workflows/main.yml)

DLRover automatically trains the Deep Learning model on the distributed cluster.
It helps model developers to focus on model arichtecture, without taking
care of any engineering stuff, say, hardware acceleration,
distributed running, etc. Now, it provides automated operation
and maintenance for deep learning training jobs on K8s/Ray. Detail features as

- **Fault-Tolerance**, the training process can continue if some nodes fails.
- **Auto-Scaling**, the training job can automatically scale up/down nodes.
- **Automatic Resource Optimization**, DLRover can automatically optimize
the job resource to improve the training performance.

## Why DLRover?

### Integration of Offline and Online Deep Learning.

Users can define a model with `tf.estimator.Estimator` and
deploy an offline job on K8s with batch data or online job on Ray 
with streaming data to train the model.
For detail to develop models, we can see the
[estimator example](docs/tutorial/estimator.md).

### No Resource Configuration to Submit a Job.

Users need not to set any resource configuration to submit a
distributed training job. The following example is an ElasticJob on K8s.

```yaml
apiVersion: elastic.iml.github.io/v1alpha1
kind: ElasticJob
metadata:
  name: dlrover-dnn-iris
spec:
  distributionStrategy: ParameterServerStrategy
  replicaSpecs:
    ps:
      template:
        spec:
          containers:
            - name: main
              image: easydl/tf-estimator:iris_dnn_v0
              command:
                - "python -m model_zoo.tf_estimator.iris_dnn_elastic"
    worker:
      template:
        spec:
          containers:
            - name: main
              image: easydl/tf-estimator:iris_dnn_v0
              command:
                - "python -m model_zoo.tf_estimator.iris_dnn_elastic"
```

### Fault Tolerance to Improve the Stable of Job.

DLRover can recover failed parameter servers and workers to
resume the training. Some failed nodes do not interrupt the 
training and hurt the convergence accuracy. The main error is
OOM of node due to user's insufficient memory configuration.
DLRover can automatically launch a Pod with more memory to recover
the OOM node. In AntGroup, DLRover manages hundreds of DL training
jobs every day on the customized Kubernetes cluster in AntGroup.
Except the failed job resulted by code errors, the rate of completed job
raise 89% with tf-operator in KubeFlow to 95%. Other unrecoverable
failure reasons of job are data error, NaN loss of the model, network breakdown
and so on.

<div align="center">
<img src="docs/figures/job-complete-rate.png" alt="Editor" width="600">
</div>

### Auto-Scaling to Improve Training Performance.

DLRover can automatically scale up/down the number of
nodes (parameter servers or workers) at runtime of a training job.
By monitoring the workload of nodes and throughput, DLRover can
diagnose the bottleneck of resource configuration.
The common bottleneck contains
node straggler, unbalanced workload of PS, insufficient CPU cores of nodes and
the insufficient number of nodes. DLRover can improve
the training performance by dynamic resource adjustment.

We use the dataset of [Kaggle CRITEO](https://www.kaggle.com/c/criteo-display-ad-challenge)
to train Wide&Deep and xDeepFM with 10 epoches on a K8s cluster.
DLRover can mitigate straggler to improve the training throughput
and shorten the job competion time (JCT).

<div align="center">
<img src="docs/figures/exp-jct-deepctr.png" alt="Editor" width="600">
</div>



### Auto-Scaling to improve Resource Utilization.

Different model training requires different resource. Users prefer to
configure their jobs with over-provision resources to 
avoid any potential risk from insufficient resources.
This usually ends up with huge resource waste. DLRover Auto-Scaling
can allocate resource by the demand of model training to reduce
the waste of resource.

<div align="center">
<img src="docs/figures/daily-job-resource-util.png" alt="Editor" width="1000">
</div>

## What's Next?

- Elastic data-parallel multi-GPU training.
- Elastic hybrid-parallel multi-GPU training.
- Auto-Parallelism of deep learning training.

## Quick Start

[Train a TensorFlow Estimator on Aliyun ACK](docs/tutorial/dlrover_cloud.md)
