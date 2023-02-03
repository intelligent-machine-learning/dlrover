# DLRover: An Automatic Distributed Deep Learning System

 [![Build](https://github.com/intelligent-machine-learning/easydl/actions/workflows/main.yml/badge.svg)](https://github.com/intelligent-machine-learning/easydl/actions/workflows/main.yml)

DLRover, as it says, is making deep learning training easy. It helps
model developers focus on model algorithm itself, without taking
care of any engineering stuff, say, hardware acceleration,
distributed running, etc. Now, it provides automated operation
and maintenance for deep learning training jobs on K8s
and Ray. Detail features as

- Fault-Tolerance, the training job can continue if some nodes fails.
- Auto-Scaling, the training job can automatically scale up/down nodes.
- Automatic Resource Optimization, DLRover can automatically optimize
the job resource to improve the training performance.

## Why DLRover

### Integration of Offline and Online Deep Learning.

Users can define a model with `tf.estimator.Estimator` and
deploy a offline job on K8s with a batch dataset or online job on Ray 
with a streaming dataset to train the model.
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

DLRover can recover failed parameter servers and workers and
resume the training. Some failed nodes do not interrupt the 
training and hurt the convergence accuracy. The main error is
OOM of Pods due to user's insufficient memory configuration.
DLRover can automatically launch a Pod with more memory to recover
the OOM Pod. In AntGroup, DLRover manages hundreds of DL training
jobs daily on the customized Kubernetes cluster in AntGroup.
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
By monitor the workload of nodes and throughput, DLRover can
diagnose the bottleneck of resource. The common bottleneck contains
node straggler, unbalanced workload of PS, insifficient CPU cores of workers and
the insifficient number of PS/workers. DLRover can improve
the training performance by dynamic resource adjustment.

We use the dataset of [Kaggle CRITEO](https://www.kaggle.com/c/criteo-display-ad-challenge)
to train Wide&Deep and xDeepFM with 10 epoches on a K8s cluster.
The resource configuration in normal experiments without straggler is 

<div align="center">
<table>
    <tr>
        <td rowspan="2">Model</td>
        <td colspan="3">PS</td>
        <td colspan="3">Worker</td>
    </tr>
    <tr>
        <td>num</td>
        <td>CPU</td>
        <td>Memory</td>
        <td>num</td>
        <td>CPU</td>
        <td>Memory</td>
    </tr>
    <tr>
        <td>Wide&Deep</td>
        <td>8</td>
        <td>16</td>
        <td>8GB</td>
        <td>24</td>
        <td>3</td>
        <td>4GB</td>
    </tr>
    <tr>
        <td>xDeepFM</td>
        <td>2</td>
        <td>16</td>
        <td>8GB</td>
        <td>8</td>
        <td>20</td>
        <td>8GB</td>
    </tr>
</table>
</div>

<div align="center">
<img src="docs/figures/exp-jct-deepctr.png" alt="Editor" width="600">
</div>

DLRover can migigate straggler to improve the training throughput
and shorten the job competion time (JCT).

### Auto-Scaling to improve Resource Utilization.

Different model training requires different resource. Users prefer to
configure their jobs with over provision resources to 
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
