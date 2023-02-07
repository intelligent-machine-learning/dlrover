# DLRover

<div align="center">
<img src="docs/figures/dlrover_logo.png" alt="Editor" width="300">
</div>

<div id="top" align="center">
DLRover: An Automatic Distributed Deep Learning System
</div>

[![Build](https://github.com/intelligent-machine-learning/easydl/actions/workflows/main.yml/badge.svg)](https://github.com/intelligent-machine-learning/easydl/actions/workflows/main.yml)

DLRover automatically trains the Deep Learning model on the distributed cluster. It helps model developers to focus on model arichtecture, without taking care of any engineering stuff, say, hardware acceleration, distributed running, etc. Now, it provides automated operation and maintenance for deep learning training jobs on K8s/Ray. Major features as

- **Automatic Resource Optimization**, Automatically optimize the job resource to improve the training performance and resources utilization.
- **Fault-Tolerance**, single node failover without restarting the entire job.
- **Auto-Scaling**, Automatically scale up/down resources at both node level and CPU/memory level.

## Why DLRover?
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

### Fault Tolerance to Improve the Stability of Job.

DLRover can recover failed parameter servers and workers to
resume training. Some failed nodes do not interrupt the training and hurt the convergence accuracy. The main error is
OOM of node due to user's insufficient memory configuration.
DLRover can automatically launch a Pod with more memory to recover the OOM node. In AntGroup, DLRover manages hundreds of DL training jobs every day on the customized Kubernetes cluster in AntGroup.
Except for the failed job resulting from code errors, the rate of completed jobs raise 89% with tf-operator in KubeFlow to 95%. Other unrecoverable failure reasons of a job are data error, NaN loss of the model, network breakdown, and so on.

<div align="center">
<img src="docs/figures/job-complete-rate.png" alt="Editor" width="600">
</div>

### Auto-Scaling to Improve Training Performance.

DLRover automatically scales up/down resources (for parameter servers or workers) at the runtime of a training job.
By monitoring the workload of nodes and throughput, DLRover can diagnose the bottleneck of the resource configuration.
The common bottleneck contains node straggler, the unbalanced workload of PS, insufficient CPU cores of nodes, and the insufficient number of nodes. DLRover can improve the training performance by dynamic resource adjustment.

We use the dataset of [Kaggle CRITEO](https://www.kaggle.com/c/criteo-display-ad-challenge)
to train Wide&Deep and xDeepFM with 10 epoches on a K8s cluster.
DLRover can mitigate straggler to improve the training throughput
and shorten the job competion time (JCT).

<div align="center">
<img src="docs/figures/exp-jct-deepctr.png" alt="Editor" width="600">
</div>



### Auto-Scaling to improve Resource Utilization.

Different model training requires different resources. Users prefer to
configure their jobs with over-provision resources to 
avoid any potential risk from insufficient resources.
This usually ends up in huge resource waste. DLRover Auto-Scaling
can allocate resources by the demand of model training to reduce
the waste of resources.

<div align="center">
<img src="docs/figures/daily-job-resource-util.png" alt="Editor" width="1000">
</div>

### Dynamic data sharding
TODO...



### Integration to Offline and Online Deep Learning.

With the data source transparency provided by dynamic data sharding, DLRover can be integrated with offline training which consumes batch data, and also supports online learning with real-time streaming data. (fed with a message queue like RocketMQ/Kafka/Pulsar/..., or executed as a training sink node inside Flink/Spark/Ray/...)

By practice, DLRover is an ideal component to build an end-to-end industrial online learning system, [estimator.md](docs/tutorial/estimator.md) provides a detailed example implemented with `tf.estimator.Estimator`.

## What's Next?

- Automatic Distributed training for GPU Synchronous jobs
  - data-parallel mode
  - hybrid-parallel mode
  - adapted hyper parameters adjustment with dynamic resources
- Full stack solution for Online Deep Learning
- ...


## Quick Start

[Train a TensorFlow Estimator on Aliyun ACK](docs/tutorial/dlrover_cloud.md)
