# DLRover

<div align="center">
<img src="docs/figures/dlrover_logo.png" alt="Editor" width="350">
</div>

<div id="top" align="center">
DLRover: An Automatic Distributed Deep Learning System
</div>

[![Build](https://github.com/intelligent-machine-learning/easydl/actions/workflows/main.yml/badge.svg)](https://github.com/intelligent-machine-learning/easydl/actions/workflows/main.yml)

DLRover automatically trains the Deep Learning model on the distributed cluster. It helps model developers to focus on model arichtecture, without taking care of any engineering stuff, say, hardware acceleration, distributed running, etc. Now, it provides automated operation and maintenance for deep learning training jobs on K8s/Ray. Major features as

- **Fault-Tolerance**, single node failover without restarting the entire job.
- **Auto-Scaling**, Automatically scale up/down resources at both node level and CPU/memory level.
- **Dynamic data sharding**, dynamic dispatch training data to each worker instead of dividing equally, faster worker more data.
- **Automatic Resource Optimization**, Automatically optimize the job resource to improve the training performance and resources utilization.

## Why DLRover?

<div align="center">
   <a href="https://www.bilibili.com/video/BV1Nk4y1N7fx/?vd_source=603516da01339dc75fb908e1cce180c7">
   <img src="docs/figures/dlrover-cover.jpg" width="700" />
   </a>
</div>

### Fault Tolerance to Improve the Stability of Job.

DLRover can restore the training when the process fails without stopping the
training job. The actions to restore training in DLRover are:

1. Diagnose the failure reason.
2. Restart the process not the node due to software errors.
3. Restart the failed nodes due to hardward errors.

#### Fault Tolerance of PyTorch Distributed Training.

DLRover supports fault tolerance of the process failure and the node failure
to restore trainig. Compared with restarting a new job, DLRover can
reduce the overhead to schedule all Pods, pull image and
install  packages on all nodes.  

|  Step to restore training |  Failure without DLRover  |     Node failure with DLRover     |    Process failure with DLRover   |
|:-------------------------:|:-------------------------:|:---------------------------------:|:---------------------------------:|
|       Restore action      |        Restart Job        |        Restart failed nodes       |      Restart training process     |
|  Schedule node, pull image and install packages   |  All nodes |       Only new nodes      |                 No                |
| Node health check         |            No             | All nodes execute a simple allgtather task | All nodes execute a allgtather simple task |
| Build communication world |            Yes            |                Yes                |                Yes                |
|   Start training process  |            Yes            |                Yes                |                Yes                |
|     Restore checkpoint    |            Yes            |                Yes                |                Yes                |

#### Fault Tolerance of TensorFlow PS Distributed Training.

DLRover can recover failed parameter servers and workers to
resume training. Compared with manual restarting jobs, DLRover
can reduce the overhead to restore the training.

|             Step to restore training            |  Failure without DLRover  |   PS failure with DLRover  | Worker failure with DLRover |
|:-----------------------------------------------:|:-------------------------:|:--------------------------:|:---------------------------:|
|                  Restore action                 |        Restart Job        |      Restart failed PS     |    Restart failed workers   |
| Schedule node, pull image and install packages | All nodes               |         Only new PS        |     Only new workers        |
|                  Start session                  |         all nodes         |          all nodes         |       Only new workers      |
|                 Initialize Graph                |            Yes            |             Yes            |       Only new workers      |
|                Restore checkpoint               |            Yes            |             Yes            |              No             |


What's more, DLRover also can automatic diagnose the reason of failure. For example,
the OOM is the common error due to user's insufficient memory configuration.
DLRover can automatically launch a Pod with more memory to recover the OOM node. In AntGroup, DLRover manages hundreds of DL training jobs every day on the customized Kubernetes cluster in AntGroup.
Except for the failed job resulting from code errors, the rate of completed jobs raise 89% with tf-operator in KubeFlow to 95%. Other unrecoverable failure reasons of a job are data error, NaN loss of the model, network breakdown, and so on.

<div align="center">
<img src="docs/figures/job-complete-rate.png" alt="Editor" width="600">
</div>


### No Resource Configuration to Submit a Job.

Compared with Training Job (e.g., TensorFlow, PyTorch etc) in Kubeflow,
Users can  submit a distributed training job without any resource configuration.

<div align="center">
<img src="docs/figures/dlrover_vs_tfjob.jpg" alt="Editor" width="600">
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

### Auto-Scaling to Improve Resource Utilization.

Different model training requires different resources. Users prefer to
configure their jobs with over-provision resources to 
avoid any potential risk from insufficient resources.
This usually ends up in huge resource waste. DLRover Auto-Scaling
can allocate resources by the demand of model training to reduce
the waste of resources.

<div align="center">
<img src="docs/figures/daily-job-resource-util.png" alt="Editor" width="1000">
</div>

### Dynamic Data Sharding For Elasticity and Fault-tolerance.

Dynamic data sharding splits the dataset into many small shards and each shard only
contains a few batches of training samples. The worker will get a shard only when it using up
samples of the last one. With the dynaic sharding, DLRover can

- recover the shard if the worker fails before using up samples of the shard.
- mitigate the worker straggler by assigning more shards to the fast worker.

### Integration to Offline and Online Deep Learning.

With the data source transparency provided by dynamic data sharding, DLRover can be integrated with offline training which consumes batch data, and also supports online learning with real-time streaming data. (fed with a message queue like RocketMQ/Kafka/Pulsar/..., or executed as a training sink node inside Flink/Spark/Ray/...)

By practice, DLRover is an ideal component to build an end-to-end industrial online learning system, [estimator.md](docs/tutorial/estimator.md) provides a detailed example implemented with `tf.estimator.Estimator`.

## How to Use DLRover to Train Your Models?

Firstly, the user need to deploy the DLRover elasticjob controller in a kubernetes
cluster by followding the [tutorial](docs/deployment/controller.md).

### Train a PyTorch Model.

Only by 2 steps, the user can use DLRover to run the training script which 
`torchrun` or `torch.distributed.run` can run.

- Install dlrover[torch] in the training image with the command 

```bash
pip install dlrover[torch]
```

- Use `dlrover-run` to run the training script. 

```bash
dlrover-run
    --nnodes=$NUM_NODES
    --nproc_per_node=$$NUM_TRAINERS
    train_scripts.py
```

- Set the image and command in an ElasticJob yaml file to submit a job.
We can refer to the example [torch_mnist_job.yaml](dlrover/examples/torch_mnist_job.yaml)
to make an ElasticJob yaml.

### Train a TensorFlow Model

We can use DLRover to train a TensorFlow by the following steps:

- Use TensorFlow estimator to develop the TensorFlow model.
- Define the input of `tf.dataset` in a training configuration of DLRover.
- Define your reader to read samples from the dataset file.

We can refer to the [estimator.md](docs/tutorial/estimator.md) to train
a model with DLRover.

## What's Next?

- Fine-grained automatic distributed training for GPU Synchronous jobs
  - hybrid-parallel mode
  - adapted hyper parameters adjustment with dynamic resources
  - more strategies for Fine-grained scenarioes
- Full stack solution for Online Deep Learning
- High performance extension library for Tensorflow/Pytorch to speed up training
- ...

## Contributing
Please refer to the [DEVELOPMENT](docs/developer_guide.md)

## Quick Start

[Train a TensorFlow Estimator on Aliyun ACK](docs/tutorial/tf_ps_on_cloud.md)

[Train a PyTorch Model on Aliyun ACK](docs/tutorial/torch_allreduce_on_cloud.md)
