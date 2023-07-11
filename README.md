# DLRover

<div align="center">
<img src="docs/figures/dlrover_logo.png" alt="Editor" width="350">
</div>

<div id="top" align="center">
DLRover: An Automatic Distributed Deep Learning System
</div>

[![Build](https://github.com/intelligent-machine-learning/easydl/actions/workflows/main.yml/badge.svg)](https://github.com/intelligent-machine-learning/easydl/actions/workflows/main.yml)

DLRover automatically trains the Deep Learning model on the distributed cluster. It helps model developers to focus on model arichtecture, without taking care of any engineering stuff, say, hardware acceleration, distributed running, etc. Now, it provides automated operation and maintenance for deep learning training jobs on K8s/Ray. Major features as

- **Automatic Resource Optimization**, Automatically optimize the job resource to improve the training performance and resources utilization.
- **Dynamic data sharding**, dynamic dispatch training data to each worker instead of dividing equally, faster worker more data.
- **Fault-Tolerance**, single node failover without restarting the entire job.
- **Auto-Scaling**, Automatically scale up/down resources at both node level and CPU/memory level.

## Why DLRover?

<div align="center">
   <a href="https://www.bilibili.com/video/BV1Nk4y1N7fx/?vd_source=603516da01339dc75fb908e1cce180c7">
   <img src="docs/figures/dlrover-cover.jpg" width="700" />
   </a>
</div>

### No Resource Configuration to Submit a Job.

Compared with TFJob in Kubeflow, Users need not to set any resource configuration to submit a
distributed training job. 
<div align="center">
<img src="docs/figures/dlrover_vs_tfjob.jpg" alt="Editor" width="600">
</div>

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

There are at two reasons why we need dynamic data sharding. The first one is that Dlrover needs to ensure the training data used to fit the model is processed as users expect. When workers recover from failure or are scaled up/down, they may consume training data twice or miss some training data due to a lack of a global coordinator. With dynamic data sharding, DLrover keeps track of every worker's data consumption and try its best to ensurse that data is delievered exact once/at least once/at most once streaming-data-splitter-and-manager.md. As a result, dynamic data sharding helps to eliminate uncertainties by ensuring data is consumed as users expect.

The second one is that dynamic data sharding reduces complexity for worker to deal with obtaining training data. As the training-master.md indicates, worker only needs to ask for data shard from the DLrover master without interacting with other worker to split data.

Dynamic data sharding can also mitigate the worker straggler. After a worker starts its training loop, it queries a shard from the TODO queue one by one. The fast worker will consumes more shards than the slow worker which is a straggler.
### Integration to Offline and Online Deep Learning.

With the data source transparency provided by dynamic data sharding, DLRover can be integrated with offline training which consumes batch data, and also supports online learning with real-time streaming data. (fed with a message queue like RocketMQ/Kafka/Pulsar/..., or executed as a training sink node inside Flink/Spark/Ray/...)

By practice, DLRover is an ideal component to build an end-to-end industrial online learning system, [estimator.md](docs/tutorial/estimator.md) provides a detailed example implemented with `tf.estimator.Estimator`.

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
