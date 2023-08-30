# Design to Scale Nodes of a Job

The design described how DLrover scales up/down nodes of a job.

## Motivation

Auto-scaling is the key to automatic maintenance of a DL training job.
DLRover should support to add/remove nodes to a job without interrupting
the training process.

## Design

Auto-scaling in DLRover contains the following steps:

- The `JobResourceOptimizer` in `JobManager` queries a `ResourcePlan`.
- The `TrainingNodeManager` (e.g. `PSManager` and `WorkerManger`)
in `JobManager` generates the `ScalePlan`.
- The `Scaler` in `JobManager` adds/removes nodes according to
 the `ScalePlan`.

### Job Resource Optimizer

`JobResourceOptimizer` splits  a job's lifecycl into
4 stages `CREATE`, `WORKER_INITIAL`, `PS_INITIAL` and `RUNNING`.

- CREATE: There is no any runtime statistics of the job. The resource
optimizer need to generate the resource plan by statis configuration
of a job. The resource plan contains the resource of chief (the first
worker) and PS.

- WORKER_INITIAL: There is runtime statistics of the chief and PS
to execute the training process. The resource optimizer need to
generate the number of workers to estimate the resource requirement
of PS.

- PS_INITIAL: At the stage, the resource optimizer need to predict
the resource plan of PS by the runtime statistics of multiple workers
and PS.

- RUNNING: At the stage, the resource optimzier monitor the bottleneck
of the job and ajust resource to mitigate the bottleneck.

At each stage, `JobResourceOptimizer` queries a `ResourcePlan` by calling its
`ResourceOptimizer`.

The `ResourcePlan` contains resource configurations of training nodes. For
exampel:

```Python
resource_plan = ResourcePlan()
resource_plan.node_group_resources["ps"] = NodeGroupResource(
    count=3, node_resource=NodeResource(cpu=8, memory=10240)
)
resource_plan.node_group_resources["worker"] = NodeGroupResource(
    count=3,
    node_resource=NodeResource(
        cpu=8, memory=10240, gpu_type="v100", gpu_num=1, priority="high"
    )
)
resource_plan.node_resources["training-edljob-worker-0"] =  NodeResource(
    cpu=8, memory=10240, gpu_type="v100", gpu_num=1, priority="high"
)
```

The `ResourceOptimizer` contains the following interfaces.

```Python
class ResourceOptimizer(metaclass=ABCMeta):
    @abstractmethod
    def generate_opt_plan(self, stage, config={}) -> ResourcePlan:
        """Generate a resource configuration plan"""
        pass

    @abstractmethod
    def generate_oom_recovery_plan(
        self, oom_nodes, stage, config={}
    ) -> ResourcePlan:
        """Generate a recovery plan for OOM nodes"""
        pass

    @abstractmethod
    def generate_resource_plan_with_optimizer(self, config={}) -> ResourcePlan:
        """Generate a resource plan by an optimizer"""
        pass
```

#### Local Optimizer for Single-Job Mode

`LocalOptimizer` is designed for Single-Job Mode when there is no optimization
service on the cluster. `LocalOptimizer` collects job runtime statistics and
store those data in memory. It can generate optimized `ResourcePlan` by
its own job runtime statistics but without the information of the cluster.

#### Brain Optimizer for Cluster Mode

`BrainOptimizer` is designed for Cluster Mode when the Brain optimization service
it deployed. `BrainOptimizer` is the client to query optimization resource plan
from the service. When using `BrainOptimizer`, job runtime statistics is
send to the Brain service by the `StatsReporter` and persisted into a database.

### Training Node Manger

`TrainingNodeManger` manages the node status and generate a `ScalePlan` by
a `ResourcePlan` base on the current the status of all nodes. The `ScalePlan`
contains nodes to be luanched or removed. For example,

```Python
scale_plan = ScalePlan()
scale_plan.node_group_resource =NodeGroupResource(
    3, 
    NodeResource(
        cpu=8, memory=10240, gpu_type="v100", gpu_num=1, priority="high"
    )
)  # Scale up the number of workers to 3 with the NodeResource.
scale_plan.launch_nodes = [
    Node(NodeType.WORKER, 0, NodeResource(cpu=9, memory=10240))
]  # New worker to be launched
scale_plan.remove_nodes = [
    Node(NodeType.WORKER, 1, NodeResource(cpu=9, memory=10240), name="worker-0")
]  # New worker to be removed
scale_plan.ps_addrs = ["ps-0:2222", "ps-1:2222"]  ## The host set of PS nodes.

```

#### PS Manager

`PSManager` manages the status of PS nodes by the `ResourcePlan`
and node events from `NodeMonitor`. `PSManager` also generate
a `ScalePlan` when migrating PS.

**Scale Up PS.**
If the number of PS in `ResourcePlan` is larger than the current number
of PS. `PSManager` will create new PS `Node`s and increase the total PS
number to the required value in the plan.
Those new nodes' status is set to NodeStatus.INITIAL. `PSManger` will set the status
by node events from `NodeMonitor`. After all PS nodes are running, `PSManager`
will update its `_next_training_ps_cluster` and inform workers to connect the
new PS cluster.

**Scale Down PS.**
If the number of PS is smaller than the current number of PS. `PSManager` will
not delete the additional PS nodes immediately. Because model parameters are
stored across PS nodes and will be lost if we delele PS nodes before
workers checkpoints model parameters on PS.
`PSManager` will  add those PS nodes which is to be removed
to a queuee `_pre_dropped_ps` and remove those PS hosts from
its `_next_training_ps_cluster`. After workers succeed to checkpoint model parameters
and connect the next PS cluster. `PSManager` will set those those PS nodes into `remove_nodes`
of a `ScalePlan`.

**Migrate PS.**
If there is a updatation in a PS node's resource in `ResourcePlan.node_resources`,
`PSManager` will create a PS `Node` with the new resource.
After the new PS node is running, `PSManager`
will update its `_next_training_ps_cluster` and notify
workers to connect new PS clusters. After workers succeed to connect new PS
cluster, `PSManager` will set the old PS node into `remove_nodes`
of a `ScalePlan`.

#### Worker Manager

`WorkerManager` manages the status of worker nodes by the `ResourcePlan`
and node events from `NodeMonitor`.

**Scale up workers.**
If the number of worker is larger than the current number of workers,
`WorkerManager` will create new `Node`s with the status `NodeStatus.INITIAL`
and update the node status by node events from `NodeMonitor`.

**Scale down workers.**
If the number of worker is smaller than the current number of workers,
`WorkerManager` will set `relaunchable=False` and `is_released=True`
for additional workers to be removed.

### Scaler

`Scalcer` manges the lifecycle of training nodes by the `ScalePlan`. It can
create/update/delete nodes to achieve the `ScalePlan`. We can implement
differenct `Scaler` for different distributed cluster.

#### Pod Scaler

`PodScaler` is implemented by K8s Python APIs to create/update/delete Pods
on a K8s cluster.

**Scale up Pods.**
`PodScaler` starts a creating thread to periodicall create Pods from a `Node` queue.
If the number of Pods in `ScalePlan.node_group_resource` is
more than the current number of Pods, `PodScaler`
will create `Node`s into a queue. If `PodScaler` fail to create a Pod, it
will replace the `Node` into the queue to retry.

**Scale down Pods.** If there are nodes in `ScalePlan.remove_nodes`, `PodScaler` will delete the Pod
by the name of the node in `remove_nodes`.

#### ElasticJob Scaler

`ElasticJobScaler` is implemented to create a `ScalePlan` CRD to notify the
[ElasticJob controller](docs/design/elastic-training-operator.md) to
reconcile Pods by the `ScalePlan` on a K8s cluster. The example of `ScalePlan` is

```yaml
apiVersion: elastic.iml.github.io/v1alpha1
kind: ScalePlan
metadata:
  name: scaleplan-sample
  labels:
    elasticjob-name: elasticjob-sample
    scale-type: manual
spec:
  ownerJob: elasticjob-sample
  replicaResourceSpecs:
    ps:
      replicas: 1
      resource:
        cpu: "0.5"
        memory: 256Mi
    worker:
      replicas: 2
      resource:
        cpu: "0.5"
        memory: 256Mi
```

Compared with `PodScaler` with k8s Python APIs, `ElasticJobScaler` is
more maintainable and simple than `PodScaler`. What's more, it is
more convenient to be intergrated into other platforms with operators
on a customized K8s cluster.
