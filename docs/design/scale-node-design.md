# Design to Scale Nodes of a Job

The design described how DLrover scales up/down nodes of a job.

## Motivation

AutoScaling is the key to automatic maintenance of a DL training job.
DLRover should support add/remove nodes to a job without interrupting
the training process to improve the training performance.

## Design

Auto-scaling in DLRover contains the following steps:
- The `JobResourceOptimizer` in `JobManager` queries a `ResourcePlan`.
- The `TrainingNodeManager` (e.g. `PSManager` and `WorkerManger`)
in `JobManager` generates the `ScalePlan`.
- The `Scaler` in `JobManager` adds/removes nodes by the `ScalePlan`.

### Job Resource Optimizer

`JobResourceOptimizer` splits the resource optimization into
4 stages `CREATE`, `WORKER_INITIAL`, `PS_INITIAL` and `RUNNING`.
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

#### Local Optimizer

`LocalOptimizer` is designed for Single-Job Mode when there is no optimization
service on the cluster. `LocalOptimizer` collects job runtime statistics and
store those data in memory. It can generate optimized `ResourcePlan` by
its own job runtime statistics but without the information of the cluster.

#### Remote Brain Optimizer

`BrainOptimizer` is designed for Cluster Mode when the Brain optimizations service
it deployed. `BrainOptimizer` is the client to query optimization resource plan
from the service. When using `BrainOptimizer`, job runtime statistics is
send to the Brain service by the `StatsReporter` and persisted into a database.

### Training Node Manger

`TrainingNodeManger` manages the node status and generate a `ScalePlan` by
a `ResourcePlan` base on the current the status of all nodes. The `ScalePlan`
contains nodes to be luanched or removed. For example,

```Python
scale_plan = ScalePlan()
scale_pan.node_group_resource =NodeGroupResource(
    3, 
    NodeResource(
        cpu=8, memory=10240, gpu_type="v100", gpu_num=1, priority="high"
    )
)  # Scale up the number of workers to 3 with the NodeResource.
scale_pan.launch_nodes = [
    Node(NodeType.WORKER, 0, NodeResource(cpu=9, memory=10240))
]  # New worker to be launched
scale_pan.remove_nodes = [
    Node(NodeType.WORKER, 1, NodeResource(cpu=9, memory=10240), name="worker-0")
]  # New worker to be removed
scale_pan.ps_addrs = ["ps-0:2222", "ps-1:2222"]  ## The host set of PS nodes.

```

#### PS Manager

`PSManager` manages the status of PS nodes by the `ResourcePlan`
and node events from `NodeMonitor`. `PSManager` also generate
a `ScalePlan` when migrating PS.

**Scale Up PS.**
If the number of PS in `ResourcePlan` is bigger than the current number
of PS. `PSManager` will create new PS `Node` by the difference and set
the node status to `NodeStatus.INITIAL`. `PSManger` will set the status
by node events from `NodeMonitor`. If all PS nodes are running, `PSManager`
will update its `_next_training_ps_cluster` which is used to notify
workers to connect new PS clusters.

**Scale Down PS.**
If the number of PS is less than the current number of PS. `PSManager` will
not delete the additional PS nodes immediately because model parameters are
stored across PS nodes. `PSManager` will place those additional PS nodes
to a queue `_pre_dropped_ps` and remove those PS hosts from 
its `_next_training_ps_cluster`. After workers succeed to connect the next
PS cluster. `PSManager` will set those additional PS nodes into `remove_nodes`
of a `ScalePlan`.

**Migrate PS.**
If there is a PS resource in `ResourcePlan.node_resources` which is different
from the current resource of the PS node, `PSManager` will create a `Node`
with the new resource. After the new PS node is running, `PSManager`
will update its `_next_training_ps_cluster` and notify
workers to connect new PS clusters. After workers succeed to connect new PS
cluster, `PSManager` will set the old PS node into `remove_nodes`
of a `ScalePlan`.

#### Worker Manager

`WorkerManager` manages the status of worker nodes by the `ResourcePlan`
and node events from `NodeMonitor`.

**Scale up workers.**
If the number of worker is bigger than the current number of workers,
`WorkerManager` will create new `Node`s with the status `NodeStatus.INITIAL`
and update the node status by node events from `NodeMonitor`.

**Scale down workers.**
If the number of worker is less than the current number of workers,
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
If the number of Pods in `ScalePlan.node_group_resource`, `PodScaler`
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
spec:
  ownerJob: elasticjob-sample
  manualScaling: True
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
