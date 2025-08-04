# GangScheduling Best Practise

The document described how to provide Gang mechanism for the multi-node distributed training.

## Background

Consider the resource limited cluster, which CPU, memory or GPU, are insufficient for the requirement of the whole training. During the training, only a part of pod can be scheduled. However, the training job can not run normally because the number of pods is always less than the min requirement. Meanwhile, the resources, which have already been allocated to the scheduled pods, cannot be released for other jobs, leading to the waste of cluster resources.
With Gang mechanism, users can declare a resource-collection-minimum number. All-or-nothing pods could be scheduled which depends on whether cluster resources reached the given limitation. The condition can be avoided that resources are occupied by the impossible training job.

## Target
1. For multi-node distributed training ElasticJobs, all-or-nothing Gang scheduling with `Koordinator`
2. `Suspend` API in v0.5.0 version. If the suspend filed of the ElasticJob is set to be true, the training will be suspended once created and no worker pods will be created until the suspend filed is set to false manually.

## Best Practise

#### Step 1: Preliminary

1. Install Koordinator (v1.6 commanded)
2. Install DLRover (v0.5.0 necessary)

#### Step 2: Submit ElasticJob with Gang scheduling annotations

1. Submit the multi-node ElasticJob in the cluster
```yaml
apiVersion: elastic.iml.github.io/v1alpha1
kind: ElasticJob
metadata:
  name: torch-mnist-gang
  namespace: dlrover
spec:
  distributionStrategy: AllreduceStrategy
  optimizeMode: single-job
  replicaSpecs:
    worker:
      replicas: 4
      template:
        metadata:
          annotations:
            gang.scheduling.koordinator.sh/name: torch-mnist-gang
            gang.scheduling.koordinator.sh/min-available: 4
        spec:
          restartPolicy: Always
          containers:
            - name: main
              # yamllint disable-line rule:line-length
              image: registry.cn-hangzhou.aliyuncs.com/intell-ai/dlrover:pytorch-example
              imagePullPolicy: Always
              command:
                - /bin/bash
                - -c
                # NODE_NUM is set into env with the value as replicas.
                - "dlrover-run --network-check --nnodes=3:$NODE_NUM \
                  --nproc_per_node=2 --max_restarts=3  \
                  examples/pytorch/mnist/cnn_train.py --num_epochs 5 \
                  --training_data /data/mnist_png/training/ \
                  --validation_data /data/mnist_png/testing/"
              resources:
                limits:
                  cpu: "4"  # turn up when using GPU
                  memory: 8Gi  # turn up when using GPU
                  nvidia.com/gpu: 1 # optional
                requests:
                  cpu: "4"  # turn up when using GPU
                  memory: 8Gi  # turn up when using GPU
                  nvidia.com/gpu: 1  # optional
```
Compare with the `examples/pytorch/mnist/elastic_job.yaml`, two gang-scheduling  annotations are added in the `spec.replicaSpecs[x].template.metadata.annotations`. The detailed explanation of the annotations as follows.

| Key | Defination |  Value |
|----|:----:|-------:|
| gang.scheduling.koordinator.sh/name | The name of the gang group. The same name in different namespaces are considered as different gang groups. | String |
| gang.scheduling.koordinator.sh/min-available | The min num for the gang group which can be scheduled |    int |

2. GangScheduling
There are 2 GPU in the cluster, which is less than the min-available of the gang group. Therefore, all worker pods are pending.
```bash
NAME                                          READY   STATUS    RESTARTS   AGE
dlrover-controller-manager-5799c85445-rfmjp   2/2     Running   0          17m
elasticjob-torch-mnist-dlrover-master-0       1/1     Running   0          3m45s
torch-mnist-edljob-worker-0                   0/1     Pending   0          3m38s
torch-mnist-edljob-worker-1                   0/1     Pending   0          3m38s
torch-mnist-edljob-worker-2                   0/1     Pending   0          3m38s
torch-mnist-edljob-worker-3                   0/1     Pending   0          3m38s
```
```bash
kubectl describe pod torch-mnist-edljob-worker-2 -n dlrover
```
The FailedScheduling events will show.
```bash
Events:
  Type     Reason            Age                From               Message
  ----     ------            ----               ----               -------
  Warning  FailedScheduling  15s                default-scheduler  Gang "dlrover/torch-mnist" gets rejected due to member Pod "torch-mnist-edljob-worker-2" is unschedulable with reason "0/13 nodes are available: 2 Insufficient koordinator.sh/gpu-core, 2 Insufficient koordinator.sh/gpu-memory-ratio, 3 node(s) had untolerated taint {node-role.kubernetes.io/control-plane: }, 8 node(s) didn't match Pod's node affinity/selector."
  Warning  FailedScheduling  10s (x2 over 13s)  default-scheduler  Gang "dlrover/torch-mnist" gets rejected due to member Pod "torch-mnist-edljob-worker-2" is unschedulable with reason "0/13 nodes are available: 2 Insufficient koordinator.sh/gpu-core, 2 Insufficient koordinator.sh/gpu-memory-ratio, 3 node(s) had untolerated taint {node-role.kubernetes.io/control-plane: }, 8 node(s) didn't match Pod's node affinity/selector."
```
3. Adjust GangScheduling configuration
Set the `annotationgang.scheduling.koordinator.sh/min-available=2` and recreate the ElasticJob. There will be two worker pods scheduled successfully and tow pods are pending.
```bash
NAME                                          READY   STATUS    RESTARTS   AGE
dlrover-controller-manager-5799c85445-rfmjp   2/2     Running   0          38m
elasticjob-torch-mnist-dlrover-master-0       1/1     Running   0          15m
torch-mnist-edljob-worker-0                   1/1     Running   0          15m
torch-mnist-edljob-worker-1                   0/1     Pending   0          15m
torch-mnist-edljob-worker-2                   0/1     Pending   0          15m
torch-mnist-edljob-worker-3                   1/1     Running   0          15m
```
#### Step 3: GangScheduling using Suspend API
If there is no Koordinate or other GangScheduling scheduler in the cluster, the Suspend API can be used instead of the gang-scheduling scheduler.
1. Set `ElasticJob.Spec.suspend=true` and create the ElasticJob.
```yaml
---
apiVersion: elastic.iml.github.io/v1alpha1
kind: ElasticJob
metadata:
  name: torch-mnist
  namespace: dlrover
spec:
  distributionStrategy: AllreduceStrategy
  optimizeMode: single-job
  suspend: true
  replicaSpecs:
    worker:
      replicas: 4
      template:
        spec:
          restartPolicy: Always
          containers:
            - name: main
              # yamllint disable-line rule:line-length
              image: registry.cn-hangzhou.aliyuncs.com/intell-ai/dlrover:pytorch-example
              imagePullPolicy: Always
              command:
                - /bin/bash
                - -c
                # NODE_NUM is set into env with the value as replicas.
                - "dlrover-run --network-check --nnodes=3:$NODE_NUM \
                  --nproc_per_node=2 --max_restarts=3  \
                  examples/pytorch/mnist/cnn_train.py --num_epochs 5 \
                  --training_data /data/mnist_png/training/ \
                  --validation_data /data/mnist_png/testing/"
              resources:
                limits:
                  cpu: "2"  # turn up when using GPU
                  memory: 3Gi  # turn up when using GPU
                  # nvidia.com/gpu: 1 # optional
                requests:
                  cpu: "2"  # turn up when using GPU
                  memory: 3Gi  # turn up when using GPU
```
2. The torch-mnist Job are suspended and only the master pod is created.
```bash
NAME          PHASE       AGE
torch-mnist   Suspended   28s
```
```bash
NAME                                          READY   STATUS    RESTARTS   AGE
dlrover-controller-manager-6dcfd55c89-sbnjq   2/2     Running   0          3m2s
elasticjob-torch-mnist-dlrover-master-0       1/1     Running   0          68s
```
3. Unsuspend the training job
During this time, you can check if the cluster resources are sufficient for the whole training.
The Suspended status will continue until set `suspend=false`. Then, the master pod continues to create worker pods to start the distributed training.
```bash
NAME          PHASE     AGE
torch-mnist   Running   2m52s
```
```bash
NAME                                          READY   STATUS    RESTARTS   AGE
dlrover-controller-manager-5799c85445-rfmjp   2/2     Running   0          38m
elasticjob-torch-mnist-dlrover-master-0       1/1     Running   0          15m
torch-mnist-edljob-worker-0                   1/1     Running   0          15m
torch-mnist-edljob-worker-1                   0/1     Pending   0          15m
torch-mnist-edljob-worker-2                   0/1     Pending   0          15m
torch-mnist-edljob-worker-3                   1/1     Running   0          15m
```