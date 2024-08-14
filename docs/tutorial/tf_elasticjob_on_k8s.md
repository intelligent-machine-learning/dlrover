# PS Training Using DLRover on Public Cloud

This document explains how to run a DLRover elastic job for PS training
with on a public cloud, namely, Alibaba Cloud Container Service for Kubernetes(ACK).

## Preliminary

- Create a Kubernetes cluster on [ACK](https://help.aliyun.com/document_detail/309552.htm?spm=a2c4g.11186623.0.0.168f6b7aegH7nI#task-2112671).
- Configure cluster credentials on your local computer.
- Create a [NAS](https://help.aliyun.com/document_detail/477380.html?spm=a2c4g.11186623.0.0.10635c83Xn7Tkh)
storage and mount it to the cluster.

## Deploy the ElasticJob CRD on ACK

1. Deploy the controller on the cluster.

```bash
make deploy IMG=easydl/elasticjob-controller:v0.1.1
```

1. Grant permission for the DLRover master to Access CRDs.

```bash
kubectl -n dlrover apply -f dlrover/go/operator/config/rbac/default_role.yaml 
```

## Submit an Auto-Scaling Job

- Submit a job to train a DeepFM model without specified resource.

```bash
kubectl -n dlrover apply -f examples/tensorflow/criteo_deeprec/autoscale_job.yaml
```

- Check the job status

```bash
kubectl -n dlrover get elasticjob deepctr-auto-scale
```

```bash
NAME                 PHASE     AGE
deepctr-auto-scale   Running   4s
```

- Check the Pod status

```bash
kubectl -n dlrover get pods -l elasticjob.dlrover/name=deepctr-auto-scale
```

```bash
NAME                                           READY   STATUS    RESTARTS   AGE
deepctr-auto-scale-edljob-chief-0              1/1     Running   0          78s
deepctr-auto-scale-edljob-evaluator-0          1/1     Running   0          78s
deepctr-auto-scale-edljob-ps-0                 1/1     Running   0          78s
elasticjob-deepctr-auto-scale-dlrover-master   1/1     Running   0          82s
```

Now, the speed is about 30 steps/s. After about 3min, DLRover scales up 3 workers
and the speed is up to 100 steps/s.

```bash
NAME                                          READY   STATUS    RESTARTS   AGE
dlrover-auto-scale-edljob-chief-0             1/1     Running   0          6m17s
dlrover-auto-scale-edljob-ps-0                1/1     Running   0          6m17s
dlrover-auto-scale-edljob-worker-0            1/1     Running   0          3m19s
dlrover-auto-scale-edljob-worker-1            1/1     Running   0          3m19s
dlrover-auto-scale-edljob-worker-2            1/1     Running   0          3m19s
```

## Submit a Mannul Scaling Job

### Submit a job with initial resource configuration

- Submit a job with the DeepFM model.

```bash
kubectl -n dlrover apply -f examples/tensorflow/criteo_deeprec/manual_job.yaml
```

- Check the job status

```bash
kubectl -n dlrover get elasticjob deepctr-manual-scaling
```

```bash
NAME                    PHASE     AGE
deepctr-manual-scaling  Running   2m20s
```

- Check the Pod status

```bash
kubectl -n dlrover get pods -l elasticjob-name=deepctr-manual-scaling
```

```bash
NAME                                               READY   STATUS    RESTARTS   AGE
deepctr-manual-scale-edljob-chief-0                1/1     Running   0          12s
deepctr-manual-scale-edljob-worker-0               1/1     Running   0          12s
deepctr-manual-scale-edljob-ps-0                   1/1     Running   0          12s
elasticjob-deepctr-manual-scaling-dlrover-master   1/1     Running   0          19s
```

### Mannually Scale Nodes of a Job

We can submit a ScalePlan CRD to scale up/down nodes of a job.
In a ScalePlan, we need to set `metadata.labels` to specify
which job to scale and `metadata.labels["scale-type"]` to "manual".
For example, the ScalePlan is to scale
workers of the job `deepctr-manual-scaling`.

```yaml
apiVersion: elastic.iml.github.io/v1alpha1
kind: ScalePlan
metadata:
  name: deepctr-manual-scale-plan-0
  labels:
    elasticjob.dlrover/name: deepctr-manual-scale
    scale-type: manual
spec:
  ownerJob: deepctr-manual-scale
  replicaResourceSpecs:
    worker:
      replicas: 2
```

After scaling, there two worker nodes:

``` bash
NAME                                             READY   STATUS    RESTARTS   AGE
deepctr-manual-scale-edljob-chief-0              1/1     Running   0          14m
deepctr-manual-scale-edljob-ps-0                 1/1     Running   0          14m
deepctr-manual-scale-edljob-worker-0             1/1     Running   0          14s
deepctr-manual-scale-edljob-worker-1             1/1     Running   0          3s
elasticjob-deepctr-manual-scale-dlrover-master   1/1     Running   0          14m
```

We can scale up PS nodes with the spec in ScalePlan like

```yaml
apiVersion: elastic.iml.github.io/v1alpha1
kind: ScalePlan
metadata:
  namespace: dlrover
  name: deepctr-manual-scale-plan-1
  labels:
    elasticjob-name: deepctr-manual-scale
    scale-type: manual
spec:
  ownerJob: deepctr-auto-scale
  replicaResourceSpecs:
    ps:
      replicas: 2
```

After scaling, there two ps nodes:

``` bash
NAME                                           READY   STATUS    RESTARTS   AGE
deepctr-auto-scale-edljob-chief-0              1/1     Running   0          7m36s
deepctr-auto-scale-edljob-ps-0                 1/1     Running   0          7m36s
deepctr-auto-scale-edljob-ps-1                 1/1     Running   0          2m50s
elasticjob-deepctr-auto-scale-dlrover-master   1/1     Running   0          7m43s
```

We can scale down PS nodes with the spec in ScalePlan like

```yaml
apiVersion: elastic.iml.github.io/v1alpha1
kind: ScalePlan
metadata:
  namespace: dlrover
  name: deepctr-manual-scale-plan-2
  labels:
    elasticjob-name: deepctr-manual-scale
    scale-type: manual
spec:
  ownerJob: deepctr-auto-scale
  replicaResourceSpecs:
    ps:
      replicas: 1
```

After scaling, there two ps nodes:

``` bash
NAME                                           READY   STATUS    RESTARTS   AGE
deepctr-auto-scale-edljob-chief-0              1/1     Running   0          9m30s
deepctr-auto-scale-edljob-ps-0                 1/1     Running   0          9m30s
elasticjob-deepctr-auto-scale-dlrover-master   1/1     Running   0          9m47s
```

We can migrate a PS with more resource like

```yaml
apiVersion: elastic.iml.github.io/v1alpha1
kind: ScalePlan
metadata:
  namespace: dlrover
  name: deepctr-manual-scale-plan-3
  labels:
    elasticjob-name: deepctr-auto-scale
    scale-type: manual
spec:
  ownerJob: deepctr-auto-scale
  migratePods:
    - name: deepctr-auto-scale-edljob-ps-0
      resource:
        cpu: "2"
        memory: 4Gi
```

During migrating, a new ps node is started. When new ps is ready, master will inform workers to accept new ps.

``` bash
NAME                                           READY   STATUS    RESTARTS   AGE
deepctr-auto-scale-edljob-chief-0              1/1     Running   0          22m
deepctr-auto-scale-edljob-ps-0                 1/1     Running   0          22m
```

After migrating, new ps joins and the old ps exit:

``` bash
NAME                                           READY   STATUS    RESTARTS   AGE
deepctr-auto-scale-edljob-chief-0              1/1     Running   0          22m
deepctr-auto-scale-edljob-ps-2                 1/1     Running   0          20s
```
