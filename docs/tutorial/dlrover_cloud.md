# DLRover on Public Cloud

This document explains how to run a DLRover elastic job with on a public cloud,
namely, Alibaba Cloud Container Service for Kubernetes(ACK).

## Preliminary

- Create a Kubernetes cluster on [ACK](https://help.aliyun.com/document_detail/309552.htm?spm=a2c4g.11186623.0.0.168f6b7aegH7nI#task-2112671). 
- Configure cluster credentials on your local computer.
- Create a [NAS](https://help.aliyun.com/document_detail/477380.html?spm=a2c4g.11186623.0.0.10635c83Xn7Tkh) storage and mount it to the cluster.

## Deploy the ElasticJob CRD on ACK

1. Deploy the controller on the cluster.

```bash
make deploy IMG=easydl/elasticjob-controller:v0.1.1
```

2. Grant permission for the DLRover master to Access CRDs.

```bash
kubectl -n dlrover apply -f dlrover/go/operator/config/rbac/default_role.yaml 
```

## Submit a Job

- Submit a job with the DeepFM model.

```bash
kubectl -n dlrover apply -f dlrover/examples/deepctr_auto_scale_job.yaml
```
You can change the initial ps and worker number by changing replica number.
```yaml
  replicaSpecs:
    ps:
      autoScale: False
      replicas: 1
---
  replicaSpecs:
    worker:
      autoScale: False
      replicas: 1
```
- Check the job status

```bash
kubectl -n dlrover get elasticjob deepctr-auto-scaling-job
```

```bash
NAME                       PHASE     AGE
deepctr-auto-scaling-job   Running   2m20s
```

- Check the Pod status

```bash
kubectl -n dlrover get pods -l elasticjob-name=deepctr-auto-scaling-job
```

```bash
NAME                                                 READY   STATUS    RESTARTS   AGE
deepctr-auto-scaling-job-edljob-chief-0              1/1     Running   0          12s
deepctr-auto-scaling-job-edljob-ps-0                 1/1     Running   0          12s
elasticjob-deepctr-auto-scaling-job-dlrover-master   1/1     Running   0          19s
```

## Mannually Scale Nodes of a Job

We can submit a ScalePlan CRD to scale up/down nodes of a job.
In a ScalePlan, we need to set `metadata.labels` to specify
which job to scale and `metadata.labels["scale-type"]` to "manual".
For example, the ScalePlan is to scale
workers of the job `deepctr-auto-scaling-job`.

```yaml
apiVersion: elastic.iml.github.io/v1alpha1
kind: ScalePlan
metadata:
  name: deepctr-auto-scaling-job
  labels:
    elasticjob-name: deepctr-auto-scaling-job
    scale-type: manual
spec:
  ownerJob: deepctr-auto-scaling-job
  replicaResourceSpecs:
    worker:
      replicas: 1
```
After scaling, there two ps nodes:

``` bash
NAME                                                 READY   STATUS    RESTARTS   AGE
deepctr-auto-scaling-job-edljob-chief-0              1/1     Running   0          7m36s
deepctr-auto-scaling-job-edljob-ps-0                 1/1     Running   0          7m36s
elasticjob-deepctr-auto-scaling-job-dlrover-master   1/1     Running   0          7m43s
```


We can scale PS nodes with the spec in ScalePlan like

```yaml
apiVersion: elastic.iml.github.io/v1alpha1
kind: ScalePlan
metadata:
  name: deepctr-auto-scaling-job
  labels:
    elasticjob-name: deepctr-auto-scaling-job
    scale-type: manual
spec:
  ownerJob: deepctr-auto-scaling-job
  replicaResourceSpecs:
    ps:
      replicas: 2
```

After scaling, there two ps nodes:

``` bash
NAME                                                 READY   STATUS    RESTARTS   AGE
deepctr-auto-scaling-job-edljob-chief-0              1/1     Running   0          7m36s
deepctr-auto-scaling-job-edljob-ps-0                 1/1     Running   0          7m36s
deepctr-auto-scaling-job-edljob-ps-1                 1/1     Running   0          2m50s
elasticjob-deepctr-auto-scaling-job-dlrover-master   1/1     Running   0          7m43s
```