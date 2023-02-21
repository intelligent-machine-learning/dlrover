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
kubectl -n dlrover apply -f dlrover/examples/deepctr_job.yaml
```

- Check the job status

```bash
kubectl -n dlrover get elasticjob deepctr-sample
```

```bash
NAME             PHASE     AGE
deepctr-sample   Running   2m20s
```

- Check the Pod status

```bash
kubectl -n dlrover get pods -l elasticjob-name=deepctr-sample
```

```bash
deepctr-sample-edljob-chief-0              1/1     Running             0          90s
deepctr-sample-edljob-ps-0                 1/1     Running             0          90s
elasticjob-deepctr-sample-dlrover-master   1/1     Running             0          94s
```

## Mannually Scale Nodes of a Job

We can submit a ScalePlan CRD to scale up/down nodes of a job.
In a ScalePlan, we need to set `metadata.labels` to specify
which job to scale and `spec.manualScaling` to `True`.
For example, the ScalePlan is to scale
workers of the job `elasticjob-sample`.

```yaml
apiVersion: elastic.iml.github.io/v1alpha1
kind: ScalePlan
metadata:
  name: scaleplan-sample
  labels:
    elasticjob-name: elasticjob-sample
spec:
  ownerJob: elasticjob-sample
  replicaResourceSpecs:
    worker:
      replicas: 3
      resource:
        cpu: "0.5"
        memory: 256Mi
  manualScaling: True
```

We can scale PS nodes with the spec in ScalePlan like

```yaml
spec:
  ownerJob: elasticjob-sample
  replicaResourceSpecs:
    ps:
      replicas: 2
      resource:
        cpu: "0.5"
        memory: 256Mi
  manualScaling: True
```

or migrate a PS with more resource like

```yaml
apiVersion: elastic.iml.github.io/v1alpha1
kind: ScalePlan
metadata:
  name: scaleplan-sample
  labels:
    elasticjob-name: elasticjob-sample
spec:
  ownerJob: elasticjob-sample
  migratePods:
    - name: elasticjob_sample-ps-0
      resource:
        cpu: 4
        memory: "1024Mi"
  manualScaling: True
```