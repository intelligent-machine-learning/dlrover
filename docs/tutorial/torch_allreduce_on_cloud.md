# AllReduce Training Using DLRover on Public Cloud

This document explains how to run a DLRover elastic job using torchrun
on a public cloud, namely, Alibaba Cloud Container Service for Kubernetes(ACK).

## Preliminary

- Create a Kubernetes cluster on [ACK](https://help.aliyun.com/document_detail/309552.htm?spm=a2c4g.11186623.0.0.168f6b7aegH7nI#task-2112671). 
- Configure cluster credentials on your local computer.
- Create a [NAS](https://help.aliyun.com/document_detail/477380.html?spm=a2c4g.11186623.0.0.10635c83Xn7Tkh) storage and mount it to the cluster.

## Deploy the ElasticJob CRD on ACK

1. Deploy the controller on the cluster.

```bash
make deploy IMG=easydl/elasticjob-controller:test
```

2. Grant permission for the DLRover master to Access CRDs.

```bash
kubectl -n dlrover apply -f dlrover/go/operator/config/rbac/default_role.yaml 
```

## Submit a Job

- Submit a job to train a CNN model with MNIST dataset.

```bash
kubectl -n dlrover apply -f dlrover/examples/torch_mnist_master_backend_job.yaml
```

- Check the job status

```bash
kubectl -n dlrover get elasticjob torch-mnist 
```

```bash
NAME          PHASE     AGE
torch-mnist   Running   19h
```

- Check the Pod status

```bash
kubectl -n dlrover get pods -l elasticjob-name=torch-mnist
```

```bash
NAME                                    READY   STATUS    RESTARTS   AGE
elasticjob-torch-mnist-dlrover-master   1/1     Running   0          26s
torch-mnist-edljob-worker-0             1/1     Running   0          29s
torch-mnist-edljob-worker-1             1/1     Running   0          32s
```

## Test Fault-tolerance

- Delete a worker.

```bash
kubectl -n dlrover delete pod torch-mnist-edljob-worker-1
```

Then, we can see there are only one worker.

```bash
NAME                                    READY   STATUS    RESTARTS   AGE
elasticjob-torch-mnist-dlrover-master   1/1     Running   0          1m12s
torch-mnist-edljob-worker-0             1/1     Running   0          1m15s
```

For a while, DLRover will restore the deleted worker.

```bash
NAME                                    READY   STATUS    RESTARTS   AGE
elasticjob-torch-mnist-dlrover-master   1/1     Running   0          1m52s
torch-mnist-edljob-worker-0             1/1     Running   0          1m55s
torch-mnist-edljob-worker-1             1/1     Running   0          32s
```
