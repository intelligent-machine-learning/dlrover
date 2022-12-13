# Introduction to develop EasyDL

The document describes how to make contribution to EasyDL.

## Submit a PR

- Fork EasyDL Repo to your owner namespace.
- `git clone git@github.com:intelligent-machine-learning/easydl.git`
- `cd easydl`
- `git remote rename origin upstream`
- `git remote add origin ${YOUR OWNER REPOT}`

Then, you can check out your developed branch, push the branch to origin,
and create a PR on github.


## Test ElasticJob on Minikube

### Preliminary

Install [minikube](https://kubernetes.io/docs/tasks/tools/) on your loptop.
And you can start minikube by the command

```bash
minikube start --vm-driver=docker --cpus 6 --memory 6144
```

### Start ElasticJob Controller

1. Deploy ElasticJob CRD on minikube.

```bash
kubectl apply -f dlrover/go/operator/config/crd/bases/
kubectl apply -f dlrover/go/operator/config/rbac/
```

2. Start ElasticJob Controller

```bash
cd dlrover/go/operator
make run
```

3. Submit an ElasticJob.

```bash
eval $(minikube docker-env)
kubectl apply -f dlrover/go/operator/config/samples/elastic_v1alpha1_elasticjob.yaml
```

4. Check traning nodes.

```bash
kubectl get pods
```

```
NAME                                  READY   STATUS    RESTARTS   AGE
elasticjob-elasticjob-sample-master   1/1     Running   0          2m47s
elasticjob-sample-edljob-chief-0      1/1     Running   0          2m42s
elasticjob-sample-edljob-ps-0         1/1     Running   0          2m42s
elasticjob-sample-edljob-worker-0     1/1     Running   0          2m42s
elasticjob-sample-edljob-worker-1     1/1     Running   0          2m42s
```