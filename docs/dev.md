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

### 1. Preliminary

Install [minikube](https://kubernetes.io/docs/tasks/tools/) on your loptop.
And you can start minikube by the command

```bash
k8s start --vm-driver=docker --cpus 6 --memory 6144
```

### 2. Run ElasticJob Controller

We can run the ElasticJob in the terminal or deploy the controller with
a docker image.

- Run the controller in the terminal.

```bash
cd dlrover/go/operator
make install
make run
```

- Deploy the controller.

```bash
make deploy IMG=easydl/elasticjob-controller:test
```

### 3. Grant Permission for the DLRover Master to Access CRDs

```bash
kubectl apply -f dlrover/go/operator/config/rbac/default_role.yaml 
```

### 4. Build the Image of DLRover Master

```bash
docker build -t easydl/dlrover-master:test -f docker/Dockerfile.
```

### 5. Submit an ElasticJob.

```bash
eval $(k8s docker-env)
kubectl -n dlrover apply -f dlrover/go/operator/config/samples/elastic_v1alpha1_elasticjob.yaml
```

Check traning nodes.

```bash
kubectl -n dlrover get pods
```

```
NAME                                  READY   STATUS    RESTARTS   AGE
elasticjob-elasticjob-sample-master   1/1     Running   0          2m47s
elasticjob-sample-edljob-chief-0      1/1     Running   0          2m42s
elasticjob-sample-edljob-ps-0         1/1     Running   0          2m42s
elasticjob-sample-edljob-worker-0     1/1     Running   0          2m42s
elasticjob-sample-edljob-worker-1     1/1     Running   0          2m42s
```