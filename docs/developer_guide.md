# Introduction to develop EasyDL

The document describes how to make contribution to EasyDL.

## Submit a PR

- Fork EasyDL Repo to your owner namespace.
- `git clone git@github.com:intelligent-machine-learning/dlrover.git`
- `cd dlrover`
- `git remote rename origin upstream`
- `git remote add origin ${YOUR OWNER REPO}`
- `git checkout -b {DEV-BRANCH}`
- `git push -u origin {DEV-BRANCH}`

Then, you create a PR on github.

## Requirements

- [Go](https://golang.org/) (1.17 or later)

## Building the operator

Create a symbolic link inside your GOPATH to the location you checked out the code

```sh
mkdir -p ${go env GOPATH}/src/github.com/intelligent-machine-learning
ln -sf ${GIT_TRAINING} ${go env GOPATH}/src/github.com/intelligent-machine-learning/dlrover
```

- GIT_TRAINING should be the location where you checked out https://github.com/intelligent-machine-learning/dlrover

Install dependencies

```sh
go mod vendor
```

## Running the Operator Locally
Running the operator locally (as opposed to deploying it on a K8s cluster) is convenient for debugging/development.

### 1. Preliminary

Install [minikube](https://kubernetes.io/docs/tasks/tools/) on your loptop.
And you can start minikube by the command

```bash
minikube start --vm-driver=docker --cpus 6 --memory 6144
```

### Configure KUBECONFIG and KUBEFLOW_NAMESPACE

We can configure the operator to run locally using the configuration available in your kubeconfig to communicate with
a K8s cluster. Set your environment:

```sh
export KUBECONFIG=$(echo ~/.kube/config)
export KUBEFLOW_NAMESPACE=$(your_namespace)
```

- KUBEFLOW_NAMESPACE is used when deployed on Kubernetes, we use this variable to create other resources (e.g. the resource lock) internal in the same namespace. It is optional, use `default` namespace if not set.


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
make deploy IMG=easydl/elasticjob-controller:master
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
eval $(minikube docker-env)
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

### 6. Create a release
Change pip version and docker image tag when creating a new release.

## Go version

On ubuntu the default go package appears to be gccgo-go which has problems see [issue](https://github.com/golang/go/issues/15429) golang-go package is also really old so install from golang tarballs instead.
