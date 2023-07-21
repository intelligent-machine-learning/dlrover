# Introduction to develop DLRover

The document describes how to make contribution to DLRover.

## Submit a PR

- Fork DLRover Repo to your owner namespace.
- `git clone git@github.com:intelligent-machine-learning/dlrover.git`
- `cd dlrover`
- `git remote rename origin upstream`
- `git remote add origin ${YOUR OWNER REPO}`
- `git checkout -b {DEV-BRANCH}`
- `git push -u origin {DEV-BRANCH}`

Then, you create a PR on github. If you has modified codes of the repo,
you need to execute `pre-commit` to check codestyle and unittest cases
by the following steps.

- ```docker run -v `pwd`:/dlrover -it easydl/dlrover:ci /bin/bash```
- `cd /dlrover`
- `pre-commit run -a`
- `python -m pytest dlrover/python/tests`
- `python -m pytest dlrover/trainer/tests`

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

#### Minikube Install

Install [minikube](https://kubernetes.io/docs/tasks/tools/) on your loptop.

#### Minikube with GPU Support

To enable GPU support, follow the docs as follows:

- Install [cri-dockerd](https://github.com/Mirantis/cri-dockerd)
and [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
- Enable [k8s-device-plugin](https://github.com/NVIDIA/k8s-device-plugin#preparing-your-gpu-nodes)

- Test your GPU by the official [gpu-pod](https://github.com/NVIDIA/k8s-device-plugin#running-gpu-jobs)

It is highly recommended to have more than one GPU resources in your workspace.

However, there is still a workaround to divide your single GPU resource into multiple ones.

For this, enable [shared-access-to-gpus with CUDA Time-Slicing](https://github.com/NVIDIA/k8s-device-plugin#shared-access-to-gpus-with-cuda-time-slicing) to get more GPU resources.

Check the doc and modify your ``nvidia-k8s-device-plugin`` or simply update the plugin by ``helm`` with the command ([See more details about getting GPU resources](https://github.com/ChenhuiHu/DLRover-Supplementary-Description-/blob/main/Obtain%20more%20GPU%20resources%20on%20a%20single%20machine.md))

```bash
$ helm upgrade -i nvdp nvdp/nvidia-device-plugin \
    --version=0.13.0 \
    --namespace nvidia-device-plugin \
    --create-namespace \
    --set-file config.map.config=./dlrover/go/operator/config/gpu/nvidia-device-plugin-gpu-shared.yaml
```

Then test your GPU resources by

```bash
$ kubectl get nodes -ojson | jq '.items[].status.capacity'
>
{
  "cpu": "8",
  "ephemeral-storage": "229336240Ki",
  "hugepages-1Gi": "0",
  "hugepages-2Mi": "0",
  "memory": "32596536Ki",
  "nvidia.com/gpu": "2", # create one more gpu resource on your laptop
  "pods": "110"
}
```

Create this deployment to test your GPU resources.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: test-gpu
spec:
  replicas: 2 # replace this to your amount of GPU resources
  selector:
    matchLabels:
      app: test-gpu
  template:
    metadata:
      labels:
        app: test-gpu
    spec:
      containers:
        - name: cuda-container
          image: nvcr.io/nvidia/k8s/cuda-sample:vectoradd-cuda10.2
          resources:
            limits:
              nvidia.com/gpu: 1 # requesting 1 GPU
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
```

```bash
NAME                                          READY   STATUS      RESTARTS      AGE
dlrover-controller-manager-6c464d59f8-np7tg   2/2     Running     0             55m
test-gpu-59c9677b99-qtxbv                     0/1     Completed   2 (24s ago)   27s
test-gpu-59c9677b99-sxd6n                     0/1     Completed   2 (24s ago)   27s

$ kubectl logs test-gpu-59c9677b99-qtxbv
>
[Vector addition of 50000 elements]
Copy input data from the host memory to the CUDA device
CUDA kernel launch with 196 blocks of 256 threads
Copy output data from the CUDA device to the host memory
Test PASSED
Done
```

#### Start Your Minikube

After preparing your minikube cluster you can start minikube with the command:

```bash
minikube start --vm-driver=docker --cpus 6 --memory 6144

# If you wish to run minikube with GPUs, recommended commands are as follows.(root privilege requried)

minikube start --driver=none --container-runtime='containerd' --apiserver-ips 127.0.0.1 --apiserver-name localhost --cpus 6 --memory 6144
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

### 4. Build the Image 

**Build the master image with codes.**

```bash
docker build -t easydl/dlrover-master:test -f docker/Dockerfile .
```

**Build the training image of PyTorch models.**

```bash
docker build -t easydl/dlrover-train:test -f docker/pytorch/mnist.dockerfile .
```

### 5. Submit an ElasticJob to test your images.

We can set the training image of the line 18 and the master image
of line 42 in the debug job `dlrover/examples/torch_debug_job.yaml`.
Then, we can submit a job with the above images.

```bash
eval $(minikube docker-env)
kubectl -n dlrover apply -f dlrover/examples/torch_debug_job.yaml
```

Check traning nodes.

```bash
kubectl -n dlrover get pods
```

```
NAME                            READY   STATUS    RESTARTS   AGE
elasticjob-torch-mnist-master   1/1     Running   0          2m47s
torch-mnist-edljob-chief-0      1/1     Running   0          2m42s
torch-mnist-edljob-worker-0     1/1     Running   0          2m42s
torch-mnist-edljob-worker-1     1/1     Running   0          2m42s
```

### 6. Create a release

Change pip version and docker image tag when creating a new release.

## Go version

On ubuntu the default go package appears to be gccgo-go which has problems see [issue](https://github.com/golang/go/issues/15429) golang-go package is also really old so install from golang tarballs instead.
