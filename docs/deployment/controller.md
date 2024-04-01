# Deploy DLRover ElasticJob Controller on a Kubernetes Cluster

Here, we introduce how to deploy the DLRover job controller directly on a
Kubernetes cluster step by step. Minikube is optional and primarily used for testing.

## 1. Preliminary

- Ensure you have [Kubernetes](https://kubernetes.io/docs/home/) installed.
If you prefer to use Minikube for testing purposes, make sure to have [Minikube](https://minikube.sigs.k8s.io/docs/start/)
installed and run `minikube start`.

## 2. Deploy DLRover ElasticJob Controller With Kubectl

```bash
# deploy from local directory
$ kubectl -n dlrover apply -k dlrover/go/operator/config/manifests/bases

# deploy from remote repo
$ deployment="git@github.com:intelligent-machine-learning/dlrover/dlrover/go/operator/config/manifests/bases/?ref=master"
$ kubectl -n dlrover apply -k $deployment
```

To verify the controller has been deployed, run the command below.
The output should show the dlrover-controller-manager pod is running.

```bash
kubectl -n dlrover get pods
```

```bash
NAME                                              READY   STATUS    RESTARTS   AGE
pod/dlrover-controller-manager-7dccdf6c4d-grmks   2/2     Running   0          6m46s
```

## 3. Test Your Controller by Submitting A Mnist Training Job

```bash
kubectl -n dlrover apply -f examples/pytorch/mnist/elastic_job.yaml
```

Check traning nodes.

```bash
kubectl -n dlrover get pods
```

```bash
NAME                                              READY   STATUS    RESTARTS   AGE
pod/dlrover-controller-manager-7dccdf6c4d-grmks   2/2     Running   0          4h49m
pod/elasticjob-torch-mnist-dlrover-master         1/1     Running   0          4h42m
pod/torch-mnist-edljob-worker-0                   1/1     Running   0          4h42m
pod/torch-mnist-edljob-worker-1                   1/1     Running   0          4h42m
```
