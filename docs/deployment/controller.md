# Depoloy dlrover job controller on Minikube cluster

Here we would like to introuce how to depoly dlrover job controller on minikube cluster step by step.

## 1.Preliminary
- Make sure you have [minikube](https://kubernetes.io/docs/tasks/tools/)  installed and run ``minikube start``

## 2. Create Namespace

```bash
$ kubectl create namespace dlrover
```

## 3. Deploy Dlrover Job Controller With Kubectl

```bash
$ kubectl apply -k /dlrover/go/config/manifests/bases
```

Check controller.

```bash
kubectl -n dlrover get pods
```

```
NAME                                              READY   STATUS    RESTARTS   AGE
pod/dlrover-controller-manager-7dccdf6c4d-grmks   2/2     Running   0          6m46s
```

kubectl -n dlrover apply -f dlrover/go/operator/config/rbac/default_role.yaml

## 3. Grant Permission for the DLRover Master to Access CRDs

```bash
kubectl apply -f dlrover/go/operator/config/rbac/default_role.yaml 
```

## 4. Test Your Controller by Submitting A Mnist Training Job

```bash
kubectl -n dlrover apply -f dlrover/examples/torch_mnist_master_backend_job.yaml
```

Check traning nodes.

```bash
kubectl -n dlrover get pods
```
```
NAME                                              READY   STATUS    RESTARTS   AGE
pod/dlrover-controller-manager-7dccdf6c4d-grmks   2/2     Running   0          4h49m
pod/elasticjob-torch-mnist-dlrover-master         1/1     Running   0          4h42m
pod/torch-mnist-edljob-worker-0                   1/1     Running   0          4h42m
pod/torch-mnist-edljob-worker-1                   1/1     Running   0          4h42m
```