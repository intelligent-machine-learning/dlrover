# Run DLRover on Minikube cluster

Here we would like to introduce how to run DLRover on minikube cluster
step by step.

## Create namespace

```shell
kubectl create namespace dlrover
```

## Deploy MySQL

To create MySQL DB as the store for DLRover

```shell
cd dlrover/go/brain/manifests/k8s
kubectl apply -f mysql-pv.yaml
kubectl apply -f mysql.yaml
```

Create tables in MySQL

```shell
kubectl exec -it mysql-pod-name --namespace dlrover -- bash
cd dlrover
mysql -uroot -proot < dlrover-tables.sql
```
