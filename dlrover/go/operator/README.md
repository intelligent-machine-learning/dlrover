# ElasticJob Operator

ElasticJob can scale in/out resources for distributed deep learning,
including the number of nodes, CPU and memory of each node.

## Description

The operator contains 2 CRDs, `elasticjob` and `scaleplan`. Users don't need
to set replica resource when they apply a `elasticjob` to train on a cluster.
The `elasticjob` controller will create a EasyDL master Pod for each
`elasticjob`. The master will generate a `scaleplan` with PS/worker resources
to notify the controller to launch Pods for the training.

## Getting Started

You’ll need a Kubernetes cluster to run against. You can use [KIND](https://sigs.k8s.io/kind)
to get a local cluster for testing, or run against a remote cluster.
**Note:** Your controller will automatically use the current context in
your kubeconfig file (i.e. whatever cluster `kubectl cluster-info` shows).

### Running on the cluster

1. Install Instances of Custom Resources:

```sh
kubectl apply -f config/crd/bases
```

We can deploy the controller with a released image.

```sh
make deploy IMG=easydl/elasticjob-controller:master
```

1. Build and push your image to the location specified by `IMG`:

```sh
make docker-build docker-push IMG=<some-registry>/operator:tag
```

1. Deploy the controller to the cluster with the image specified by `IMG`:

```sh
make deploy IMG=<some-registry>/operator:tag
```

### Uninstall CRDs

To delete the CRDs from the cluster:

```sh
make uninstall
```

### Undeploy controller

UnDeploy the controller to the cluster:

```sh
make undeploy
```

## Contributing

You can feel free to submit PullRequest to support features or fix bugs.

### How it works

This project aims to follow the Kubernetes [Operator pattern](https://kubernetes.io/docs/concepts/extend-kubernetes/operator/)

It uses [Controllers](https://kubernetes.io/docs/concepts/architecture/controller/)
which provides a reconcile function responsible for synchronizing resources
untile the desired state is reached on the cluster

### Test It Out

1. Install the CRDs into the cluster:

```sh
make install
```

1. Run your controller (this will run in the foreground, so switch to
new terminal if you want to leave it running):

```sh
make run
```

**NOTE:** You can also run this in one step by running: `make install run`

### Modifying the API definitions

If you are editing the API definitions, generate the manifests
such as CRs or CRDs using:

```sh
make manifests
```
