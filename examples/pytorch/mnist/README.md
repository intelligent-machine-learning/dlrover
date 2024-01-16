# Train a CNN Model with MNIST dataset using DLRover

The document describes how to use DLRover to train a Pytorch CNN model
with MNIST dataset.

## Prepare Data

You can directly use `datasets.MNIST()` to create a dataset, or you can manually download it as follows:

- Download the dataset from [Kaggle MNIST Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset).
- Untar the dataset into a directory like `data/mnist_png`.

There are 2 sub-directories in the directory.

```text
|-data
    |-mnist_png
        |-testing
        |-training
```

## Train on a Single Node with Mutliple GPUs

Firstly, we need to install dlrover and the dependencies of the model by

```bash
pip install dlrover -U
```

Then, we can use `dlrover-run` to start the training by

```bash
dlrover-run --nproc_per_node=${GPU_NUM} \
    examples/pytorch/mnist/cnn_train.py --num_epochs 5 
```

or

```bash
dlrover-run --nproc_per_node=${GPU_NUM} \
    examples/pytorch/mnist/cnn_train.py --num_epochs 5 \
    --training_data data/mnist_png/training/ \
    --validation_data data/mnist_png/testing/ 
```

`GPU_NUM` is the number of GPUs on the node.

## Train on Multiple Nodes with Mutliple GPUs

If we want to train the model on multiple nodes, we need to firstly
deploy the DLRover ElasticJob CRD on the k8s cluster with the
[tutorial](../../../docs/tutorial/torch_elasticjob_on_k8s.md).

### Prepare Docker Image

Build the docker image with the command

```bash
docker build -t ${IMAGE_NAME} -f examples/pytorch/mnist/mnist.dockerfile .
```

## Traing on Mutliple Nodes

Set the `${IMAGE_NAME}` in to line 18 of `elastic_job.yaml` and
use `kubectl` to submit an elastic job.

```bash
kubectl -n dlrover apply -f examples/pytorch/mnist/elastic_job.yaml
```

We can use the FSDP strategy to pretrain when we have GPUs and only
use the command in the containers of worker.

```bash
dlrover-run --network-check --nnodes=3:$NODE_NUM\
    --nproc_per_node=2 --max_restarts=3  \
    examples/pytorch/mnist/cnn_train.py --use_fsdp --num_epochs 5 \
    --training_data /data/mnist_png/training/ \
    --validation_data /data/mnist_png/testing/
```
