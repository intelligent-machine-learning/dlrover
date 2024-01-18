# Pretrain Nano-GPT using DLRover

The document describes how to use DLRover to pretrain a Nano-GPT model
on a k8s cluster.

## Prepare Data

Now that you have your data, let's run the preparation script as follows:

```bash
python  examples/pytorch/nanogpt/prepare.py \
    --src_data_path=examples/pytorch/nanogpt/data.txt \
    --output_dir=data/nanogpt/
```

This command generates a train.bin, val.bin and meta.pkl and file in `data/nanogpt/`.

## Train on a Single Node with Mutliple GPUs

Firstly, we need to install dlrover and the dependencies of the model by

```bash
pip install dlrover -U
```

Then, we can use `dlrover-run` to start the training by

```bash
dlrover-run --nproc_per_node=${GPU_NUM} \
    examples/pytorch/nanogpt/train.py \
    --data_dir data/nanogpt/
```

`GPU_NUM` is the number of GPUs on the node.

## Train on Multiple Nodes with Mutliple GPUs

If we want to train the Nanogpt with DLRover on multiple nodes, we need to firstly
deploy the DLRover ElasticJob CRD on the k8s cluster with the
[tutorial](../../../docs/tutorial/torch_elasticjob_on_k8s.md).

### Prepare Docker Image

Build the docker image with the command

```bash
docker build -t ${IMAGE_NAME} -f examples/pytorch/nanogpt/nanogpt.dockerfile .
```

## Traing on Mutliple Nodes

Set the `${IMAGE_NAME}` in to line 18 of `elastic_job.yaml` and
use `kubectl` to submit an elastic job.

```bash
kubectl -n dlrover apply -f examples/pytorch/nanogpt/elastic_job.yaml
```

We can use the FSDP strategy to pretrain when we have GPUs and only
use the command in the containers of worker.

```bash
dlrover-run --nnodes=$NODE_NUM \
    --nproc_per_node=${GPU_NUM_PER_NODE} --max_restarts=1  \
    ./examples/pytorch/nanogpt/fsdp_train.py  \
    --data_dir '/data/nanogpt/' --epochs 50 --checkpoint_step 50 \
```
