# Fine-tuning llama2 using DLRover

The document describes how to use DLRover to fine-tune the llama2.

## Prepare Data

- Download the [BTC tweets sentiments](https://www.kaggle.com/datasets/aisolutions353/btc-tweets-sentiment)
dataset from Kaggle and untar the dataset into the file BTC_Tweets_Updated.csv.
- Convert the dataset by `python prepare_data.py BTC_Tweets_Updated.csv`.

For convenience, there is a small sampling dataset with 500 samples
in the `examples/pytorch/llama2/btc_tweets_sentiment.json`

## Train on a Single Node with Mutliple GPUs

Firstly, we need to install dlrover and the dependencies of the model by

```bash
pip install dlrover -U
pip install -r examples/pytorch/llama2/requirements.txt
```

Then, we can use `dlrover-run` to start the training by

```bash
dlrover-run --nproc_per_node=${GPU_NUM} examples/pytorch/llama2/fine_tuning.py 
```

`GPU_NUM` is the number of GPUs on the node.

## Train on Multiple Nodes with Mutliple GPUs

If we want to train the llama2 with DLRover on multiple nodes, we need to firstly
deploy the DLRover ElasticJob CRD on the k8s cluster with the
[tutorial](../../../docs/tutorial/torch_elasticjob_on_k8s.md).

### Prepare Docker Image

Build the docker image with the command

```bash
docker build -t registry.cn-hangzhou.aliyuncs.com/intell-ai/dlrover:llama-finetuning \
-f examples/pytorch/llama2/llama2.dockerfile .
```

## Traing on Mutliple Nodes

Use `kubectl` to submit an elastic job.

```bash
kubectl -n dlrover apply -f examples/pytorch/llama2/elastic_job.yaml
```

## Train on a Single Node with Mutliple Ascend-NPUs

Firstly, we need to set npu environment.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

Then, we need to install dlrover and the dependencies of the model by

```bash
pip install dlrover -U
pip install -r examples/pytorch/llama2/npu_requirements.txt
```

Now we can use `dlrover-run` to start the training by

```bash
dlrover-run --nproc_per_node=${NPU_NUM} examples/pytorch/llama2/fine_tuning.py 
```

`NPU_NUM` is the number of NPUs on the node.
