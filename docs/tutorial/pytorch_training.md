# Introduction to Develop PyTorch DDP Model with DLRover

The document describes how to develop PyTorch models and train the model
with elasticity using DLRover. Users only need to make some simple changes
of native PyTorch training codes. We have provided the
[CNN example](../../model_zoo/pytorch/mnist_cnn.py) to show how to
train a CNN model with the MNIST dataset.

## Develop a Torch Model with DLRover. 

### Setup the Environment Using ElasticTrainer

Users need to set up the environment through `ElasticTrainer`. 

The `ElasticTrainer` will mark the rank-0 node as PyTorch MASTER
and the node's IP as `MASTER_ADDR`. Note that, the ranks of all nodes
are not fixed during elasticity and the rank-0 node is always marked as MASTER.

```python
from dlrover.trainer.torch.elastic import ElasticTrainer

ElasticTrainer.setup()
```

### Develop the ElasticDataset.

At first, users need to write the path of the sample into a `Text` file.
The path can be a location path of a file or a linke to download
the sample data from a remote storage. For example, we can create a Text
file to storage the location path and lable of MNIST dataset.

```text
/data/mnist_png/training/9/37211.png,9
/data/mnist_png/training/9/51194.png,9
/data/mnist_png/training/9/374.png,9
/data/mnist_png/training/1/29669.png,1
/data/mnist_png/training/1/19782.png,1
/data/mnist_png/training/1/42786.png,1
/data/mnist_png/training/8/41017.png,8
/data/mnist_png/training/8/13037.png,8
/data/mnist_png/training/8/7101.png,8
```

Then, we create a dataset to support elastic training
using the Text file.

```python
from dlrover.trainer.torch.elastic_dataset import ElasticDataset


class ElasticMnistDataset(ElasticDataset):
    def __init__(self, path, batch_size, epochs, shuffle, checkpoint_path):
        """The dataset supports elastic training.

        Args:
            path: str, the path of dataset meta file. For example, if the image
                is stored in a folder. The meta file should be a
                text file where each line is the absolute path of a image.
            batch_size: int, the size of batch samples to compute gradients
                in a trainer process.
            epochs: int, the number of epoch.
            shuffle: bool, whether to shuffle samples in the dataset.
            checkpoint_path: the path to save the checkpoint of shards
                int the dataset.
        """
        super(ElasticMnistDataset, self).__init__(
            path,
            batch_size,
            epochs,
            shuffle,
            checkpoint_path,
        )

    def read_sample(self, index):
        """
        Read the sample by the index. Users can get the location
        path by the index from the text file and write codes
        to read sample data.
        """
        pass
```

### Wrap the Training Step using ElasticTrainer

To keep the total batch size fixed during elastic training,
users need to create an `ElasticTrainer` to wrap the model, optimizer
and scheduler. `ElasticTrainer` can keep the total batch size
fixed by accumulating gradients if the number of worker decreases.
For example, there are only 4 workers and the user set 8 workers,
each worker will accumulate gradients with 2 micro-batches before
synchronizing gradients. The number of micro-batch to update gradient
is also 8.

```python
model = Net()
optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
step_size = int(train_dataset.dataset_size / args.batch_size)
scheduler = StepLR(optimizer, step_size=step_size, gamma=0.5)
model = DDP(model)

# Initialize the ElasticTrainer 
elastic_trainer = ElasticTrainer(model, train_loader)
optimizer, scheduler = elastic_trainer.prepare(optimizer, scheduler)

# Load checkpoint to restore the model and optimizer states.
load_checkpoint(model, optimizer)

for _, (data, target) in enumerate(train_loader):
    model.train()

    # Run the step in the context of ElasticTrainer
    with elastic_trainer.step():
        target = target.type(torch.LongTensor)
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
         # Save checkpoint periodically.
        if elastic_trainer.num_steps % 200 == 0:
            model_checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(model_checkpoint, "model.pt")

            # Checkpoint the dataset when checkpointing the model.
            dataset.save_checkpoint()
```

## Submit an ElasticJob on the Kubernetes to Train the model.

### Build the Image with the Model.

You can install dlrover in your image like

```bash
pip install dlrover[torch] -U
```
or build your image with the dockerfile.

```dockerfile
ROM python:3.8.14 as base

WORKDIR /dlrover
RUN apt update
RUN apt install -y libgl1-mesa-glx libglib2.0-dev vim
RUN pip install deprecated pyparsing -i https://pypi.org/simple
RUN pip install torch opencv-python torchvision

RUN pip install dlrover -U
COPY ./model_zoo ./model_zoo
```

### Run the Training code with torchrun.

If we want to use the DLRover job master as the rendezvous backend,
we need to execute `python -m dlrover.python.elastic_agent.torch.prepare`
before `trochrun`. The `RendezvousBackend` of job master can support
the fault-tolerance of rank-0 which is not supported
in `C10dRendezvousBackend`.

```yaml
spec:
  distributionStrategy: AllreduceStrategy
  replicaSpecs:
    worker:
      replicas: 2
      template:
        spec:
          restartPolicy: Never
          containers:
            - name: main
              # yamllint disable-line rule:line-length
              image: registry.cn-hangzhou.aliyuncs.com/intell-ai/dlrover:torch113-mnist
              imagePullPolicy: Always
              command:
                - /bin/bash
                - -c
                - "python -m dlrover.python.elastic_agent.torch.prepare \
                  && torchrun --nnodes=1:$WORKER_NUM --nproc_per_node=1
                  --max_restarts=3 --rdzv_backend=dlrover-master \
                  model_zoo/pytorch/mnist_cnn.py \
                  --training_data /data/mnist_png/training/elastic_ds.txt \
                  --validation_data /data/mnist_png/testing/elastic_ds.txt"
```
