# Introduction to Develop PyTorch DDP Model with DLRover

The document describes how to develop PyTorch models and train the model
with elasticity using DLRover. Users only need to make some simple changes
of native PyTorch training codes. We have provided the
[CNN example](../../model_zoo/pytorch/mnist_cnn.py) to show how to
train a CNN model with the MNIST dataset.

## Develop a Torch Model with DLRover. 

Using elastic training of DLRover, users only need to set the
`ElasticDistributedSampler` into their training `DataLoader`
and checkpoint the sampler when checkpointing the model.

### Setup ElasticDistributedSampler into the Dataloader.


```Python
from dlrover.trainer.torch.elastic_sampler import ElasticDistributedSampler

train_data = torchvision.datasets.ImageFolder(
    root="mnist/training/",
    transform=transforms.ToTensor(),
)
#  Setup sampler for elastic training.
sampler = ElasticDistributedSampler(dataset=train_data)
train_loader = DataLoader(
    dataset=train_data,
    batch_size=32,
    num_workers=2,
    sampler=sampler,
)
```

### Save and Restore Checkpoint of  ElasticDistributedSampler

Checkpoint `ElasticDistributedSampler` when checkpointing
the model.

```python
checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "sampler": train_loader.sampler.state_dict(
        step, train_loader.batch_size
    ),  # Checkpoint sampler
}
torch.save(checkpoint, CHEKPOINT_PATH)
```

Restore `ElasticDistributedSampler` when restoring the model
from a checkpoint file.

```Python
checkpoint = load_checkpoint(CHEKPOINT_PATH)
model.load_state_dict(checkpoint.get("model", {}))
optimizer.load_state_dict(checkpoint.get("optimizer", {}))
#  Restore sampler from checkpoint.
train_loader.sampler.load_state_dict(checkpoint.get("sampler", {})
```

Then, we create a dataset to support elastic training
using the Text file.

### Wrap the Training Step using ElasticTrainer to Fix Batch Size

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
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "sampler": train_loader.sampler.state_dict(
                        train_step, train_loader.batch_size
                    ),  # Checkpoint sampler
            }
            torch.save(model_checkpoint, "model.pt")
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

### Run the Training code with dlrover-run.

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
              image: registry.cn-hangzhou.aliyuncs.com/intell-ai/dlrover:torch201-mnist
              imagePullPolicy: Always
              command:
                - /bin/bash
                - -c
                - "dlrover-run --nnodes=1:$WORKER_NUM --nproc_per_node=1
                  --max_restarts=3 \
                  model_zoo/pytorch/mnist_cnn.py \
                  --training_data /data/mnist_png/training/elastic_ds.txt \
                  --validation_data /data/mnist_png/testing/elastic_ds.txt"
```
