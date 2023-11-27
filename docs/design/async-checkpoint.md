# Fastly Save and Load Checkpoint to Resume Training

The design to introduce how to speed up saving and loading the checkpoint
to resume training.

## Background

It is very difficult to keep training a foundational model during a long time period
without any interruption. Hardware failure, OS failure or network breakdown
may often happen during the training. Users usually periodically checkpoint
the model and optimizer states to the storage and resume the training
after an interruption. Now, It takes a few minutes to use  `torch.save` to
checkpoint a large foundational model. Frequent checkpointing may waste
many hours of the accelerator (GPU, NPU, TPU) since the training must stop
until the checkpointing is completed. On the other hand, infrequent checkpoints
could reduce the overhead but take the risk of wasting more training steps
when resuming the training from the latest checkpoint file. Reducing the checkpointing
overhead is necessary to improve the training efficiency and the accelerator utilization
when we train a foundation model using thousands of accelerators.

## Target

- Reduce the checkpointing time overhead to block the training for
  DDP, FSDP, DeepSpeed or Megatron.
- Speed up loading the checkpoint file when the training restarts.
- Automatically configure the checkpoint period to improve the efficient
  training time as soon as possible if the failure probability is given.

## Design

Save a checkpoint in PyTorch contains the following steps:

- Allocate CPU memory to place parameters.
- Copy parameters from GPU to CPU memory.
- Using `pickle` to serialize parameters.
- Write the serialized data into a file in the storage.

By experiments to save a checkpoint of 3GB GPT model from A100 to a disk, we find
the 1st step needs 2.3s, the 2nd step needs 0.12s, the 3rd and 4th steps
need 6.5s. So, it is very fast to copy the model parameters from GPU to the CPU memory.
So, we can copy the weights from GPU to CPU memory in the training loop
and asynchronously save weights from CPU memory to the storage. The solution
will block the training to save checkpoint with a little time.

### Asynchronously Save Checkpoint by a daemon subprocess of the GPU process

We can start a daemon subprocess in the training process to save the

- Start a thread to save states from GPU to CPU memory.

1. Make the memory buffer to place Torch tensors of states.

```Python

import torch.distributed.checkpoint._traverse as _traverse


def alloc_memory(path, value):
    print(path[0])
    buffer[path[0]] = torch.empty_like(value.cpu(), pin_memory=True)


def make_state_dict_buffer(state_dict):
    _traverse.traverse_state_dict(state_dict, alloc_memory)

```

- Write state dict to pinned memory.

```python

import torch.distributed.checkpoint._traverse as _traverse


def copy_state_to_memory(path, value):
    buffer[path[0]].copy_(value)


def copy_state_dict_to_buffer(state_dict):
    _traverse.traverse_state_dict(state_dict, copy_state_to_memory)
```

- Start a subprocess to asynchronously save states to storage.

We can start a subprocess in the training process and share the
tensor buffer by the shared memory of multiprocessing.

```Python
import multiprocessing

manager = multiprocessing.Manager()
share_memory = manager.dict()
step_queue = multiprocessing.Queue(maxsize=1)


def periodically_save():
    while True:
        step = step_queue.get()
        step = step_queue.pop()
        path = os.path.join(ckpt_dir, str(step))
        torch.save(shard_memory["model"], path)


def save_checkpoint_step(step):
    step_queue.put(step, block=False)
```

### Asynchronously Save Checkpoint by an independent CPU process

If we start a daemon subprocess of the GPU training process, the daemon
subprocess will exit if the training process fails. In this case, the
parameters in CPU memory of the daemon subprocess will loss. So, we can
start an independent CPU process to share the memory with the training
process to save checkpoint to storage.

Allocate the small shared memory to place the meta information of the model and optimizer.
The meta mainly contains the tensor size of the model and optimizer. The process
to save checkpoint can allocate another shared memory with the meta.

### Load checkpoint from the multiple-level storage

If the training process fails and the elastic agent of PyTorch can restart the
training process to resume the training, the training process can load the checkpoint
from the shared memory not from the storage. Loading from the memory is much faster
than the storage.

## The ElasticAgent Asynchronously Saves the Checkpoint into Storage

If we start a daemon subprocess of the GPU training process, the daemon
subprocess will exit if the training process fails. In this case, the
parameters in CPU memory of the daemon subprocess will be cleaned. So, the elastic agent
in the main process can allocate the shared memory with the training
process. The training process saves the state dict to the shared memory and
the agent save them into the storage.

### The Classes Design

As we need a global thread to keep and sync the checkpointing state in storage, and an agent
thread per node to save the checkpointing state into the storage, we design three classes to
implement the checkpointing process.

<div align="center">
<img src="../figures/async-ckpt-classes.jpg" alt="Async Checkpoint Classes" width="1000">
</div>

- **AgentCkptManger**
  - One instance runs in each agent process.
      memory and the storage.
  - Get the Shared lock of shared memory and save the checkpoint state into the storage.
  - One of Agent check if all agents finish the writing and commit the checkpoint.

- **TrainCkptManger**
  - One instance runs in each training process.
  - Is responsible for coping the checkpointing state from GPU to shared memory.
  - Notifies the AgentCkptManger to save the checkpoint state into the storage.

### Async Checkpointing Saving Steps

<div align="center">
<img src="../figures/async-ckpt-steps.jpg" alt="Async Checkpoint Classes" width="1000">
</div>

The agent and training process need to do the following steps:

1. The agent monitors the training process to create the mata of model and
  optimizer state dict.
2. TrainCkptManager acquires the shared lock and update the meta of model and optimizer
  state dict.
3. TrainCkptManager copy the state dict from GPU to the shared memory.
4. TrainCkptManager releases the shared lock.
5. TrainCkptManager notifies the AgentCkptManager to save the checkpointing state into the
  storage.
6. AgentCkptManager acquires the shared lock and write the checkpoint state into the storage.
7. AgentCkptManager in rank 0 checks if all agents finish the writing and commit the checkpoint.
8. AgentCkptManager releases the shared lock.

The following figure shows the checkpointing process in sequence diagram.

<div align="center">
<img src="../figures/async-ckpt-sequence.jpg" alt="Async Checkpoint Classes" width="1000">
</div>

### Last Words when the Training Process Fails

When any of the training processes fails, or the agent is killed by SIGTERM, we can automatically
save the latest checkpoint state into the storage.

### Consistency of the Checkpointing State

For the shared memory checkpointing consistency and correctness, we use a shared lock
to protect the shared memory. If the Agent writing process takes too long time, for training
efficiency, the TrainCkptManager will skip a memory checkpointing and keep training.

For the storage checkpointing consistency and correctness, every agent will write the checkpoint
to a temporary directory, and one of the AgentCkptManager will commit the checkpointing after all agents
finish the writing.

## Checkpoint APIs Design

```Python

class AsyncCheckpointEngine(object):
    """
    Attributes:
        checkpoint_dir: str, the directory to save the checkpoint.
        max_to_keep: int, the number of checkpoint files to keep.
        save_mem_interval: int, the interval of iteration steps to save the model and
            optimizer states into the CPU memory.
        save_storage_interval: int, the interval of iteration steps to save the model
            and optimizer states from CPU memory to the storage.
        auto_save: bool, the checkpoint manager will automatically configure the
            interval to save checkpoint into memory and storage according to
            the time of iteration step.
    """

    def __init__(
        self,
        checkpoint_dir,
        max_to_keep=1,
        save_mem_interval=0,
        save_storage_interval=0,
        auto_save=False,
    ):
        pass

    def save(self, step, state_dict):
        """
        Save the state dict if the step is multiple of save_mem_interval.

        Args:
            step: the iteration step in the training loop.
            state_dict: a dictionary.
        """
        pass

    def load(self, resume_path=""):
        """
        Load the state dict from the CPU memory if the state dict is complete in
        CPU memory. Otherwise, the function will load the state dict from the storage. 

        Args:
            resume_path: str, If the resume_path is an empty
                string, the function will load the latest checkpoint file in the checkpoint
                directory.
        
        Returns:
            A dict.
        """
        pass
```
