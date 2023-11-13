# Fastly Save and Load Checkpoint to Resume Training

The design to introduce how to speed up saving and loading the checkpoint
to resume training.

## Background

It is very difficult to keep training a fundational model during a long time period
without any interruption. Hareware failure, OS failure or network breakdwon
may oftern happen during the training. Users usually periodically checkpoint
the model and optimizer states to the storage and resume the training
after an interruption. Now, It takes a few minutes to use  `torch.save` to
checkpoint a large fundational model. Frequent checkpointing may waste
many hours of the accelerator (GPU, NPU, TPU) since the training must stop until the checkpointing is completed.On the other hand, infrequent checkpoints could reduce the overhead but take the risk of wasting more training steps when resuming the training from the latest checkpoint file. Reducing the checkpointing
overhead is necessary to improve the training efficiency and the accelerator utilization
when we train a fundation model using thouands of accelerators.

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
So, we can copy the weights from GPU to CPU memory in the trainin loop
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

- Start a subprocess to asynchrounously save states to storage.

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
