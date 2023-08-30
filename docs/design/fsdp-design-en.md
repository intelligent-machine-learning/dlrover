# DLRover's FSDP Elastic Training Design Document

## 1. Background

As the scale of machine learning models has grown, the necessity for distributed
training has become paramount. DLRover is especially proficient in addressing
challenges such as node failures, dynamic resource allocation, and automatic
recovery. While PyTorch provides support through FSDP (Fully Sharded Data Parallel),
DLRover still faces specific issues related to dynamic resources and fault tolerance.
These problems often manifest as OOM (Out of Memory) errors and complications in
reloading sharded checkpoints when the world size alters."

### 1.1 Motivation

DLRover aims to support FSDP at the system level for:

1. **System-Level FSDP Elastic Training:** DLRover should recognize FSDP
training, aiding users in auto-saving and loading checkpoints.
2. **Enhanced PyTorch FSDP Resharding:** PyTorch's save method requires
refinement for dynamic resharding when world size varies.

### 1.2 Main Challenges

FSDP implementation in DLRover presents hurdles:

1. **System-Level Interface for FSDP Elastic Training:** The
`ElasticTrainer` design should simplify user interaction yet remain
extensible. It should support easy FSDP elastic training configuration,
custom checkpoint strategies, and shared storage paths without task
conflicts.

2. **Efficient Load/Save with Resharding:** DLRover's elasticity
demands sharded checkpoint saves. PyTorch's current methods aren't
optimal for save/load with resharding, especially during world size
changes.

## 2. High-Level Design

Revise DLRover's `ElasticTrainer` to save/load model shards with FSDP.

### 2.1 Attributes Added to ElasticTrainer

- **use_fsdp:** Option for the FSDP training strategy.
- **shared_storage_path:** Path shared among worker pods containing FSDP
data.
- **checkpoint_interval:** A `CheckpointInterval` instance to define
checkpoint save strategy, either by epochs or steps.

```python
class CheckpointInterval:
    def __init__(self, steps=None, epochs=None):
        if steps and epochs:
            raise ValueError("Only one of 'steps' or 'epochs' should be set.")
        self.steps = steps
        self.epochs = epochs

    def should_save(self, current_step=None, current_epoch=None):
        if self.steps and current_step and current_step % self.steps == 0:
            return True
        if self.epochs and current_epoch and current_epoch % self.epochs == 0:
            return True
        return False
```

### 2.2 New and Modified Public Functions in ElasticTrainer

- **Add `epoch` function:** Introduces a context manager decorator for executing specific operations before
and after each epoch. Notably, it resets and saves model and optimizer state shards under the FSDP strategy.
  
- **Modify `step` function:** Updated to act as a context manager decorator. It verifies whether state shards
for the model and optimizer should be saved after each training step.

- **Modify `prepare` function:** Enhanced to inspect at initialization whether state shards for the model and
optimizer should be pre-loaded. If shard inconsistencies exist due to varying worker numbers, resharding is
conducted during load.

### 2.3 ElasticTrainer Support for Resharding During Save/Load

Considering the fluid nature of elastic training, existing PyTorch support for parameter resharding is somewhat lacking.
To address this, we've added `_save_fsdp_state` and `_load_fsdp_state` functions in `ElasticTrainer`.
These functions manage the saving of shard metadata and execute resharding procedures during load operations.

## 3. Detailed Design of ElasticTrainer

### 3.1 Introduction of Attributes

In `ElasticTrainer`, we're introducing `shared_storage_path` and `use_fsdp`.

1. `use_fsdp: bool` - Sourced from the constructor, it defaults to `False`.
If activated, the trainer checks the existence of `shared_storage_path`.
When the `checkpoint_interval` is met, model parameters and optimizer states are saved there.

2. `shared_storage_path: str` - Obtained from the constructor and defaults to `None`.
This path temporarily houses checkpoints for facilitating elastic training. It can also store other data types.

    ```bash
    shared_storage_path/
    │
    └── fsdp_checkpoint/
        │
        ├── <job1_name>_<timestamp>/
        │   ├── epoch_1/
        │   ├── epoch_2/
        │   ├── ...
        │   └── step_100/
        │
        ├── <job2_name>_<timestamp>/
        │   ├── epoch_1/
        │   ├── epoch_2/
        │   ├── ...
        │   └── step_150/
        │
        └── <job3_name>_<timestamp>/
            ├── epoch_1/
            ├── epoch_2/
            ├── ...
            └── step_200/
    ```

3. `checkpoint_interval`

```python
class ElasticTrainer:
    def __init__(self, checkpoint_interval: CheckpointInterval, ...):  # Other parameters
        ...
        self.checkpoint_interval = checkpoint_interval

class CheckpointInterval:
    def __init__(self, steps=None, epochs=None):
        if steps and epochs:
            raise ValueError("Only one of 'steps' or 'epochs' should be set.")
        self.steps = steps
        self.epochs = epochs

    def should_save(self, current_step=None, current_epoch=None):
        if self.steps and current_step and current_step % self.steps == 0:
            return True
        if self.epochs and current_epoch and current_epoch % self.epochs == 0:
            return True
        return False
```

### 3.2 Add `epoch` function

The current usage process of the Elastic Trainer is:

```python
from dlrover.trainer.torch.elastic import ElasticTrainer

model, optimizer, scheduler = ...

elastic_trainer = ElasticTrainer(model)
optimizer, scheduler = elastic_trainer.prepare(optimizer, scheduler)
for epoch in range(start_epoch, epochs):
    elastic_trainer.reset()
        for _, (data, target) in enumerate(train_loader):
            ...
            with elastic_trainer.step():
                ...
```

After modification, the Elastic Trainer's usage process becomes:

```python
from dlrover.trainer.torch.elastic import ElasticTrainer

model, optimizer, scheduler = ...

elastic_trainer = ElasticTrainer(model)
optimizer, scheduler = elastic_trainer.prepare(optimizer, scheduler)
for epoch in range(start_epoch, epochs):
    with elastic_trainer.epoch():
        for _, (data, target) in enumerate(train_loader):
            ...
            with elastic_trainer.step():
                ...
```

We will decorate the `epoch` function with `@contextmanager`.
`ElasticTrainer.epoch` will handle the following at the start of each epoch:

1. Invoke the `reset` function, setting `trainer.gradient_state.num_steps` to zero.

Upon conclusion, it will:

1. If `checkpoint_interval` meets the `should save` condition,
save model parameters and optimizer state as a checkpoint to `shared_storage_path`.

```python
class ElasticTrainer(object):
    @contextmanager
    def epoch(self, epoch: int):
        self._before_epoch()
        yield 
        self._after_epoch(epoch: int)

    def _after_epoch(self, epoch: int):
        # save checkpoint to self.shared_storage_path
        if self.checkpoint_interval.should_save(current_epoch=epoch):
            ....
```

### 3.3 Modify `step` function

If `checkpoint_interval` meets the `should save` condition,
save model parameters and optimizer state as a checkpoint to `shared_storage_path`.

Before modification:

```python
class ElasticTrainer(object):
    @contextmanager
    def step(self, fix_total_batch_size=True):
        self._before_step(fix_total_batch_size)
        context = contextlib.nullcontext
        if not self.gradient_state.sync_gradients:
            context = getattr(self.model, "no_sync", context)

        with context():
            yield
            self._after_step()

    def _after_step(self):
        if self.gradient_state.sync_gradients:
            self.gradient_state.num_steps += 1
```

After modification:

```python
class ElasticTrainer(object):
    @contextmanager
    def step(self, fix_total_batch_size=True):
        self._before_step(fix_total_batch_size)
        context = contextlib.nullcontext
        if not self.gradient_state.sync_gradients:
            context = getattr(self.model, "no_sync", context)

        with context():
            yield
            self._after_step()

    def _after_step(self):
        # save checkpoint to self.shared_storage_path
        if self.checkpoint_interval.should_save(current_step=self.num_steps):
            ....
        if self.gradient_state.sync_gradients:
            self.gradient_state.num_steps += 1
```

### 3.4 Modify `prepare` function

```python
class ElasticTrainer(object):
    def prepare(self, optimizer, lr_scheduler=None):
        """
        Prepare optimizer and learning rate scheduler for elastic training.
        """
        if self.load_from_checkpoint:
            self._load_model()
            self._load_optim()
        #########################################################
        self._set_gradient_accumulation_steps()
        optimizer = _ElasticOptimizer(optimizer)
        if lr_scheduler:
            lr_scheduler = _ElasticLRScheduler(lr_scheduler)
            return optimizer, lr_scheduler
        else:
            return optimizer
```

### 3.5 Resharding Support for Save/Load

**Background:**

Restoring model states in elastic fault tolerance relies on checkpoints.
Modern large model training employs the FSDP parallel technique. FSDP offers two checkpoint-saving methods:

1. `rank0_only`: Rank-0 node aggregates model parameters and optimizer states, then commits to disk.
2. Sharding method: Each rank saves its model parameters and optimizer states.

While `rank0_only` aids elastic fault tolerance, it can trigger OOM errors and bears notable write delays.
The sharding method doesn't meet elastic fault tolerance training needs.

Concerning `rank0_only`:

1. Loading all parameters and states in rank-0 can induce OOM.
2. Aggregating then sequentially writing to disk in rank-0 is inefficient.

Regarding the sharding method:

1. Ranks saving the checkpoint should mirror those loading it. But in elastic scenarios, rank count can vary.

**Objective:**

Refine the sharding technique to accommodate resharding when rank counts shift.

**Design:**

Document start and end positions for each parameter. Retain this data as checkpoint metadata in an isolated file.
During loading, perform resharding using these parameters.

![fsdp](../figures/fsdp-resharding.png)

<aside>
As operations on non-flattened data (Tensor) are needed when saving and loading the model,
we need to save a triplet for the model parameters: (original_shape, start, end).
</aside>

File structure will look like:

```bash
ckpt
├── optim_meta
├── optim_param.00000-00002
└── optim_param.00001-00002
```

We will use `LOCAL_STATE_DICT` for saving checkpoints.

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullOptimStateDictConfig

class ElasticTrainer(object):
    def _save_fsdp_state(self):
        # save checkpoint to self.shared_storage_path
        folder_name = ...
        save_dir = os.path.join(self.shared_storage_path, folder_name)
        writer = FileSystemWriter(save_dir)
        with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
            state_dict = model.state_dict()
            fsdp_osd = FSDP.sharded_optim_state_dict(self.model, self.optim)
            flattened_osd = FSDP.flatten_sharded_optim_state_dict(
                fsdp_osd, self.model, self.optim
            )
        # save a checkpoint ...
```

**Upcoming Improvements:**

1. Asynchronous writing: Utilize a separate thread for checkpoint saving, ensuring training remains non-blocking.
2. When the nodegroup is stable, avoid meta data rewrites.

## 3. Deploying PV

To enable Worker Pods shared storage, the storage solution should support the **`ReadWriteMany`** mode.
This mode permits multiple pods on different nodes to access a singular storage volume simultaneously.

Popular storage solutions with **`ReadWriteMany`** mode include:

1. **NFS (Network File System)**:
A widely-used system supporting multiple operations. By setting up an NFS server and leveraging NFS-provisioner,
one can dynamically provision PersistentVolumes in Kubernetes.
2. **CephFS**: A scalable, distributed file system. Its CephFS layer allows multiple operations.
3. **GlusterFS**:
An open-source, scalable network file system offering expansive storage pools and access from multiple nodes.

For shared storage across master and worker pods in a multi-node Kubernetes cluster:

1. Ensure a proper deployment and operation of the storage backend (e.g., NFS, CephFS, GlusterFS).
2. Set up a **`PersistentVolume`** and **`PersistentVolumeClaim`** using that storage,
specifying **`ReadWriteMany`** in the PVC.
3. Integrate this PVC into master and worker pod definitions.

<aside>
Often, manual PV creation isn't needed. Many Kubernetes setups, like AWS EBS, Google Cloud Persistent Disk,
or Azure Disk Storage, auto-create PVs upon PVC generation. But if automated storage provisioning is absent or a
distinct storage configuration is required, manual PV creation becomes essential.
In such cases, specific parameters are defined in the PV, paving the way for PVC usage.
</aside>
