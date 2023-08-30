# DLRover's FSDP Elastic Training Design Document

## 1. Background

As large-scale machine learning models continue to grow, distributed training has become a core technology. DLRover has shown advantages in various scenarios, especially in node failure, dynamic resource allocation, and automatic recovery. While PyTorch natively supports the FSDP (Fully Sharded Data Parallel) strategy, DLRover encounters issues with dynamic resources and fault tolerance, such as OOM (Out of Memory) and the inability to reload sharded checkpoints when the world size changes.

**Motivation**

The primary motivations for supporting FSDP at the system level in DLRover are:

1. **System-Level Support for FSDP Elastic Training:** DLRover needs to be aware of FSDP training to help users automatically save and load checkpoints.
2. **Extend PyTorch FSDP Resharding During Elastic Training:** The PyTorch save method needs improvements to support dynamic resharding when the world size changes.

**Main Challenges**

During the implementation of FSDP in DLRover, we primarily address the following technical challenges:

1. **System-Level Interface Design for FSDP Elastic Training:** The design of the `ElasticTrainer` interface needs to simplify user input while maintaining extensibility. Users should be able to easily configure FSDP elastic training tasks, provide custom checkpoint save strategies, and specify shared storage paths. The key is designing a path that avoids conflicts between tasks.
  
2. **Load/Save with Resharding:** The elastic nature of DLRover requires checkpoints to be saved in shards. PyTorch does not fully support saving and loading with resharding, especially when the world size changes. Extending PyTorch's capabilities to efficiently reshard during saving and loading is essential.

## 2. High-Level Design

Modify DLRover's `ElasticTrainer` module to save and load model shards at intervals under the FSDP strategy.

### 2.1 Attributes Added to ElasticTrainer

- **use_fsdp:** Whether to adopt the FSDP training strategy
- **shared_storage_path:** Specifies the shared storage path between worker pods; FSDP-related data is part of this.
- **checkpoint_interval:** An instance of `CheckpointInterval` to specify the checkpoint save strategy, either by the number of epochs or steps.

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

- **Add `epoch` function:** A context manager decorator to perform certain operations before and after each epoch, such as reset and save model and optimizer state shards under the FSDP strategy.
- **Modify `step` function:** A context manager decorator to check whether model and optimizer state shards need to be saved after each step.
- **Modify `prepare` function:** Check during initialization if model and optimizer state shards need to be loaded first. If shards are inconsistent with the current number of workers, perform resharding during load.

### 2.3 ElasticTrainer Support for Resharding During Save/Load

Given the dynamic nature of elastic training, current PyTorch support for parameter resharding is limited. Thus, we need to save shard meta information during save and perform resharding operations during load. Add `_save_fsdp_state` and `_load_fsdp_state` functions in `ElasticTrainer` for this purpose.

## 3. Detailed Design of ElasticTrainer

### 3.1 Add `shared_storage_path`, `use_fsdp`, and `checkpoint_interval` Attributes

We will only add `shared_storage_path` and `use_fsdp` attributes in `ElasticTrainer`.

1. `use_fsdp: bool`, obtained from the constructor, with a default value of `False`. When set to `True`, `ElasticTrainer` will check if `shared_storage_path` exists and will save model parameters and optimizer states to `shared_storage_path` when it meets the `checkpoint_interval`.
2. `shared_storage_path: str`, obtained from the constructor, with a default value of `None`. This path is used to temporarily store checkpoints for this training session to achieve elastic training (this shared path can also store other data).

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

We will decorate the `epoch` function with `@contextmanager`. `ElasticTrainer.epoch` will handle the following at the start of each epoch:

1. Invoke the `reset` function, setting `trainer.gradient_state.num_steps` to zero.

Upon conclusion, it will:

1. If `checkpoint_interval` meets the `should save` condition, save model parameters and optimizer state as a checkpoint to `shared_storage_path`.

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

If `checkpoint_interval` meets the `should save` condition, save model parameters and optimizer state as a checkpoint to `shared_storage_path`.

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

### 3.5 Support resharding for save/load

**Background:**

Elastic fault tolerance depends on checkpoints to restore model states. The present large model training adopts the FSDP parallel approach. There are two ways FSDP saves checkpoints:

1. `rank0_only`: The rank-0 node collects all model parameters and optimizer states, then saves them to disk.
2. Sharding method: Every rank saves its respective model parameters and optimizer states.

While `rank0_only` supports elastic fault tolerance, it might cause OOM errors and has significant write latency. The sharding approach can't satisfy elastic fault tolerance training's requirements.

For `rank0_only`:

1. Loading all model parameters and optimizer states in rank-0 might lead to OOM.
2. Collecting all model parameters and optimizer states in rank-0, then writing them to disk one by one, is time-consuming.

For the sharding method:

1. The number of ranks saving the checkpoint must match the number loading it. But in elastic fault tolerance, rank numbers may change.

**Objective:**

Enhance the sharding method to support resharding when the number of ranks changes.

**Design:**

Save the start and end positions for each parameter. Store this info as checkpoint meta data in a separate file. Then reshard based on these parameters during load.

![fsdp](../figures/fsdp-resharding.png)

<aside>
As operations on non-flattened data (Tensor) are needed when saving and loading the model, we need to save a triplet for the model parameters: (original_shape, start, end).
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

**Two upcoming improvements:**

1. Asynchronous writing: Use a separate thread to save the checkpoint to prevent blocking the training process.
2. If the nodegroup remains unchanged, don't rewrite the meta information. This is because when shards don't change, meta data remains the same.

## 3. Deploying PV

To enable Worker Pods to share storage, the chosen storage solution must support the **`ReadWriteMany`** access mode. This is because the **`ReadWriteMany`** access mode allows pods on multiple nodes to access the same storage volume simultaneously.

Here are some common storage solutions that support the **`ReadWriteMany`** access mode:

1. **NFS (Network File System)**: NFS is a popular file system that supports multiple read and writes. You can set up an NFS server and use the NFS-provisioner to dynamically provision PersistentVolumes in Kubernetes.
2. **CephFS**: Ceph is a highly scalable distributed file system, and its CephFS layer supports multiple read and write modes.
3. **GlusterFS**: Gluster is a free and open-source scalable network file system that can provide a large storage pool and supports concurrent access from multiple nodes.

To achieve shared storage functionality for master and worker pods in a multi-node Kubernetes cluster, one needs to:

1. Ensure the storage backend (e.g., NFS, CephFS, GlusterFS, etc.) is deployed and running correctly in the cluster.
2. Create a **`PersistentVolume`** and **`PersistentVolumeClaim`** to utilize that storage. Ensure the **`ReadWriteMany`** access mode is specified in the PVC definition.
3. Utilize this PVC in the definitions of both the master and worker pods.

<aside>
In most cases, there's no need to manually create PVs. Many Kubernetes clusters have automated storage provisioning configured, such as AWS EBS, Google Cloud Persistent Disk, or Azure Disk Storage, which automatically provisions a new PV when a PVC is created. However, if your Kubernetes environment doesn't have automated storage provisioning or if you require a specific storage configuration, then manual PV creation becomes necessary. In such scenarios, specific parameters and configurations can be specified in the PV, after which a PVC can be created to utilize this PV.
</aside>
