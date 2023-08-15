# DLRover 支持 FSDP 弹性训练设计文档

## 1. 背景

随着大规模机器学习模型的持续增长，分布式训练已经成为该领域的核心技术。DLRover系统已展现出其在多个方面的优势，特别是在节点故障、资源动态分配和自动恢复等关键场景下。尽管PyTorch原生支持Fully Sharded Data Parallelism (FSDP) 策略，但DLRover作为一个天然的弹性训练系统，当面临动态的资源扩缩容和容错场景时，简单的使用PyTorch的FSDP可能会导致问题。

**动机**

考虑DLRover系统级别上支持FSDP的主要动机如下：

1. **系统弹性与FSDP策略的结合**：虽然PyTorch提供了FSDP的支持，但在DLRover的弹性环境中，例如节点的动态扩缩容，需要更为深入的整合以确保FSDP策略的正常运作。
2. **确保训练的稳定性**：DLRover在弹性扩缩容和容错时，可能会面临因worldsize变化而导致的checkpoint加载问题。系统级别的FSDP支持可以确保这些场景下模型的稳定训练。

**主要挑战**

在DLRover中支持FSDP的道路上，我们预计会遭遇以下技术难点：

1. **模型恢复的复杂性**：由于DLRover的弹性特性，如何在worldsize发生变化时，基于checkpoint准确恢复模型状态，成为了一个关键技术问题。
2. **参数reshard的需求**：考虑到弹性训练的动态性，目前PyTorch在分片参数上的reshard支持有限，DLRover需要找到一种有效的方法来适应这种变化，保障训练的连续性。

综上所述，为DLRover系统引入对FSDP的深度支持不仅是技术上的延展，更是为了确保在复杂、动态的弹性训练环境中，大模型的训练可以稳定、高效地进行。

## 2. 概要设计

将主要对 DLRover 系统中的 ElasticTrainer 模块、Master 模块进行修改。ElasticTrainer 相当于是运行在 worker pod 中的 Agent，因此将主要修改这个模块使其能够支持在fsdp训练策略下，没过一段step或者epoch后保存模型分片，然后在pod数量发生变化时reaload 模型和优化器状态的分片。修改Master主要是因为Worker pod是由master拉起，因此我们需要在master创建 Worker 这部分代码中进行修改，使其能够传入必要的环境变量等。

### **2.1 ElasticTrainer 增加的属性**

- **use_fsdp：**是否采用 FSDP 训练策略
- **shared_storage_path：**指定 worker pod 之间共享存储的路径，fsdp 的相关数据是其中的一部分
- **checkpoint_interval：**CheckpointInterval 的实例用来表达存储 checkpoint 的策略。保存策略由用户指定，可以按照多少个epoch来保存，也可以按照多少个step来保存

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

### **2.2 ElasticTrainer 新增和修改的 public 函数**

- **增加** **epoch 函数**：这是一个 contextmanager 装饰器，在 epoch 开始前后做一些操作。比如rest 和在 fsdp 策略下保存模型和优化器状态参数分片。
- **修改** **step 函数**：这是一个 contextmanager 装饰器，在 step 结束后检测是否需要保存模型和优化器状态参数分片。
- **prepare 函数的修改**：初始化时需要检查是否需要先load模型和优化器状态参数分片，如果分片和当前worker数不一致需要reshard load

### **2.3 ElasticTrainer 对 save/load 时 reshard 的支持**

考虑到弹性训练的动态性，目前 PyTorch 在分片参数上的 reshard 支持有限，因此需要支持在 save 时保存分片的 meta 信息，在 load 时进行 reshard 操作等。在ElasticTrainer中增加：_save_fsdp_state 和 _load_fsdp_state 函数来实现

### 2.4 部署 PV

非云上环境需要。为了让 Worker Pods 共享存储，选择的存储解决方案必须支持 **`ReadWriteMany`** 访问模式。

### 2.5 Job yaml 增加 volumeMounts 和 env

增加共享存储和是否使用 fsdp 等相关环境变量，用来传递给 worker pod

## 3. ElasticTrainer 详细设计

### 3.1 增加 shared_storage_path 、use_fsdp 和 **checkpoint_interval** 属性

为了能够让用户编写 fsdp 弹性训练代码的时候可以尽量少的去感知内部实现，

我们将在 ElasticTrainer 中增加 shared_storage_path 和 use_fsdp 属性

1. use_fsdp: bool ，use_fsdp 从环境变量中获取，默认值是 False。当为True 时 ElasticTrainer 将检测 shared_storage_path 是否存在，并且在 trainer 每轮 epoch 结束的时候将模型参数和优化器状态 checkpoint，保存到shared_storage_path。
2. shared_storage_path：str，从环境变量中获取，默认值是None。用来暂存这次训练用于实现弹性训练而存放checkpoint的path（这个共享路径还可以存储其他数据）。
    
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
    
3. load_from_checkpoint：bool，从环境变量中获取，默认值是 False。当为 True 时，将在 Elastic Trainer 的 prepare 函数中，load checkpoint，更新 model 和 optimizer 的 state。
    1. 当计算资源变更的时候，master 拉起的新的 worker 会将此变量置为 True。第一次拉起时为 False

```python
class ElasticTrainer:
    def __init__(self, ...):  # 其他参数
        ...
        self.checkpoint_interval = CheckpointInterval()

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

> 考虑从环境变量获取而不是构造函数传参是因为 Trainer 运行在 pod 中，拉起 pod 时可以指定环境变量，而且三个参数应该对用户透明。
> 

### 3.2 增加 epoch 函数

目前的 Elastic Trainer 的使用流程如下：

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

修改后 Elastic Trainer 的使用流程：

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

我们将用 @contextmanager 装饰 epoch 函数，ElasticTrainer.epoch 负责在每轮epoch开始的时候完成以下操作：

1. 调用 reset 函数，置零 trainer 的 gradient_state.num_steps

在结束时完成以下操作：

1. 在 checkpoint_interval 满足 should save 条件时，将模型参数和优化器状态 checkpoint，保存到shared_storage_path

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

### 3.3  修改 step 函数

在 checkpoint_interval 满足 should save 条件时，将模型参数和优化器状态 checkpoint，保存到shared_storage_path

修改前：

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

修改后：

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

### 3.4 修改 prepare 函数

```python
class ElasticTrainer(object):
	def prepare(self, optimizer, lr_scheduler=None):
	        """
	        Prepare optimizer and learning rate scheduler in elastic training.
	        """
					# If the trainer is configured to load from a checkpoint,
					# load both the model state and the optimizer state.
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

### 2.4 支持 reshard 的 save/load

**背景：**

弹性容错需要依赖 checkpoint 来恢复模型状态。当前大模型训练采用 FSDP 的并行方式，FSDP 保存 checkpoint 的方案有两种：

1. rank0_only：由 rank-0 节点获取所有的模型参数和优化器状态存入磁盘;
2. sharding 方式：所有 rank 各自保存其模型参数和优化器状态。

但是这两个方案都没法满足弹性容错训练的需求。

rank0_only：

1. rank-0 需要加载所有的模型参数和优化器状态，可能导致 OOM。
2. rank-0 需要通过 Allgather 获取所有模型参数和优化器状态并依次写入磁盘，耗时过长。

sharding 方式：

1. 保存 checkpoint 的 rank 数量必须和加载 checkpoint 的 rank 数量必须一致。而弹性容错作业中并不能保证 rank 数量不变。

**目标：**

对于 sharding 方式进行改进，使其支持 rank 数量改变时的 resharding

**设计：**

保存每个参数的起始位置（start，end）将这些信息作为 checkpoint 的 meta 信息进行保存到单独文件中。然后在 load 时依据这些参数进行 reshard。

![fsdp](../figures/fsdp-resharding.png)

<aside>
由于模型在 save 和 load 时需要对非 flatten 数据操作（Tensor），所以对模型参数我们需要保存的是三元组（original_shape，start，end）
</aside>

保存的文件树样例如下：

```bash
ckpt
├── optim_meta
├── optim_param.00000-00002
└── optim_param.00001-00002
```

我们将使用 **LOCAL_STATE_DICT** 来保存checkpoint

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

**后续的两个改进：**

1. 异步写入，在保存checkpoint这个操作的时候单开一个线程去写入，不阻塞训练过程
2. meta信息在nodegroup没有变更的时候不重复写入，因为分片没有改变时候meta信息也没有改变

## 3. CRD

### 3.1 Job yaml 增加 volumeMounts 和 env

用户需要现在集群里创建或使用一个 PVC

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: dlrover-shared-pvc
spec:
  accessModes:
  - ReadWriteMany
  resources:
    requests:
      storage: 1Gi
```

```yaml
apiVersion: elastic.iml.github.io/v1alpha1
kind: ElasticJob
metadata:
  name: torch-mnist-fsdp
  namespace: dlrover
spec:
  distributionStrategy: AllreduceStrategy
  optimizeMode: single-job
  replicaSpecs:
    worker:
      replicas: 4
      template:
        spec:
          restartPolicy: Never
					volumes:
					  - name: dlrover-shared-storage
					    persistentVolumeClaim:
					      claimName: dlrover-shared-pvc
          containers:
            - name: main
						volumeMounts:
						    - name: dlrover-shared-storage
						      mountPath: /data/shared-worker-data
						env:                       
						    - name: USE_FSDP
						      value: "true"
						    - name: SHARED_STORAGE_PATH
						      value: "/data/shared-worker-data"
```

用户提交的CRD里需要增加的有：

1. 一个 PVC 的 volume和 volumeMounts
2. 两个环境变量：USE_FSDP 和 SHARED_STORAGE_PATH

## 4. 部署PV

为了让 Worker Pods 共享存储，选择的存储解决方案必须支持 **`ReadWriteMany`** 访问模式。这是因为 **`ReadWriteMany`** 访问模式允许多个节点上的 pods 同时访问同一个存储卷。

一些常见的支持 **`ReadWriteMany`** 的存储解决方案：

1. **NFS (Network File System)**: NFS 是一个常用的支持多读写的文件系统。你可以设置一个 NFS 服务器，并使用 NFS-provisioner 在 Kubernetes 中动态提供 PersistentVolumes。
2. **CephFS**: Ceph 是一个高度可扩展的分布式文件系统，其上层的 CephFS 支持多读写模式。
3. **GlusterFS**: Gluster 是一个自由开源的可扩展的网络文件系统，可以提供大量的存储池，并支持多节点并发访问。

要在多节点 Kubernetes 集群中实现 master 和 worker pods 共享存储的功能，需要：

1. 存储后端（例如 NFS、CephFS、GlusterFS 等）已经在集群中部署并运行正常。
2. 创建 **`PersistentVolume`** 和 **`PersistentVolumeClaim`** 以使用该存储。确保在 PVC 的定义中指定 **`ReadWriteMany`** 访问模式。
3. 在 master 和 worker pods 的定义中使用这个 PVC

<aside>
在多数情况下，不需要手动创建PV。很多 Kubernetes 集群配置了自动化的存储供应，例如AWS EBS、Google Cloud Persistent Disk 或 Azure Disk Storage，这些存储解决方案会在PVC创建时自动供应一个新的 PV。
但是，如果 Kubernetes 环境没有自动存储供应或你需要特定的存储配置，那么需要手动创建 PV。在这种情况下，你可以在 PV 中指定特定的参数和配置，然后再创建 PVC 来使用这个 PV。

</aside>