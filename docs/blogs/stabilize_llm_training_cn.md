# DLRover在Kubernetes上千卡级大模型训练稳定性保障的技术实践

## 背景

如今大语言模型（LLM）的分布式训练节点规模越来越大，训练耗时长。比如 OpenAI 在 1024 个
NVIDIA A100 GPU 上训练 GPT-3 大约需要 34 天。训练节点越多，耗时越长，训练期间节点故障概率就越大，况且
A100 GPU 的故障率也相对较高。所以大规模训练作业难免会遇到节点故障。据我们在蚂蚁 GPU
训练集群上观察，一个月内，单卡的故障率约8%，那么一天单卡的故障率约为0.27%。常见的故障原因有
Xid、ECC、NVLINK error 和 NCCL error 故障等。对于一个千卡训练作业来说，
卡故障导致一天内训练失败的概率高达到 93%。所以训练作业几乎每天都会失败。作业失败后，
用户需要手动重启作业，运维成本很高。如果用户重启不及时，中间间隔的时间就会导致 GPU 卡空闲，浪费昂贵的算力资源。
有些故障会导致机器不可用，从而导致可用节点数量不能达到用户指定的数量。这时，训练就不能启动，
用户需要手动减少节点数量后重新提交作业。待故障机修复后，用户又需要手动增加作业的节点数来重启作业。
这样增大了用户的运维成本，也导致了新节点无法及时加入训练。
为此，DLRover 在 Kubernetes 上基于 Torch Elastic 开发了弹性训练功能，实现
PyTorch 分布式训练的自动容错和弹性。具体功能如下：

1. 出现故障后，快速执行节点健康检测，定位故障机并将其隔离，然后重启 Pod 来替换故障节点。
2. 健康检测通过后，重启训练子进程来自动恢复模型训练，无需重启作业或者所有Pod。
3. 节点故障导致可用机器少于作业配置，自动缩容来继续训练。集群新增机器后，自动扩容来恢复节点数量。

在 DLRover 弹性容错应用在蚂蚁大模型训练前，一周内千卡训练运行时间占 60.8%，有效训练时间约 32.9%。
有效训练时间 = 模型迭代的步数 * 每步的时间，除此之外，训练运行时间还包括checkpoint 保存时间和训练回退时间等。
DLRover 上线后，一周内千卡训练运行时间占比提升至 83.6%，有效训练时间提升至 58.9%。

## 故障对训练的影响

故障发生后，训练会中断，用户需要看下作业日志来排查故障原因。如果是网络抖动造成的 NCCL timeout 等问题，
用户只需重启训练作业。如果是机器故障导致的训练失败，用户则需要排查是哪些机器有问题并将问题机器从集群中下线掉，
然后再重新提交作业。如下图所示：

<div align="center">
<img src="../figures/ft_llm_training/training-failure.png" alt="Editor" width="600">

<text>图1: 训练故障及处理方式 </text>
</div>

手动排查故障机器是比较麻烦的事情，一般采用二分法用 nccl-test 来找故障机器。如果我们用了 256 台机器，则至少需要
运行 14 次 nccl-test 来定位故障机器，耗时费力。

## PyTorch 分布式训练的弹性与容错

### PyTorch 容错框架

为了实现训练的弹性与容错，Torch Elastic 和 Elastic Horovod 都可以在故障发生后自动恢复训练进程。
二者显著的区别在于节点数量变化后是否需要重启训练子进程来恢复训练。Torch Elastic 感知到新节点加入后会立刻重启所有节点的子进程，
集合通信组网，然后从 checkpoint 文件里恢复训练状态来继续训练。而 Elastic Horovod 则是每个训练子进程在每个 step 后检查新节点加入
，子进程不退出的情况下重新集合通信组网，然后有rank0将模型广播给所有rank。二者的优劣对比如下：

|  | Torch Elastic | Elastic Horovod |
| --- | --- | --- |
| 是否需要修改训练代码 | 否 | 需要 hvd.elastic.run 来包装train 函数，以便重组网 |
| 支持的backend | NCCL/gloo | gloo |
| 模型保存于恢复方式 | checkpoint 到磁盘 | 各个进程保持到内存 |
| 支持的训练模式 | DDP/FSDP | DDP |
| 支持的模型大小 | 大 | 小，只能是单机能存下的 |

通过上述对比可以看出，Torch Elastic 重启训练子进程的方案对用户更加友好，支持更多的分布式训练策略和模型。
而FSDP和NCCL是当前大模型分布式训练使用最为广泛的技术。所以 DLRover 选择使用 Torch Elastic 重启子进程的方案来实现 Kubernetes 集群上分布式训练的弹性容错。

### TorchElastic 动态组网

动态组网是指训练进程可以自动根据动态变化的节点数量来组网通信，无需固定给各个节点指定组网的 rank 和 world size。
动态组网是弹性容错训练缩必须的，因为弹性容错作业中，节点的失败、扩容或者缩容都会导致节点的 rank 和 world size 变化。
所以我们无法在作业启动前给节点指定 rank 和 world size。

TorchElastic 使用 Dynamic Rendezvous 机制来协助子进程组网。
每个节点上运行一个 ElasticAgent，ElasticAgent 会从一个共享存储中获取作业节点的 host group，然后将自己的 host 加入 group
并同步到共享存储里。这个共享存储当前默认使用 TCPStore。接着，ElasticAgent 不断从共享存储里获取查询 host group，
直到 host group 里的节点数量达到最小节点数量 min_nodes 且一段时间内没有变化，即认为所有节点都准备好了。然后，
ElasticAgent 就可以从 host group 里获取自己的节点rank （PyTorch 中称为 group rank） 和 world size。这样，
ElasticAgent 就可以给拉起的子进程配置 local rank、global rank 和 world size了。有了这些信息，子进程就可以进程集合通信组网。
但是使用 Torch Elastic 原生方案中，我们发现一些问题：

1. 节点不能容错。TCPStore 在一个训练节点上，如果该节点挂了，重组网就没法继续了。
2. 节点 rank 是随机的。 ElasticAgent 同步 host 到共享存储的顺序是随机的，导致节点 rank 的随机。
在训练代码中，用户一般会将模型迭代信息输出在 rank-0 的日志里，比如 step、loss 和耗时等。
用户只能通过进程日志寻找 rank-0 ，对于多节点的作业，这是比较麻烦的。
3. Torch Elastic 的动态组网不能控制组网的节点数量。比如 LLM 模型训练中，
用户可能会将4个节点作为一个数据预处理的组，那么弹性伸缩需要保证节点数量是4的整数倍。
而 Torch Elastic 只要发现有一个新节点加入就会立刻重启训练。

## DLRover 训练故障自愈

### 分布式训练故障自愈流程

为了实现训练故障自愈，DLRover 实现了自动的节点检查，训练进程自动重启，大幅提升了训练的稳定性和人工运维成本。DLRover
的故障资源流程如下：

<div align="center">
<img src="../figures/ft_llm_training/dlrover_failure_recovery.jpg" alt="Editor" width="600">

<text> 图2：DLRover 训练故障自愈流程 </text>
</div>

### 分布式训练架构

为了在 Kubernetes 集群上使用 DLRover 的训练故障自愈功能，DLRover 开发了 ElasticJob。ElasticJob 采用
master-worker 架构，master Pod 是一个轻量级的 CPU 节点，主要进程节点故障检测、节点管理和worker 组网管理。
worker 是运行 AI 训练的节点，其主要包含一个基于 TorchElastic 定制的 ElasticAgent。该 ElasticAgent 主要
从 master 获取组网信息、启动和监控训练进程和上报心跳信息。如下图所示：

<div align="center">
<img src="../figures/dlrover-elasticjob.png" alt="Editor" width="600">

<text>图3：DLRover 弹性训练架构 </text>
</div>

### 节点检测流程

DLRover 在重启训练前在每个 GPU 上启动子进程来运行一个轻量级的检测任务来检查机器。该检测任务包含一个简单的 GEMM 和 allgather 任务。
详细见[GPU 检测脚本](../../dlrover/trainer/torch/node_check/nvidia_gpu.py)
和[昇腾芯片检测脚本](../../dlrover/trainer/torch/node_check/ascend_npu.py)。
DLRover 启动训练任务前的检测流程如下。

<div align="center">
<img src="../figures/ft_llm_training/dlrover_node_check.jpg" alt="Editor" width="600">

<text>图4：DLRover 节点检测流程 </text>
</div>

job master 先将所有节点两两划分为多个
group, group 内的节点上执行 allgather 任务并将运行耗时上报给 job master。
如果有 group 里的有节点执行检测任务失败，则此 group 的节点为潜在故障机，否则为正常机器。
然后开始第二轮测试，master 会将潜在故障节点和正常节点再次两两划分 group。每个 group 的节点继续执行 GEMM 和 allgather。
如果某个节点两轮执行都失败了，则该节点为故障节点，或者两轮执行的耗时都明显长于其他节点，则该节点为慢节点。
比如作业有6个节点，第一轮的划分结果为 [{1,2}, {3,4}, {5,6}]， {5, 6} 执行检测任务失败了，
那么节点 5 和 6 就是潜在故障节点。为此第二轮的划分为[{1,2}, {3,5}, {4,6}] 。如果{4,6} 失败了，
说明节点 6 就是故障节点。然后，DLRover 会重新拉起一个 Pod，替换节点6。

<div align="center">
<img src="../figures/ft_llm_training/node_health_check.jpg" alt="Editor" width="600">

<text>图5：DLRover 节点检测示例 </text>
</div>

### DLRover 动态组网与弹性扩缩容

为了解决上面提到的 TorchElastic 动态组网的问题，DLRover 重新实现了 TorchElastic 的 ElasticAgent。
DLRover 的 ElasticAgent 在组网时将自己的节点 RANK 发送给 job master，job master 根据收到的节点 RANK
信息来划分组网的 group。这样 job master 就可以控制组网 group 中的节点信息，比如可以实现 group 中节点
数量满足一定条件；有新节点加入后给节点发送新的 group 来重新组网从实现弹性扩缩容。例如，
训练作业配置了6个节点，由于机器故障导致 Pod-5 失败了，重新拉起的 Pod-6 因为没有资源而 pending。此时，
master 收到的节点信息为 {0, 1, 2, 3, 4, 5}。但是用户要求节点是 2 的整数倍，那么master可以将
Pod-4 从 group 中踢出，然后发送给 Pod-0/1/2/3。而 Pod-4 会等着 Pod-6 起来后再加入训练实现扩容。如下图所示：

<div align="center">
<img src="../figures/ft_llm_training/dlrover_scale_node.jpg" alt="Editor" width="600">

<text>图6：DLRover 节点自动扩缩容 </text>
</div>

### DLRover 错误日志收集

在 PyTorch 分布式训练中，一个节点的进程出错后，Torch Elastic 会停止所有节点的进程。
各个进程的日志都是单独存在各自日志文件中。为了找到训练失败是哪个进程出错导致的，我们需要搜索所有进程的日志
。这个工作对于千卡作业是十分耗时且繁琐的。为此，我们在 ElasticAgent 中开发了错误日志收集供功能。
当 ElasticAgent 发现子进程失败后，后将其错误日志的 message 发送给 job master。
job master 会在其日志中展示具体哪个节点的那个进程失败了，以及错误日志。
这样用户只需看下 job master 的节点日志就可以定位训练失败原因了。同时我们也支持将错误信息上报给用户。

```json
任务 torch-train 训练进程失败 torch-train-edljob worker-116 restart 0 fails: {
  "784": {
    "local_rank": 0,
    "exitcode": -6,
    "message": {
      "message": "RuntimeError: CUDA error: uncorrectable NVLink error detected during the execution\nCUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.\nFor debugging consider passing CUDA_LAUNCH_BLOCKING=1.\nCompile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.\n",
      "extraInfo": {
        "py_callstack": "Traceback (most recent call last):\n"  
        "timestamp": "1689298411"
      }
    },
    "datetime": "2023-07-14 09:33:31"
  }
} 
```

## 弹性容错在千亿级大模型训练的应用效果

在使用 DLRover 弹性容错之前，Torch 大模型训练只要出错就要重启训练作业。为了及时重启作业，
用户写了个程序每隔10min 来检测作业状态。如果失败，就会重启作业。

<div align="center">
<img src="../figures/ft_llm_training/dlrover_ft_result.jpg" alt="Editor" width="600">

<text>图7：千卡作业运行时间占比 </text>
</div>

下面对比了训练失败时使用 DLRover 弹性容错前后的耗时。

|| 没有弹性容错 | DLRover 弹性容错 |  |
| --- | --- | --- | --- |
| 训练恢复步骤 | 任何故障 | 机器硬件故障 | 软件故障 |
| 作业启动，包括作业失败检测、作业提交 | 约 10 min | 无需重启作业 | 无需重启作业 |
| Pod 调度、镜像下载、安装依赖包等 | 资源充足，所有 Pod 成功拉起的时间约5min。 | 资源充足，单个 Pod 拉起约 2min。 | 无需拉起 Pod |
|  | 资源不足，只能等到资源足够或者手动减少资源重启。 | 自动缩容约 3min。 | 无需拉起 Pod |
| 节点健康检测 | 无节点健康检测 | 验证节点约 2-3min | 验证节点约 1min |
|  | 节点网络故障，手动检测约 2h | 自动检测约 5min | 自动检测 5min |
|  | 其他硬件错误，取决于SRE 巡检的周期。 | 自动检测约 5min | 自动检测5min |
| checkpoint 保存 | 10 min | 1 min | 1 min |
| checkpoint 加载 | 8 min | 4min | 4min |

## Kubernetes 上提交 GPT 弹性容错作业

1. 在 Kubernetes 集群上部署 DLRover ElasticJob CRD。

GO 版本： GO 1.18.

```python
git clone git@github.com:intelligent-machine-learning/dlrover.git
cd dlrover/dlrover/go/operator/
make deploy IMG=easydl/elasticjob-controller:master
```

2. 在构造训练镜像的 dockerfile 中安装 dlrover[torch]。

```python
FROM registry.cn-hangzhou.aliyuncs.com/easydl/dlrover-train:torch201-py38  as base

WORKDIR /dlrover

RUN apt-get install sudo
RUN pip install dlrover[torch] -U
COPY ./model_zoo ./model_zoo

```

3. 在 ElasticJob 的container 的 command里使用 dlrover-run 在运行训练脚本。
在镜像 registry.cn-hangzhou.aliyuncs.com/easydl/dlrover-train:nanogpt-test
我们已经准备好了代码和训练数据，可以直接用如下 ElasticJob 来提交示例作业。

```yaml
apiVersion: elastic.iml.github.io/v1alpha1
kind: ElasticJob
metadata:
  name: torch-nanogpt
  namespace: dlrover
spec:
  distributionStrategy: AllreduceStrategy
  optimizeMode: single-job
  replicaSpecs:
    worker:
      replicas: 2
      template:
        spec:
          restartPolicy: Never
          containers:
            - name: main
              # yamllint disable-line rule:line-length
              image: registry.cn-hangzhou.aliyuncs.com/easydl/dlrover-train:nanogpt-test
              imagePullPolicy: IfNotPresent
              command:
                - /bin/bash
                - -c
                - "dlrover-run --nnodes=1:$NODE_NUM \
                  --nproc_per_node=1 --max_restarts=1  \
                  examples/pytorch/nanogpt/train.py \
                  --data_dir '/data/nanogpt/'"
              resources:
                limits:
                  cpu: "8"
                  memory: 16Gi
                  # nvidia.com/gpu: 1 # optional
                requests:
                  cpu: "4"
                  memory: 16Gi
                  # nvidia.com/gpu: 1 # optional

```

## 总结 & 未来计划

DLRover 目前已经在蚂蚁千亿模型训练训练上落地，将GPU故障导致训练暂停时间由 30%降低到了约 12%。
我们希望 DLRover 在大规模分布式训练上提供智能化运维功能，降低用户运维成本，提升训练的稳定性。
后续我们将介绍蚂蚁在千亿模型训练上的 PyTorch 性能优化方案的扩展包 ATorch，ATorch 旨在提升大规模
GPU 训练的硬件算力效率 HFU (Hardware Flops Utilization) 和训练的稳定性，当前蚂蚁千亿大模型训练使用
Atorch 的 HFU 为 49.6%。我们欢迎不同机构的开发者也能根据自身特点，同我们一起共建 DLRover 项目，推进分布式自动化。
