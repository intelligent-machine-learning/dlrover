# TFPlus
<div id="top" align="center">


   <h3> <a href="https://arxiv.org/pdf/2107.14432.pdf">Optimizer Paper </a> |
   <a href="./tutorials_ZH.md"> Tutorials </a> |
   <a href="./optimizer_api_ZH.md"> API Doc </a> 
   <p></p>

   [![GitHub Repo stars](https://img.shields.io/github/stars/intelligent-machine-learning/dlrover?style=social)](https://github.com/intelligent-machine-learning/dlrover/stargazers)
   [![Build](https://github.com/intelligent-machine-learning/dlrover/actions/workflows/main.yml/badge.svg)](https://github.com/intelligent-machine-learning/dlrover/actions/workflows/main.yml)
   [![CodeFactor](https://www.codefactor.io/repository/github/intelligent-machine-learning/dlrover/badge)](https://www.codefactor.io/repository/github/intelligent-machine-learning/dlrover)


   | [English](../README.md) | [中文](./README-zh-Hans.md) |

</div>

TFPlus 是蚂蚁集团自研的高性能 TensorFlow 扩展库，支持了蚂蚁集团搜推广等核心业务，支撑着万亿超大规模的稀疏训练。TFPlus 积累了核心的稀疏场景的功能及性能优化。针对稀疏模型在 IO、算子、图优化、分布式、集合通信等方面进行了深度的性能优化，同时提供了稀疏场景下特有的优化器、容错、弹性、增量更新等功能。<br />TFPlus Opensource 是 TFPlus 的社区开源版本，其包含了蚂蚁集团超大规模的稀疏训练的核心能力，其主要的特性如下：

- 以插件化的方式，提供高效的 TF 算子扩展；
- 提供推荐场景下高性能稀疏 Embedding 训练的支持：Kv Variable；
- 提供高性能自研深度学习优化器：Group Adam、Group Adagrad。
> 我们将推进 DLRover 开源体系建设，逐步将 TFPlus 的全部功能进行开源。

## TFPlus 安装
我们提供了两种方式来安装 TFPlus：一种是直接安装我们在github发布的whl包，另一种是在本地构建。
### Github release 安装
首先，你需要安装 TensorFlow 2.13.0。值得注意的是，TFPlus 当前只对 TensorFlow 的 CPU 版本进行了优化支持。  
然后前往[Github仓库](https://github.com/intelligent-machine-learning/dlrover)的[release页面](https://github.com/intelligent-machine-learning/dlrover/releases)获取最新的tfplus的whl下载地址。
```shell
pip install tensorflow-cpu==2.13.0

# 通过 pip 安装最新版本的 TFPlus：
pip install [Github release path]

# 为了确认 TFPlus 安装成功，你可以尝试导入 TFPlus 并打印其版本：
python -c "import tfplus; print(tfplus.__version__)"
```
目前，TFPlus 的 PyPI 版本仅在 Linux 下进行了测试和支持。如果安装不成功，请在本地安装我们的镜像在镜像内使用或者直接在本地进行构建，见[本地构建安装](#local-build)。
<a id="local-build"></a>
### 本地构建安装
另外一种可选的安装方式是在本地构建，这需要更多的时间（取决于你的机器性能）。<br /> 你可以通过运行以下命令来下载和启动我们的 Docker 镜像：
```shell
git clone https://github.com/intelligent-machine-learning/dlrover.git
cd dlrover
# For GPU image: easydl/tfplus:tf213_dev_gpu
docker run -it --net=host -v ${PWD}:/v -w /v easydl/tfplus:tf213_dev /bin/bash

# 然后，在启动的 Docker 容器中执行构建脚本：
cd tfplus/dev/scripts
bash build_and_test.sh
```
### 注意事项

- TFPlus 正在使用的是 TensorFlow 2.13.0，因此你需要首先安装对应版本的 TensorFlow；
- TFPlus 当前仅支持 CPU 训练。我们计划在未来版本中添加 GPU 支持；
- TFPlus 当前仅在 Linux 平台下进行了测试和支持；
- 尽管 TFPlus 基于 TF2 进行功能扩展，但是当前只支持 TF1 下的静态图模式（Graph Execution）训练。TF2 默认的 Eager Execution 目前还不支持。
## 使用方法
使用 Kv Variable 的 Embedding 能力，您只需要在构建 Tensor 变量时候进行简单替换：
```python
# 使用 get_kv_variable 替换 tf.compat.v1.get_variable 构建 embedding
# get_kv_variable 只需要指定 embedding_dim，不需要定义特征数量
from tfplus.kv_variable.python.ops.variable_scope import get_kv_variable
user_embeddings = get_kv_variable(
    name="user_dynamic_embeddings",
    key_dtype=tf.int64,
    embedding_dim=32,
    initializer=tf.compat.v1.keras.initializers.RandomNormal(-1, 1))

# 使用 embedding_lookup 进行 embedding 特征查找
import tfplus
from tensorflow.python.ops.embedding_ops import embedding_lookup
user_id_weights = embedding_lookup(params=self.user_embeddings,
                                   ids=user_id_val,
                                   name="user-id-weights")
```
使用 Kv Variable 后，对应的优化器需要同样替换成 TFPlus 实现的优化器
```python

# TFPlus 目前支持四种优化器
# 包含两种常用的 Adam、Adagrad 以及两种基于 Group Lasso 的 Group Adam、Group Adagrad (Sparse Group Ftrl)
from tfplus.kv_variable.python.training import AdagradOptimizer
from tfplus.kv_variable.python.training import AdamOptimizer
from tfplus.kv_variable.python.training import GroupAdamOptimizer
from tfplus.kv_variable.python.training import SparseGroupFtrlOptimizer

opt = GroupAdamOptimizer(
       learning_rate = 0.001,
       initial_accumulator_value=0.0,
       beta1=0.9,
       beta2=0.999,
       epsilon=1e-8,
       l1_regularization_strength=0.0,
       l2_regularization_strength=0.0,
       l21_regularization_strength=0.0,
       use_locking=False,
       name="GroupAdam",
       accum_name=None,
       linear_name=None)

```
关于 Kv Variable 和优化器的详细使用方法和端到端训练样例，请参考：
-  [Tutorials](./tutorials_ZH.md)
-  [Kv Variable Api Doc](./kv%20variable_api_ZH.md)
-  [Optimizer Api Doc](./optimizer_api_ZH.md)
## 插件化 TF 算子扩展
TFPlus Opensource 兼容 **TensorFlow 2.13.0**，在 TF2 的基础上以插件化的方式扩展了面向稀疏 CTR 模型的稀疏 Embedding 算子（我们称为 KvVariable）以及兼容 Kv Variable 算子的高性能自研优化器。<br /> 目前，TFPlus Opensource 插件库支持以下能力

- Kv Variable（核心 Embedding 能力）
   - 高性能 Embedding Ops
   - Embedding 权重动态扩容和分区
   - 单机训练和 PS/Worker 集群训练
- 高性能优化器
   - 兼容 Kv Variable 的常用优化器
      - Adam
      - Adagrad
   - 基于 Sparse Group Lasso 的自研深度学习优化器
      - Group Adam
      - Group Adagrad
## Kv Variable
Kv Variable 是基于 TensorFlow 的 Embedding 参数服务器实现，在蚂蚁集团主要用于解决深度学习中稀疏 Embedding 的计算、存储问题，通过分布式哈希存储的方式，支持稀疏特征参数的计算、查询、插入、动态扩容、分片以及弹性。<br />Tensorflow 提供的 tf.Variable 来构建一个基础的 Tensor，使用稠密矩阵的方式进行存储和访问，需要提前定义好 shape。在 TFPlus 中，我们实现了一个全新的面向 Embedding 的 Variable，即 Kv Variable。其内部数据类型采用 Key-Value 格式存储, 并支持以下功能：

- KvVariable 的键（Key）支持多种数据类型，包括 [int, int32, int64, string]；
- KvVariable 不需定义特征数量（即 Embedding 的第 0 维），只需要定义 embedding_dimension，新出现的特征会被动态加入到训练中，并支持频次过滤；
- KvVariable 可以用于推荐场景的 Embedding 计算，也可以用于 NLP 的文本输入的 Embedding 构造；
- KvVariable 支持数据分片（即 Partitioned Variable），对超大规模的 Embedding 层进行分片，支持自定义 partition_strategy 使用不同的分片策略，将特征均匀分布到不同分片上；
- Kv Variable 参数的导入、导出完全兼容 Tensorflow 官方的接口和格式，无需做额外的代码修改。
## 自研高性能优化器
由于 Kv Variable 自定义 Embedding 算子的加入，梯度更新的优化器算子也要进行对应的修改。TFPlus Opensource 提供了业界常用的优化器以及蚂蚁集团自研的高性能优化器的对应实现。

- 兼容 Kv Variable 的常用优化器
   - Adam
   - Adagrad
- 基于 Sparse Group Lasso 的自研深度学习优化器
   - Group Adam
   - Group Adagrad

Lasso 和 Group Lasso 可以用来对模型进行稀疏化压缩，自动选择重要特征。我们将 Sparse Group Lasso 功能添加到深度学习的一系列常用优化器中，开发了一类新的优化器。Group Lasso 系列优化器在蚂蚁集团内部搜索、广告和推荐场景中已经得到了广泛的使用。具体的算法见 Algortihm 1。

<br />![image.png](images/Algorithm1.png)<br /> 

上述算法中当 $\lambda_{1},\ \lambda_{2}, \ \lambda_{21}$（分别对应 $\ell_{1}, \ \ell_{2}, \ \ell_{21}$ 的正则系数）都为 $0$ 时，Group Lasso 类优化器即退化为常用优化器，对应关系如下：

| **Group Lasso 类优化器 ** | ** 常用优化器 ** |
| --- | --- |
| Group AdaGrad (Sparse Group Ftrl) | AdaGrad |
| Group Adam | Adam |

> 关于 Sparse Group Lasso 优化器更多的细节，可以查看我们的论文 [Adaptive Optimizers with Sparse Group Lasso for Neural Networks in CTR Prediction](https://arxiv.org/pdf/2107.14432.pdf)，ECML PKDD '21。

## 支持本项目
我们希望通过与社区共享研究，可以使更多人受益，并开启更多有关深度学习优化研究的可能性。感谢您对我们工作的关注和支持。也请欢迎对我们的代码作出贡献或提出问题和意见。<br /> 如果您在研究中使用了我们的优化器并发现其有用，我们鼓励您引用我们的文章。您的引用是对我们工作的一种极好的赞同和支持。<br /> 以下是您可以使用的引用信息：
```bibtex
@inproceedings{yue2021adaptive,
  title={Adaptive Optimizers with Sparse Group Lasso for Neural Networks in CTR Prediction},
  author={Yue, Y and Liu, Y and Tong, S and others},
  booktitle={Machine Learning and Knowledge Discovery in Databases. Research Track: European Conference, ECML PKDD 2021, Bilbao, Spain, September 13–17, 2021, Proceedings, Part III 21.},
  pages={314-329},
  year={2021},
  publisher={Springer International Publishing},
}
```
## License
Apache License 2.0
