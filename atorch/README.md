# ATorch
<div id="top" align="center">

   <img src="docs/img/atorch.png" alt="Editor" width="500">

   ATorch: Make large model training more efficient and reproducible for everyone.



   [![GitHub Repo stars](https://img.shields.io/github/stars/intelligent-machine-learning/dlrover?style=social)](https://github.com/intelligent-machine-learning/dlrover/stargazers)
   [![Build](https://github.com/intelligent-machine-learning/dlrover/actions/workflows/main.yml/badge.svg)](https://github.com/intelligent-machine-learning/dlrover/actions/workflows/main.yml)
   [![PyPI Status Badge](https://badge.fury.io/py/atorch.svg)](https://pypi.org/project/atorch/)

</div>


## Table of Contents
<ul>
 <li><a href="#Features">Features</a> </li>
  <li><a href="#Installation">Installation</a></li>
 <li><a href="#Getting-Started">Getting Started</a></li>
 <li><a href="#Contributing">Contributing</a></li>

</ul>


ATorch is an extension library of PyTorch developed by Ant Group's AI Infrastructure team. By decoupling model definition from training optimization strategy, ATorch supports efficient and easy-to-use model training experience. The design principle is to minimally disrupt the native PyTorch programming style. Through its API, ATorch provides performance optimizations in aspects such as I/O, preprocessing, computation, and communication (including automatic optimization). ATorch has supported large-scale pretraining of LLMs with over 100 billion parameters and thousands of A100/H100 GPUs. 

## Features

![atorch_diagram](docs/img/atorch_fig.png)
* Easy-to-use interface
  * [auto_accelerate](docs/auto_accelerate_api.md) API
  * ATorchTrainer (ongoing work)
* Solutions for large-scale model training
  * support efficient large model initialization, checkpoint save/load, and restart with elastic resources.
* Automatic/semi-automatic optimization
  * Acceleration Engine for automatic optimization
  * Semi-automatic optimization supports custom optimization
* Hybrid parallelism support (arbitrary combination of fsdp/zero/ddp/tp/sp/pp)
* High performance operators
  * Flash attention 2 with custom mask support
  * Transformer ops
  * High-performance MOE
  * sub-graph compilation
* Checkpointing
* Mixed precision
* Communication optimization
  * Cached sharding
* Effective optimizers for fast training convergence
  * [AGD optimizer](docs/README-AGD.md)
  * [WSAM optimizer](docs/README-WSAM.md)
* IO/Preprocessing
  * CPU/GPU coworker to speedup data preprocessing 
  * IO optimization for different dataset
* Elastic and fault tolerance
  * Hardware error detection and migration (with dlrover)
  * GPU elastic training support
  * HangDetector (detecting and automatically restarting distributed training if it hangs)

## Installation

ATorch supports PyTorch with version >= 1.12, and version 2.1 or above is preferred.
For example, you can use docker image <code>registry.cn-hangzhou.aliyuncs.com/atorch/atorch-open-20240430:pt210</code>) which has PyTorch 2.1 installed.

### Install From PyPI
Install atorch in any PyTorch-preinstalled environment (such as a container created with the docker image above) with <code>pip</code>: 

```
pip install atorch
```

### Install From Source Files

```
# clone repository
git clone https://github.com/intelligent-machine-learning/dlrover.git
cd dlrover/atorch
# build package, optional set version.
bash dev/scripts/build.sh [version]
# install the created package in dist directory. Note that if version is set, file name is different.
pip install dist/atorch-0.1.0.dev0-py3-none-any.whl
```


## Getting Started

### Run Examples


- To run [auto_accelerate examples](examples/auto_accelerate):
```
cd dlrover/atorch/examples/auto_accelerate
# Single process train
python train.py --model_type toy
# Distributed train
python -m atorch.distributed.run  --nproc_per_node 2  train.py --model_type llama --distributed --load_strategy --use_fsdp --use_amp --use_module_replace --use_checkpointing
```

- [Llama2 pretrain/finetune examples](examples/llama2)

- [Optimizer (AGD, WSAM) Examples](examples/optimizer)

### Documentations

[auto_accelerate](docs/auto_accelerate_api.md)

[AGD optimizer](docs/README-AGD.md)

[WSAM optimizer](docs/README-WSAM.md)




## Contributing
Contributions are welcome! If you have any suggestions, ideas, or bug reports, please open an issue or submit a pull request.

## CI/CD

We leverage the power of [GitHub Actions](https://github.com/features/actions) to automate our development, release and deployment workflows. Please check out this [documentation](.github/workflows/README.md) on how the automated workflows are operated.


