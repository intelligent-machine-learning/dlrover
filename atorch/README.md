# ATorch
<div id="top" align="center">

   <img src="docs/img/atorch.png" alt="Editor" width="500">

   ATorch: Make large model training more efficient and reproducible for everyone.



   [![GitHub Repo stars](https://img.shields.io/github/stars/intelligent-machine-learning/dlrover?style=social)](https://github.com/intelligent-machine-learning/dlrover/stargazers)
   [![Build](https://github.com/intelligent-machine-learning/dlrover/actions/workflows/main.yml/badge.svg)](https://github.com/intelligent-machine-learning/dlrover/actions/workflows/main.yml)

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
* Usability
  * Fast deployment of runtime environment (images and installation packages)
* Solutions for large-scale model training
* Automated optimization
  * auto_accelerate for automatic optimization
* IO/Preprocessing
  * Recommended storage for training data
  * Accessing the Pangu cluster
  * CPU/GPU cooperation to optimize data preprocessing
* Customized operator optimization 
  * High-performance MOE
  * Flash Attention 2
  * Transformer operator
* Mixed precision
* Communication optimization
  * Cashed sharding
* Hybrid parallelism
* Compilation optimization
* Elastic fault tolerance
  * HangDetector (detecting and automatically restarting distributed training if it hangs)
  * GPU elastic training
  * Hardware error detect and migration



## Installation

ATorch supports PyTorch with version >= 1.12, and verion 2.1 or above is preferred.
For example, you can use docker image <code>easydl/atorch:iml_pt210</code> which has PyTorch 2.1 installed.

### Install From pypi
Install atorch in any PyTorch-preinstalled environment (such as a container created with the docker image above) with <code>pip</code>: 

```
pip install atorch
```

### Install From Source Files

```
# clone repository
git clone https://github.com/intelligent-machine-learning/dlrover.git
cd dlrover/atorch
# build package
sh dev/scripts/build.sh
# install the created package in dist directory
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


