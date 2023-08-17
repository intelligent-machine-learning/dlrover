# ATorch
<div id="top" align="center">

   <img src="docs/img/atorch.png" alt="Editor" width="500">

   ATorch: Make LLMs training more efficient and reproducible for everyone.

   <h3> <a href="https://www.placeholder.com/"> Paper </a> |
   <a href="https://www.placeholder.com/"> Documentation </a> |
   <a href="https://www.placeholder.com/"> Examples </a> |
   <a href="https://www.placeholder.com/"> Blog </a></h3>

   [![GitHub Repo stars](https://img.shields.io/github/stars/intelligent-machine-learning/dlrover?style=social)](https://github.com/intelligent-machine-learning/dlrover/stargazers)
   [![Build](https://github.com/intelligent-machine-learning/dlrover/actions/workflows/main.yml/badge.svg)](https://github.com/intelligent-machine-learning/dlrover/actions/workflows/main.yml)
   [![CodeFactor](https://www.codefactor.io/repository/github/intelligent-machine-learning/dlrover/badge)](https://www.codefactor.io/repository/github/intelligent-machine-learning/dlrover)
   [![HuggingFace badge](https://img.shields.io/badge/%F0%9F%A4%97HuggingFace-Join-yellow)](https://huggingface.co/)
   [![slack badge](https://img.shields.io/badge/Slack-join-blueviolet?logo=slack&amp)](https://join.slack.com/)
   [![WeChat badge](https://img.shields.io/badge/微信-加入-green?logo=wechat&amp)](https://huggingface.com/)


   | [English](README.md) | [中文](docs/README-zh-Hans.md) |

</div>

## Latest News
* TODO

## Table of Contents
<ul>
 <li><a href="#Why-ATorch">Why ATorch</a> </li>
 <li><a href="#Features">Features</a> </li>
 <li>
   <a href="#ATorch-Applications">ATorch Applications</a>
   <ul>
     <li><a href="#ATorchPretrain">ATorch: Super backend engine supportted Ant LLMs training with over 60% HFU</a></li>
   </ul>
 </li>
 <li>
   <a href="#Parallel-Training-Demo">Parallel Training Demo</a>
   <ul>
     <li><a href="#LLaMA2">LLaMA2</a></li>
     <li><a href="#BERT">BERT</a></li>
   </ul>
 </li>
 <li>
   <a href="#Single-GPU-Training-Demo">Single GPU Training Demo</a>
   <ul>
     <li><a href="#LLaMA2-Single">LLaMA2-Single</a></li>
   </ul>
 </li>
 <li>
   <a href="#Installation">Installation</a>
   <ul>
     <li><a href="#PyPI">PyPI</a></li>
     <li><a href="#Install-From-Source">Install From Source</a></li>
   </ul>
 </li>
 <li><a href="#Community">Community</a></li>
 <li><a href="#Contributing">Contributing</a></li>
 <li><a href="#Cite-Us">Cite Us</a></li>
</ul>

## Why ATorch
ATorch is an extension library of PyTorch developed by Ant Group's AI Infrastructure team. By decoupling model definition from training optimization strategy, ATorch supports efficient and easy-to-use model training experience. The design principle is to minimally disrupt the native PyTorch programming style. Through its API, ATorch provides performance optimizations in aspects such as I/O, preprocessing, computation, and communication (including automatic optimization). ATorch has supported large-scale pretraining of LLMs with over 100 billion parameters and thousands of A100/H100 GPUs. We aim to open source it and make these capabilities reproducible for everyone. We also welcome contributions.

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

## ATorch Applications

### ATorch Pretrain LLMs with over thousands GPUs (HFU > 50%)

* Improve the stalibity of training over thousands GPUs by [fault-tolerance and elasticity](../docs/blogs/stabilize_llm_training_cn.md).

### Finetune your LLMs with ATorch RLHF (3x trlx)
TODO

## Major Model results
TODO

### LLaMA2
TODO

### GPT2
TODO

### GLM
TODO

### CLIP
TODO

## Installation
TODO

## Contributing
TODO
## CI/CD

We leverage the power of [GitHub Actions](https://github.com/features/actions) to automate our development, release and deployment workflows. Please check out this [documentation](.github/workflows/README.md) on how the automated workflows are operated.

## Cite Us
