# ATorch Introduction

ATorch is an extension library of PyTorch developed by Ant Financial. It provides efficient and easy-to-use model training and optimization capabilities. The design principle is to minimally invade the native PyTorch programming style, and through API interfaces, provide performance optimizations in aspects such as IO/preprocessing, computation, and communication (including automatic optimization), as well as support for large model training.

* Usability
  * Fast deployment of runtime environment (images and installation packages)
  * Script submission of distributed training jobs
* Automated optimization
  * auto_accelerate for automatic optimization
* IO/preprocessing
  * Recommended storage for training data
  * Accessing the Pangu cluster
  * CPU/GPU cooperation to optimize data preprocessing
* Customized operator optimization 
  * High-performance MOE
  * Flash Attention and Transformer operator
* Mixed precision
* Distributed VRAM optimization
* Computation/communication pipelining optimization
* Hybrid parallelism
* Compilation optimization
* Solutions for large-scale model training
* Elastic fault tolerance
  * HangDetector (detecting and automatically restarting distributed training if it hangs)
  * GPU elastic training