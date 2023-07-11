# Demo
This demo is modified from [nanoGPT](https://github.com/karpathy/nanoGPT) and aimed to test optimizers running on NPU.
Please refer to https://github.com/karpathy/nanoGPT#reproducing-gpt-2 for dataset preparation. To start training, you can use the following command:
```
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```


