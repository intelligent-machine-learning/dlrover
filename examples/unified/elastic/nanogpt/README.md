# Pretrain Nano-GPT with DLRover Unified API (Ray-based)

This document describes how to use the DLRover Unified API to pretrain a Nano-GPT model.

## Prepare Data

Once your data is ready, run the preparation script as follows:

```bash
python examples/pytorch/nanogpt/prepare.py \
    --src_data_path=examples/pytorch/nanogpt/data.txt \
    --output_dir=data/nanogpt/
```

This command will generate `train.bin`, `val.bin`, and `meta.pkl` files in the `data/nanogpt/` directory.

## Base Training

Use the `0_base_train.py` script to submit a training job to the Ray cluster:

```bash
python -m examples.unified.elastic.nanogpt.0_base_train \
    --data_dir data/nanogpt/
```

The `run` function is the main entry point for training (inside multi-processing worker).  
Use `launch` or `launch_use_api` to submit the job to Ray, start DLRover, and begin training.  
If no Ray cluster is detected, a local Ray cluster will be started automatically.
