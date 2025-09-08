# 01. Get Started: Run NanoGPT with DLRover

This guide shows how to submit and run a unified DLRover job using the
NanoGPT example.

## Prerequisites

- Python 3.10+
- Ray Cluster:
  Need cluster for multi-node training, reference [Ray Cluster](https://docs.ray.io/en/latest/cluster/getting-started.html) for guidance.
  If there is no cluster, DLRover will automatically start a local cluster for single-node training.
- If using GPUs: torch CUDA and appropriate drivers installed

## Quick Start

1. Install DLRover with Ray and Torch extras:

```fish
python -m pip install --upgrade 'dlrover[ray,torch]'
```

2. Prepare NanoGPT data:

```fish
python examples/pytorch/nanogpt/prepare.py \
  --src_data_path=examples/pytorch/nanogpt/data.txt \
  --output_dir=data/nanogpt/
```

This will create `train.bin`, `val.bin`, and `meta.pkl` under
`data/nanogpt/`.

3. Run the training example locally:

```fish
python -m examples.unified.elastic.nanogpt.train \
  --data_dir data/nanogpt/
```

## How it works

### 1) Submit Job

The submission script parses arguments, builds a job using the Builder API,
and submits it. DLRover starts the PrimeMaster which manages the core lifecycle of job
and creates Worker processes.

A typical Builder API usage:

```python
def launch_use_api():
    args = arg_parser()

    job = (
        DLJobBuilder()
        .config(DictConfig(args.__dict__))
        .node_num(2)
        .device_per_node(2)
        .device_type("GPU" if torch.cuda.is_available() else "CPU")
        .train("examples.unified.elastic.nanogpt.train.run")
        .nnodes(2)
        .nproc_per_node(2)
        .end() #optional, for type hinting
        .build()
    )

    job.submit(job_name="nanogpt")
```

### 2) Worker Start

After the Master finishes node checks it launches Worker processes. Each
Worker executes the configured entrypoint.

```python
def run():
    args: Any = current_worker().job_info.user_config

    if getattr(args, "use_ray_dataloader", False):
        from dlrover.python.unified.api.runtime.ray_dataloader_iter import (
            patch_dataloader_ray,
        )

        patch_dataloader_ray()

    train_params = setup_train_params(args)
    train(args, train_params)
```

## Troubleshooting & Tips

- For GPU runs verify CUDA, drivers and NCCL configuration.
- For local debugging reduce `nnodes` and `nproc_per_node` before scaling.

## Next steps

- Builder API and advanced configuration: `unified_api_guide.md`
- Multi-role training (OpenRLHF example): `multi_role_training.md`
- Runtime SDK: `runtime_sdk.md`
- Design proposal and architecture: `../design/unified-mpmd-control-proposal`
