# Four-GPU MoE In-Place Restart Demo

This example starts with a small replicated-expert MoE model trained with
DDP. It then demonstrates the first healthy-rank recovery prototype:

- Four CUDA processes use a reconfigurable PyTorch `nccl2` process group.
- Rank 2 exits at a configured training step.
- The local supervisor leaves ranks 0, 1, and 3 running.
- Only rank 2 is started again.
- All four ranks reconfigure the same default process group.
- Every rank restores the same committed checkpoint.
- The DDP wrapper is rebuilt around the existing raw model and optimizer.
- Training resumes and reaches the configured final step.

The experts are replicated on every rank. The only distributed communication
is DDP over the default process group. This deliberately avoids TP, PP, EP,
and expert all-to-all recovery in the first prototype.

## Requirements

- Linux with four NVIDIA GPUs on one host
- PyTorch 2.14 or newer
- The experimental `nccl2` distributed backend
- NCCL 2.28 or newer; NCCL 2.32 is recommended

PyTorch 2.14 marks `nccl2` and process-group reconfiguration as experimental.
Pin and test the exact PyTorch build used by the cluster.

The local supervisor disables the legacy asynchronous error processes so that
`nccl2` can revoke the failed communicator and return control to the training
process.

```bash
python -m pip install -r examples/in_place_restart/requirements.txt
```

## Build a Docker image

The recommended base image is `nvcr.io/nvidia/pytorch:26.09-py3`. NVIDIA's
26.09 release contains PyTorch `2.14.0a0`, CUDA 13.4.1, and NCCL 2.31. NCCL
2.31 is sufficient for PyTorch's `nccl2` reconfigure implementation, which
requires NCCL 2.28 or newer.

The host must have a driver compatible with CUDA 13.4 and the NVIDIA Container
Toolkit installed. If the NGC registry requires authentication, use an NGC API
key:

```bash
docker login nvcr.io
# Username: $oauthtoken
# Password: <NGC_API_KEY>
```

Build the image from the repository root:

```bash
docker build \
  -f examples/in_place_restart/Dockerfile \
  -t dlrover-inplace-moe:26.09 \
  .
```

The build installs DLRover into the NGC PyTorch environment and verifies at
image-build time that:

- PyTorch is the expected 2.14 build;
- the `nccl2` backend is available;
- `dist._reconfigure()` and reconfigure handles are exposed;
- the linked NCCL version is at least 2.28.

The image does not run:

```text
pip install -r examples/in_place_restart/requirements.txt
```

That command could replace NVIDIA's optimized PyTorch build with a generic
PyPI wheel. The NGC base image already contains the required framework.

To require NCCL 2.32 explicitly, build against NGC 26.10:

```bash
docker build \
  -f examples/in_place_restart/Dockerfile \
  --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:26.10-py3 \
  -t dlrover-inplace-moe:nccl232 \
  .
```

NGC 26.10 contains NCCL 2.32.3 but PyTorch `2.15.0a0`, so rerun the complete
recovery test before treating that image as supported.

Run the demonstration with all four GPUs:

```bash
mkdir -p /tmp/dlrover-moe-artifacts
docker run --rm \
  --gpus '"device=0,1,2,3"' \
  --shm-size=16g \
  -v /tmp/dlrover-moe-artifacts:/artifacts \
  dlrover-inplace-moe:26.09 \
  python examples/in_place_restart/run_in_place_demo.py \
  --work-dir /artifacts
```

Before running the recovery demo, execute the image preflight:

```bash
docker run --rm \
  --gpus '"device=0,1,2,3"' \
  --shm-size=16g \
  dlrover-inplace-moe:26.09 \
  python examples/in_place_restart/verify_image.py
```

The preflight checks the PyTorch and NCCL versions, `nccl2` backend,
reconfigure APIs, visible GPUs, `/dev/shm`, DLRover installation, and an actual
four-rank initial reconfiguration followed by an all-reduce.

Use `--ipc=host` instead of `--shm-size=16g` when host IPC is permitted and
the cluster policy expects that configuration. The example starts all four
workers inside this one container; it does not launch a separate container per
rank.

For Kubernetes, run the local demonstration in one Pod that requests all four
GPUs:

```yaml
resources:
  limits:
    nvidia.com/gpu: 4
```

For DLRover jobs, override the container command with the appropriate
`dlrover-run` invocation. The image includes both the DLRover package and this
example directory.

Use a digest rather than only the moving `26.09-py3` tag when promoting an
image to a shared cluster, for example:

```text
nvcr.io/nvidia/pytorch@sha256:<digest>
```

## Run the in-place restart demo

From the DLRover repository root:

```bash
python examples/in_place_restart/run_in_place_demo.py
```

The default sequence is:

1. Train ranks 0-3.
2. Commit checkpoints at steps 0, 10, 20, and later checkpoints.
3. Exit rank 2 at step 23.
4. Start a replacement for rank 2 with the same logical rank.
5. Reconfigure the `nccl2` default group.
6. Restore the step-20 checkpoint on all ranks.
7. Continue and finish 40 steps.

A temporary work directory is printed at the end. It contains:

```text
checkpoints/
  checkpoint-00000020.pt
  latest.json
recovery/
  command.json
rendezvous/
  store
status/
  rank_0.json
  rank_1.json
  rank_2.json
  rank_3.json
```

The final status verifies that:

- ranks 0, 1, and 3 retained their original PIDs;
- rank 2 has its replacement PID;
- all ranks completed at least one recovery;
- all ranks reached the final training step.

The supervisor prints the process replacement and continuity explicitly:

```text
[supervisor] initial processes: rank 0=pid 101, rank 1=pid 102,
rank 2=pid 103, rank 3=pid 104
[supervisor] FAILED rank 2: old_pid=103, exit_code=17
[supervisor] healthy processes still running:
rank 0=pid 101, rank 1=pid 102, rank 3=pid 104
[supervisor] REPLACED rank 2: old_pid=103, new_pid=205
[supervisor] RECOVERED rank 0: pid=101, epoch=1, step=20
[supervisor] RECOVERED rank 1: pid=102, epoch=1, step=20
[supervisor] RECOVERED rank 2: pid=205, epoch=1, step=20
[supervisor] RECOVERED rank 3: pid=104, epoch=1, step=20
```

The final summary marks ranks 0, 1, and 3 as `CONTINUED` and rank 2 as
`REPLACED`.

Use a persistent work directory to inspect the artifacts:

```bash
python examples/in_place_restart/run_in_place_demo.py \
  --work-dir /tmp/dlrover-moe-recovery
```

The work directory must be empty for a new demonstration.

## Checkpoint-only baseline

The training script can be started by `dlrover-run` without in-place
reconfiguration. This is useful for checking deterministic data generation
and checkpoint restore before testing process-group recovery.

```bash
mkdir -p /tmp/dlrover-moe-baseline
dlrover-run --standalone --nproc_per_node=4 \
  examples/in_place_restart/train.py \
  --backend nccl \
  --work-dir /tmp/dlrover-moe-baseline \
  --steps 40 \
  --checkpoint-interval 10
```

Do not use the local supervisor with `--backend nccl`. In-place recovery
requires the `nccl2` backend and `enable_reconfigure=True`.

## How recovery works

The worker performs these phases:

```text
RUNNING
  -> stop old-epoch training
  -> detach the DDP wrapper while retaining raw model parameters
  -> revoke in-flight nccl2 work
  -> exchange opaque reconfigure handles
  -> dist._reconfigure() on the same ProcessGroup object
  -> restore checkpoint on the same raw model and optimizer
  -> rebuild DDP around the same parameters
  -> wait for every rank
  -> RESUME
```

The ordered reconfigure handle list assigns ranks by position. The replacement
rank's handle is therefore placed at logical rank 2, preserving the original
rank mapping.

Healthy ranks retain:

- process PID and Python runtime;
- CUDA context;
- raw model parameter tensors;
- optimizer object;
- dataset and training configuration.

The demonstration rebuilds the DDP wrapper after recovery. This clears any
partially completed DDP reducer state and is safer for the first prototype.
It does not rebuild the model or process.

## DLRover integration

The local supervisor in this directory models the control-plane work that
DLRover must perform:

- detect the failed logical rank;
- preserve healthy worker processes;
- start only the failed rank's replacement process;
- select a globally committed checkpoint;
- allocate a monotonic recovery epoch and fresh 64-bit reconfigure UUID;
- publish a `RECOVER` command out of band;
- wait for all ranks to become ready before releasing training.

The file-based command and status protocol keeps the example self-contained.
Production integration should replace it with a DLRover master-to-worker
control channel, such as a per-process Unix socket or an existing agent RPC.

The standalone demo uses a `FileStore`. A production multi-node deployment
must use a control-plane-owned rendezvous that does not disappear when a
worker, including rank 0, fails.

## Known limitations

- `nccl2`, `_supports_reconfigure`, `_get_reconfigure_handle`, and
  `_reconfigure` are experimental PyTorch APIs.
- The example supports one default NCCL process group only.
- The local supervisor injects one failure and does not handle a second rank
  failure during recovery.
- Experts are replicated; this is not expert-parallel training.
- CUDA graphs and third-party NCCL communicators are not covered.
- The local supervisor and file-based protocol are for the example only.
- Heartbeat detection lives in DLRover and is outside this demo's scope.
