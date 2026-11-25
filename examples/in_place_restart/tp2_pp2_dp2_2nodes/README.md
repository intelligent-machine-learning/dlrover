# Two-Node TP2 x PP2 x DP2 MoE Recovery

This is the two-node variant of the hybrid in-place restart demo:

```text
World size = 8
TP = 2
PP = 2
DP = 2
EP = 1
GPUs per node = 4
```

The global rank layout is:

```text
rank = DP * 4 + PP_stage * 2 + TP_rank
```

The nodes own these logical ranks:

```text
node 0: ranks 0, 1, 2, 3
node 1: ranks 4, 5, 6, 7
```

The process groups are the same as the single-node hybrid demo:

```text
TP: [0,1] [2,3] [4,5] [6,7]
PP: [0,2] [1,3] [4,6] [5,7]
DP: [0,4] [1,5] [2,6] [3,7]
```

The default injected failure is rank 5, which is
`TP1-PP0-DP1` on node 1. Its TP and PP peers are local to node 1, while its
DP peer rank 1 is on node 0. The node 1 agent starts rank 5's replacement;
all other ranks keep their original processes and PIDs.

## Shared directory

Both node agents and all workers must use the same absolute path for
`--work-dir`. The path must be on a filesystem shared by both nodes and must
support the POSIX file operations used by `torch.distributed.FileStore`.
Both containers must run with write permission for the same directory. The
node clocks should be synchronized because the recovery command contains a
wall-clock deadline.

For example:

```bash
export RUN_DIR=/nas/shared/dlrover/two-node-run-001
mkdir -p "$RUN_DIR"
```

Do not use each node's local `/tmp` path. The shared directory contains the
rendezvous store, checkpoint shards, recovery command, and node status files:

```text
agents/
  node_0.json
  node_1.json
  failures/
checkpoints/
  checkpoint-00000020/
  latest.json
recovery/
  command.json
rendezvous/
  store
status/
  rank_0.json ... rank_7.json
```

The demonstration requires a fresh directory. Node 0 initializes it; node 1
waits for node 0's `run.json` marker.

## Launch

Run these commands from the same source checkout on the two nodes. Start
node 0 first, then start node 1 while node 0 is still waiting for the initial
all-rank rendezvous.

Node 0:

```bash
bash examples/in_place_restart/tp2_pp2_dp2_2nodes/run_node0.sh \
  --work-dir "$RUN_DIR" \
  --fail-rank 5 \
  --fail-at-step 23
```

Node 1:

```bash
bash examples/in_place_restart/tp2_pp2_dp2_2nodes/run_node1.sh \
  --work-dir "$RUN_DIR" \
  --fail-rank 5 \
  --fail-at-step 23
```

The shell scripts select the two logical node ranks. The Python agent
launches four local workers with:

```text
WORLD_SIZE=8
LOCAL_WORLD_SIZE=4
LOCAL_RANK=0,1,2,3
NODE_RANK=0 or 1
```

The training implementation is shared with
`../tp2_pp2_dp2/hybrid_moe_train.py`; this directory only supplies the
two-node launch and control layer. The `train.py` file is a compatibility
entry point for launching the same worker under a DLRover-compatible runner.

For a real cross-node NCCL connection, the cluster must allow the required
NCCL/NVLink/TCP ports and must select the correct network interface. Pass
the interface explicitly when auto-detection is not appropriate:

```bash
bash run_node0.sh \
  --work-dir "$RUN_DIR" \
  --nccl-socket-ifname eth0
```

For InfiniBand, also pass `--nccl-ib-hca`, for example `mlx5_0:1`.
Use the network settings required by the cluster; the two-node demo does not
guess them.

## What happens during recovery

1. Both node agents launch their four workers and publish initial PIDs in
   `agents/node_0.json` and `agents/node_1.json`.
2. The training script creates the default, TP, PP, and DP `nccl2` groups and
   writes a checkpoint manifest.
3. Rank 5 exits at step 23. Node 1 records the failure in
   `agents/failures/`.
4. Node 0 detects the failure, waits for a committed checkpoint, and writes
   `recovery/command.json`. The command contains one ordered handle slot per
   logical rank; the replacement's handle is placed at the original rank-5
   slot.
5. Node 1 starts only rank 5's replacement process. Node 0 starts no rank-5
   process because rank 5 belongs to node 1.
6. Every healthy worker keeps its process, CUDA context, model parameters,
   optimizer, and dataset state. Its recovery watcher observes the command
   and aborts local `nccl2` operations that are blocked on the failed rank.
7. The replacement enters the same recovery path: it creates group shells,
   exchanges reconfigure handles, and reconfigures the default, TP, PP, and
   DP groups.
8. All ranks validate their local groups, restore their own checkpoint shard
   at step 20, rebuild only the DDP wrapper, and resume training to step 40.
9. Node 0 verifies all eight final statuses: ranks other than 5 retain their
   initial PIDs, rank 5 has a replacement PID, and every rank reports the
   same recovery count and final step.

The healthy ranks do not recreate their model, optimizer, CUDA context, or
Python process. Reconfiguration changes the communicator state and rebuilds
the DDP reducer; it is not a full training-process restart.

Expected supervisor output includes:

```text
[node-agent 1] FAILED rank 5/TP1-PP0-DP1: pid=..., exit_code=17
[node-agent 0] requested recovery epoch 1 for rank 5/TP1-PP0-DP1
[node-agent 0] healthy ranks retain their initial processes: ...
[node-agent 1] REPLACED rank 5/TP1-PP0-DP1: old_pid=..., new_pid=...
[node-agent 0] RECOVERED rank 0/TP0-PP0-DP0: pid=..., ...
...
Two-node in-place restart summary
```

## Docker

After rebuilding the image from this checkout, the existing Dockerfile is
sufficient because it copies all of `examples/in_place_restart`. On each node,
run a container with four GPUs and mount the shared directory at the same
path in both containers. For example:

First run the four-rank local image check on each node:

```bash
python examples/in_place_restart/verify_image.py --world-size 4
```

This checks the PyTorch/NCCL versions, `nccl2` reconfigure APIs, four visible
GPUs, and a local four-rank reconfigure. It does not prove that NCCL traffic
or the shared filesystem works between the two hosts; the two-node run below
does that.

```bash
docker run --rm -it --network=host \
  --gpus '"device=0,1,2,3"' \
  --shm-size=16g \
  -v "$RUN_DIR:$RUN_DIR" \
  dlrover-inplace-moe:26.09 \
  bash examples/in_place_restart/tp2_pp2_dp2_2nodes/run_node0.sh \
  --work-dir "$RUN_DIR" \
  --nccl-socket-ifname eth0
```

Use `run_node1.sh` in the container on node 1. Both containers must see the
same `RUN_DIR`; mount the shared filesystem, not a container-local path.
