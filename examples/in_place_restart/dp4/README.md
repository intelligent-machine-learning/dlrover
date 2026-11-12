# Four-GPU DDP MoE Recovery

This is the original replicated-expert MoE demo:

```text
World size = 4
DP = 4
TP = 1
PP = 1
EP = 1
```

It uses one reconfigurable default `nccl2` process group. The default demo
fails rank 2 at step 23, restarts only rank 2, restores checkpoint step 20,
and continues to step 40.

From the repository root:

```bash
ARTIFACT_DIR=$(mktemp -d /tmp/dlrover-dp4.XXXXXX)

python examples/in_place_restart/dp4/run_in_place_demo.py \
  --work-dir "$ARTIFACT_DIR"
```

Files:

- `train.py`: checkpoint-only DLRover-compatible entry point
- `moe_train.py`: training and recovery implementation
- `run_in_place_demo.py`: four-rank local supervisor
- `run.sh`: convenience launcher
