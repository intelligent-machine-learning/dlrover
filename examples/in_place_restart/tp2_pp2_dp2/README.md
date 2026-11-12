# Eight-GPU TP2 x PP2 x DP2 MoE Recovery

This demo uses:

```text
World size = 8
TP = 2
PP = 2
DP = 2
EP = 1
```

Process groups:

```text
TP: [0,1] [2,3] [4,5] [6,7]
PP: [0,2] [1,3] [4,6] [5,7]
DP: [0,4] [1,5] [2,6] [3,7]
```

Rank 3 is `TP1-PP1-DP0`. It fails at step 23, is replaced with a new process,
and all eight ranks reconfigure the default, TP, PP, and DP groups before
restoring checkpoint step 20.

From the repository root:

```bash
ARTIFACT_DIR=$(mktemp -d /tmp/dlrover-hybrid.XXXXXX)

python examples/in_place_restart/tp2_pp2_dp2/run_in_place_demo.py \
  --work-dir "$ARTIFACT_DIR"
```

Files:

- `train.py`: checkpoint-only DLRover-compatible entry point
- `hybrid_moe_train.py`: TP/PP/DP training and recovery implementation
- `run_in_place_demo.py`: eight-rank local supervisor
- `run.sh`: convenience launcher
