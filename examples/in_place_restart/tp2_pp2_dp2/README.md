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
restoring checkpoint step 20. The same trainer also supports a restart-all
baseline that replaces every rank.

From the repository root:

```bash
ARTIFACT_DIR=$(mktemp -d /tmp/dlrover-hybrid.XXXXXX)

python examples/in_place_restart/tp2_pp2_dp2/run_in_place_demo.py \
  --work-dir "$ARTIFACT_DIR"
```

Run the restart-all baseline with the same failure point:

```bash
RESTART_DIR=$(mktemp -d /tmp/dlrover-hybrid-restart.XXXXXX)

python examples/in_place_restart/tp2_pp2_dp2/run_in_place_demo.py \
  --work-dir "$RESTART_DIR" \
  --recovery-mode restart-all
```

Use `--recovery-mode in-place` (the default) to run only the in-place case.
For a single run that executes both modes and compares them, use:

```bash
COMPARE_DIR=$(mktemp -d /tmp/dlrover-hybrid-compare.XXXXXX)

python examples/in_place_restart/tp2_pp2_dp2/run_recovery_comparison.py \
  --work-dir "$COMPARE_DIR" \
  --steps 40 \
  --checkpoint-interval 10 \
  --fail-rank 3 \
  --fail-at-step 23
```

The comparison run writes:

```text
in_place/metrics/loss_rank_2.jsonl
in_place/metrics/summary.json
restart_all/metrics/loss_rank_2.jsonl
restart_all/metrics/summary.json
comparison.json
```

It reports the maximum absolute and relative loss difference after the
restored checkpoint, failure-to-resume time for both modes, the number of
process restarts, and the time saved by in-place recovery. Losses are
compared after the checkpoint step because steps 21 and 22 are replayed after
recovery.

For timing measurements, run the comparison several times with fresh
directories and alternate `--run-order in-place-first` with
`--run-order restart-all-first`. The small demo is sensitive to process-start
and filesystem-cache effects, so one timing sample is not enough for a
production performance claim.

For the same topology on two nodes with four GPUs per node, use
`../tp2_pp2_dp2_2nodes/README.md`. That variant keeps this training
implementation and replaces the single-node supervisor with two node agents.

Files:

- `train.py`: checkpoint-only DLRover-compatible entry point
- `hybrid_moe_train.py`: TP/PP/DP training and recovery implementation
- `run_in_place_demo.py`: eight-rank local supervisor
- `run_recovery_comparison.py`: in-place versus restart-all comparison
- `run.sh`: convenience launcher
