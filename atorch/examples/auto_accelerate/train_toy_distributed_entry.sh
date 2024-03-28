#!/bin/bash
NUM_GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
WORLD_SIZE=${WORLD_SIZE:-1}
NUM_GPUS=$((NUM_GPUS_PER_NODE * WORLD_SIZE))

python -m atorch.distributed.run --nnodes="$WORLD_SIZE" \
    --nproc_per_node="$NUM_GPUS_PER_NODE" \
    train.py --model_type toy \
    --distributed \
    2>&1 | tee log_toy_distributed_"${WORLD_SIZE}"n"${NUM_GPUS}"g.txt 
