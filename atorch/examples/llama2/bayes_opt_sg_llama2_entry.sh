#!/bin/bash

pip install -r requirements.txt

PRETRAINED_MODEL_DIR=""
DATASET_DIR=""
NUM_GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
WORLD_SIZE=${WORLD_SIZE:-1}
NUM_GPUS=$((NUM_GPUS_PER_NODE * WORLD_SIZE))
PER_DEVICE_TRAIN_BATCH_SIZE=4
TOTAL_TRAIN_BATCH_SIZE=$((NUM_GPUS_PER_NODE * WORLD_SIZE * PER_DEVICE_TRAIN_BATCH_SIZE))
export BO_SG_MAX_IETR=12
export RANDOM_SAMPLE=4


python -m atorch.distributed.run --nnodes="$WORLD_SIZE" \
    --nproc_per_node="$NUM_GPUS_PER_NODE" \
    bayes_opt_sg_llama2.py \
    --dataset_path $DATASET_DIR \
    --config_name $PRETRAINED_MODEL_DIR \
    --tokenizer_name $PRETRAINED_MODEL_DIR \
    --total_train_batch_size  $TOTAL_TRAIN_BATCH_SIZE \
    --block_size 2048 \
    --seed 42 \
    --preprocessing_num_workers 12 \
    --ignore_mismatched_sizes \
    2>&1 | tee log_llama2_"${WORLD_SIZE}"n"${NUM_GPUS}"g.txt 
