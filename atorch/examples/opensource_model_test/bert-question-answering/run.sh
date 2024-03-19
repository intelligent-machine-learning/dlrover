#!/bin/bash
set -exo pipefail
DATASET_DIR=/path/to/squad_v2
PRETRAINED_MODEL_DIR=/path/to/bert-large-cased
METRIC_SCRIPT_PATH=./metrics/squad_v2/squad_v2.py

NUM_GPUS=8
PER_DEVICE_TRAIN_BATCH_SIZE=16
PER_DEVICE_EVAL_BATCH_SIZE=$PER_DEVICE_TRAIN_BATCH_SIZE
WORLD_SIZE=${WORLD_SIZE:-1}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"23456"}

python -m torch.distributed.run \
    --nnodes=$WORLD_SIZE \
    --nproc_per_node $NUM_GPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    run_qa_no_atorch.py \
    --model_name_or_path $PRETRAINED_MODEL_DIR \
    --dataset_path $DATASET_DIR \
    --max_seq_length 384 \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
    --version_2_with_negative \
    --metrics_path $METRIC_SCRIPT_PATH \
    --doc_stride 128 \
    --seed 42 \
    --preprocessing_num_workers 1 \
    --num_train_epochs 3 \
    --pad_to_max_length \
    --fp16 \
    --trust_remote_code 2>&1 | tee ./bert_large_qa.log