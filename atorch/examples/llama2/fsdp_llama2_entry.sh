#!/bin/bash

set -x

source ./dataset_model.sh

WORLD_SIZE=${WORLD_SIZE:-1}
NUM_GPUS_PER_NODE=$(nvidia-smi -L | wc -l)

PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}
BLOCK_SIZE=${BLOCK_SIZE:-4096}

if [ -z "$USE_LORA" ]; then
	LORA_OPT=""
else
	LORA_OPT="
        --peft_type lora \
        --lora_r 16 \
        --lora_alpha 16 \
        --lora_target_modules q_proj v_proj k_proj o_proj \
        --lora_dropout 0.05 \
    "
fi

if [ -z "$USE_FP8" ]; then
	FP8_OPT=""
else
	FP8_OPT="--fp8"
fi

python -m atorch.distributed.run \
    --nnodes="$WORLD_SIZE" \
    --nproc_per_node="$NUM_GPUS_PER_NODE" \
    fsdp_llama2.py \
    --block_size $BLOCK_SIZE \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --dataset_path $DATASET_PATH \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --precision bf16_amp \
    --gradient_checkpointing \
    $LORA_OPT $FP8_OPT