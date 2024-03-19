#!/bin/sh
#****************************************************************#
# ScriptName: run.sh
# Author: $SHTERM_REAL_USER@alibaba-inc.com
# Create Date: 2024-03-14 13:29
# Modify Author: $SHTERM_REAL_USER@alibaba-inc.com
# Modify Date: 2024-03-14 13:29
# Function: 
#***************************************************************#
set -ex

NNODES=${WORLD_SIZE:-"1"}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"20001"} 
RANK=${RANK:-"0"}
NUM_DEVICES=$(nvidia-smi -L | wc -l)

CURRENT_PATH=$(realpath .)

export PYTHONPATH=$PYTHONPATH:$CURRENT_PATH/src

torchrun --nnodes=${NNODES}  --nproc_per_node=${NUM_DEVICES} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} --node_rank=${RANK} examples/finetuning.py \
    --use_fast_kernels \
    --enable_fsdp \
    --low_cpu_fsdp \
    --context_length 4096 \
    --fsdp_config.pure_bf16 \
    --fsdp_config.fsdp_activation_checkpointing False \
    --batch_size_training 1 \
    --batching_strategy packing \
    --gradient_accumulation_steps 1 \
    --num_epochs 3 \
    --model_name /datacube_nas/workspace/sichuan/pretrained_models/Llama-2-7b-hf/ \
    --dataset alpaca_dataset \
    --output_dir /tmp