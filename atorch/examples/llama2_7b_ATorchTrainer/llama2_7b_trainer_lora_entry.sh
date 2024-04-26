#!/bin/bash
set -exo pipefail

HOME=$(echo ~)
if [ ! -d dataset_and_weight/ ]; then
    mkdir dataset_and_weight/
    bash prepare_dataset_and_weight.sh
fi

WORLD_SIZE=${WORLD_SIZE:-$WORKER_NUM}
NODE_RANK=${RANK:-0}
TIME_STAMP=$(date '+%Y%m%d-%H%M%S')

OUTPUT_DIR=${1}
if [ ! -d ${OUTPUT_DIR} ]; then
    mkdir -p ${OUTPUT_DIR}
fi
NODE_NAME=${POD_NAME:-"master-0"}

DATASET_DIR=/dataset_and_weight/AlpacaDataCleaned/alpaca_data_cleaned.json
PRETRAINED_MODEL_DIR=$HOME/.cache/Llama-2-7b-hf/
EVALUATE_SCRIPT=/dataset_and_weight/evaluate/metrics/perplexity/perplexity.py

NUM_GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
NUM_GPUS=$((NUM_GPUS_PER_NODE * WORLD_SIZE))
PER_DEVICE_TRAIN_BATCH_SIZE=16
TOTAL_TRAIN_BATCH_SIZE=$((NUM_GPUS_PER_NODE * WORLD_SIZE * PER_DEVICE_TRAIN_BATCH_SIZE))

if [ ${NODE_RANK} -eq "0" ]; then
    cp ${0} llama2_clm_atorch_trainer.py ${OUTPUT_DIR}
    if [ -e requirements.txt ]; then
        pip install -U -r requirements.txt
        cp requirements.txt ${OUTPUT_DIR}
    fi
    pip list >> ${OUTPUT_DIR}/env.txt
    nvidia-smi >> ${OUTPUT_DIR}/log_llama2_lora_atorch_trainer_${WORLD_SIZE}n${NUM_GPUS}g.txt
    printenv >> ${OUTPUT_DIR}/log_llama2_lora_atorch_trainer_${WORLD_SIZE}n${NUM_GPUS}g.txt
fi

python -m atorch.distributed.run --fault_tolerant --max_restarts=1 \
    --nnodes="$WORLD_SIZE" \
    --nproc_per_node="$NUM_GPUS_PER_NODE" \
    llama2_clm_atorch_trainer.py \
    --peft_type "lora" \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_target_modules "q_proj" "k_proj" "v_proj" "o_proj" \
    --gradient_checkpointing true \
    --dataset_path $DATASET_DIR \
    --model_name_or_path $PRETRAINED_MODEL_DIR \
    --evaluate_script $EVALUATE_SCRIPT \
    --num_train_epochs 6 \
    --block_size 512 \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --learning_rate 3e-4 \
    --weight_decay 0.0 \
    --warmup_steps 100 \
    --seed 42 \
    --do_train \
    --do_eval \
    --max_grad_norm 0 \
    --bf16 \
    --save_strategy "no" \
    --evaluation_strategy "epoch" \
    --seed 42 \
    --preprocessing_num_workers 6 \
    --dataloader_num_workers 0 \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_output_dir \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --logging_nan_inf_filter false \
    --remove_unused_columns false \
    --enable_torch_profiler false \
    --save_load_by_streaming false \
    --save_base_model false \
    --logging_dir ${OUTPUT_DIR} \
    --report_to none \
    2>&1 | tee -a ${OUTPUT_DIR}/log_llama2_lora_atorch_trainer_${WORLD_SIZE}n${NUM_GPUS}g.txt