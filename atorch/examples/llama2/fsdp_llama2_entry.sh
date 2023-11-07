set -x

source dataset_model.sh

WORLD_SIZE=${WORLD_SIZE:-1}

PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-16}
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

python -m atorch.distributed.run --fault_tolerant --max_restarts=0 \
    --nnodes="$WORLD_SIZE" \
    --nproc_per_node=8 \
    fsdp_llama2.py \
    --block_size $BLOCK_SIZE \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --dataset_path $DATASET_PATH \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --precision bf16_amp \
    --gradient_checkpointing \
    $LORA_OPT