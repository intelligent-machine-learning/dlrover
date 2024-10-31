#!/bin/bash
set -exo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1

TIME_STAMP=$(date '+%Y%m%d-%H%M%S')

OUTPUT_DIR=${OUTPUT_DIR:-"/tmp/llama2_atorch_trainer/"}
if [ ! -d ${OUTPUT_DIR} ]; then
    mkdir -p ${OUTPUT_DIR}
fi
NODE_NAME=${POD_NAME:-"master-0"}

export PYTHONPATH="$(dirname $0)/../../":"/hetero_infer/jinshi.cl/code/Megatron-LM-core_r0.6.0":$PYTHONPATH

DATA_PATH="/hetero_infer/jinshi.cl/code/Megatron-LM-core_r0.6.0/wikitext-2-raw-v1-llama2/llama2_tokenized_train_text_document"
TOKENIZER_PATH="/hetero_infer/jinshi.cl/code/tokenizers/llama_tokenizer/tokenizer.model"

GPUS_PER_NODE=${GPUS_PER_NODE:-8}

if [[ $POD_NAME =~ "edljob" ]]; then
    WORLD_SIZE=${WORKER_NUM:-1}
    NODE_RANK=${RANK:-0}
    MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
    RANDOM_PORT=$[$RANDOM + 20000]
    MASTER_PORT=${MASTER_PORT:-$RANDOM_PORT}
    GPU_NUM=$((${GPUS_PER_NODE}*${WORLD_SIZE}))
    echo "---> from edl runtime, WORLD_SIZE: ${WORLD_SIZE}, NODE_RANK: ${NODE_RANK}"
    LAUNCHER=" \
        python -m atorch.distributed.run --fault_tolerant --network-check \
        --max_restarts=1 \
        --nnode=$WORLD_SIZE \
        --nproc_per_node=$GPUS_PER_NODE \
        --rdzv_conf join_timeout=300 \
        "
else
    WORLD_SIZE=${WORLD_SIZE:-1}
    NODE_RANK=${RANK:-0}
    MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
    RANDOM_PORT=$[$RANDOM + 20000]
    MASTER_PORT=${MASTER_PORT:-$RANDOM_PORT}
    GPU_NUM=$((${GPUS_PER_NODE}*${WORLD_SIZE}))
    echo "---> from pytorch runtime, WORLD_SIZE: ${WORLD_SIZE}, NODE_RANK: ${NODE_RANK}, MASTER_ADDR: ${MASTER_ADDR}, MASTER_PORT: ${MASTER_PORT}"
    LAUNCHER=" \
    torchrun \
    --nproc_per_node ${GPUS_PER_NODE} \
    --nnodes ${WORLD_SIZE} \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    "
fi

DATA_ARGS=(
    --data_path $DATA_PATH
    --tokenizer_model $TOKENIZER_PATH
    # For GPT2
    # --vocab_file $VOCAB_FILE
    # --merge_file $MERGE_FILE
)

CHECKPOINTING_ARGS=(
    # --flash_checkpoint
    # --resume_from_checkpoint $OUTPUT_DIR
    # --save_to_ais_ckpt
    # --ais_ckpt_memo youyou_240926001
)

PROFILING_ARGS=(
    # --profiler_type "nv"
    # --profiler_file_path $OUTPUT_DIR/profiler_output
    # --profiler_schedule_wait 1
    # --profiler_schedule_warmup 1
    # --profiler_schedule_active 1
    # --profiler_schedule_repeat 1
    # --profiler_schedule_skip_first 20
)

TRAINING_ARGS=(
    --output_dir $OUTPUT_DIR
    --overwrite_output_dir
    --do_train
    --do_eval
    --distributed_type "megatron"
    --num_train_epochs 6
    --block_size 512
    --per_device_train_batch_size 2
    --per_device_eval_batch_size 2
    --preprocessing_num_workers 6
    --learning_rate 2e-5
    --weight_decay 0.0
    --warmup_ratio 0.03
    --seed 42
    --max_grad_norm 0
    --bf16
    --save_strategy "steps"
    --save_steps 1000
    --save_total_limit 3
    --evaluation_strategy "steps"
    --eval_steps 2000
    --logging_strategy "steps"
    --logging_steps 1
    --logging_nan_inf_filter false
    --log_params_std true
    --log_grad_diff_for_debug true
    --tensorboard_dir $OUTPUT_DIR/runs/$TIME_STAMP
    --dataloader_num_workers 0
    --gradient_checkpointing
)

CMD="${LAUNCHER[@]} \
    $(dirname $0)/llama2_trainer_megatron.py \
    ${DATA_ARGS[@]} \
    ${CHECKPOINTING_ARGS[@]} \
    ${PROFILING_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
"
echo ${CMD}
${CMD} 2>&1 | tee -a ${OUTPUT_DIR}/log_${NODE_RANK}.log
