#!/bin/bash
set -ex

export PYTHONPATH=$PWD/ant_patches:$PYTHONPATH

export OMP_NUM_THREADS=1
#export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG_SUBSYS=INIT   # disable aistudio default nccl env

GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
#GPUS_PER_NODE=4
#export CUDA_VISIBLE_DEVICES=4,5,6,7

lines=`echo $POD_NAME | grep edljob | wc -l`
if [ $lines -eq 0 ]; then
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
else
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
fi

MODEL_SIZE=${MODEL_SIZE:-8}
if   [[ ${MODEL_SIZE} == 8 ]];   then HIDDEN_SIZE=4096;  NUM_HEADS=32;  NUM_QUERY_GROUPS=8; NUM_LAYERS=32;  FFN_HIDDEN_SIZE=14336; MAX_POSITION_EMBEDDINGS=131072; VOCAB_SIZE=128256;
elif [[ ${MODEL_SIZE} == 70 ]];  then HIDDEN_SIZE=8192;  NUM_HEADS=64;  NUM_QUERY_GROUPS=8; NUM_LAYERS=80;  FFN_HIDDEN_SIZE=28672; MAX_POSITION_EMBEDDINGS=131072; VOCAB_SIZE=128256;
elif [[ ${MODEL_SIZE} == 405 ]]; then HIDDEN_SIZE=16384; NUM_HEADS=128; NUM_QUERY_GROUPS=8; NUM_LAYERS=126; FFN_HIDDEN_SIZE=53248; MAX_POSITION_EMBEDDINGS=131072; VOCAB_SIZE=128256;
elif [[ ${MODEL_SIZE} == 1 ]];   then HIDDEN_SIZE=2048;  NUM_HEADS=32;  NUM_QUERY_GROUPS=8; NUM_LAYERS=16;  FFN_HIDDEN_SIZE=8192;  MAX_POSITION_EMBEDDINGS=131072; VOCAB_SIZE=128256;
elif [[ ${MODEL_SIZE} == 3 ]];   then HIDDEN_SIZE=3072;  NUM_HEADS=24;  NUM_QUERY_GROUPS=8; NUM_LAYERS=28;  FFN_HIDDEN_SIZE=8192;  MAX_POSITION_EMBEDDINGS=131072; VOCAB_SIZE=128256;
else echo "invalid MODEL_SIZE: ${MODEL_SIZE}"; exit 1
fi

DEVICE_MODEL=$(nvidia-smi -i 0 -q | grep "Product Name" | awk -F: '{ print $2 }')
DEVICE_MODEL=$(echo "$DEVICE_MODEL" | xargs)  # drop white space
DEVICE_MODEL=NVIDIA

JOB_DIR="/tmp/llama3_${MODEL_SIZE}B_${DEVICE_MODEL}_${GPU_NUM}p"
echo $JOB_DIR
mkdir -p ${JOB_DIR}
CHECKPOINT_PATH=${JOB_DIR} #<Specify path>
TENSORBOARD_LOGS_PATH=${JOB_DIR}


cp -r ${0} ${JOB_DIR}
#pip list > ${JOB_DIR}/pip_list.txt

DATASET_DIR="/dnn_training_sys/dataset/nlp/fineweb-edu/CC-MAIN-2024-10/"
DATASET0="${DATASET_DIR}/CC-MAIN-2024-10_0000_text_document"
DATASET1="${DATASET_DIR}/CC-MAIN-2024-10_0001_text_document"
DATASET2="${DATASET_DIR}/CC-MAIN-2024-10_0002_text_document"
DATASET4="${DATASET_DIR}/CC-MAIN-2024-10_0004_text_document"

DATA_PATH="0.25 ${DATASET0} 0.25 ${DATASET1} 0.25 ${DATASET2} 0.25 ${DATASET4}"
LOG_PATH="${JOB_DIR}/debug_llama_${RANK}.txt"



GPT_MODEL_ARGS=(
    --num-layers $NUM_LAYERS
    --hidden-size $HIDDEN_SIZE
    --ffn-hidden-size $FFN_HIDDEN_SIZE
    --num-attention-heads $NUM_HEADS
    --group-query-attention
    --num-query-groups $NUM_QUERY_GROUPS
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS
    --vocab-size $VOCAB_SIZE
    --position-embedding-type "rope"
    --rotary-base 500000
    --rotary-percent 1.0
    --swiglu
    --untie-embeddings-and-output-weights
    --normalization "RMSNorm"
    --norm-epsilon "1e-05"
    --disable-bias-linear
    --transformer-impl "transformer_engine"
)

TRAINING_ARGS=(
    --micro-batch-size 2
    --global-batch-size 128
    --seq-length "4096"
    --train-iters 800
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --init-method-std 0.006
    --clip-grad 1.0
    --bf16
    --lr "3.0e-4"
    --lr-decay-style cosine
    --min-lr "3.0e-5"
    --lr-warmup-fraction 0.001
    --lr-decay-iters 20000
    --seed 42
)


if [ "$DEVICE_MODEL" = "A800-SXM4-80GB" ] || [ "$DEVICE_MODEL" = "A100-SXM4-80GB" ]; then
    # Ampere GPUs do not support multicast. If `--tp-comm-overlap` is set on Ampere-arch GPUs, this env must be set.
    export UB_SKIPMC=1  
fi
export NVTE_FLASH_ATTN=1

# deterministic computation
#export PYTORCH_JIT=0 
#export NVTE_TORCH_COMPILE=0
#export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
#export NCCL_ALGO="Ring"
#export CUBLAS_WORKSPACE_CONFIG=":4096:8"

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 4
    --use-distributed-optimizer
	--no-async-tensor-model-parallel-allreduce
    --sequence-parallel
	--manual-gc
	--manual-gc-interval 50
)

    #--pipeline-model-parallel-size 4
    #--num-layers-per-virtual-pipeline-stage 2

    #--overlap-param-gather
    #--overlap-grad-reduce
# some optional args

#    --use-distributed-optimizer
#    --overlap-param-gather
#    --overlap-grad-reduce
#    --context-parallel-size 2
#    --tp-comm-overlap
# --decoder-first-pipeline-num-layers
# --decoder-last-pipeline-num-layers


DATA_ARGS=(
    --mock-data
    --tokenizer-type "NullTokenizer"
)


EVAL_AND_LOGGING_ARGS=(
    --save-interval 1000 
    --eval-interval 1000 
    --save $CHECKPOINT_PATH
    --load $CHECKPOINT_PATH
    --ckpt-format "torch_dist"
    --async-save
    --eval-iters 250
    --log-interval 1
    --log-throughput
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard
    --log-world-size-to-tensorboard
    --log-validation-ppl-to-tensorboard
)

KERNEL_ARGS=(
    --use-flash-attn
    --no-masked-softmax-fusion
    --attention-softmax-in-fp32	
)
# 
#    --deterministic-mode
#    --use-flash-attn
#    --cross-entropy-loss-fusion

PROFILING_ARGS=(
    --profile
    --use-pytorch-profiler
    --profile-ranks 0
    --profile-step-start 10
    --profile-step-end 20
)

CMD="${LAUNCHER} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${KERNEL_ARGS[@]} \
	"

    #${PROFILING_ARGS[@]} \
echo ${CMD}
nohup ${CMD} > ${LOG_PATH} 2>&1 &
#${CMD} 2>&1 | tee ${LOG_PATH}
