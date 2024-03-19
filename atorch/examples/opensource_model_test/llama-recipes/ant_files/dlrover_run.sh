set -euxo

NUM_DEVICES=$(nvidia-smi -L | wc -l)

export PYTHONPATH=$PYTHONPATH:./src/

TORCH_NCCL_AVOID_RECORD_STREAMS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
dlrover-run --network-check --max_restarts=0 --nnodes=$NODE_NUM --nproc_per_node=${NUM_DEVICES} examples/finetuning.py \
    --use_fast_kernels \
    --enable_fsdp \
    --low_cpu_fsdp \
    --context_length \
    --fsdp_config.pure_bf16 \
    --fsdp_config.fsdp_activation_checkpointing False \
    --batch_size_training 1 \
    --batching_strategy packing \
    --gradient_accumulation_steps 1 \
    --num_epochs 3 \
    --model_name /datacube_nas/workspace/sichuan/pretrained_models/Llama-2-7b-hf/ \
    --dataset alpaca_dataset \
    --show_throughput \
    --output_dir /tmp
