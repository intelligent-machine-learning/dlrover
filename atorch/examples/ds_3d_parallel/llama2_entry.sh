set -x

DATASET_PATH=${DATASET_PATH:-/path/to/wikitext-2-raw-v1}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-/path/to/Llama-2-7b-hf}
# MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-/path/to/Llama-2-70b-hf}

WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29678}

PIPELINE_PARALLEL_SIZE=${PIPELINE_PARALLEL_SIZE:-4}
MODEL_PARALLEL_SIZE=${MODEL_PARALLEL_SIZE:-8}
BLOCK_SIZE=${BLOCK_SIZE:-4096}

# ds config
script_path=$(realpath $BASH_SOURCE)
script_dir=$(dirname $script_path)
DS_CONFIG="$script_dir/ds_config.json"
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
ACCU_STEPS=${ACCU_STEPS:-8}
cat <<EOT > $DS_CONFIG
{
    "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
    "gradient_accumulation_steps": $ACCU_STEPS,
    "steps_per_print": 50,
    "gradient_clipping": 1.0,
    "zero_optimization": {
      "stage": 1
    },
    "zero_allow_untested_optimizer": true,
    "fp16": {
      "enabled": true,
      "loss_scale": 0,
      "loss_scale_window": 1000,
      "initial_scale_power": 16,
      "hysteresis": 2,
      "min_loss_scale": 1
    },
    "activation_checkpointing": {
      "partition_activations": false,
      "contiguous_memory_optimization": false
    },
    "wall_clock_breakdown": false,
    "pipeline": {
      "activation_checkpoint_interval": 1
    }
  }
EOT


nohup python -u -m atorch.distributed.launch --nproc_per_node 8 --nnodes $WORLD_SIZE \
  --node_rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT ds_3d_llama2.py \
  --pipeline_parallel_size $PIPELINE_PARALLEL_SIZE \
  --model_parallel_size $MODEL_PARALLEL_SIZE \
  --block_size $BLOCK_SIZE \
  --ds_config $DS_CONFIG \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --dataset_path $DATASET_PATH >> output.log 2>&1 &
