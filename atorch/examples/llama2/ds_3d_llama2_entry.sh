#!/bin/bash
set -x

source ./dataset_model.sh

WORLD_SIZE=${WORLD_SIZE:-1}
NUM_GPUS_PER_NODE=$(nvidia-smi -L | wc -l)

PIPELINE_PARALLEL_SIZE=${PIPELINE_PARALLEL_SIZE:-2}
MODEL_PARALLEL_SIZE=${MODEL_PARALLEL_SIZE:-2}
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


python -u -m atorch.distributed.run \
  --nnodes=$WORLD_SIZE --nproc_per_node=$NUM_GPUS_PER_NODE ds_3d_llama2.py \
  --pipeline_parallel_size $PIPELINE_PARALLEL_SIZE \
  --model_parallel_size $MODEL_PARALLEL_SIZE \
  --block_size $BLOCK_SIZE \
  --ds_config $DS_CONFIG \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --dataset_path $DATASET_PATH
