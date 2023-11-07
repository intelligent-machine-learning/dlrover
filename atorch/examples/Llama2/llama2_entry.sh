set -x

# Dataset path, would download in `example_utils.py` if not exist
DATASET_PATH=${DATASET_PATH:-~/.cache//wikitext-2-raw-v1}

# Llama model path, download and convert it if not exist
MODEL_SIZE=${MODEL_SIZE-7B}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-~/.cache//Llama-2-`echo $MODEL_SIZE|tr '[:upper:]' '[:lower:]'`-hf}
if ! [[ -d $MODEL_NAME_OR_PATH && \
        -f ${MODEL_NAME_OR_PATH%/}/config.json && \
        -f ${MODEL_NAME_OR_PATH%/}/tokenizer_config.json && \
        -f ${MODEL_NAME_OR_PATH%/}/tokenizer.json && \
        -f ${MODEL_NAME_OR_PATH%/}/tokenizer.model ]]; then
  echo "$MODEL_NAME_OR_PATH not cached."
  pushd /tmp
  git clone https://github.com/shawwn/llama-dl.git
  pushd llama-dl
  sed 's/MODEL_SIZE="7B,13B,30B,65B"/MODEL_SIZE="'$MODEL_SIZE'"/g' llama.sh > llama$MODEL_SIZE.sh
  sh llama$MODEL_SIZE.sh
  pip install transformers, sentencepiece
  python -m transformers.models.llama.convert_llama_weights_to_hf --input_dir=. --model_size=$MODEL_SIZE --output_dir=$MODEL_NAME_OR_PATH
  popd
  popd
fi

WORLD_SIZE=${WORLD_SIZE:-1}

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


nohup python -u -m atorch.distributed.run --fault_tolerant \
  --nnodes=$WORLD_SIZE --nproc_per_node=8 ds_3d_llama2.py \
  --pipeline_parallel_size $PIPELINE_PARALLEL_SIZE \
  --model_parallel_size $MODEL_PARALLEL_SIZE \
  --block_size $BLOCK_SIZE \
  --ds_config $DS_CONFIG \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --dataset_path $DATASET_PATH >> output.log 2>&1 &
