#!/bin/bash

HOME=$(echo ~)

# Dataset path, would download in `example_utils.py` if not exist
DATASET_PATH=${DATASET_PATH:-$HOME/.cache/wikitext-2-raw-v1}

# Llama model path, download and convert it if not exist
MODEL_SIZE=${MODEL_SIZE-7B}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-$HOME/.cache/Llama-2-`echo $MODEL_SIZE|tr '[:upper:]' '[:lower:]'`-hf}
if ! [[ -d $MODEL_NAME_OR_PATH && \
        -f ${MODEL_NAME_OR_PATH%/}/config.json && \
        -f ${MODEL_NAME_OR_PATH%/}/tokenizer_config.json && \
        -f ${MODEL_NAME_OR_PATH%/}/tokenizer.json && \
        -f ${MODEL_NAME_OR_PATH%/}/tokenizer.model ]]; then
  echo "$MODEL_NAME_OR_PATH not cached."
  mkdir -p $HOME/.cache/
  pushd $HOME/.cache/
  git clone https://github.com/shawwn/llama-dl.git
  pushd llama-dl
  sed 's/MODEL_SIZE="7B,13B,30B,65B"/MODEL_SIZE="'$MODEL_SIZE'"/g' llama.sh > llama$MODEL_SIZE.sh
  bash llama$MODEL_SIZE.sh
  pip install transformers sentencepiece
  python -m transformers.models.llama.convert_llama_weights_to_hf --input_dir=. --model_size=$MODEL_SIZE --output_dir=$MODEL_NAME_OR_PATH
  popd
  popd
fi