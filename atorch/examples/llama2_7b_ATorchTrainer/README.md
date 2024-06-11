# Llama2 7B Finetune By ATorchTrainer

This document presents 2 examples of using ATorchTrainer api to finetune the HuggingFace Llama-2-7b-hf model, using the ways of fsdp or fsdp with lora mainly.

- Note: 
    - Llama2 model and alpaca dataset is used in the examples. The training script will automatically download them for you. Note that downloading may take quite some time.


## ATorchTrainer FSDP

Fully Sharded Data Parallel (FSDP) is a default training config in ATorchTrainer. This is implemented by calling auto_accelerate API with load_strategy argument, and load_strategy specifies the training optimization method combination.

### Scripts

- training file [llama2_clm_atorch_trainer.py](llama2_clm_atorch_trainer.py)

- launch script [llama2_7b_trainer_entry.sh](llama2_7b_trainer_entry.sh)

```bash
cd dlrover/atorch/examples/llama2_7b_ATorchTrainer
pip install -r requirements.txt

WORLD_SIZE=8 bash llama2_7b_trainer_entry.sh output_dir
```


## ATorchTrainer FSDP with LoRA

LoRA is compatible by ATorchTrainer FSDP training, you can load peft lora model firstly.

### Scripts

- training file [llama2_clm_atorch_trainer.py](llama2_clm_atorch_trainer.py)

- launch script [llama2_7b_trainer_lora_entry.sh](llama2_7b_trainer_lora_entry.sh)

```bash
cd dlrover/atorch/examples/llama2_7b_AtorchTrainer
pip install -r requirements.txt

WORLD_SIZE=8 bash llama2_7b_trainer_lora_entry.sh output_dir
```