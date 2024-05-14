# Running auto_accelerate Examples

## Source files

[train.py](./train.py): main training code for training model using auto_accelerate API.

[modeling.py](./modeling.py): model definition, including loss_func.

[data.py](./data.py): fake datasets for training.


## Usage

```
train.py [-h] --model_type MODEL_TYPE [--datasize DATASIZE] [--epoch EPOCH] [--hidden_size HIDDEN_SIZE] [--head_num HEAD_NUM] [--layer_num LAYER_NUM] [--seq_length SEQ_LENGTH] [--batchsize BATCHSIZE] [--distributed] [--user_created_dataloader] [--load_strategy] [--optim_grouped_params] [--log_interval LOG_INTERVAL] [--use_fsdp] [--use_amp] [--use_checkpointing] [--use_module_replace] [--use_local_sgd] [--local_sgd_sync_interval LOCAL_SGD_SYNC_INTERVAL] [--local_sgd_warmup_steps LOCAL_SGD_WARMUP_STEPS] [--outer_optim_class OUTER_OPTIM_CLASS]
```

model_type: toy, gpt2, or llama


- toy: a small model with 8 linear layers.
- gpt2: GPT2 model, model size is determined by arguments hidden_size, head_num, layer_num, seq_length.
- llama: Llama2 model, model size is determined by arguments hidden_size, head_num, layer_num, seq_length.

Optional args:

+ datasize: fake dataset size for training, default 200.
+ epoch: training epoch, default 2.
+ hidden_size: for llm (gpt2, llama) model, default 32.
+ head_num: for llm (gpt2, llama) model, default 32.
+ layer_num: for llm (gpt2, llama) model, default 32.
+ seq_length: for llm (gpt2, llama) model, default 32.
+ batchsize: total batchsize in one training iteration, default 8.
+ user_created_dataloader: if set, auto_accelerate will not create dataloader, and user is responsible to provide dataloader. Default False.
+ distributed: if set, running distributed training, default False.
+ load_strategy: if set, set load_strategy to use semi-automatic mode, default False.
+ optim_grouped_params: if set, use grouped params for optimizer, default False.
+ log_interval: log print interval in training, default 10.
+ use_fsdp: if set, add fsdp optimization method in load_strategy. Default False.
+ use_amp: if set, add amp_native optimization method in load_strategy. Default False.
+ use_checkpointing: if set, add checkpoint optimization method in load_strategy. Default False.
+ use_module_replace: if set, add module_replace optimization method in load_strategy. Default False.
+ use_local_sgd: if set, use local sgd for model wrap and training. Default False.
+ local_sgd_sync_interval: model synchronization interval in local sgd training mode. Default 1.
+ local_sgd_warmup_steps: if set, the model will be first trained synchronously in a certain number of steps. Default 0.
+ outer_optim_class: if set, the model will be synchronized using outer optimizer. Only can be set to `none` or `sgd`. Default none.

To launch distributed training, such as 8 process per node training for llama,  run:

```
python -m atorch.distributed.run --nproc_per_node 8 train.py --model_type llama --distributed [other_args]
```

## Examples

Training toy model in fully-automatic mode.
```
python train.py --model_type toy
```
with distributed training.

```
python -m atorch.distributed.run --nproc_per_node 2 train.py --model_type toy --distributed
```

Train gpt2 model in semi-automatic mode, using (fsdp, amp_native, module_replace) optimization strategy.

```
python -m atorch.distributed.run --nproc_per_node 8  train.py --model_type gpt2 --distributed --hidden_size 64 --head_num 4 --layer_num 4 --seq_length 32 --load_strategy --use_fsdp --use_amp --use_module_replace
```

Train llama model in semi-automatic mode, using (fsdp, amp_native, module_replace, checkpoint) optimization strategy, and user provides dataloader.
```
python -m atorch.distributed.run  --nproc_per_node 8  train.py --model_type llama --distributed --hidden_size 64 --head_num 4 --layer_num 4 --seq_length 32 --load_strategy --use_fsdp --use_amp --use_module_replace --use_checkpointing --user_created_dataloader
```
And adding fp8 for training with GPU with fp8 capability (sm >=8.9).
```
python -m atorch.distributed.run  --nproc_per_node 8  train.py --model_type llama --distributed --hidden_size 64 --head_num 4 --layer_num 4 --seq_length 32 --load_strategy --use_fsdp --use_amp --use_module_replace --use_checkpointing --use_fp8 --user_created_dataloader
```

Train llama model in local sgd mode.
```
python -m atorch.distributed.run --nproc_per_node 8 train.py --model_type llama --distributed --hidden_size 64 --head_num 4 --layer_num 4 --seq_length 32 --load_strategy --use_fsdp --use_amp --use_module_replace --use_local_sgd --local_sgd_sync_interval 5 --local_sgd_warmup_steps 10 --outer_optim_class sgd
```
