config文件位置：
src/llama_recipes/configs

常用命令：
```
export PYTHONPATH=$PYTHONPATH:./src/
```

```bash
TORCH_NCCL_AVOID_RECORD_STREAMS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nnodes 1 --nproc_per_node 8 examples/finetuning.py --use_fast_kernels --enable_fsdp --low_cpu_fsdp --fsdp_config.pure_bf16 --fsdp_config.fsdp_activation_checkpointing False --batch_size_training 1 --model_name /datacube_nas/workspace/sichuan/pretrained_models/Llama-2-7b-hf/ --dataset alpaca_dataset 
```