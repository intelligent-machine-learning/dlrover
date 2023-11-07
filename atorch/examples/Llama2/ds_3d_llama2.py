import argparse
import datetime

import torch
from example_utils import compute_llama2_training_flops, get_data_iter, print_rank_0, report_memory, sync_and_time
from transformers import AutoConfig, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaModel

import atorch
from atorch.auto import auto_accelerate
from atorch.auto.opt_lib.ds_3d_parallel_optimization import DeepSpeed3DParallelConfig
from atorch.common.util_func import divide
from atorch.modules.distributed_modules.cross_entropy import vocab_parallel_cross_entropy
from atorch.utils.manual_tp_utils import TPInfo
from atorch.utils.meta_model_utils import record_module_init


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain llama2 with atorch ds 3d.")
    parser.add_argument(
        "--ds_config",
        type=str,
        default=None,
        help="Deepspeed config json file path.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="A dir containing dataset with .arrow format.",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=100,
        help=("Max steps for training."),
    )
    parser.add_argument(
        "--model_parallel_size", type=int, default=1, help="size of the model parallel. specific for tensor parallel"
    )
    parser.add_argument("--pipeline_parallel_size", type=int, default=1, help="size of the pipeline parallel.")
    args = parser.parse_args()

    return args


def get_llama_tpinfo():
    llama_tpinfo = TPInfo()
    llama_tpinfo.shard_col("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "mlp.gate_proj", "mlp.up_proj")
    llama_tpinfo.shard_row("self_attn.o_proj", "mlp.down_proj")
    llama_tpinfo.shard_vocab("embed_tokens")
    llama_tpinfo.shrink({"self_attn": {"num_heads", "num_key_value_heads", "hidden_size"}})
    return llama_tpinfo


def get_llama_3d_parallel_cfg(args):
    def batch_fn(data):
        tokens = data["input_ids"]
        position_ids = data["position_ids"]
        attention_mask = data["attention_mask"]
        labels = data["labels"]
        loss_mask = data["loss_mask"]

        return (tokens, attention_mask, position_ids), (labels, loss_mask)

    ds_config = args.ds_config

    ds_3d_parallel_cfg = DeepSpeed3DParallelConfig(
        tpinfo=get_llama_tpinfo(),
        ds_config=ds_config,
        batch_fn=batch_fn,
    )
    return ds_3d_parallel_cfg


def my_loss_func(logits, labels):
    labels, loss_mask = labels[0], labels[1]
    logits = logits.float()
    losses = vocab_parallel_cross_entropy(logits, labels).view(-1)
    loss = torch.sum(losses * loss_mask.view(-1))
    if loss_mask.sum().item() > 0:
        loss = loss / loss_mask.sum()
    return loss


def optim_param_func(model):
    no_decay = ["bias", "LayerNorm.weight", "layernorm.weight", "norm.weight"]  # llama has "norm" as final norm
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 1e-1,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


def main():
    args = parse_args()
    atorch.init_distributed("nccl", set_cuda_device_using_local_rank=True)

    tensor_size = args.model_parallel_size
    pipeline_size = args.pipeline_parallel_size
    data_size = divide(atorch.world_size(), tensor_size * pipeline_size)
    print_rank_0(f"3D parallel: tensor {tensor_size}, data {data_size}, pipeline {pipeline_size}")

    llama_config = AutoConfig.from_pretrained(args.model_name_or_path)
    with record_module_init():
        meta_model = LlamaModel(llama_config)

    strategy = [
        ("parallel_mode", ([("tensor", tensor_size), ("data", data_size), ("pipeline", pipeline_size)], None)),
        ("deepspeed_3d_parallel", get_llama_3d_parallel_cfg(args)),
        "module_replace",  # attn fa
    ]
    status, result, best_strategy = auto_accelerate(
        meta_model,
        torch.optim.AdamW,
        optim_args={"lr": 1e-5},
        optim_param_func=optim_param_func,
        loss_func=my_loss_func,
        load_strategy=strategy,
        ignore_dryrun_on_load_strategy=True,
    )
    assert status, f"auto_accelerate failed. status: {status}, result: {result}, best_strategy: {best_strategy}"
    model = result.model

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    data_iter = get_data_iter(args.dataset_path, tokenizer, args.block_size, model.train_micro_batch_size_per_gpu())

    print_rank_0(f"Global batch size: {model.train_batch_size()}")
    flops_per_iter, _ = compute_llama2_training_flops(
        model.train_batch_size(),
        args.block_size,
        llama_config.hidden_size,
        llama_config.vocab_size,
        llama_config.intermediate_size,
        llama_config.num_hidden_layers,
        use_gradient_checkpointing=model.module.activation_checkpoint_interval > 0,
    )
    timestamp = sync_and_time()
    for iter in range(args.max_steps):
        loss = model.train_batch(data_iter)
        new_timestamp = sync_and_time()
        elapsed_time, timestamp = new_timestamp - timestamp, new_timestamp
        print_rank_0(
            f"{datetime.datetime.now()} iter: {iter}, loss: {loss.item():.4f}, "
            f"elapsed time: {elapsed_time:.3f}s, "
            f"TFLOPs: {flops_per_iter / elapsed_time / atorch.world_size() / (10**12):.2f}"
        )
        report_memory("Mem")


if __name__ == "__main__":
    main()
