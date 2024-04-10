import argparse
import datetime
from contextlib import nullcontext

import torch
from example_utils import compute_llama2_training_flops, get_data_iter, print_rank_0, report_memory, sync_and_time
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

import atorch
from atorch.auto import auto_accelerate
from atorch.common.log_utils import default_logger as logger
from atorch.utils.meta_model_utils import init_empty_weights_with_disk_offload


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain llama2 with atorch fsdp.")
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
        "--init_emtpy_offload",
        action="store_true",
        help="If passed, use init_empty_weights_with_disk_offload.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "bf16_amp", "fp16_amp", "bf16"],
        default="bf16_amp",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=0,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--peft_type",
        type=str,
        default=None,
        help="Whether use peft and use what type of peft.",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="Lora attention dimension.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="The alpha parameter for Lora scaling.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="The dropout probability for Lora layers.",
    )
    parser.add_argument(
        "--lora_target_modules",
        nargs="*",
        default=["q_proj", "v_proj"],
        help="The names of the modules to apply Lora to.",
    )
    parser.add_argument(
        "--peft_task_type",
        type=str,
        default=TaskType.CAUSAL_LM,
        choices=[TaskType.SEQ_CLS, TaskType.SEQ_2_SEQ_LM, TaskType.CAUSAL_LM, TaskType.TOKEN_CLS],
        help="Peft task type.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Use gradient checkpointing or not.",
    )
    parser.add_argument(
        "--fp8",
        action="store_true",
        help="Use fp8 or not.",
    )
    args = parser.parse_args()

    return args


def get_peft_config(args):
    """
    Returns:
        config(PeftConfig)
    """
    if args.peft_type == "lora":
        peft_config = LoraConfig(
            task_type=args.peft_task_type,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
        )
    else:
        raise NotImplementedError(f"Not support {args.peft_type}")
    return peft_config


def my_loss_func(_, outputs):
    if isinstance(outputs, dict):
        return outputs["loss"]


def my_prepare_input(batch, device):
    batch = {k: v.to(device=device, non_blocking=True) for k, v in batch.items()}
    return batch


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

    llama_config = AutoConfig.from_pretrained(args.model_name_or_path)
    with init_empty_weights_with_disk_offload() if args.init_emtpy_offload else nullcontext():
        model = AutoModelForCausalLM.from_config(llama_config)

    # Lora support
    if args.peft_type is not None:
        peft_config = get_peft_config(args)
        logger.info(f"Load Peft {args.peft_type} model ......")
        if args.gradient_checkpointing and args.peft_type == "lora":
            # Make Lora and gradient checkpointing compatible
            # https://github.com/huggingface/peft/issues/137
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        model = get_peft_model(model, peft_config)

    strategy = [
        ("parallel_mode", ([("data", atorch.world_size())], None)),
        "module_replace",
        (
            "fsdp",
            {
                "sync_module_states": True,
                "atorch_wrap_cls": (LlamaDecoderLayer,),
                "limit_all_gathers": True,
                "use_orig_params": True,
                "forward_prefetch": True,
                "wrap_trainable_outmost": args.peft_type is not None,
            },
        ),
    ]
    if "amp" in args.precision:
        low_precision_dtype = torch.bfloat16 if args.precision == "bf16_amp" else torch.float16
        amp_opt = ("amp_native", {"dtype": low_precision_dtype})
        strategy.append(amp_opt)
    elif args.precision == "bf16":
        strategy.append(("half", "bf16"))
    if args.gradient_checkpointing:
        checkpoint_config = {"wrap_class": (LlamaDecoderLayer,), "no_reentrant": True}
        strategy.append(("checkpoint", checkpoint_config))
    if args.fp8:
        if args.peft_type is not None:
            logger.warning("fp8 ignored as fp8 for lora training is not implemented yet.")
        else:
            strategy.append(("fp8", {"include": ("layers",)}))
    status, result, best_strategy = auto_accelerate(
        model,
        torch.optim.AdamW,
        loss_func=my_loss_func,
        prepare_input=my_prepare_input,
        model_input_format="unpack_dict",
        optim_args={"lr": 1e-5},
        optim_param_func=optim_param_func,
        load_strategy=strategy,
        ignore_dryrun_on_load_strategy=True,
    )
    assert status, "auto_accelerate failed"
    logger.info(f"Best strategy is: {best_strategy}")
    model = result.model
    optimizer = result.optim
    loss_func = result.loss_func
    prepare_input = result.prepare_input

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    data_iter = get_data_iter(
        args.dataset_path, tokenizer, args.block_size, args.per_device_train_batch_size, pre_shift=False
    )

    global_batch_size = args.per_device_train_batch_size * atorch.world_size()
    print_rank_0(f"Global batch size: {global_batch_size}")
    flops_per_iter, _ = compute_llama2_training_flops(
        global_batch_size,
        args.block_size,
        llama_config.hidden_size,
        llama_config.vocab_size,
        llama_config.intermediate_size,
        llama_config.num_hidden_layers,
        use_gradient_checkpointing=args.gradient_checkpointing,
        use_lora=args.peft_type is not None,
    )

    # train
    device = torch.device("cuda:{}".format(atorch.local_rank()))
    model.train()
    timestamp = sync_and_time()
    for iter, batch in zip(range(args.max_steps), data_iter):
        batch = prepare_input(batch, device)
        outputs = model(**batch)
        loss = loss_func(None, outputs)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
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
