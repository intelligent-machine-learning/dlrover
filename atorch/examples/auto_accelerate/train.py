import argparse
import time

import torch
from data import get_dataloader_args, get_dataset
from modeling import ModelType, get_loss_func, get_model, get_model_input_format, get_model_type, get_module_type
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import atorch
from atorch.auto.accelerate import auto_accelerate
from atorch.auto.model_context import get_data_partition_rank_and_size
from atorch.common.util_func import data_to_device


def optim_grouped_param_func(model):
    no_decay = "bias"
    parameters = [
        {
            "params": [p for n, p in model.named_parameters() if no_decay not in n],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if no_decay in n],
            "weight_decay": 0.0,
        },
    ]
    return parameters


def parse_args():
    parser = argparse.ArgumentParser(description="Process arguments")
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--datasize", type=int, default=200, required=False)
    parser.add_argument("--epoch", type=int, default=2, required=False)
    parser.add_argument("--hidden_size", type=int, default=32, required=False)
    parser.add_argument("--head_num", type=int, default=4, required=False)
    parser.add_argument("--layer_num", type=int, default=3, required=False)
    parser.add_argument("--seq_length", type=int, default=16, required=False)
    parser.add_argument("--batchsize", type=int, default=8, required=False)
    parser.add_argument("--in_size", type=int, default=16, required=False)
    parser.add_argument("--out_size", type=int, default=8, required=False)
    parser.add_argument("--distributed", default=False, action="store_true")
    parser.add_argument("--user_created_dataloader", default=False, action="store_true")
    parser.add_argument("--load_strategy", default=False, action="store_true")
    parser.add_argument("--optim_grouped_params", default=False, action="store_true")
    parser.add_argument("--log_interval", type=int, default=10, required=False)
    parser.add_argument("--use_fsdp", default=False, action="store_true")
    parser.add_argument("--use_amp", default=False, action="store_true")
    parser.add_argument("--use_fp8", default=False, action="store_true")
    parser.add_argument("--use_checkpointing", default=False, action="store_true")
    parser.add_argument("--use_module_replace", default=False, action="store_true")
    # local sgd related
    parser.add_argument("--use_local_sgd", default=False, action="store_true")
    parser.add_argument("--local_sgd_sync_interval", type=int, default=1, required=False)
    parser.add_argument("--local_sgd_warmup_steps", type=int, default=0, required=False)
    parser.add_argument("--outer_optim_class", type=str, choices=["none", "sgd"], default="none", required=False)

    return parser.parse_args()


def train(args):
    # get model type
    model_type = get_model_type(args.model_type)
    if model_type is None:
        print(f"{args.model_type} not supported model type.")
        return

    # init distributed if distributed training
    if args.distributed:
        if torch.cuda.is_available():
            atorch.init_distributed("nccl", set_cuda_device_using_local_rank=True)
        else:
            atorch.init_distributed("gloo")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # get model, loss_func
    if model_type == ModelType.TOY:
        model_config = {
            "in_features": args.in_size,
            "out_features": args.out_size,
            "num_linears": args.layer_num,
        }
    else:
        model_config = {
            "hidden_size": args.hidden_size,
            "head_num": args.head_num,
            "layer_num": args.layer_num,
            "seq_length": args.seq_length,
        }
    model = get_model(model_type, model_config)
    print("Get model with class ", model.__class__)
    loss_func = get_loss_func(model_type)

    dataset = get_dataset(
        model_type,
        seq_length=args.seq_length,
        input_size=args.in_size,
        output_size=args.out_size,
        datasize=args.datasize,
    )
    dataloader_args = get_dataloader_args(model_type, batch_size=args.batchsize)

    strategy = None
    if args.load_strategy:
        # data parallel if distributed
        strategy = ["parallel_mode"] if args.distributed else []
        if args.use_local_sgd:
            device_counts = torch.cuda.device_count()
            if device_counts < 4 or device_counts % 2:
                raise RuntimeError(
                    "If using local sgd, make sure the number of GPUs is an even number and not less than 4."
                )
            parallel_config = ([("zero", device_counts // 2), ("data", 2)], None)
            strategy = [("parallel_mode", parallel_config)] if args.distributed else []
        # module_replace
        if torch.cuda.is_available() and args.use_module_replace:
            strategy.append("module_replace")
        # fsdp
        if args.use_fsdp:
            fsdp_config = {
                "sync_module_states": True,
                "limit_all_gathers": True,
                "forward_prefetch": True,
                "atorch_wrap_cls": (get_module_type(model_type),),
            }
            # use_orig_params if grouped parameters are used in optim.
            if args.optim_grouped_params:
                fsdp_config["use_orig_params"] = True
            if args.use_local_sgd:
                fsdp_config["use_local_sgd"] = True
                fsdp_config["local_sgd_sync_interval"] = args.local_sgd_sync_interval
                fsdp_config["local_sgd_warmup_steps"] = args.local_sgd_warmup_steps
                fsdp_config["outer_optim_class"] = torch.optim.SGD if args.outer_optim_class == "sgd" else None
                fsdp_config["outer_optim_kwargs"] = {
                    "lr": 0.7,
                    "momentum": 0.8,
                    "nesterov": True,
                }
                fsdp_config["outer_optim_cpu_offload"] = True
            strategy.append(("fsdp", fsdp_config))
        # mixed precision
        if torch.cuda.is_available() and args.use_amp:
            amp_config = {"dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16}
            strategy.append(("amp_native", amp_config))
        # checkpoint
        if args.use_checkpointing:
            checkpoint_modules = (get_module_type(model_type),)
            checkpoint_config = {"wrap_class": checkpoint_modules, "no_reentrant": True}
            strategy.append(("checkpoint", checkpoint_config))
        # fp8
        if args.use_fp8:
            if model_type == ModelType.LLAMA:
                strategy.append(("fp8", {"include": ("layers",)}))
            elif model_type == ModelType.TOY:
                if args.in_size % 16 != 0 or args.out_size % 16 != 0 or args.batchsize % 16 != 0:
                    print(
                        "fp8 is ignored. To use fp8 for toy model, "
                        + "in_size({}), out_size({}) and batchsize({}) must be multiples of 16!".format(
                            args.in_size, args.out_size, args.batchsize
                        )
                    )
                else:
                    strategy.append("fp8")
            else:
                print("fp8 is ignored for gpt2 model")

    # optimizer
    if model_type == ModelType.LLAMA:
        optim_func = atorch.optimizers.AGD
    else:
        optim_func = torch.optim.AdamW
    optim_args = {"lr": 0.001}
    optim_param_func = optim_grouped_param_func if args.optim_grouped_params else None

    # Move data to device
    prepare_input = data_to_device

    model_input_format = get_model_input_format(model_type)

    # auto_accelerate
    status, res, best_strategy = auto_accelerate(
        model,
        optim_func=optim_func,
        dataset=dataset if not args.user_created_dataloader else None,
        loss_func=loss_func,
        prepare_input=prepare_input,
        model_input_format=model_input_format,
        optim_args=optim_args,
        optim_param_func=optim_param_func,
        dataloader_args=dataloader_args if not args.user_created_dataloader else None,
        load_strategy=strategy,
        ignore_dryrun_on_load_strategy=args.load_strategy,
    )
    assert status
    # res is a namedtuple of (model, optim, dataloader, loss_func, prepare_input, lr_scheduler)
    model = res.model
    optim = res.optim
    dataloader = res.dataloader
    loss_func = res.loss_func
    prepare_input = res.prepare_input

    if args.user_created_dataloader:
        sampler = None
        if args.distributed:
            rank, dp_size = get_data_partition_rank_and_size()
            if dp_size > 1:
                # strong scaling for batchsize, so adjust per-process batchsize
                dataloader_args["batch_size"] = dataloader_args["batch_size"] // dp_size
                shuffle = dataloader_args.get("shuffle", False)
                if shuffle:
                    dataloader_args["shuffle"] = False

                sampler = DistributedSampler(dataset, shuffle=shuffle, num_replicas=dp_size, rank=rank)
        dataloader = DataLoader(dataset, sampler=sampler, **dataloader_args)

    global_step = 0
    start_time = time.time()
    for _ in range(args.epoch):
        for batch in dataloader:
            optim.zero_grad()
            batch = prepare_input(batch, device)
            if model_input_format == "unpack_dict":
                outputs = model(**batch)
            elif model_input_format == "unpack_dequence":
                outputs = model(*batch)
            else:
                outputs = model(batch)
            loss = loss_func(batch, outputs)
            loss.backward()
            optim.step()
            global_step += 1
            if global_step % args.log_interval == 0 and (atorch.rank() is None or atorch.rank() == 0):
                cur_time = time.time()
                time_per_step = (cur_time - start_time) / args.log_interval
                print(f"[step={global_step-1}]: {time_per_step} sec/step")
                start_time = cur_time

    print("Finished training!")


if __name__ == "__main__":
    args = parse_args()
    train(args)
