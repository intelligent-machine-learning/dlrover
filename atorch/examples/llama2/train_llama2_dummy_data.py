import argparse
import functools
import time
from datetime import timedelta

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaForCausalLM

import atorch
from atorch.auto.accelerate import auto_accelerate
from atorch.auto.model_context import get_data_partition_rank_and_size
from atorch.common.util_func import data_to_device
from atorch.distributed.distributed import destroy_parallel_group
from atorch.utils.meta_model_utils import init_empty_weights_with_disk_offload


def llama_loss_func(inputs, output):
    return output.loss


def prepare_input(data, device):
    return data_to_device(data, device)


def get_model(config, meta_init=False):
    if meta_init:
        with init_empty_weights_with_disk_offload(ignore_tie_weights=False):
            model = LlamaForCausalLM(config)
    else:
        model = LlamaForCausalLM(config)
    return model


# copy from pytorch/benchmarks/distributed/pipeline/benchmark_dataset.py with some modification
def collate_sentences_lm(samples, input_name, label_name):
    if len(samples) == 0:
        return {}

    src_tokens = torch.stack([s["source"] for s in samples], 0)
    tgt_tokens = torch.stack([s["target"] for s in samples], 0)

    batch = {
        input_name: src_tokens,
        label_name: tgt_tokens,
    }
    return batch


# copy from pytorch/benchmarks/distributed/pipeline/benchmark_dataset.py with some modification
class BenchmarkLMDataset(Dataset):
    def __init__(
        self,
        vocab_size=10000,
        max_source_positions=1024,
        total_samples=10000,
    ):
        self.vocab_size = vocab_size
        self.max_source_positions = max_source_positions
        self.total_samples = total_samples

    def __getitem__(self, index):
        length = self.max_source_positions
        source = torch.randint(1, self.vocab_size, (length,))
        target = source.clone()
        return {
            "source": source,
            "target": target,
        }

    def __len__(self):
        return self.total_samples


def get_dataset(seq_length=128, vocab_size=32000, datasize=1000):
    return BenchmarkLMDataset(vocab_size=vocab_size, max_source_positions=seq_length, total_samples=datasize)


def get_dataloader(dataset, batch_size, rank, dp_size, use_distributed):
    dataloader_args = {"batch_size": batch_size, "drop_last": True, "shuffle": False, "num_workers": 2}
    input_name = "input_ids"
    label_name = "labels"
    dataloader_args["collate_fn"] = functools.partial(
        collate_sentences_lm, input_name=input_name, label_name=label_name
    )
    if use_distributed:
        sampler = DistributedSampler(dataset, shuffle=False, num_replicas=dp_size, rank=rank)
    else:
        sampler = None
    dataloader = DataLoader(dataset, sampler=sampler, **dataloader_args)
    return dataloader


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
    parser.add_argument("--hidden_size", type=int, default=256, required=False)
    parser.add_argument("--intermediate_size", type=int, default=512, required=False)
    parser.add_argument("--head_num", type=int, default=4, required=False)
    parser.add_argument("--key_value_head_num", type=int, default=4, required=False)
    parser.add_argument("--layer_num", type=int, default=3, required=False)
    parser.add_argument("--seq_length", type=int, default=64, required=False)

    parser.add_argument("--batchsize_per_gpu", type=int, default=8, required=False)
    parser.add_argument("--max_train_step", type=int, default=20, required=False)

    parser.add_argument("--optim_grouped_params", default=False, action="store_true")
    parser.add_argument("--log_interval", type=int, default=10, required=False)
    parser.add_argument("--timeout_sec", type=int, default=1800, required=False)

    parser.add_argument("--use_fsdp", default=False, action="store_true")
    parser.add_argument("--use_amp", default=False, action="store_true")
    parser.add_argument("--use_fp8", default=False, action="store_true")
    parser.add_argument("--use_checkpointing", default=False, action="store_true")
    parser.add_argument("--use_module_replace", default=False, action="store_true")
    parser.add_argument("--use_meta_init", default=False, action="store_true")
    parser.add_argument("--use_distributed_dataloader", default=False, action="store_true")
    parser.add_argument("--max_checkpoint_module_num", type=int, default=-1, required=False)

    return parser.parse_args()


def get_strategy(args):
    strategy = ["parallel_mode"]
    if args.use_module_replace:
        strategy.append("module_replace")
    if args.use_fsdp:
        fsdp_config = {
            "sync_module_states": True,
            "limit_all_gathers": True,
            "forward_prefetch": True,
            "atorch_wrap_cls": (LlamaDecoderLayer,),
        }
        strategy.append(("fsdp", fsdp_config))
    if args.optim_grouped_params:
        fsdp_config["use_orig_params"] = True
    if args.use_amp:
        amp_config = {"dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16}
        strategy.append(("amp_native", amp_config))
    if args.use_checkpointing:
        checkpoint_modules = (LlamaDecoderLayer,)
        checkpoint_config = {"wrap_class": checkpoint_modules, "no_reentrant": True}
        if args.max_checkpoint_module_num >= 0:
            checkpoint_config["max_checkpoint_module_num"] = args.max_checkpoint_module_num
        strategy.append(("checkpoint", checkpoint_config))
    if args.use_fp8:
        strategy.append(("fp8", {"include": ("layers",)}))
    return strategy


def train(args):
    timeout = timedelta(seconds=args.timeout_sec)
    atorch.init_distributed("nccl", set_cuda_device_using_local_rank=True, timeout=timeout)
    if atorch.rank() == 0:
        print("Args is ", args)

    model_config = LlamaConfig()
    c_s = f"hidden_size={args.hidden_size},num_attention_heads={args.head_num},num_hidden_layers={args.layer_num},"
    c_s += f"num_key_value_heads={args.key_value_head_num},max_position_embeddings={args.seq_length},"
    c_s += f"intermediate_size={args.intermediate_size}"
    model_config.update_from_string(c_s)

    st = time.time()
    model = get_model(model_config, meta_init=args.use_meta_init)
    if atorch.rank() == 0:
        print("Get model time : ", time.time() - st)

    if atorch.rank() == 0:
        print(model_config)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of parameters: {total_params}")

    optim_func = torch.optim.AdamW
    optim_args = {"lr": 0.001}
    optim_param_func = optim_grouped_param_func if args.optim_grouped_params else None

    prepare_input = data_to_device

    strategy = get_strategy(args)

    st = time.time()
    # auto_accelerate
    status, res, _ = auto_accelerate(
        model,
        optim_func=optim_func,
        dataset=None,
        loss_func=llama_loss_func,
        prepare_input=prepare_input,
        model_input_format="unpack_dict",
        optim_args=optim_args,
        optim_param_func=optim_param_func,
        dataloader_args=None,
        load_strategy=strategy,
    )
    assert status
    if atorch.rank() == 0:
        print("auto_accelerate time : ", time.time() - st)

    model = res.model
    optim = res.optim
    dataloader = res.dataloader
    loss_func = res.loss_func
    prepare_input = res.prepare_input

    rank, dp_size = get_data_partition_rank_and_size()
    batchsize_per_gpu = args.batchsize_per_gpu
    total_batchsize = batchsize_per_gpu * dp_size
    total_data_size = args.max_train_step * total_batchsize

    if args.use_distributed_dataloader:
        datasize = total_data_size
    else:
        datasize = total_data_size // dp_size
    dataset = get_dataset(seq_length=args.seq_length, vocab_size=model_config.vocab_size, datasize=datasize)
    dataloader = get_dataloader(dataset, batchsize_per_gpu, rank, dp_size, args.use_distributed_dataloader)

    global_step = 0
    start_time = time.time()
    device = "cuda"

    step_time_stats = []
    throughput_stats = []
    max_reserved_stats = []
    max_allocated_stats = []

    for batch in dataloader:
        optim.zero_grad()
        batch = prepare_input(batch, device)
        outputs = model(**batch)
        loss = loss_func(batch, outputs)
        loss.backward()
        optim.step()
        global_step += 1
        if global_step % args.log_interval == 0 and atorch.rank() == 0:
            cur_time = time.time()
            time_per_step = (cur_time - start_time) / args.log_interval
            sample_per_second = total_batchsize / time_per_step
            print(f"[step={global_step-1}]: {time_per_step} sec/step, throughput is {sample_per_second} sample/sec.")
            mem_reserved = torch.cuda.max_memory_reserved() / 1e6
            mem_allcoated = torch.cuda.max_memory_allocated() / 1e6
            print(f"max_memory_reserved={mem_reserved} MB, max_memory_allocated={mem_allcoated} MB.")
            torch.cuda.reset_peak_memory_stats()
            step_time_stats.append(time_per_step)
            throughput_stats.append(sample_per_second)
            max_reserved_stats.append(mem_reserved)
            max_allocated_stats.append(mem_allcoated)
            start_time = cur_time

    if atorch.rank() == 0:
        print("Finished training!")
        total_valid_stats_num = len(step_time_stats) - 1
        if total_valid_stats_num > 0:
            avg_step_time = sum(step_time_stats[1:]) / total_valid_stats_num
            avg_throughput = sum(throughput_stats[1:]) / total_valid_stats_num
            avg_throughput_token = batchsize_per_gpu * args.seq_length / avg_step_time
            avg_max_reserved = max(max_reserved_stats)
            avg_max_allocated = max(max_allocated_stats)
            print(f"Average : {avg_step_time} sec/step.")
            print(f"Average thoughput: {avg_throughput} sample/sec, {avg_throughput_token} token/gpu/sec.")
            print(f"max_memory_reserved={avg_max_reserved} MB, max_memory_allocated={avg_max_allocated} MB.")
    destroy_parallel_group()


if __name__ == "__main__":
    args = parse_args()
    train(args)
