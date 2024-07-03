import argparse
import functools
import time
from datetime import timedelta

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers.models.llama import modeling_llama
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaForCausalLM

import atorch
from atorch.auto.accelerate import auto_accelerate
from atorch.auto.model_context import get_data_partition_rank_and_size
from atorch.common.util_func import data_to_device
from atorch.distributed.distributed import destroy_parallel_group
from atorch.modules.moe.grouped_gemm_moe import Grouped_GEMM_MoE
from atorch.utils.meta_model_utils import init_empty_weights_with_disk_offload


class TopNRouter(torch.nn.Module):
    """
    This implementation is equivalent to the standard
    TopN MoE with full capacity without dropp tokens.
    """

    def __init__(self, config, top_k=2, norm_prob=True):
        super().__init__()
        self.num_experts = config.num_experts
        self.classifier = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.top_k = top_k
        self.norm_prob = norm_prob

    def _cast_classifier(self):
        r"""
        `bitsandbytes` `Linear8bitLt` layers does not support manual casting Therefore we need to check if they are an
        instance of the `Linear8bitLt` class by checking special attributes.
        """
        if not (hasattr(self.classifier, "SCB") or hasattr(self.classifier, "CB")):
            self.classifier = self.classifier.to(self.dtype)

    def _compute_router_probabilities(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_logits = self.classifier(hidden_states)
        # router_logits = torch.rand_like(router_logits)

        return router_logits

    def _route_tokens(self, router_logits: torch.Tensor):
        router_probs = nn.functional.softmax(router_logits, dim=-1, dtype=torch.float)

        topk_weight, topk_experts_index = torch.topk(router_probs, self.top_k, dim=-1)
        if self.norm_prob:
            topk_weight /= topk_weight.sum(dim=-1, keepdim=True)
        return topk_weight, topk_experts_index

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_logits = self._compute_router_probabilities(hidden_states)
        router_probs, topk_experts_index = self._route_tokens(router_logits)
        return router_probs, router_logits, topk_experts_index


class _MLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = F.silu

    def forward_w1_fn(self, x):
        return self.act_fn(self.gate_proj(x))

    def forward_w2_fn(self, x, w1_fn_input_x):
        x = x * self.up_proj(w1_fn_input_x)
        x = self.down_proj(x)
        return x

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class _SparseMLP(nn.Module):
    def __init__(self, config, use_expert_parallelism=True):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.num_shared_expert = config.num_shared_expert
        self.intermediate_size = config.intermediate_size
        self.use_expert_parallelism = use_expert_parallelism

        self.shared_expert_overlapping = config.shared_expert_overlapping

        self.router = TopNRouter(config, config.top_k)

        self.experts = Grouped_GEMM_MoE(
            hidden_size=config.hidden_size,
            expert_intermediate_size=self.intermediate_size,
            output_dropout_prob=0.1,
            num_experts=self.num_experts,
            topk=config.top_k,
            use_swiglu=True,
            use_bias=False,
            initializer_range=config.initializer_range,
            use_expert_parallelism=use_expert_parallelism,
            expert_parallel_group=None,
        )

        if config.num_shared_expert > 0:
            self.shared_experts = _MLP(config, intermediate_size=self.intermediate_size * config.num_shared_expert)
        else:
            self.shared_experts = None

    def forward(self, hidden_states):
        router_probs, router_logits, top_expert_index = self.router(hidden_states)
        identify = hidden_states

        if self.shared_experts is not None and self.use_expert_parallelism:
            hidden_shape = hidden_states.shape
            temp_hidden_states = hidden_states.view(-1, hidden_shape[-1])
            if self.shared_expert_overlapping:
                shared_experts_fn = (self.shared_experts.forward_w1_fn, self.shared_experts.forward_w2_fn)
                se_fn2_additional_input = temp_hidden_states
            else:
                shared_experts_fn = (None, None)
                se_output = self.shared_experts(temp_hidden_states)
                se_output = se_output.view(hidden_shape)
                se_fn2_additional_input = None

            hidden_states = self.experts(
                hidden_states,
                router_probs,
                top_expert_index,
                *shared_experts_fn,
                se_fn2_additional_input=se_fn2_additional_input,
            )

            if not self.shared_expert_overlapping:
                hidden_states = hidden_states + se_output
        else:
            hidden_states = self.experts(hidden_states, router_probs, top_expert_index)
        hidden_states = hidden_states.to(identify.dtype)

        if self.shared_experts is not None and not self.use_expert_parallelism:
            hidden_states = hidden_states + self.shared_experts(identify)

        return hidden_states


modeling_llama.LlamaMLP = _SparseMLP


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


def pre_create_group(ep_size):
    from atorch.common.util_func import divide
    from atorch.distributed.distributed import create_parallel_group

    world_size = torch.distributed.get_world_size()
    fsdp_mode = ([("data", torch.distributed.get_world_size())], None)
    create_parallel_group(fsdp_mode)
    ep_mode = ([("expert", ep_size), ("expert_fsdp", divide(world_size, ep_size))], None)
    create_parallel_group(ep_mode)


def parse_args():
    parser = argparse.ArgumentParser(description="Process arguments")
    parser.add_argument("--ep_size", type=int, default=1, required=False)
    parser.add_argument("--num_experts", type=int, default=8, required=False)
    parser.add_argument("--top_k", type=int, default=2, required=False)
    parser.add_argument("--num_shared_expert", type=int, default=2, required=False)
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
    parser.add_argument("--shared_expert_overlapping", default=False, action="store_true")
    parser.add_argument("--max_checkpoint_module_num", type=int, default=-1, required=False)
    parser.add_argument("--record_timeline", default=False, action="store_true")
    parser.add_argument("--timeline_dir", type=str, default="timeline_dir", required=False)

    return parser.parse_args()


def get_strategy(args):
    if args.ep_size > 1:
        strategy = []
    else:
        strategy = ["parallel_mode"]

    if args.use_module_replace:
        strategy.append("module_replace")
    if args.use_fsdp:
        atorch_wrap_cls = (LlamaDecoderLayer,)
        fsdp_config = {
            "sync_module_states": True,
            "limit_all_gathers": True,
            "forward_prefetch": True,
        }

        if args.ep_size > 1:
            from torch.distributed.fsdp.wrap import CustomPolicy

            from atorch.distributed.distributed import parallel_group

            experts_cls = Grouped_GEMM_MoE

            def moe_fsdp_policy_fn(module):
                if isinstance(module, atorch_wrap_cls):
                    # non experts fsdp wrap
                    return {"process_group": parallel_group("data")}
                elif isinstance(module, experts_cls):
                    # experts fsdp wrap
                    return {"process_group": parallel_group("expert_fsdp")}
                return False

            moe_fsdp_policy = CustomPolicy(moe_fsdp_policy_fn)
            fsdp_config["auto_wrap_policy"] = moe_fsdp_policy

        else:
            fsdp_config["atorch_wrap_cls"] = atorch_wrap_cls
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


def train_model(
    args,
    dataloader,
    optim,
    model,
    loss_func,
    total_batchsize,
    step_time_stats,
    throughput_stats,
    max_reserved_stats,
    max_allocated_stats,
    prof=None,
):
    global_step = 0
    start_time = time.time()
    device = "cuda"

    for batch in dataloader:
        optim.zero_grad()
        batch = prepare_input(batch, device)
        outputs = model(**batch)
        loss = loss_func(batch, outputs)
        loss.backward()
        optim.step()
        if prof is not None:
            prof.step()
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
    model_config.num_experts = args.num_experts
    model_config.num_shared_expert = args.num_shared_expert
    model_config.top_k = args.top_k
    model_config.shared_expert_overlapping = args.shared_expert_overlapping

    st = time.time()
    if args.ep_size > 1:
        pre_create_group(args.ep_size)
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

    step_time_stats = []
    throughput_stats = []
    max_reserved_stats = []
    max_allocated_stats = []

    if args.record_timeline:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=5,  # Skip first N steps, profiler is disabled
                warmup=5,  # Warmup steps, profiler is enabled, but results are discarded
                active=5,  # Profiler works and records events.
                repeat=1,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(args.timeline_dir),
        ) as prof:
            train_model(
                args,
                dataloader,
                optim,
                model,
                loss_func,
                total_batchsize,
                step_time_stats,
                throughput_stats,
                max_reserved_stats,
                max_allocated_stats,
                prof=prof,
            )
    else:
        train_model(
            args,
            dataloader,
            optim,
            model,
            loss_func,
            total_batchsize,
            step_time_stats,
            throughput_stats,
            max_reserved_stats,
            max_allocated_stats,
        )

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
