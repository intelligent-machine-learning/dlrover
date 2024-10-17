import functools
import os
import shutil
import time
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook import PostLocalSGDState, post_localSGD_hook
from torch.utils.data import Dataset
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM

import atorch
from atorch.auto import auto_accelerate
from atorch.common.util_func import data_to_device, find_free_port
from atorch.distributed.distributed import parallel_group
from atorch.utils.version import torch_version


class LlamaDataset(Dataset):
    def __init__(
        self,
        vocab_size=10000,
        max_source_positions=1024,
        total_samples=10000,
    ):
        self.vocab_size = vocab_size
        self.max_source_positions = max_source_positions
        self.total_samples = total_samples
        self.sizes = [self.max_source_positions] * self.total_samples

    def __getitem__(self, index):
        length = self.sizes[index]
        source = torch.randint(1, self.vocab_size, (length,))
        target = source.clone()
        return {
            "source": source,
            "target": target,
        }

    def __len__(self):
        return self.total_samples


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


def create_test_model(model_config, ckpt_path, dataset, loss_func, dataloader_args, strategy):
    new_model = LlamaForCausalLM(model_config)
    model_checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
    new_model.load_state_dict(model_checkpoint)

    new_status, new_res, _ = auto_accelerate(
        new_model,
        optim_func=torch.optim.AdamW,
        dataset=dataset,
        loss_func=loss_func,
        prepare_input=data_to_device,
        model_input_format="unpack_dict",
        optim_args={"lr": 0.001},
        optim_param_func=None,
        dataloader_args=dataloader_args,
        load_strategy=strategy,
        ignore_dryrun_on_load_strategy=True,
    )
    assert new_status
    return new_res


def save_ckpt(model, optim, save_dir):
    model_state_dict = model.state_dict()
    model_state_dict = (
        torch.utils._pytree.tree_map(lambda x: x.cpu() if isinstance(x, torch.Tensor) else x, model_state_dict)
        if model_state_dict is not None
        else model_state_dict
    )
    optim_state_dict = optim.state_dict()
    optim_state_dict = (
        torch.utils._pytree.tree_map(lambda x: x.cpu() if isinstance(x, torch.Tensor) else x, optim_state_dict)
        if optim_state_dict is not None
        else optim_state_dict
    )

    dp_rank = torch.distributed.get_rank()

    if dp_rank == 0 and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    if dp_rank == 0:
        save_model_full_path = os.path.join(save_dir, "model.pt")
        torch.save(model_state_dict, save_model_full_path)
        opt_save_full_path = os.path.join(save_dir, "optimizer.pt")
        torch.save(optim_state_dict, opt_save_full_path)


def run_ddp_local_sgd(
    rank,
    world_size,
    local_sgd_warmup_steps,
    outer_optim_class,
    save_dir,
    experiment_name,
    reducer=None,
    clip=None,
    cpu_offload=True,
):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["NPROC_PER_NODE"] = str(world_size)
    atorch.init_distributed("nccl", set_cuda_device_using_local_rank=True)
    device = torch.cuda.current_device()

    # HACK we have to import it here
    # Because this will further import from local_sgd.HSDP for configs
    # The HSDP import relies on torch.distributed._tensor which causes CPU test fail
    # TODO might have to move configs out of local_sgd.HSDP
    from atorch.local_sgd.DDP import OuterOptimPeriodicModelAverager, StatefulPostLocalSGDOptimizer

    model_config = LlamaConfig(
        hidden_size=64, num_attention_heads=4, num_hidden_layers=4, num_key_value_heads=4, max_position_embeddings=32
    )
    model = LlamaForCausalLM(model_config)

    def loss_func(_, output):
        return output.loss

    dataset = LlamaDataset(vocab_size=model_config.vocab_size, max_source_positions=32, total_samples=200)
    dataloader_args = {
        "batch_size": 4,
        "drop_last": True,
        "shuffle": True,
        "num_workers": 2,
        "collate_fn": functools.partial(collate_sentences_lm, input_name="input_ids", label_name="labels"),
    }

    # atorch auto accelerate
    parallel_config = ([("data", 2)], None)
    strategy = [("parallel_mode", parallel_config)]

    from atorch.local_sgd.HSDP import GTAConfigs, LocalSGDConfigs, OuterOptimizerConfigs

    local_sgd_configs = LocalSGDConfigs(
        local_sgd_sync_interval=5,
        local_sgd_warmup_steps=local_sgd_warmup_steps,
        gradient_accumulation_steps=1,
        clip_pseudo_grad=clip,
        cpu_offload=cpu_offload,
        skip_anomaly=True,
        ewma_warmup_steps=2,
    )
    outer_optim_configs = OuterOptimizerConfigs(
        outer_optim_class=torch.optim.SGD if outer_optim_class == "sgd" else None,
        outer_optim_kwargs={
            "lr": 0.9,
            "momentum": 0.8,
            "nesterov": True,
        },
    )
    gta_configs = None
    if reducer is not None:
        # test for gta sum sufficies
        gta_configs = GTAConfigs(
            reducer=reducer,
            consensus_method="sum",
            sparsification_method=None,
            normalize=True,
            density=None,
            int8_mask=False,
        )

    # auto_accelerate
    status, res, _ = auto_accelerate(
        model,
        optim_func=torch.optim.AdamW,
        dataset=dataset,
        loss_func=loss_func,
        prepare_input=data_to_device,
        model_input_format="unpack_dict",
        optim_args={"lr": 0.001},
        optim_param_func=None,
        dataloader_args=dataloader_args,
        load_strategy=strategy,
        ignore_dryrun_on_load_strategy=True,
    )
    assert status

    model = res.model
    optim = res.optim
    dataloader = res.dataloader
    loss_func = res.loss_func
    prepare_input = res.prepare_input
    local_sgd_state = PostLocalSGDState(
        process_group=parallel_group("data"),
        start_localSGD_iter=local_sgd_warmup_steps,
        subgroup=None,
        post_local_gradient_allreduce=False,
    )
    model.register_comm_hook(local_sgd_state, post_localSGD_hook)
    averager = OuterOptimPeriodicModelAverager(
        process_group=parallel_group("data"),
        local_sgd_config=local_sgd_configs,
        gta_config=gta_configs,
        outer_optim_config=outer_optim_configs,
    )

    optim = StatefulPostLocalSGDOptimizer(
        optim=optim,
        averager=averager,
    )

    for batch in dataloader:
        optim.zero_grad()
        batch = prepare_input(batch, device)
        outputs = model(**batch)
        loss = loss_func(batch, outputs)
        loss.backward()
        optim.step()

    # Test checkpoint methods
    save_ckpt(model.module, optim, os.path.join(save_dir, experiment_name))
    load_model_path = os.path.join(save_dir, experiment_name, "model.pt")
    while not os.path.exists(load_model_path):
        time.sleep(1)
    dist.barrier()
    # model checkpoints
    new_res = create_test_model(model_config, load_model_path, dataset, loss_func, dataloader_args, strategy)
    new_model = new_res.model
    new_optim = new_res.optim

    new_local_sgd_state = PostLocalSGDState(
        process_group=parallel_group("data"),
        start_localSGD_iter=local_sgd_warmup_steps,
        subgroup=None,
        post_local_gradient_allreduce=False,
    )
    new_model.register_comm_hook(new_local_sgd_state, post_localSGD_hook)
    new_averager = OuterOptimPeriodicModelAverager(
        process_group=parallel_group("data"),
        local_sgd_config=local_sgd_configs,
        gta_config=gta_configs,
        outer_optim_config=outer_optim_configs,
    )

    new_optim = StatefulPostLocalSGDOptimizer(
        optim=new_optim,
        averager=new_averager,
    )

    new_optim.load_state_dict(torch.load(os.path.join(save_dir, experiment_name, "optimizer.pt")))

    for batch in dataloader:
        new_optim.zero_grad()
        batch = prepare_input(batch, device)
        outputs = new_model(**batch)
        loss = loss_func(batch, outputs)
        loss.backward()
        new_optim.step()


class TestDDPLocalSGD(unittest.TestCase):
    def setUp(self):
        self.save_dir = "/tmp/local_sgd_ckpt/"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def tearDown(self):
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)

    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2 or torch_version()[:2] != (2, 1),  # type: ignore
        "Must have at least 2 GPUs and torch == 2.1.x. for local sgd test.",
    )
    def test_ddp_diloco_gta_sum(self):
        world_size = 2
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(
            run_ddp_local_sgd,
            args=(world_size, 0, "sgd", self.save_dir, "diloco_gta", "gta"),
            nprocs=world_size,
            join=True,
        )
        os.environ["MASTER_ADDR"] = ""
        os.environ["MASTER_PORT"] = ""


if __name__ == "__main__":
    unittest.main()
