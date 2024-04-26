import functools
import os
import shutil
import unittest

import torch
import torch.multiprocessing as mp
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.utils.data import Dataset
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaForCausalLM

import atorch
from atorch.auto import auto_accelerate
from atorch.common.util_func import data_to_device, find_free_port
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


def compare_dicts(dict1, dict2):
    if len(dict1) != len(dict2):
        return False

    for key in dict1:
        if key not in dict2:
            return False
        val1 = dict1[key]
        val2 = dict2[key]
        if isinstance(val1, dict) and isinstance(val2, dict):
            if not compare_dicts(val1, val2):
                return False
        elif isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
            if not torch.equal(val1, val2):
                return False
        else:
            if val1 != val2:
                return False
    return True


def compare_fsdp_modules(fsdp_modules1, fsdp_modules2):
    if len(fsdp_modules1) != len(fsdp_modules2):
        return False
    for i in range(len(fsdp_modules2)):
        if fsdp_modules1[i].global_step != fsdp_modules2[i].global_step:
            return False
        if fsdp_modules1[i].use_outer_optim != fsdp_modules2[i].use_outer_optim:
            return False
        if fsdp_modules1[i].use_outer_optim and fsdp_modules1[i].outer_optimizer is not None:
            if not compare_dicts(
                fsdp_modules1[i].outer_optimizer.state_dict(), fsdp_modules2[i].outer_optimizer.state_dict()
            ):
                return False
            if not torch.equal(fsdp_modules1[i].last_synced_params, fsdp_modules2[i].last_synced_params):
                return False
    return True


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
    from torch.distributed.fsdp import FullOptimStateDictConfig, FullStateDictConfig, StateDictType

    state_dict_context = FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
    )

    with state_dict_context:
        model_state_dict = model.state_dict()
        optim_state_dict = FSDP.optim_state_dict(model, optim)

        dp_rank = torch.distributed.get_rank()

        if dp_rank == 0 and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        if dp_rank == 0:
            save_model_full_path = os.path.join(save_dir, "model.pt")
            torch.save(model_state_dict, save_model_full_path)
            opt_save_full_path = os.path.join(save_dir, "optimizer.pt")
            torch.save(optim_state_dict, opt_save_full_path)

    FSDP.save_local_sgd_state_dict(
        model=model,
        rank0_only=True,
        full_state_dict=True,
        cpu_offload=True,
        save_dir=os.path.join(save_dir, "full_rank0"),
    )

    FSDP.save_local_sgd_state_dict(
        model=model,
        rank0_only=False,
        full_state_dict=True,
        cpu_offload=True,
        save_dir=os.path.join(save_dir, "full_not_rank0"),
    )

    FSDP.save_local_sgd_state_dict(
        model=model,
        full_state_dict=False,
        cpu_offload=True,
        save_dir=os.path.join(save_dir, "sharded"),
    )


def run_hsdp_local_sgd(
    rank, world_size, local_sgd_warmup_steps, outer_optim_class, save_dir, experiment_name, reducer=None
):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["NPROC_PER_NODE"] = str(world_size)
    atorch.init_distributed("nccl", set_cuda_device_using_local_rank=True)
    device = torch.cuda.current_device()

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
    parallel_config = ([("zero", torch.distributed.get_world_size() // 2), ("data", 2)], None)
    strategy = [("parallel_mode", parallel_config)]

    fsdp_config = {
        "sync_module_states": True,
        "limit_all_gathers": True,
        "atorch_wrap_cls": (LlamaDecoderLayer,),
        "use_local_sgd": True,
        "local_sgd_sync_interval": 5,
        "local_sgd_warmup_steps": local_sgd_warmup_steps,
        "gradient_accumulation_steps": 1,
        "outer_optim_class": torch.optim.SGD if outer_optim_class == "sgd" else None,
        "outer_optim_kwargs": {
            "lr": 0.9,
            "momentum": 0.8,
            "nesterov": True,
        },
        "outer_optim_cpu_offload": True,
    }
    if reducer is not None:
        # test for gta sum sufficies
        fsdp_config.update(
            {
                "reducer": reducer,
                "consensus_method": "sum",
                "sparsification_method": None,
                "normalize": True,
                "density": None,
                "int8_mask": False,
            }
        )
    strategy.append(("fsdp", fsdp_config))

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

    for batch in dataloader:
        optim.zero_grad()
        batch = prepare_input(batch, device)
        outputs = model(**batch)
        loss = loss_func(batch, outputs)
        loss.backward()
        optim.step()

    # Test checkpoint methods
    save_ckpt(model, optim, os.path.join(save_dir, experiment_name))
    # model checkpoints
    new_res = create_test_model(
        model_config, os.path.join(save_dir, experiment_name, "model.pt"), dataset, loss_func, dataloader_args, strategy
    )
    new_model = new_res.model

    model_param = dict(model.named_parameters())
    new_model_param = dict(new_model.named_parameters())

    for name1, param1 in model_param.items():
        if name1 in new_model_param:
            param2 = new_model_param[name1]
            assert torch.equal(param1, param2)

    # local sgd checkpoints
    model_fsdp_modules = FSDP.fsdp_modules(model)
    ## full rank0 settings
    test_res = create_test_model(
        model_config, os.path.join(save_dir, experiment_name, "model.pt"), dataset, loss_func, dataloader_args, strategy
    )
    test_model = test_res.model
    FSDP.load_local_sgd_state_dict(
        test_model,
        rank0_only=True,
        full_state_dict=True,
        load_dir=os.path.join(save_dir, experiment_name, "full_rank0"),
    )
    test_model_fsdp_modules = FSDP.fsdp_modules(test_model)
    assert compare_fsdp_modules(test_model_fsdp_modules, model_fsdp_modules)

    ## full not-rank0 settings
    test_res = create_test_model(
        model_config, os.path.join(save_dir, experiment_name, "model.pt"), dataset, loss_func, dataloader_args, strategy
    )
    test_model = test_res.model
    FSDP.load_local_sgd_state_dict(
        test_model,
        rank0_only=False,
        full_state_dict=True,
        load_dir=os.path.join(save_dir, experiment_name, "full_not_rank0"),
    )
    test_model_fsdp_modules = FSDP.fsdp_modules(test_model)
    assert compare_fsdp_modules(test_model_fsdp_modules, model_fsdp_modules)

    ## sharded settings
    test_res = create_test_model(
        model_config, os.path.join(save_dir, experiment_name, "model.pt"), dataset, loss_func, dataloader_args, strategy
    )
    test_model = test_res.model
    FSDP.load_local_sgd_state_dict(
        test_model,
        full_state_dict=False,
        load_dir=os.path.join(save_dir, experiment_name, "sharded"),
    )
    test_model_fsdp_modules = FSDP.fsdp_modules(test_model)
    assert compare_fsdp_modules(test_model_fsdp_modules, model_fsdp_modules)


class TestHSDPLocalSGD(unittest.TestCase):
    def setUp(self):
        self.save_dir = "/tmp/local_sgd_ckpt/"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def tearDown(self):
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)

    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.device_count() < 4 or torch_version() != (2, 1, 0),  # type: ignore
        "Must have at least 4 GPUs and torch == 2.1.0. for local sgd test.",
    )
    def test_hsdp_local_sgd(self):
        world_size = 4
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(
            run_hsdp_local_sgd,
            args=(world_size, 0, "none", self.save_dir, "local_sgd"),
            nprocs=world_size,
            join=True,
        )
        os.environ["MASTER_ADDR"] = ""
        os.environ["MASTER_PORT"] = ""

    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.device_count() < 4 or torch_version() != (2, 1, 0),  # type: ignore
        "Must have at least 4 GPUs and torch == 2.1.0. for local sgd test.",
    )
    def test_hsdp_post_local_sgd(self):
        world_size = 4
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(
            run_hsdp_local_sgd,
            args=(world_size, 10, "none", self.save_dir, "post_local_sgd"),
            nprocs=world_size,
            join=True,
        )
        os.environ["MASTER_ADDR"] = ""
        os.environ["MASTER_PORT"] = ""

    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.device_count() < 4 or torch_version() != (2, 1, 0),  # type: ignore
        "Must have at least 4 GPUs and torch == 2.1.0. for local sgd test.",
    )
    def test_hsdp_diloco(self):
        world_size = 4
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(
            run_hsdp_local_sgd,
            args=(world_size, 0, "sgd", self.save_dir, "diloco"),
            nprocs=world_size,
            join=True,
        )
        os.environ["MASTER_ADDR"] = ""
        os.environ["MASTER_PORT"] = ""

    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.device_count() < 4 or torch_version() != (2, 1, 0),  # type: ignore
        "Must have at least 4 GPUs and torch == 2.1.0. for local sgd test.",
    )
    def test_hsdp_diloco_gta_sum(self):
        world_size = 4
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(
            run_hsdp_local_sgd,
            args=(world_size, 0, "sgd", self.save_dir, "diloco_gta", "gta"),
            nprocs=world_size,
            join=True,
        )
        os.environ["MASTER_ADDR"] = ""
        os.environ["MASTER_PORT"] = ""

    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.device_count() < 4 or torch_version() != (2, 1, 0),  # type: ignore
        "Must have at least 4 GPUs and torch == 2.1.0. for local sgd test.",
    )
    def test_hsdp_post_diloco(self):
        world_size = 4
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(
            run_hsdp_local_sgd,
            args=(world_size, 10, "sgd", self.save_dir, "post_diloco"),
            nprocs=world_size,
            join=True,
        )
        os.environ["MASTER_ADDR"] = ""
        os.environ["MASTER_PORT"] = ""


if __name__ == "__main__":
    unittest.main()
