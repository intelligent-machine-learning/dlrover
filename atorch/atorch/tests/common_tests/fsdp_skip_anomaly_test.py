import functools
import os
import unittest

import torch
import torch.multiprocessing as mp
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


def run_fsdp_skip_anomaly(rank, world_size):
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
    # technically no difference between hsdp and fsdp for this purpose
    parallel_config = ([("zero", torch.distributed.get_world_size() // 2), ("data", 2)], None)
    strategy = [("parallel_mode", parallel_config)]

    from atorch.local_sgd.HSDP.configs import AnomalyConfigs

    fsdp_config = {
        "sync_module_states": True,
        "limit_all_gathers": True,
        "atorch_wrap_cls": (LlamaDecoderLayer,),
        "anomaly_configs": AnomalyConfigs(
            skip_anomaly=True, ewma_alpha=0.02, ewma_warmup_steps=1, ewma_threshold=0.1
        ),  # trick threshold
    }
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

    # just run for several rounds is fine
    for batch in dataloader:
        optim.zero_grad()
        batch = prepare_input(batch, device)
        outputs = model(**batch)
        loss = loss_func(batch, outputs)
        loss.backward()
        optim.step()


class TestFSDPSkipAnomaly(unittest.TestCase):
    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.device_count() < 4 or torch_version()[:2] != (2, 1),  # type: ignore
        "Must have at least 4 GPUs and torch == 2.1.x. for local sgd test.",
    )
    def test_fsdp_skip_anomaly(self):
        world_size = 4
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(
            run_fsdp_skip_anomaly,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )
        os.environ["MASTER_ADDR"] = ""
        os.environ["MASTER_PORT"] = ""


if __name__ == "__main__":
    unittest.main()
