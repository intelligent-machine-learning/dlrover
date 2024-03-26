import copy
import os
import tempfile
import unittest

import deepspeed
import torch
import torch.multiprocessing as mp
from transformers import OPTConfig, OPTForCausalLM

from atorch.common.util_func import find_free_port
from atorch.rl.ds_hybrid_engine.initialize import initialize
from atorch.rl.model_utils.load_init_model import load_ds_model_with_zero3_partition
from atorch.tests.utils.test_utils import init_dist

try:
    from transformers import MixtralConfig, MixtralForCausalLM
except Exception:
    MixtralConfig = None
    MixtralForCausalLM = None


def _load_ds_model_with_zero3_partition(rank, world_size):
    init_dist(rank, world_size)
    torch.manual_seed(0)
    opt_config = OPTConfig()
    opt_model = OPTForCausalLM(opt_config)
    opt_model_copy = copy.deepcopy(opt_model).to(rank)
    folder = tempfile.mkdtemp()
    model_class = OPTForCausalLM
    model_config_class = OPTConfig
    model_config_folder = folder
    opt_model.save_pretrained(folder)
    ds_config = {"train_batch_size": 4}
    model = load_ds_model_with_zero3_partition(
        model_class, model_config_class, model_config_folder=model_config_folder, ds_config_dict_or_path=ds_config
    )
    with deepspeed.zero.GatheredParameters(list(model.parameters(recurse=True)), modifier_rank=0):
        for p1, p2 in zip(model.parameters(), opt_model_copy.parameters()):
            assert torch.allclose(p1, p2, rtol=1e-05, atol=1e-08)
    assert model is not None
    # opt model tie lm_head.weight with embed_tokens.weight by default
    config = {
        "train_batch_size": 32,
        "train_micro_batch_size_per_gpu": 2,
        "steps_per_print": 10,
        "zero_optimization": {"stage": 3, "stage3_param_persistence_threshold": 0},
        "bf16": {
            "enabled": True,
        },
        "gradient_clipping": 1.0,
    }
    kwargs = {}
    kwargs["config"] = config
    kwargs["model"] = model
    m, _, _, _ = deepspeed.initialize(**kwargs)
    assert m.model.decoder.embed_tokens.weight.ds_tensor.data_ptr() == m.lm_head.weight.ds_tensor.data_ptr()


@unittest.skipIf(torch.cuda.device_count() < 2, "run with gpu_num >=2")
class TestLoadDSModel(unittest.TestCase):
    def test_load_ds_model_with_zero3_partition(self):
        world_size = 2
        os.environ["MASTER_ADDR"] = "localhost"  #
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(
            _load_ds_model_with_zero3_partition,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )


def _load_ds_model_with_zero3_partition_mixtral(rank, world_size):
    init_dist(rank, world_size)
    torch.manual_seed(0)
    mixtral_config = MixtralConfig()
    mixtral_config.num_hidden_layers = 2
    mixtral_model = MixtralForCausalLM(mixtral_config)
    mixtral_model_copy = copy.deepcopy(mixtral_model).to(rank)
    folder = tempfile.mkdtemp()
    model_class = MixtralForCausalLM
    model_config_class = MixtralConfig
    model_config_folder = folder
    mixtral_model.save_pretrained(folder)
    ds_config = {"train_batch_size": 4}
    model = load_ds_model_with_zero3_partition(
        model_class, model_config_class, model_config_folder=model_config_folder, ds_config_dict_or_path=ds_config
    )
    with deepspeed.zero.GatheredParameters(list(model.parameters(recurse=True)), modifier_rank=0):
        for p1, p2 in zip(model.parameters(), mixtral_model_copy.parameters()):
            assert torch.allclose(p1, p2, rtol=1e-05, atol=1e-08)
    assert model is not None
    config = {
        "train_batch_size": 32,
        "train_micro_batch_size_per_gpu": 2,
        "steps_per_print": 10,
        "zero_optimization": {"stage": 3, "stage3_param_persistence_threshold": 0},
        "bf16": {
            "enabled": True,
        },
        "gradient_clipping": 1.0,
    }
    kwargs = {}
    kwargs["config"] = config
    kwargs["model"] = model
    m, _, _, _ = initialize(**kwargs)


@unittest.skipIf(True, "works with transformers>=transformers==4.36.2")
class TestLoadDSMixtralModel(unittest.TestCase):
    def test_load_ds_model_with_zero3_partition_mixtral(self):
        world_size = 2
        os.environ["MASTER_ADDR"] = "localhost"  #
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(
            _load_ds_model_with_zero3_partition_mixtral,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )
