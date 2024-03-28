import json
import os
import tempfile
import unittest

import torch
import torch.multiprocessing as mp
from deepspeed import DeepSpeedEngine
from deepspeed.ops.adam import DeepSpeedCPUAdam

import atorch
from atorch.common.util_func import find_free_port
from atorch.rl.config import AtorchRLConfig
from atorch.rl.model_engine import ModelEngine
from atorch.utils.version import torch_version


def init_dist(rank, world_size):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["NPROC_PER_NODE"] = str(world_size)

    atorch.init_distributed("nccl")
    torch.cuda.device(atorch.local_rank())


def _run_test_model_engine_load_init_model(rank, world_size):
    init_dist(rank, world_size)
    config_path = "./atorch/tests/test_define_rl_models/independent_models/model_def.yaml"
    atorch_rl_config = AtorchRLConfig.load_yaml(config_path)
    model_engine = ModelEngine(atorch_rl_config)
    assert model_engine is not None
    assert model_engine.models_strategies["actor"][1] == "amp_native"
    assert model_engine.models_strategies["actor"][2] == ("fsdp")
    model_engine.apply_strategy_to_child_model("critic")
    assert isinstance(model_engine.auto_accelerated_models["critic"], torch.nn.Module)
    assert model_engine.models_strategies["critic"] == "torch_native"
    critic_optimizer = model_engine.auto_accelerated_optimizer["critic"]
    assert issubclass(type(critic_optimizer), torch.optim.Optimizer)
    model_engine.apply_strategy_to_child_model("reward_model")
    assert isinstance(model_engine.auto_accelerated_models["reward_model"], torch.nn.Module)
    model_engine.apply_strategy_to_child_model("ref_model")
    assert isinstance(model_engine.auto_accelerated_models["ref_model"], torch.nn.Module)
    model_engine.apply_strategy_to_child_model("reward_model")
    assert isinstance(model_engine.auto_accelerated_models["reward_model"], torch.nn.Module)
    atorch.reset_distributed()


def _model_engine_apply_strategy(rank, world_size):
    init_dist(rank, world_size)
    # TODO: test whether the strategies are applied successfully
    # and models are wrapped by correspondding backbend
    config_path = "./atorch/tests/test_define_rl_models/independent_models/model_def.yaml"
    atorch_rl_config = AtorchRLConfig.load_yaml(config_path)
    model_engine = ModelEngine(atorch_rl_config)
    model_engine.models_strategies["actor"] = ["amp_native", ("fsdp", {"atorch_wrap_cls": {"Linear"}})]
    model_engine.apply_strategy_to_child_model("actor")
    actor = model_engine.auto_accelerated_models["actor"]
    assert isinstance(actor.linear, torch.distributed.fsdp.fully_sharded_data_parallel.FullyShardedDataParallel)


def _model_engine_support_ds_backend_with_offload_optimizer_and_param(rank, world_size):
    init_dist(rank, world_size)
    config_path = "./atorch/tests/test_define_rl_models/independent_models/model_def.yaml"
    atorch_rl_config = AtorchRLConfig.load_yaml(config_path)
    model_engine = ModelEngine(atorch_rl_config)
    # ds_config
    ds_json = {
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 1,
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {"device": "cpu"},
            "offload_param": {"device": "cpu"},
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
        },
        "gradient_clipping": 1.0,
    }
    folder = tempfile.TemporaryDirectory()
    config_path = os.path.join(folder.name, "ds_config.json")
    with open(config_path, "w") as f:
        json.dump(ds_json, f)
    model_type = "actor"
    model = torch.nn.Linear(2, 3)
    model_engine.models_strategies[model_type] = config_path
    model_engine.models[model_type] = model

    model_engine.apply_strategy_to_child_model(model_type)
    assert isinstance(model_engine.auto_accelerated_models[model_type], DeepSpeedEngine)
    for i in range(10):
        inputs = torch.ones(4, 2).to(atorch.local_rank()).bfloat16()
        outputs = model_engine.auto_accelerated_models[model_type](inputs)
        loss = outputs.sum()
        model_engine.auto_accelerated_models[model_type].backward(loss)
        model_engine.auto_accelerated_models[model_type].step()
    assert isinstance(model_engine.auto_accelerated_models[model_type].client_optimizer, DeepSpeedCPUAdam)


def _model_engine_support_ds_backend_with_offload_param(rank, world_size):
    init_dist(rank, world_size)
    config_path = "./atorch/tests/test_define_rl_models/independent_models/model_def.yaml"
    atorch_rl_config = AtorchRLConfig.load_yaml(config_path)
    model_engine = ModelEngine(atorch_rl_config)
    # ds_config
    ds_json = {
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 1,
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 2,
            "offload_param": {"device": "cpu"},
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
        },
        "gradient_clipping": 1.0,
    }

    folder = tempfile.TemporaryDirectory()
    config_path = os.path.join(folder.name, "ds_config.json")
    with open(config_path, "w") as f:
        json.dump(ds_json, f)
    model_type = "actor"
    model = torch.nn.Linear(2, 3)
    model_engine.models_strategies[model_type] = config_path
    model_engine.models[model_type] = model
    model_engine.apply_strategy_to_child_model(model_type)
    assert isinstance(model_engine.auto_accelerated_models[model_type], DeepSpeedEngine)
    for i in range(10):
        inputs = torch.ones(4, 2).to(atorch.local_rank()).bfloat16()
        outputs = model_engine.auto_accelerated_models[model_type](inputs)
        loss = outputs.sum()
        model_engine.auto_accelerated_models[model_type].backward(loss)
        model_engine.auto_accelerated_models[model_type].step()
    assert isinstance(model_engine.auto_accelerated_models[model_type].client_optimizer, torch.optim.Adam)


@unittest.skipIf(torch.cuda.device_count() < 2 or torch_version() < (2, 0, 0), "run with gpu_num >=2")  # type: ignore
class TestModelEngine(unittest.TestCase):
    def test_run_test_model_engine_load_init_model(self):
        os.environ["MASTER_ADDR"] = "localhost"  #
        os.environ["MASTER_PORT"] = str(find_free_port())
        world_size = 2
        mp.spawn(
            _run_test_model_engine_load_init_model,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )

    def test_model_engine_apply_strategy(self):
        world_size = 2
        os.environ["MASTER_ADDR"] = "localhost"  #
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(
            _model_engine_apply_strategy,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )

    def test_model_engine_support_ds_backend_with_offload_param(self):
        world_size = 2
        os.environ["MASTER_ADDR"] = "localhost"  #
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(
            _model_engine_support_ds_backend_with_offload_param,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )

    def test_model_engine_support_ds_backend_with_offload_optimizer_and_param(self):
        world_size = 2
        os.environ["MASTER_ADDR"] = "localhost"  #
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(
            _model_engine_support_ds_backend_with_offload_optimizer_and_param,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )


if __name__ == "__main__":
    unittest.main()
