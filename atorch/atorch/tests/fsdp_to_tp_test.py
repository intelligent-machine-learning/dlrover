import os
import unittest

import torch
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import atorch
from atorch.common.log_utils import default_logger as logger
from atorch.distributed.distributed import create_parallel_group
from atorch.modules.distributed_modules.layers import RowParallelLinear
from atorch.modules.distributed_modules.transformer import MegatronGLMModel
from atorch.tests.glm.modeling_glm import GLMConfig, GLMModel

logger.setLevel("INFO")
os.environ["NCCL_DEBUG"] = "ERROR"


def init_dist(rank, world_size):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["NPROC_PER_NODE"] = str(world_size)
    if torch.cuda.is_available():
        atorch.init_distributed("nccl")
    else:
        atorch.init_distributed("gloo")


def _run_linear_fsdp_to_tp(rank, world_size):
    init_dist(rank, world_size)
    config = [("tensor", world_size)]
    create_parallel_group((config, None))
    pg, ranks = atorch.distributed.distributed.parallel_group_and_ranks("tensor")

    device = torch.device(rank)
    linear = torch.nn.Linear(4, 2).to(device)
    torch.cuda.set_device(device)
    sharded_module = FSDP(linear).to(device)
    input_x = torch.tensor([[1.0, 2.0, 2.0, 2.0], [1.0, 2.0, 2.0, 2.0], [1.0, 2.0, 2.0, 2.0], [1.0, 2.0, 2.0, 2.0]]).to(
        device
    )

    print(input_x)
    with FSDP.summon_full_params(sharded_module):
        weight = sharded_module.weight
        print("shard", weight.device)
        row_parallel_linear = RowParallelLinear(
            orig_module=sharded_module, process_group=pg, ranks=ranks, defer_init=False
        )
        row_parallel_linear.to(device)
        print("row", row_parallel_linear.weight.device)
        if rank == 0:
            assert torch.norm(weight[:, 0:2].to(device) - row_parallel_linear.weight, p=-1) == 0

    atorch.reset_distributed()


def _run_megatron_glm_fsdp_to_tp(rank, world_size):
    init_dist(rank, world_size)
    config = [("tensor", world_size)]
    create_parallel_group((config, None))
    pg, ranks = atorch.distributed.distributed.parallel_group_and_ranks("tensor")

    device = torch.device(rank)

    config = GLMConfig()
    torch.manual_seed(0)
    glm_model = GLMModel(config).to(device)

    class FakeInput:
        input_ids = torch.ones((4, 10), dtype=torch.long).to(device)
        attention_mask = torch.ones((4, 10)).to(device)

    glm_model.eval()
    res1 = glm_model(input_ids=FakeInput.input_ids, attention_mask=FakeInput.attention_mask)

    with FSDP.summon_full_params(glm_model):

        megatron_glm_model = MegatronGLMModel(
            glm_model.config,
            orig_module=glm_model,
            process_group=pg,
            ranks=ranks,
            defer_init=False,
            orig_module_dst_device="cpu",
        ).to(device)
        megatron_glm_model.eval()
        res2 = megatron_glm_model(input_ids=FakeInput.input_ids, attention_mask=FakeInput.attention_mask)

    res = torch.norm(res1.last_hidden_states - res2.last_hidden_states, p=-1)
    assert res == 0
    atorch.reset_distributed()


class TestMegatronLinearFSDPToTP(unittest.TestCase):
    @unittest.skipIf(torch.cuda.device_count() < 2, "gpu_num >=2")
    def test_run_linear_fsdp_to_tp(self):

        world_size = 2
        os.environ["MASTER_ADDR"] = "localhost"  #
        os.environ["MASTER_PORT"] = "5000"
        mp.spawn(
            _run_linear_fsdp_to_tp,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )
        os.environ["MASTER_ADDR"] = ""
        os.environ["MASTER_PORT"] = ""


class TestMegatronGLMFSDPToTP(unittest.TestCase):
    @unittest.skipIf(torch.cuda.device_count() < 2, "gpu_num >=2")
    def test_run_megatron_glm_fsdp_to_tp(self):

        world_size = 2
        os.environ["MASTER_ADDR"] = "localhost"  #
        os.environ["MASTER_PORT"] = "5000"
        mp.spawn(
            _run_megatron_glm_fsdp_to_tp,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )
        os.environ["MASTER_ADDR"] = ""
        os.environ["MASTER_PORT"] = ""


if __name__ == "__main__":
    unittest.main()
