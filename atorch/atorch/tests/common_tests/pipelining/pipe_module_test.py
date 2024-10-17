import os
import unittest

import torch
import torch.multiprocessing as mp

import atorch
from atorch.common.util_func import find_free_port
from atorch.distributed import distributed as _distributed_context
from atorch.distributed.distributed import create_parallel_group, destroy_parallel_group
from atorch.pipeline_parallel.pipe_module import PipeModule, PipeModuleConfig, make_pipe_module
from atorch.pipeline_parallel.pipe_partition import TieWeightInfo
from atorch.tests.toy_modules.toy_module import (
    decoder_loss_func,
    get_llama_config,
    get_llama_input_output_mapping,
    get_llama_model_chunk,
)
from atorch.utils.version import torch_version

skip = None
if torch_version() >= (2, 4, 0):  # type: ignore
    skip = False
else:
    skip = True


def module_create_fn_pipeline(rank):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    res = atorch.init_distributed(backend, set_cuda_device_using_local_rank=backend == "nccl")
    assert res
    world_size = torch.distributed.get_world_size()
    gpu_partition = ([("pipe", world_size)], None)
    create_parallel_group(gpu_partition)

    model_config = get_llama_config()

    config = PipeModuleConfig()
    config.model_config = model_config
    config.total_layer_num = world_size * 2
    config.virtual_pp_size = 2
    config.input_output_mapping = get_llama_input_output_mapping()

    pipe_module = make_pipe_module(
        model_provider=get_llama_model_chunk,
        loss_func=decoder_loss_func,
        distributed_context=_distributed_context,
        config=config,
    )

    assert isinstance(pipe_module, PipeModule)
    assert len(pipe_module.modules) == config.virtual_pp_size
    assert pipe_module.modules[0].layer_num == 1

    destroy_parallel_group()
    gpu_partition = ([("pipe", world_size // 2), ("data", 2)], None)
    create_parallel_group(gpu_partition)

    # when world_size=4, pp size is 2, vpp=2, total stage=4.
    # total_layer_num=8, thus average 2 layers per stage.
    config.manual_stage_partition = [1, 3, 3, 1]
    pipe_module = make_pipe_module(
        model_provider=get_llama_model_chunk,
        loss_func=decoder_loss_func,
        distributed_context=_distributed_context,
        config=config,
    )
    assert len(pipe_module.modules) == config.virtual_pp_size
    if rank == 0 or rank == 2:
        assert pipe_module.modules[0].module.layer_num == 1
        assert pipe_module.modules[1].module.layer_num == 3
    else:
        assert pipe_module.modules[0].module.layer_num == 3
        assert pipe_module.modules[1].module.layer_num == 1

    torch.distributed.barrier()
    atorch.reset_distributed()


class PipeModuleTest(unittest.TestCase):
    def test_tie_weight_info(self):
        tw_info = TieWeightInfo()
        self.assertTrue(tw_info.num() == 0)
        info0 = [(0, "embedding"), (3, "embedding")]
        info1 = ["layer0.weight", "layer8.weight", "layer10.weight"]
        tw_info.add(info0)
        tw_info.add(info1)
        self.assertTrue(tw_info.num() == 2)
        self.assertEqual(tw_info[0], info0)
        self.assertEqual(tw_info[1], info1)

    @unittest.skipIf(torch.cuda.device_count() < 4 or skip, "Requires 4 gpus.")
    def test_pp_module(self):
        world_size = 4
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["WORLD_SIZE"] = str(world_size)
        mp.spawn(
            module_create_fn_pipeline,
            nprocs=world_size,
            join=True,
        )
