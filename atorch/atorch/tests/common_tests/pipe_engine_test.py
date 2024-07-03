import copy
import functools
import os
import unittest

import torch
import torch.multiprocessing as mp

import atorch
from atorch.common.util_func import data_to_device, find_free_port
from atorch.distributed import distributed as _distributed_context
from atorch.distributed.distributed import (
    create_parallel_group,
    destroy_parallel_group,
    get_data_partition_rank_and_size,
)
from atorch.pipeline_parallel.pipe_engine import PipeEngine
from atorch.pipeline_parallel.pipe_module import PipeModule, PipeModuleConfig, make_pipe_module
from atorch.pipeline_parallel.scheduler import PipeSchedulerType, _PipeState
from atorch.tests.toy_modules.toy_module import (
    decoder_loss_func,
    get_llama_config,
    get_llama_dataloader,
    get_llama_dataset,
    get_llama_input_output_mapping,
    get_llama_model_chunk,
)


def _create_llama_pipe_module(layer_num=2, virtual_pp_size=None, model_config=None, loss_func=decoder_loss_func):
    config = PipeModuleConfig()
    config.model_config = model_config if model_config is not None else get_llama_config()
    config.total_layer_num = layer_num
    config.virtual_pp_size = virtual_pp_size
    config.input_output_mapping = get_llama_input_output_mapping()

    module = make_pipe_module(
        model_provider=get_llama_model_chunk,
        loss_func=loss_func,
        distributed_context=_distributed_context,
        config=config,
    )
    return module, config


def pipe_module_create_fn(rank):
    os.environ["RANK"] = str(rank)
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    res = atorch.init_distributed(backend, set_cuda_device_using_local_rank=backend == "nccl")
    assert res
    world_size = torch.distributed.get_world_size()
    gpu_partition = ([("pipe", world_size)], None)
    create_parallel_group(gpu_partition, use_atorch_pipe=True)

    virtual_pp_size = 2

    pipe_module, config = _create_llama_pipe_module(world_size * 2, virtual_pp_size=virtual_pp_size)

    assert isinstance(pipe_module, PipeModule)
    assert len(pipe_module.modules) == virtual_pp_size

    return pipe_module, config


def pipe_engine_create_fn_pp(rank):
    pipe_module, config = pipe_module_create_fn(rank)

    pipe_engine_config = {
        "pp_size": _distributed_context.parallel_group_size("pipe"),
        "virtual_pp_size": config.virtual_pp_size,
        "scheduler": "OneForwardOneBackwardInterleaving",
    }
    pipe_engine = PipeEngine(pipe_module, pipe_engine_config)
    assert isinstance(pipe_engine, PipeEngine)
    assert isinstance(pipe_engine.model, PipeModule)
    assert isinstance(pipe_engine.pipe_state, _PipeState)

    assert len(pipe_engine.model.modules) == config.virtual_pp_size
    assert pipe_engine.config.pp_size == pipe_engine_config["pp_size"]
    assert pipe_engine.config.virtual_pp_size == pipe_engine_config["virtual_pp_size"]
    assert pipe_engine.config.scheduler is PipeSchedulerType(pipe_engine_config["scheduler"])

    torch.distributed.barrier()
    atorch.reset_distributed()


def pipe_engine_create_fn_no_pipeline(
    world_size, micro_batchsize=8, global_batchsize=8, rank=0, seq_length=256, vocab_size=32000
):
    os.environ["RANK"] = str(rank)

    model_config = get_llama_config(seq_length=seq_length, vocab_size=vocab_size)
    decoder_loss_func_with_vocab_size = functools.partial(decoder_loss_func, vocab_size=vocab_size)

    pipe_module, _ = _create_llama_pipe_module(
        world_size * 2, model_config=model_config, loss_func=decoder_loss_func_with_vocab_size
    )

    assert isinstance(pipe_module, PipeModule)
    assert len(pipe_module.modules) == 1

    pipe_engine_config = {
        "scheduler": "ForwardBackwardNoPipelining",
        "micro_batchsize": micro_batchsize,
        "global_batchsize": global_batchsize,
    }
    pipe_engine = PipeEngine(pipe_module, pipe_engine_config)
    assert isinstance(pipe_engine, PipeEngine)
    assert isinstance(pipe_engine.model, PipeModule)
    assert isinstance(pipe_engine.pipe_state, _PipeState)

    assert len(pipe_engine.model.modules) == 1
    assert pipe_engine.config.scheduler is PipeSchedulerType(pipe_engine_config["scheduler"])

    return pipe_engine


class PipeEngineTest(unittest.TestCase):
    @unittest.skipIf(torch.cuda.device_count() < 4, "Requires 4 gpus.")
    def test_init_pipe_engine(self):
        world_size = 4
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["WORLD_SIZE"] = str(world_size)
        mp.spawn(
            pipe_engine_create_fn_pp,
            nprocs=world_size,
            join=True,
        )

    @unittest.skipIf(torch.cuda.device_count() < 1, "Requires 4 gpus.")
    def test_init_pipe_engine_no_pipeline(self):
        world_size = 1
        pipe_engine_create_fn_no_pipeline(world_size)

    def _check_parsed_config(self, parsed_config, config, skip_virtual=False, skip_pp=False):
        assert parsed_config.scheduler == PipeSchedulerType(config["scheduler"])
        assert parsed_config.micro_batchsize == config["micro_batchsize"]
        assert parsed_config.global_batchsize == config["global_batchsize"]
        assert not parsed_config.return_average_loss
        assert parsed_config.device == "cuda"

        if skip_pp:
            return

        assert parsed_config.pp_size == config["pp_size"]
        if skip_virtual:
            return

        assert parsed_config.virtual_pp_size == config["virtual_pp_size"]

    def test_parse_config(self):
        config = {"scheduler": "OneForwardOneBackward", "pp_size": 4, "micro_batchsize": 8, "global_batchsize": 8}
        parsed_config = PipeEngine._parse_config(config)
        self._check_parsed_config(parsed_config, config, skip_virtual=True)

        config = {
            "scheduler": "OneForwardOneBackwardInterleaving",
            "virtual_pp_size": 2,
            "pp_size": 8,
            "micro_batchsize": 8,
            "global_batchsize": 8,
        }
        parsed_config = PipeEngine._parse_config(config)
        self._check_parsed_config(parsed_config, config)

        config = {"scheduler": "ForwardBackwardNoPipelining", "micro_batchsize": 8, "global_batchsize": 8}
        parsed_config = PipeEngine._parse_config(config)
        self._check_parsed_config(parsed_config, config, skip_pp=True)

    def _train_toy_by_pipe_engine(
        self,
        world_size,
        seq_length=256,
        micro_batchsize=8,
        max_train_step=125,
        use_distributed_dataloader=False,
    ):
        world_size = 1
        vocab_size = 32000

        rank, dp_size = get_data_partition_rank_and_size()
        global_batchsize = micro_batchsize * dp_size
        total_data_size = max_train_step * global_batchsize

        # Create Engine
        pipe_engine = pipe_engine_create_fn_no_pipeline(
            world_size, micro_batchsize, global_batchsize, rank, seq_length, vocab_size=vocab_size
        )

        # Prepare Dataloader
        if use_distributed_dataloader:
            data_size = total_data_size
        else:
            data_size = total_data_size // dp_size

        dataset = get_llama_dataset(data_size=data_size, vocab_size=vocab_size)
        dataloader = get_llama_dataloader(dataset, batch_size=micro_batchsize, rank=rank, dp_size=dp_size)

        # Optimizer
        optimizer = torch.optim.AdamW(pipe_engine.model.parameters(), lr=0.001)

        # Train
        for _ in range(max_train_step):
            optimizer.zero_grad()
            data_iter = iter(dataloader)
            pipe_engine.train_batch_step(data_iter)
            for m in pipe_engine.model.modules:
                assert m.lm_head is not None
                assert m.lm_head.weight.grad is not None
            optimizer.step()

        destroy_parallel_group()

    @unittest.skipIf(torch.cuda.device_count() < 1, "Requires 1 gpu.")
    def test_train_toy_no_pipelining(self):
        self._train_toy_by_pipe_engine(world_size=1, max_train_step=25)

    @unittest.skipIf(torch.cuda.device_count() < 1, "Requires 1 gpu.")
    def test_no_pipeline_weight_update(self):
        vocab_size = 1024
        data_size = 4
        batch_size = 4
        micro_batch_size = batch_size // 2
        layer_num = 2
        device = "cuda"
        decoder_loss_func_with_vocab_size = functools.partial(decoder_loss_func, vocab_size=vocab_size)

        pipe_module, _ = _create_llama_pipe_module(layer_num, loss_func=decoder_loss_func_with_vocab_size)

        module_copy = copy.deepcopy(pipe_module).to(device)

        dataset = get_llama_dataset(data_size=data_size, vocab_size=vocab_size)
        dataloader = get_llama_dataloader(dataset, batch_size=batch_size, rank=0, dp_size=1)
        data_iter = iter(dataloader)
        one_batch = next(data_iter)
        one_batch = data_to_device(one_batch, device)

        optimizer = torch.optim.AdamW(module_copy.parameters(), lr=0.001)

        optimizer.zero_grad()
        outputs, loss_func = module_copy(**one_batch)
        loss = loss_func(one_batch, outputs)
        loss.backward()
        optimizer.step()

        pipe_engine_config = {
            "scheduler": "ForwardBackwardNoPipelining",
            "micro_batchsize": micro_batch_size,
            "global_batchsize": batch_size,
            "deallocate_pipeline_outputs": False,
        }
        pipe_engine = PipeEngine(pipe_module, pipe_engine_config)
        dataloader = get_llama_dataloader(dataset, batch_size=micro_batch_size, rank=0, dp_size=1)
        data_iter = iter(dataloader)
        optimizer = torch.optim.AdamW(pipe_engine.model.parameters(), lr=0.001)

        optimizer.zero_grad()
        pipe_engine.train_batch_step(data_iter)
        optimizer.step()

        # then check weight for pipe_module with module_copy
        for p1, p2 in zip(pipe_module.parameters(), module_copy.parameters()):
            self.assertTrue(torch.allclose(p1, p2, atol=1e-4))
