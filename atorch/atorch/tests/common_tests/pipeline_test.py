"""This is to verify that pipe model has the same behavior as the normal modules,
in composition with amp.

"""
import copy
import os
import random
import unittest

import numpy as np
import torch
import torch.distributed.rpc as torch_rpc
import torch.multiprocessing as mp
from torch.nn import MSELoss
from torch.utils.data import Dataset

import atorch
from atorch.auto.model_context import ModelContext
from atorch.auto.opt_lib.amp_optimization import AmpNativeOptimization
from atorch.common.util_func import find_free_port
from atorch.distributed.distributed import create_parallel_group, destroy_parallel_group, parallel_group
from atorch.modules.distributed_modules.compilers.pipe_compiler.distributed_pippy_compiler import (
    DeviceSafeDriver,
    SafeStage,
)
from atorch.utils.meta_model_utils import deepcopy_checkpoint_name
from atorch.utils.pipe_file_utils import atorch_load_pipe_checkpoint, atorch_save_pipe_checkpoint


def seed_everything(seed=42):
    random.seed(seed)  # Python's random module
    np.random.seed(seed)  # Numpy
    torch.manual_seed(seed)  # PyTorch

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # CUDA
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class MyModule(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.layer = torch.nn.Linear(in_features, out_features, bias=bias)
        self.layers = torch.nn.ModuleList([torch.nn.Linear(out_features, out_features, bias=bias) for _ in range(16)])

    def forward(self, input_):
        data = torch.nn.functional.gelu(self.layer(input_[0]))
        for op in self.layers:
            data = op(data)
            data = torch.nn.functional.gelu(data)
        return data


class ToyDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.ones((7, 32)), torch.zeros((7, 16))


def prepare_input(data, device):
    return data[0].to(device), data[1].to(device)


def my_loss_func(data, outputs):
    loss_fct = MSELoss()
    loss = loss_fct(outputs.view(-1), data[-1].view(-1))
    return loss


def create_model_context(data_size=512, batch_size=16, loss_func=None):
    model = MyModule(32, 16, True)
    dataset = ToyDataset(data_size)
    model_context = ModelContext(
        model=model,
        optim_func=torch.optim.SGD,
        dataset=dataset,
        prepare_input=prepare_input,
        dataloader_args={"batch_size": batch_size, "drop_last": True},
        optim_args={"lr": 0.001},
        loss_func=loss_func,
    )
    return model_context


def generate_pipe_configs(model_context, pipe_config=dict()):
    """Allowing pipe_configs gives us the flexibility to test different schedules"""
    from atorch.auto.opt_lib.pipeline_parallel_optimization import PipelineParallelOptimization

    pipe_opt = PipelineParallelOptimization()
    status, best_config, model_context = pipe_opt.tune(model_context, config=pipe_config, strategy=[])
    assert status, "PipelineParallelOptimization failed to tune"
    return best_config


def init_pipe_distributed(rank, world_size):
    # set random state
    seed_everything()

    if not torch.distributed.is_initialized():
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["NPROC_PER_NODE"] = str(world_size)
        if torch.cuda.is_available():
            atorch.init_distributed("nccl")
            torch.cuda.set_device(rank)
        else:
            atorch.init_distributed("gloo")
    if parallel_group("pipe") is None:
        gpu_partition = ([("pipe", world_size)], None)
        create_parallel_group(gpu_partition)


def run_two_steps_return_the_last(rank, model_context):
    model = model_context.model
    dataloader = model_context.dataloader
    loss_func = model_context.loss_func
    if loss_func is not None:
        optimizer = model_context.optim
    prepare_input = model_context.prepare_input
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"

    for step, batch in enumerate(dataloader):
        batch = prepare_input(batch, device)
        if isinstance(model, DeviceSafeDriver):
            if rank == 0 or rank == "0":
                outputs = model(batch)
                if loss_func is not None:
                    loss = loss_func(batch, outputs)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                if step == 2:
                    return outputs if loss_func is None else loss
            else:
                return torch.zeros((1,))
        else:
            outputs = model(batch)
            if loss_func is not None:
                loss = loss_func(batch, outputs)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            if step == 2:
                # For PipeStage mode the last stage returns the loss
                if isinstance(model, SafeStage):
                    if rank == atorch.world_size() - 1:
                        return outputs if loss_func is None else loss
                    else:
                        return torch.zeros((1,))
                else:
                    return outputs if loss_func is None else loss


def run_pipeline(rank, world_size, pipe_config=dict(), amp_config=None, loss_func=None):
    from atorch.auto.opt_lib.pipeline_parallel_optimization import PipelineParallelOptimization

    init_pipe_distributed(rank, world_size)
    model_context = create_model_context(loss_func=loss_func)
    model_context_copy = copy.deepcopy(model_context)
    deepcopy_checkpoint_name(model_context_copy.model, model_context.model)
    if torch.cuda.is_available():
        model_context.model.cuda()
        model_context_copy.model.cuda()
    # the result is deterministic on each rank
    best_config = generate_pipe_configs(model_context_copy, pipe_config)
    if amp_config is not None:
        best_config["compiler_configs"]["amp_config"] = amp_config
    PipelineParallelOptimization.apply_wrapper(model_context_copy, "pipe", best_config)

    model_context.update_dataloader()
    model_context_copy.update_dataloader()
    if loss_func is not None:
        model_context.update_optim()
        model_context_copy.update_optim()

    if amp_config is not None:
        AmpNativeOptimization.apply_wrapper(model_context, "amp_native", amp_config)

    outputs = run_two_steps_return_the_last(rank, model_context)

    pipe_outputs = run_two_steps_return_the_last(rank, model_context_copy)
    outputs = (
        torch.zeros((1,))
        if pipe_outputs.device == torch.device("cpu") and torch.all(pipe_outputs == torch.zeros((1,)))
        else outputs
    )
    if loss_func is not None:
        chunks = pipe_config.get("chunks", 1)
        pipe_outputs /= chunks
    torch_rpc.api._wait_all_workers()
    assert torch.allclose(
        outputs, pipe_outputs, rtol=1e-4, atol=1e-4
    ), f"Expected: ({outputs.shape}) {outputs} but pipe outputs ({pipe_outputs.shape}): {pipe_outputs}"
    destroy_parallel_group()


def run_save_and_load(rank, world_size, pipe_config=dict(), amp_config=None, loss_func=None):
    from atorch.auto.opt_lib.pipeline_parallel_optimization import PipelineParallelOptimization

    pipe_config["use_c10d"] = True
    init_pipe_distributed(rank, world_size)
    model_context = create_model_context(loss_func=loss_func)
    deepcopy_checkpoint_name(model_context.model, model_context.model)
    if torch.cuda.is_available():
        model_context.model.cuda()

    # the result is deterministic on each rank
    best_config = generate_pipe_configs(model_context, pipe_config)
    if amp_config is not None:
        best_config["compiler_configs"]["amp_config"] = amp_config
    PipelineParallelOptimization.apply_wrapper(model_context, "pipe", best_config)
    model_context.update_dataloader()
    if loss_func is not None:
        model_context.update_optim()

    optimizer = model_context.optim if loss_func is not None else None
    # first save model
    atorch_save_pipe_checkpoint(model_context.model, optimizer=optimizer)

    # Force the submod into meta
    model_context.model.to("meta")

    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    atorch_load_pipe_checkpoint(model_context.model, optim=optimizer, device=device)
    # Then run a train step to verify the reload success
    run_two_steps_return_the_last(rank, model_context)


class TestPipeMLP(unittest.TestCase):
    def tearDown(self):
        destroy_parallel_group()
        return super().tearDown()

    @unittest.skipIf(True, "Enable CPU test until aci image upgraded")
    def test_cpu_inference(self):
        world_size = 2
        pipe_config = {"nstages": world_size, "chunks": world_size}
        amp_config = None
        loss_func = None
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(
            run_pipeline,
            args=(world_size, pipe_config, amp_config, loss_func),
            nprocs=world_size,
            join=True,
        )
        os.environ["MASTER_ADDR"] = ""
        os.environ["MASTER_PORT"] = ""

    @unittest.skipIf(True, "Enable CPU test until aci image upgraded")
    def test_cpu_interleaved_inference(self):
        world_size = 2
        pipe_config = {"nstages": 2 * world_size, "chunks": world_size}
        amp_config = None
        loss_func = None
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(
            run_pipeline,
            args=(world_size, pipe_config, amp_config, loss_func),
            nprocs=world_size,
            join=True,
        )
        os.environ["MASTER_ADDR"] = ""
        os.environ["MASTER_PORT"] = ""

    @unittest.skipIf(True, "Enable CPU test until aci image upgraded")
    def test_cpu_train(self):
        world_size = 2
        pipe_config = {"nstages": world_size, "chunks": world_size}
        amp_config = None
        loss_func = my_loss_func
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(
            run_pipeline,
            args=(world_size, pipe_config, amp_config, loss_func),
            nprocs=world_size,
            join=True,
        )
        os.environ["MASTER_ADDR"] = ""
        os.environ["MASTER_PORT"] = ""

    @unittest.skipIf(True, "Enable CPU test until aci image upgraded")
    def test_cpu_interleaved_train(self):
        world_size = 2
        pipe_config = {"nstages": 2 * world_size, "chunks": world_size}
        amp_config = None
        loss_func = my_loss_func
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(
            run_pipeline,
            args=(world_size, pipe_config, amp_config, loss_func),
            nprocs=world_size,
            join=True,
        )
        os.environ["MASTER_ADDR"] = ""
        os.environ["MASTER_PORT"] = ""

    @unittest.skipIf(True, "Default to disable pippy related tests as its not stable")
    def test_gpu_inference(self):
        world_size = 2
        pipe_config = {"nstages": world_size, "chunks": world_size}
        amp_config = None
        loss_func = None
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(
            run_pipeline,
            args=(world_size, pipe_config, amp_config, loss_func),
            nprocs=world_size,
            join=True,
        )
        os.environ["MASTER_ADDR"] = ""
        os.environ["MASTER_PORT"] = ""

    @unittest.skipIf(True, "Default to disable pippy related tests as its not stable")
    def test_gpu_interleaved_inference(self):
        world_size = 2
        pipe_config = {"nstages": 2 * world_size, "chunks": world_size}
        amp_config = None
        loss_func = None
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(
            run_pipeline,
            args=(world_size, pipe_config, amp_config, loss_func),
            nprocs=world_size,
            join=True,
        )
        os.environ["MASTER_ADDR"] = ""
        os.environ["MASTER_PORT"] = ""

    @unittest.skipIf(True, "Test covered in amp mode pipe teset")
    def test_gpu_train(self):
        world_size = 2
        pipe_config = {"nstages": world_size, "chunks": world_size}
        amp_config = None
        loss_func = my_loss_func
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(
            run_pipeline,
            args=(world_size, pipe_config, amp_config, loss_func),
            nprocs=world_size,
            join=True,
        )
        os.environ["MASTER_ADDR"] = ""
        os.environ["MASTER_PORT"] = ""

    @unittest.skipIf(True, "Test covered in amp mode pipe teset")
    def test_gpu_interleaved_train(self):
        world_size = 2
        pipe_config = {"nstages": 2 * world_size, "chunks": world_size}
        amp_config = None
        loss_func = my_loss_func
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(
            run_pipeline,
            args=(world_size, pipe_config, amp_config, loss_func),
            nprocs=world_size,
            join=True,
        )
        os.environ["MASTER_ADDR"] = ""
        os.environ["MASTER_PORT"] = ""

    @unittest.skipIf(True, "Default to disable pippy related tests as its not stable")
    def test_gpu_train_fp16(self):
        world_size = 2
        pipe_config = {"nstages": world_size, "chunks": world_size}
        amp_config = {"dtype": torch.float16, "skip_if_nonfinite": None}
        loss_func = my_loss_func
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(
            run_pipeline,
            args=(world_size, pipe_config, amp_config, loss_func),
            nprocs=world_size,
            join=True,
        )
        os.environ["MASTER_ADDR"] = ""
        os.environ["MASTER_PORT"] = ""

    @unittest.skipIf(True, "Default to disable pippy related tests as its not stable")
    def test_gpu_train_bf16(self):
        world_size = 2
        pipe_config = {"nstages": world_size, "chunks": world_size}
        amp_config = {"dtype": torch.bfloat16, "skip_if_nonfinite": None}
        loss_func = my_loss_func
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(
            run_pipeline,
            args=(world_size, pipe_config, amp_config, loss_func),
            nprocs=world_size,
            join=True,
        )
        os.environ["MASTER_ADDR"] = ""
        os.environ["MASTER_PORT"] = ""

    @unittest.skipIf(True, "Default to disable pippy related tests as its not stable")
    def test_gpu_interleaved_train_bf16(self):
        world_size = 2
        pipe_config = {"nstages": 2 * world_size, "chunks": world_size}
        amp_config = {"dtype": torch.bfloat16, "skip_if_nonfinite": None}
        loss_func = my_loss_func
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(
            run_pipeline,
            args=(world_size, pipe_config, amp_config, loss_func),
            nprocs=world_size,
            join=True,
        )
        os.environ["MASTER_ADDR"] = ""
        os.environ["MASTER_PORT"] = ""

    @unittest.skipIf(True, "Default to disable pippy related tests as its not stable")
    def test_gpu_train_c10d(self):
        world_size = 2
        pipe_config = {"nstages": world_size, "chunks": world_size, "use_c10d": True}
        amp_config = None
        loss_func = my_loss_func
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(
            run_pipeline,
            args=(world_size, pipe_config, amp_config, loss_func),
            nprocs=world_size,
            join=True,
        )
        os.environ["MASTER_ADDR"] = ""
        os.environ["MASTER_PORT"] = ""

    @unittest.skipIf(True, "Default to disable pippy related tests as its not stable")
    def test_gpu_train_c10d_bf16(self):
        world_size = 2
        pipe_config = {"nstages": world_size, "chunks": world_size, "use_c10d": True}
        amp_config = {"dtype": torch.bfloat16, "skip_if_nonfinite": None}
        loss_func = my_loss_func
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(
            run_pipeline,
            args=(world_size, pipe_config, amp_config, loss_func),
            nprocs=world_size,
            join=True,
        )
        os.environ["MASTER_ADDR"] = ""
        os.environ["MASTER_PORT"] = ""

    @unittest.skipIf(True, "Default to disable pippy related tests as its not stable")
    def test_gpu_train_c10d_interleaved_bf16(self):
        world_size = 2
        pipe_config = {"nstages": 2 * world_size, "chunks": world_size, "use_c10d": True}
        amp_config = {"dtype": torch.bfloat16, "skip_if_nonfinite": None}
        loss_func = my_loss_func
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(
            run_pipeline,
            args=(world_size, pipe_config, amp_config, loss_func),
            nprocs=world_size,
            join=True,
        )
        os.environ["MASTER_ADDR"] = ""
        os.environ["MASTER_PORT"] = ""

    @unittest.skipIf(True, "Default to disable pippy related tests as its not stable")
    def test_gpu_save_load_c10d(self):
        world_size = 2
        pipe_config = {"nstages": world_size, "chunks": world_size, "use_c10d": True}
        amp_config = None
        loss_func = my_loss_func
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(
            run_save_and_load,
            args=(world_size, pipe_config, amp_config, loss_func),
            nprocs=world_size,
            join=True,
        )
        os.environ["MASTER_ADDR"] = ""
        os.environ["MASTER_PORT"] = ""


if __name__ == "__main__":
    unittest.main()
