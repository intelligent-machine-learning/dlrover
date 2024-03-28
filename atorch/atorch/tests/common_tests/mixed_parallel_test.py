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
from atorch.common.util_func import find_free_port
from atorch.distributed.distributed import destroy_parallel_group, parallel_group_and_ranks, parallel_rank
from atorch.modules.distributed_modules.compilers.pipe_compiler.distributed_pippy_compiler import (
    DeviceSafeDriver,
    SafeStage,
)
from atorch.utils.meta_model_utils import deepcopy_checkpoint_name


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


def generate_mixed_configs(model_context, mixed_config=dict()):
    """Allowing pipe_configs gives us the flexibility to test different schedules"""
    from atorch.auto.opt_lib.mixed_parallel_optimization import MixedParallelOptimization

    mixed_opt = MixedParallelOptimization()
    status, best_config, model_context = mixed_opt.tune(model_context, config=mixed_config, strategy=None)
    assert status, "MixedParallelOptimization failed to tune"
    return best_config


def init_distributed(rank, world_size):
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


def run_two_steps_return_the_last(rank, model_context):
    model = model_context.model
    dataloader = model_context.dataloader
    loss_func = model_context.loss_func
    if loss_func is not None:
        optimizer = model_context.optim
    prepare_input = model_context.prepare_input
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"

    pipe_rank = parallel_rank("pipe")
    _, pipe_ranks = parallel_group_and_ranks("pipe")
    for step, batch in enumerate(dataloader):
        batch = prepare_input(batch, device)
        if isinstance(model, DeviceSafeDriver):
            if pipe_rank == 0 or pipe_rank == "0":
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
                    if rank == pipe_ranks[-1]:
                        return outputs if loss_func is None else loss
                    else:
                        return torch.zeros((1,))
                else:
                    return outputs if loss_func is None else loss


# Leave the AMP test to pure pipeline
def run_mixed_parallel(rank, world_size, mixed_config=dict(), loss_func=None):
    from atorch.auto.opt_lib.mixed_parallel_optimization import MixedParallelOptimization

    init_distributed(rank, world_size)
    model_context = create_model_context(loss_func=loss_func)
    model_context_copy = copy.deepcopy(model_context)
    deepcopy_checkpoint_name(model_context_copy.model, model_context.model)
    if torch.cuda.is_available():
        model_context.model.cuda()
        model_context_copy.model.cuda()
    # the result is deterministic on each rank
    best_config = generate_mixed_configs(model_context_copy, mixed_config)

    MixedParallelOptimization.apply_wrapper(model_context_copy, "mp", best_config)
    model_context_copy.apply_wrappers(is_pre_wrapper=True)
    model_context.update_dataloader()
    model_context_copy.update_dataloader()
    if loss_func is not None:
        model_context.update_optim()
        model_context_copy.update_optim()
    model_context_copy.apply_wrappers(is_pre_wrapper=False)

    outputs = run_two_steps_return_the_last(rank, model_context)

    mixed_outputs = run_two_steps_return_the_last(rank, model_context_copy)

    outputs = (
        torch.zeros((1,))
        if mixed_outputs.device == torch.device("cpu") and torch.all(mixed_outputs == torch.zeros((1,)))
        else outputs
    )

    if loss_func is not None:
        chunks = mixed_config["pipe_config"].get("chunks", 1)
        mixed_outputs /= chunks
    torch_rpc.api._wait_all_workers()
    assert torch.allclose(
        outputs, mixed_outputs, rtol=1e-4, atol=1e-4
    ), f"Expected: ({outputs.shape}) {outputs} but mixed outputs ({mixed_outputs.shape}): {mixed_outputs}"
    destroy_parallel_group()


class TestMixedMLP(unittest.TestCase):
    def tearDown(self):
        destroy_parallel_group()
        return super().tearDown()

    @unittest.skipIf(True, "Default to disable pippy related tests as its not stable")
    def test_pipe_ddp_train(self):
        world_size = 4
        # test interleaved/microbatching schedule
        # TODO test interleaved mode again later
        mixed_config = {
            "parallel_mode": ([("pipe", 2), ("data", 2)], None),
            "prop_mode": "interpreter",
            "use_fake_mode": False,
            "pipe_config": {"use_c10d": True, "chunks": 4, "nstages": 2},
            "tesnor_config": {"shard_planner": "base"},
        }
        loss_func = my_loss_func
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        mp.spawn(
            run_mixed_parallel,
            args=(world_size, mixed_config, loss_func),
            nprocs=world_size,
            join=True,
        )
        os.environ["MASTER_ADDR"] = ""
        os.environ["MASTER_PORT"] = ""


if __name__ == "__main__":
    unittest.main()
