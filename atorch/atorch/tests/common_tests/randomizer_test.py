import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import atorch
from atorch.common.log_utils import default_logger as logger
from atorch.common.util_func import find_free_port
from atorch.distributed.distributed import create_parallel_group, parallel_group, parallel_group_size, parallel_rank
from atorch.modules.distributed_modules.randomizer import get_MDPRInstance, get_randomizer, init_randomizer

logger.setLevel("INFO")
os.environ["NCCL_DEBUG"] = "ERROR"


def assert_group(tensor, group_name, same=True):
    tensor_list = [torch.empty_like(tensor) for _ in range(parallel_group_size(group_name))]
    tensor_list[parallel_rank(group_name)] = tensor
    dist.all_gather(tensor_list, tensor, group=parallel_group(group_name))
    for tensor in tensor_list[1:]:
        all_same = torch.eq(tensor, tensor_list[0]).all()
        assert (
            all_same if same else not all_same
        ), f"Tensor {'same' if same else 'differet'} check failed:\n\n{tensor}\n\n{tensor_list[0]}"


def assert_model(model1, model2):
    sd1 = model1.state_dict()
    sd2 = model2.state_dict()
    for name1, name2 in zip(sd1, sd2):
        assert name1 == name2
        assert torch.all(sd1[name1] == sd2[name2])


def _run_randomizer(rank):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = "4"
    res = atorch.init_distributed("nccl", set_cuda_device_using_local_rank=True)
    if not res:
        raise Exception("init failed")
    create_parallel_group(([("tensor", 2), ("data", 2)], None))
    init_randomizer()
    device = torch.cuda.current_device()

    # cuda weight init
    with get_randomizer("data").fork():
        m = torch.nn.Linear(10, 20, device=device)
    assert_group(m.weight.detach(), "data", same=True)
    assert_group(m.weight.detach(), "tensor", same=False)

    # cpu weight init
    with get_randomizer("data", "tensor").fork():
        m = torch.nn.Linear(10, 20)
    assert_group(m.weight.detach().to(device), "data", same=True)
    assert_group(m.weight.detach().to(device), "tensor", same=True)

    # dropout pattern
    drop = torch.nn.Dropout(0.1)
    with get_randomizer("tensor").fork():
        t = drop(torch.ones(10, device=device))
    assert_group(t, "data", same=False)
    assert_group(t, "tensor", same=True)
    # again with former randomizer
    with get_randomizer("data").fork():
        t = drop(torch.ones(10, device=device))
    assert_group(t, "data", same=True)
    assert_group(t, "tensor", same=False)

    # test whold different
    drop2 = torch.nn.Dropout(0.1)
    with get_randomizer().fork():
        m = torch.nn.Linear(10, 20, device=device)
        t = drop2(torch.ones(10, device=device))
    assert_group(m.weight.detach().to(device), "data", same=False)
    assert_group(m.weight.detach().to(device), "tensor", same=False)
    assert_group(t, "data", same=False)
    assert_group(t, "tensor", same=False)

    atorch.reset_distributed()


def _run_randomizer_states(rank):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = "4"
    res = atorch.init_distributed("nccl", set_cuda_device_using_local_rank=True)
    if not res:
        raise Exception("init failed")
    create_parallel_group(([("tensor", 2), ("data", 2)], None))
    init_randomizer()
    device = torch.cuda.current_device()

    # set up some group
    with get_randomizer("data").fork():
        _ = torch.nn.Linear(10, 20, device=device)
    with get_randomizer("data", "tensor").fork():
        _ = torch.nn.Linear(10, 20, device=device)
    with get_randomizer().fork():
        _ = torch.nn.Linear(10, 20, device=device)

    # get states
    _stated_randomizers = get_MDPRInstance().get_states()
    with get_randomizer("data", "tensor").fork():
        m1 = torch.nn.Linear(10, 20, device=device)
    with get_randomizer().fork():
        m2 = torch.nn.Linear(10, 20, device=device)

    # do something else
    with get_randomizer("data").fork():
        _ = torch.nn.Dropout(0.1)(torch.ones(10, device=device))
    with get_randomizer().fork():
        _ = torch.nn.Dropout(0.1)(torch.ones(10, device=device))

    # set states and compare
    get_MDPRInstance().set_states(_stated_randomizers)
    with get_randomizer("data", "tensor").fork():
        new_m1 = torch.nn.Linear(10, 20, device=device)
    with get_randomizer().fork():
        new_m2 = torch.nn.Linear(10, 20, device=device)
    assert_model(new_m1, m1)
    assert_model(new_m2, m2)


class TestRandomizer(unittest.TestCase):
    @unittest.skipIf(torch.cuda.device_count() < 4, "run with cpu or gpu_num >=4")
    def test_randomizer(self):

        world_size = 4
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(
            _run_randomizer,
            nprocs=world_size,
            join=True,
        )
        os.environ["MASTER_ADDR"] = ""
        os.environ["MASTER_PORT"] = ""

    @unittest.skipIf(torch.cuda.device_count() < 4, "run with cpu or gpu_num >=4")
    def test_randomizer_states(self):

        world_size = 4
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(
            _run_randomizer_states,
            nprocs=world_size,
            join=True,
        )
        os.environ["MASTER_ADDR"] = ""
        os.environ["MASTER_PORT"] = ""


if __name__ == "__main__":
    unittest.main()
