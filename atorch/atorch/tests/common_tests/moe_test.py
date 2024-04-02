import os
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from atorch.common.log_utils import default_logger as logger
from atorch.modules.moe import Experts, MOELayer, SwitchGate, TopkGate

logger.setLevel("ERROR")
os.environ["NCCL_DEBUG"] = "ERROR"


def _run_SwitchGate_moe(rank, world_size, tmp_file):
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    dist.init_process_group(
        init_method="file://" + tmp_file,
        rank=rank,
        backend=backend,
        world_size=world_size,
    )

    # build moe
    s, b, z, e, m = 100, 32, 4, 48, 64
    input_tensor = torch.rand(s, b, m).to(device)
    gate = SwitchGate(m, e, need_l_aux=True)
    assert e % world_size == 0
    experts = Experts(e // world_size, m, 128)
    moe_ori = MOELayer(gate, experts, num_prototypes=z).to(device)
    moe_dispatchV2 = MOELayer(gate, experts, num_prototypes=z, dispatchV2=True).to(device)
    moe_outer_batch = MOELayer(gate, experts, num_prototypes=z, outer_batch=True).to(device)
    moe_outer_batch_dispatchV2 = MOELayer(gate, experts, num_prototypes=z, outer_batch=True, dispatchV2=True).to(device)

    # forward
    out_ori = moe_ori(input_tensor)
    out_dispatchV2 = moe_dispatchV2(input_tensor)
    l_aux_ori = moe_ori.l_aux
    l_aux_dispatchV2 = moe_dispatchV2.l_aux
    out_outer_batch = moe_outer_batch(input_tensor)
    out_outer_batch_dispatchV2 = moe_outer_batch_dispatchV2(input_tensor)
    l_aux_outer_batch = moe_outer_batch.l_aux
    l_aux_outer_batch_dispatchV2 = moe_outer_batch_dispatchV2.l_aux

    # assert dispatchV2 consistency, isclose for torch float precision
    assert torch.all(torch.isclose(out_ori, out_dispatchV2)), (out_ori - out_dispatchV2).abs().max()
    assert l_aux_ori == l_aux_dispatchV2
    assert torch.all(torch.isclose(out_outer_batch, out_outer_batch_dispatchV2)), (
        (out_outer_batch - out_outer_batch_dispatchV2).abs().max()
    )
    assert l_aux_outer_batch == l_aux_outer_batch_dispatchV2
    dist.destroy_process_group()


def _run_TopkGate_moe(rank, world_size, tmp_file):
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    dist.init_process_group(
        init_method="file://" + tmp_file,
        rank=rank,
        backend=backend,
        world_size=world_size,
    )

    # build moe
    s, b, z, e, m = 100, 32, 4, 48, 64
    input_tensor = torch.rand(s, b, m).to(device)
    gate = TopkGate(m, e, k=2, need_l_aux=True)
    assert e % world_size == 0
    experts = Experts(e // world_size, m, 128)
    moe_ori = MOELayer(gate, experts, num_prototypes=z).to(device)
    moe_dispatchV2 = MOELayer(gate, experts, num_prototypes=z, dispatchV2=True).to(device)

    # forward
    out_ori = moe_ori(input_tensor)
    out_dispatchV2 = moe_dispatchV2(input_tensor)
    l_aux_ori = moe_ori.l_aux
    l_aux_dispatchV2 = moe_dispatchV2.l_aux

    # assert dispatchV2 consistency, isclose for torch float precision
    assert torch.all(torch.isclose(out_ori, out_dispatchV2)), (out_ori - out_dispatchV2).abs().max()
    assert l_aux_ori == l_aux_dispatchV2
    dist.destroy_process_group()


class TestMOELayer(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.mkstemp()[1]

    def tearDown(self):
        try:
            os.remove(self.temp)
        except FileNotFoundError:
            pass

    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        "No gpu available for cuda tests",
    )
    def test_SwitchGate_moe(self):
        world_size = 2
        mp.spawn(
            _run_SwitchGate_moe,
            args=(world_size, self.temp),
            nprocs=world_size,
            join=True,
        )

    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        "No gpu available for cuda tests",
    )
    def test_TopkGate_moe(self):
        world_size = 2
        mp.spawn(
            _run_TopkGate_moe,
            args=(world_size, self.temp),
            nprocs=world_size,
            join=True,
        )


if __name__ == "__main__":
    unittest.main()
