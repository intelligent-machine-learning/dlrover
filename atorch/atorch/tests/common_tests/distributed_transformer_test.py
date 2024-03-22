import os
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from atorch.common.log_utils import default_logger as logger
from atorch.modules.distributed_transformer import DistributedSoftmax
from atorch.modules.distributed_transformer.commu_utils import AllGatherQMicro, ReduceScatterContext

logger.setLevel("INFO")
os.environ["NCCL_DEBUG"] = "ERROR"


def _run_distributed_softmax(rank, world_size, tmp_file):
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

    # batch size, num heads, q dim, local sequence, hidden size
    bs, nh, qd, ls = 10, 12, 32, 512
    dtype = torch.float32
    torch.manual_seed(0)
    s = torch.rand((bs, nh, qd, ls * world_size), requires_grad=True, dtype=dtype).to(device).split(ls, dim=-1)[rank]
    s.retain_grad()
    p = DistributedSoftmax.apply(s)
    dummy_label = torch.rand((bs, nh, qd, ls * world_size), dtype=dtype).to(device)
    dummy_label_local = dummy_label.split(ls, dim=-1)[rank]
    (p - dummy_label_local).sum().backward()

    torch.manual_seed(0)
    ss = torch.rand((bs, nh, qd, ls * world_size), requires_grad=True, dtype=dtype).to(device)
    ss.retain_grad()
    assert torch.all(torch.isclose(ss.split(ls, dim=-1)[rank], s))
    pp = ss.softmax(-1)
    pp_local = pp.split(ls, dim=-1)[rank]

    logger.info(
        f"Rank [{rank}] p: max diff {(pp_local - p).abs().max():.3e}"
        f" ,not close ratio "
        f"{(~torch.isclose(pp_local, p)).sum() / p.numel():.3%}"
    )
    (pp - dummy_label).sum().backward()
    ss_grad = ss.grad.split(ls, dim=-1)[rank]
    logger.info(
        f"Rank [{rank}] s' grad: max diff {(ss_grad - s.grad).abs().max():.3e}"
        f" ,not close ratio "
        f"{(~torch.isclose(ss_grad, s.grad)).sum() / s.numel():.3%}"
    )
    dist.destroy_process_group()


def _run_allgather_reducescatter(rank, world_size, tmp_file):
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

    # batch size, num heads, q dim, local sequence, hidden size
    bs, nh, qd, ls = 10, 12, 32, 512
    dtype = torch.float32
    torch.manual_seed(0)

    inp = torch.rand((bs, nh, qd, ls), requires_grad=True, dtype=dtype, device=device)
    lin = torch.nn.Linear(ls, ls).to(device)
    allgathered = AllGatherQMicro.apply(inp)
    inter = lin(allgathered)
    reduced = ReduceScatterContext.apply(inter)
    logger.info(
        f"Rank [{rank}] inp: {inp.requires_grad},\n"
        f"allgathered: {allgathered.requires_grad},\n"
        f"inter: {inter.requires_grad},\nreduced: {reduced.requires_grad}"
    )
    dist.destroy_process_group()


class TestDistributedTransformer(unittest.TestCase):
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
    def test_distributed_softmax(self):
        world_size = 2
        mp.spawn(
            _run_distributed_softmax,
            args=(world_size, self.temp),
            nprocs=world_size,
            join=True,
        )

    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        "No gpu available for cuda tests",
    )
    def test_allgather_reducescatter(self):
        world_size = 2
        mp.spawn(
            _run_allgather_reducescatter,
            args=(world_size, self.temp),
            nprocs=world_size,
            join=True,
        )


if __name__ == "__main__":
    unittest.main()
