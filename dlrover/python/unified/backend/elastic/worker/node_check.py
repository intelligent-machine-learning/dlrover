import torch

from dlrover.trainer.torch.node_check.utils import bm_allreduce, matmul


def run_comm_check() -> float:
    """Run matmul and allreduce to check communication performance."""

    use_cuda = torch.cuda.is_available()

    # warmup
    _ = matmul(use_cuda, round_num=3, verbose=False)

    t = matmul(use_cuda, round_num=500, verbose=True)
    shape = 1 << 24
    t += bm_allreduce(shape, use_cuda)
    return t
