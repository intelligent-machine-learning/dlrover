import os
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from torch_npu.contrib import transfer_to_npu  # noqa: F401


def run(rank, size):
    """Distributed function to be implemented later."""
    # group = dist.new_group([0, 1])
    tensors = []
    bufs = []
    for i in range(10):
        if i > 5:
            size = 1024 + i
        else:
            size = 102 + i
        tensor = torch.ones(size, size).cuda().half()
        buf = [torch.zeros(size, size).cuda().half() for _ in range(2)]
        tensors.append(tensor)
        bufs.append(buf)
    count = 0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []

    # Waits for everything to finish running

    while True:
        if count % 100 == 0 and count != 0:
            print(f"{count}, {np.array(times).mean()}")
            times = []

        tensor = tensors[count % 10]
        buf = bufs[count % 10]
        start.record()
        dist.all_gather(buf, tensor, async_op=False)
        dist.all_reduce(tensor, async_op=False)
        dist.reduce_scatter(tensor, buf, async_op=False)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
        count += 1


def init_process(rank, size, fn, backend="hccl"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29589"
    torch.cuda.set_device(rank)
    device = torch.device(f"npu:{rank}")
    dist.init_process_group(backend, rank=rank, world_size=size)
    with torch.device(device=device):
        fn(rank, size)


if __name__ == "__main__":
    size = 2
    rank = int(sys.argv[1])
    os.environ["LOCAL_RANK"] = sys.argv[1]
    os.environ["RANK"] = sys.argv[1]
    os.environ["WORLD_SIZE"] = str(size)
    os.environ["LOCAL_WORLD_SIZE"] = str(size)
    init_process(rank, size, run)
