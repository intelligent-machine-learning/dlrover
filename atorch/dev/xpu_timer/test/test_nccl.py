import os
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist


def run(rank, size):
    """Distributed function to be implemented later."""
    # group = dist.new_group([0, 1])
    size = 1024
    tensor = torch.ones(size, size).cuda()
    buf = [torch.zeros(size, size).cuda() for _ in range(2)]
    a = torch.randn(1024, 1024).cuda().bfloat16()
    b = torch.randn(1024, 1024).cuda().bfloat16()
    c = torch.randn(3, 1024, 1024).cuda().bfloat16()
    d = torch.randn(3, 1024, 1024).cuda().bfloat16()
    count = 0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []

    # Waits for everything to finish running

    while True:
        if rank == 1:
            time.sleep(random.random() / 4)
        if count % 100 == 0:
            print(f"{count}, {np.array(times).mean()}")
            times = []

        dist.all_gather(buf, tensor, async_op=True)
        dist.all_reduce(tensor, async_op=True)
        torch.matmul(a, b)
        start.record()
        torch.matmul(c, d)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
        count += 1
        # if count > 1000:
        #    break


def init_process(rank, size, fn, backend="nccl"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29509"
    torch.cuda.set_device(rank)
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 2
    rank = int(sys.argv[1])
    os.environ["RANK"] = sys.argv[1]
    os.environ["WORLD_SIZE"] = str(size)
    init_process(rank, 2, run)
