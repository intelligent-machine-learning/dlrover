import torch.distributed as dist
import torch
import time
from datetime import timedelta
import os
from dlrover.python.common.log import default_logger as logger


def bm_all_gather(shape):
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    data = torch.randn(shape, dtype=torch.float32).to(f'cuda:{local_rank}')
    tensor_list = [
        torch.zeros_like(data).to(f'cuda:{local_rank}') for _ in range(world_size)
    ]
    start = int(time.time())
    for _ in range(10):
        dist.all_gather(tensor_list, data)
    end = time.time()
    if local_rank == 0:
        logger.info(f"Networker test cost {end - start}s")


def main():
    shape = 1 << 20
    bm_all_gather(shape)


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        dist.init_process_group("nccl", timeout=timedelta(seconds=60))
    else:
        dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    main()
    dist.destroy_process_group()
