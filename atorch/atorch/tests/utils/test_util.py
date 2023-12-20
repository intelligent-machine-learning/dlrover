import os

import torch

import atorch


def init_dist(rank, world_size):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["NPROC_PER_NODE"] = str(world_size)

    atorch.init_distributed("nccl")
    torch.cuda.device(atorch.local_rank())
