import os

from .distributed.distributed import coworker_size, init_distributed, local_rank, rank, reset_distributed, world_size

os.environ["PIPPY_PIN_DEVICE"] = "0"
