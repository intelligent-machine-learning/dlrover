import os

from . import normalization, optim
from .distributed.distributed import coworker_size, init_distributed, local_rank, rank, reset_distributed, world_size
from .utils.metrics_reporter import report_import
from .zero import ZeroDDP

report_import()

os.environ["PIPPY_PIN_DEVICE"] = "0"
