import hashlib
import logging
import os
import sys
import time
import traceback
import typing  # type: ignore # noqa: F401

import torch

import atorch

_DEFAULT_LOGGER = "atorch.logger"

_DEFAULT_FORMATTER = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] " "[%(filename)s:%(lineno)d:%(funcName)s] %(message)s"
)

_ch = logging.StreamHandler(stream=sys.stderr)
_ch.setFormatter(_DEFAULT_FORMATTER)

_DEFAULT_HANDLERS = [_ch]

_LOGGER_CACHE = {}  # type: typing.Dict[str, logging.Logger]


def get_logger(name, level="INFO", handlers=None, update=False):
    if name in _LOGGER_CACHE and not update:
        return _LOGGER_CACHE[name]
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = handlers or _DEFAULT_HANDLERS
    logger.propagate = False
    return logger


default_logger = get_logger(_DEFAULT_LOGGER)


logged_messages = set()


# Function to log a message only once
def log_once(message):
    call_stack = repr(traceback.extract_stack()[:-1])
    message_hash = hashlib.md5((call_stack + message).encode("utf-8")).hexdigest()
    if message_hash not in logged_messages:
        logging.info(message)
        logged_messages.add(message_hash)


class DashBoardWriter(object):
    def __init__(self, logdir="./"):
        from torch.utils.tensorboard import SummaryWriter

        self.rank = int(os.environ.get("RANK", 0))
        if self.rank == 0:
            self.writer = SummaryWriter(logdir)

    def add_scalars(self, stats, n_iter, name=None):

        key_val_list = []

        def find_key_value(prefix, stats):
            for key, value in stats.items():
                if not isinstance(value, dict):
                    key_val_list.append((prefix + key, value))
                else:
                    find_key_value(prefix + key + "/", value)

        find_key_value("", stats)

        for item in key_val_list:
            key, val = item
            if self.rank == 0:
                if isinstance(val, torch.Tensor):
                    val = val.detach().clone().float().cpu().item()
                self.writer.add_scalar(key, val, n_iter)

    def flush(self):
        if self.rank == 0:
            self.writer.flush()


class TimeStats:
    def __init__(self, name):
        self.name = name
        self.time_statics = dict()

    def __getitem__(self, key):
        return self.time_statics.get(key, None)

    def __setitem__(self, key, value):
        self.time_statics[key] = value

    def to_dashboard(self, dashboard_writer=None, n_iter=None):
        dashboard_writer.add_scalars(self.time_statics, n_iter, name=self.name)
        dashboard_writer.flush()


class Timer:
    def __init__(self, name, time_stats=None):
        self.name = name
        self.time_stats = time_stats

    def start(self):
        self.start_time = time.time()

    def end(self):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time

    def __enter__(self):
        self.start()

    def __exit__(self, type, value, trace):
        self.end()
        if self.time_stats:
            self.time_stats[self.name] = self.elapsed_time
            if os.environ.get("ATORCH_DEBUG"):
                rank = atorch.local_rank()
                default_logger.info("{} on rank {} cost: {} s".format(self.name, rank, self.elapsed_time))
