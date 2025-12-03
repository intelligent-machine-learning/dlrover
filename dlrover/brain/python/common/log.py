import logging
import os
import sys
import typing

from dlrover.python.common.constants import BasicClass

logging.basicConfig(level=logging.INFO)

_DEFAULT_LOGGER = "dlrover.logger"

_LOGGER_LEVEL_RANGE = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

_DEFAULT_FORMATTER = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s"
)

_ch = logging.StreamHandler(stream=sys.stderr)
_ch.setFormatter(_DEFAULT_FORMATTER)

_DEFAULT_HANDLERS = [_ch]

_LOGGER_CACHE: typing.Dict[str, logging.Logger] = {}


def get_log_level():
    log_level = os.getenv(BasicClass.LOG_LEVEL_ENV)
    if not log_level or log_level not in _LOGGER_LEVEL_RANGE:
        return "INFO"
    return log_level


def get_logger(name, handlers=None, update=False):
    __setup_extra_logger()

    if name in _LOGGER_CACHE and not update:
        return _LOGGER_CACHE[name]
    logger = logging.getLogger(name)
    logger.setLevel(get_log_level())
    logger.handlers = handlers or _DEFAULT_HANDLERS
    logger.propagate = False
    return logger


def __setup_extra_logger():
    # tornado logger
    logging.getLogger("tornado.access").setLevel(logging.WARNING)


default_logger = get_logger(_DEFAULT_LOGGER)