# Copyright 2022 The DLRover Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
import typing
from logging.handlers import RotatingFileHandler

from dlrover.python.common.constants import BasicClass

logging.basicConfig(level=logging.INFO)

_DEFAULT_LOGGER = "dlrover.logger"

_LOGGER_LEVEL_RANGE = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

_DEFAULT_FORMATTER = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s"
)

_ch = logging.StreamHandler(stream=sys.stderr)
_ch.setFormatter(_DEFAULT_FORMATTER)

_DEFAULT_HANDLERS: typing.List[logging.Handler] = [_ch]

_LOGGER_CACHE: typing.Dict[str, logging.Logger] = {}


def get_log_level():
    log_level = os.getenv(BasicClass.LOG_LEVEL_ENV)
    if not log_level or log_level not in _LOGGER_LEVEL_RANGE:
        return "INFO"
    return log_level


def get_base_log_dir():
    log_dir = os.getenv(BasicClass.LOG_ROOT_DIR_ENV, "")
    return log_dir


def get_agent_log_dir():
    log_dir = os.getenv(BasicClass.LOG_AGENT_DIR_ENV, "")
    return log_dir


def get_base_log_file():
    log_dir = get_base_log_dir()
    log_file = ""
    if log_dir:
        log_file = os.path.join(log_dir, "dlrover.log")
        os.makedirs(log_dir, exist_ok=True)
    return log_file


def get_file_handler_max_bytes():
    max_bytes_str = os.getenv(BasicClass.LOG_ROTATE_MAX_BYTES_ENV, "")
    if (
        max_bytes_str
        and max_bytes_str.isdigit()
        and int(max_bytes_str) >= 1024 * 1024
    ):
        # must >= 1mb
        return int(max_bytes_str)
    return 200 * 1024 * 1024  # default: 200MB


def get_file_handler_backup_count():
    backup_count_str = os.getenv(BasicClass.LOG_ROTATE_BACKUP_COUNT_ENV, "")
    if (
        backup_count_str
        and backup_count_str.isdigit()
        and int(backup_count_str) >= 1
    ):
        # must >= 1
        return int(backup_count_str)
    return 5  # default: 5


def get_logger(
    name,
    handlers: typing.Optional[typing.List[logging.Handler]] = None,
    update=False,
):
    __setup_extra_logger()

    if name in _LOGGER_CACHE and not update:
        return _LOGGER_CACHE[name]
    logger = logging.getLogger(name)
    logger.setLevel(get_log_level())

    if handlers is None:
        base_log_file = get_base_log_file()
        if base_log_file:
            file_handler = RotatingFileHandler(
                base_log_file,
                maxBytes=get_file_handler_max_bytes(),
                backupCount=get_file_handler_backup_count(),
            )
            file_handler.setFormatter(_DEFAULT_FORMATTER)
            handlers = [file_handler] + _DEFAULT_HANDLERS
        else:
            handlers = _DEFAULT_HANDLERS
    else:
        handlers.extend(_DEFAULT_HANDLERS)

    logger.handlers = list(handlers)
    logger.propagate = False
    return logger


def __setup_extra_logger():
    # tornado logger
    logging.getLogger("tornado.access").setLevel(logging.WARNING)


default_logger = get_logger(_DEFAULT_LOGGER)
