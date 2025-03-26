# Copyright 2025 The DLRover Authors. All rights reserved.
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

import json
import logging
import os
import sys
import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from dlrover.python.common.singleton import Singleton

ENV_PREFIX = "DLROVER_EVENT"
DEFAULT_EVENT_EXPORTER = "TEXT_FILE"
DEFAULT_FILE_DIR = "/tmp/dlrover"
DEFAULT_TEXT_FORMATTER = "LOG"


def parse_bool(val) -> Optional[bool]:
    if isinstance(val, bool):
        return val
    elif isinstance(val, str):
        if val.lower() in ["true", "1", "yes", "y", "on", "enable", "enabled"]:
            return True
        elif val.lower() in [
            "false",
            "0",
            "no",
            "n",
            "off",
            "disable",
            "disabled",
        ]:
            return False
        else:
            return None
    else:
        return None


def parse_int(val) -> Optional[int]:
    if isinstance(val, int):
        return val
    elif isinstance(val, str):
        return int(val)
    else:
        return None


def parse_str(val) -> Optional[str]:
    if isinstance(val, str):
        return val
    else:
        return None


@lru_cache(maxsize=1)
def get_pid() -> str:
    pid = os.getpid()
    if pid == 0:
        return "empty"
    return str(pid)


@lru_cache(maxsize=1)
def get_rank() -> str:
    rank = os.getenv("RANK", "0")
    if rank == "":
        return "empty"
    return rank


def get_env_key(key):
    key = key.upper()
    return f"{ENV_PREFIX}_{key}"


def is_dlrover_event_enabled():
    config = Config.singleton_instance()
    return True if config.enable is True else False


@dataclass
class Config(Singleton):

    event_exporter: str = DEFAULT_EVENT_EXPORTER
    async_exporter: bool = False
    queue_size: int = 1024
    file_dir: str = DEFAULT_FILE_DIR
    text_formatter: str = DEFAULT_TEXT_FORMATTER
    hook_error: bool = False

    rank: str = "empty"
    pid: str = "empty"

    enable: bool = True
    debug_mode: bool = False

    def __init__(self, config_file: Optional[str] = None):

        self.rank = get_rank()
        self.pid = get_pid()

        if config_file is not None:
            self._load_from_file(config_file)

        self._init_from_env()

    def _load_from_file(self, config_file: str):
        try:
            with open(config_file, "r") as f:
                config = json.load(f)

            self._set_if_valid(config, "event_exporter", parser=parse_str)
            self._set_if_valid(config, "async_exporter", parser=parse_bool)
            self._set_if_valid(config, "queue_size", parser=parse_int)
            self._set_if_valid(config, "file_dir", parser=parse_str)
            self._set_if_valid(config, "text_formatter", parser=parse_str)
            self._set_if_valid(config, "enable", parser=parse_bool)
            self._set_if_valid(config, "debug_mode", parser=parse_bool)
            self._set_if_valid(config, "hook_error", parser=parse_bool)

        except Exception:
            # we don't have logger now, so we can't log the error
            pass

    def _init_from_env(self):
        self._set_if_valid(
            os.environ,
            "event_exporter",
            key_converter=get_env_key,
            parser=parse_str,
        )
        self._set_if_valid(
            os.environ,
            "async_exporter",
            key_converter=get_env_key,
            parser=parse_bool,
        )
        self._set_if_valid(
            os.environ,
            "queue_size",
            key_converter=get_env_key,
            parser=parse_int,
        )
        self._set_if_valid(
            os.environ, "file_dir", key_converter=get_env_key, parser=parse_str
        )
        self._set_if_valid(
            os.environ,
            "text_formatter",
            key_converter=get_env_key,
            parser=parse_str,
        )
        self._set_if_valid(
            os.environ, "enable", key_converter=get_env_key, parser=parse_bool
        )
        self._set_if_valid(
            os.environ,
            "debug_mode",
            key_converter=get_env_key,
            parser=parse_bool,
        )
        self._set_if_valid(
            os.environ,
            "hook_error",
            key_converter=get_env_key,
            parser=parse_bool,
        )

    def _set_if_valid(self, dict_val, key, key_converter=None, parser=None):

        if key_converter is not None:
            converted_key = key_converter(key)
        else:
            converted_key = key

        val = dict_val.get(converted_key)
        if val is not None:
            if parser is not None:
                try:
                    val = parser(val)
                except Exception:
                    pass

        if val is not None:
            setattr(self, key, val)

    def init_logger(self):
        logger = logging.getLogger("training_event_default")
        if self.debug_mode:
            level = logging.DEBUG
        else:
            level = logging.INFO

        logger.setLevel(level)

        if self.event_exporter == "TEXT_FILE":
            if not os.path.exists(self.file_dir):
                os.makedirs(self.file_dir, exist_ok=True)
            handler = logging.FileHandler(
                os.path.join(
                    self.file_dir, f"events_sys_{self.rank}_{self.pid}.log"
                )
            )
        elif self.event_exporter == "CONSOLE":
            handler = logging.StreamHandler(sys.stdout)
        else:
            raise ValueError(f"Invalid event exporter: {self.event_exporter}")

        handler.setLevel(level)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
        return logger


__default_logger_lock = threading.Lock()


def get_default_logger():
    if not hasattr(get_default_logger, "_logger"):
        with __default_logger_lock:
            if not hasattr(get_default_logger, "_logger"):
                config = Config.singleton_instance()
                get_default_logger._logger = config.init_logger()
                get_default_logger._logger.info(f"event config: {config}")
    return get_default_logger._logger
