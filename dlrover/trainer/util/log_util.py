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
from logging.handlers import RotatingFileHandler

import tensorflow.compat.v1 as tf

tf.logging.set_verbosity(tf.logging.INFO)


def get_log_file_path_from_env():
    return os.getenv("DLROVER_TRAINER_LOG_DIR", "./log")


DEFAULT_LEVEL = logging.INFO

DEFAULT_FORMATTER = logging.Formatter(
    "[%(asctime)s] [%(levelname)s]"
    "[%(filename)s:%(lineno)d:%(funcName)s] %(message)s"
)


class LogFactory(object):
    def __init__(self):
        self.file_handler = None
        self.stream_handler = None
        self.log_file_path = get_log_file_path_from_env() + str(os.getpid())
        self.formatter = None
        self.log_level = logging.INFO
        self.handlers = []
        self.logger = logging.getLogger("dlrover.trainer")

    def set_formatter(self, formatter=None):
        self.formatter = DEFAULT_FORMATTER or formatter
        for h in self.handlers:
            h.setFormatter(self.formatter)

    def set_log_file_path(self, file_path=None):
        self.log_file_path = file_path

    def update_log_level(self, level=None):
        self.log_level = level

    def set_log_level(self, level=None):
        for h in self.handlers:
            h.setLevel(self.log_level)
        self.logger.setLevel(level=self.log_level)

    def set_stream_handler(self):
        self.stream_handler = logging.StreamHandler()
        self.handlers.append(self.stream_handler)

    def set_file_handler(self):
        self.file_handler = RotatingFileHandler(
            self.log_file_path,
            encoding="utf8",
            maxBytes=512 * 1024 * 1024,
            backupCount=3,
        )
        self.handlers.append(self.file_handler)

    def update_default_logger(self):
        self.handlers = []
        self.set_file_handler()
        self.set_stream_handler()
        self.set_formatter()
        self.set_log_level()
        for h in self.logger.handlers:
            del h
        self.logger.handlers = []
        for h in self.handlers:
            self.logger.addHandler(h)

        logging.getLogger("tensorflow").handlers = self.handlers

    def get_default_logger(self):
        self.update_default_logger()
        return self.logger


_factory = LogFactory()
default_logger = _factory.get_default_logger()
