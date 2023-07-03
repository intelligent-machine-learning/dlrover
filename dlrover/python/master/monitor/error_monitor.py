# Copyright 2023 The DLRover Authors. All rights reserved.
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

from abc import ABCMeta, abstractmethod
from typing import Dict

from dlrover.python.common.constants import NodeErrorMessage
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.node import Node


class ErrorMonitor(metaclass=ABCMeta):
    @abstractmethod
    def handle_process_error(
        self, node: Node, restart_count: int, error_data: str
    ):
        """Handle the error of training processes."""
        pass

    @abstractmethod
    def handle_node_error(self, node: Node, error_data: str):
        """Handle the error of node."""
        pass


class ErrorLogMonitor(ErrorMonitor):
    """The monitor logs the error data."""

    def __init__(self):
        self._restart_errors: Dict[int, str] = {}

    def handle_process_error(
        self, node: Node, restart_count: int, error_data: str
    ):
        if restart_count not in self._restart_errors:
            self._restart_errors[restart_count] = error_data
            logger.error(
                f"{node.type}-{node.id} {restart_count} fails: {error_data}"
            )

    def handle_node_error(self, node: Node, error_data: str):
        if error_data == NodeErrorMessage.NETWORKER_ERROR:
            logger.error(f"{node.name} is breakdown.")
