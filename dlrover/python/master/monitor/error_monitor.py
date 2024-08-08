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

from dlrover.python.common.constants import TrainingExceptionLevel
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.node import Node


class ErrorMonitor(metaclass=ABCMeta):
    @abstractmethod
    def process_error(
        self, node: Node, restart_count: int, error_data: str, level: str
    ) -> bool:
        """
        Handle the error of training processes.

        Args:
            node: a Node instance.
            restart_count: the restart count of training on the node.
            error_data: The error message data.
            level: the error level.

        Returns:
            bool: wether to relaunch the node.
        """
        pass

    @abstractmethod
    def report_event(
        self,
        event_type: str,
        instance: str,
        action: str,
        msg: str,
        labels: Dict[str, str],
    ):
        pass


class SimpleErrorMonitor(ErrorMonitor):
    """The monitor logs the error data."""

    def __init__(self):
        self._restart_errors: Dict[int, str] = {}

    def process_error(
        self, node: Node, restart_count: int, error_data: str, level: str
    ) -> bool:
        if level == TrainingExceptionLevel.PROCESS_ERROR:
            return self._handle_process_error(node, restart_count, error_data)
        elif level == TrainingExceptionLevel.NODE_ERROR:
            return self._handle_node_error(node, error_data)
        elif level == TrainingExceptionLevel.RDZV_ERROR:
            logger.error(f"Rendezvous fails with reason {error_data}")
        elif level == TrainingExceptionLevel.WARNING:
            logger.warning(error_data)
        elif level == TrainingExceptionLevel.ERROR:
            logger.error(error_data)
        return False

    def report_event(
        self,
        event_type: str,
        instance: str,
        action: str,
        msg: str,
        labels: Dict[str, str],
    ):
        pass

    def _handle_process_error(
        self, node: Node, restart_count: int, error_data: str
    ):
        if restart_count not in self._restart_errors:
            self._restart_errors[restart_count] = error_data
            logger.error(
                f"{node.type}-{node.id} restart {restart_count} "
                f"fails: {error_data}"
            )
        return False

    def _handle_node_error(self, node: Node, error_data: str):
        logger.info(f"{node.type}-{node.id} is down. Reason: {error_data}")
        return True


class K8sJobErrorMonitor(ErrorMonitor):
    """The monitor logs the error data."""

    def __init__(self, namespace="", cordon_node_eanbled=False):
        from dlrover.python.scheduler.kubernetes import k8sClient

        self.cordon_node_eanbled = cordon_node_eanbled
        self._k8s_client = k8sClient.singleton_instance(namespace)
        self._restart_errors: Dict[int, str] = {}

    def process_error(
        self, node: Node, restart_count: int, error_data: str, level: str
    ) -> bool:
        if level == TrainingExceptionLevel.PROCESS_ERROR:
            return self._handle_process_error(node, restart_count, error_data)
        elif level == TrainingExceptionLevel.NODE_ERROR:
            return self._handle_node_error(node, error_data)
        elif level == TrainingExceptionLevel.RDZV_ERROR:
            logger.error(f"Rendezvous fails with reason {error_data}")
        elif level == TrainingExceptionLevel.WARNING:
            logger.warning(error_data)
        elif level == TrainingExceptionLevel.ERROR:
            logger.error(error_data)
        return False

    def report_event(
        self,
        event_type: str,
        instance: str,
        action: str,
        msg: str,
        labels: Dict[str, str],
    ):
        pass

    def _handle_process_error(
        self, node: Node, restart_count: int, error_data: str
    ):
        if restart_count not in self._restart_errors:
            self._restart_errors[restart_count] = error_data
            logger.error(
                f"{node.type}-{node.id} on {node.host_name} "
                f"restart {restart_count} fails: {error_data}"
            )
        return False

    def _handle_node_error(self, node: Node, error_data: str):
        logger.info(
            f"{node.name} on {node.host_name} is down. "
            f"Reason: {error_data}"
        )
        if self.cordon_node_eanbled:
            succeed = self._k8s_client.cordon_node(node.host_name)
            if succeed:
                logger.info(f"Node {node.name} is marked unschedulable.")
        return True
