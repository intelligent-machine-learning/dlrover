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

import abc
import collections

from dlrover.python.common.constants import NodeType
from dlrover.python.common.log_utils import default_logger as logger
from dlrover.python.master.node_watcher.base_watcher import Node

ClusterContext = collections.namedtuple("ClusterContext", ("node_manager"))


class NodeEventCallback(metaclass=abc.ABCMeta):
    """
    The interface for the observers that are interested in the node event.
    The subclass observers can override the following methods to handle
    various events.
    """

    @classmethod
    def log_callback_exception(cls, func):
        def wrapper(self, *args, **kwargs):
            try:
                func(self, *args, **kwargs)
            except Exception as e:
                logger.warning(
                    "Fail to call {}.{} ".format(
                        self.__class__.__name__, func.__name__
                    ),
                    e,
                )

        return wrapper

    @abc.abstractmethod
    def on_node_started(self, node: Node, cluster_context):
        """
        The handler for the node started event.
        Args:
            node: A Node object. It's the node that just becomes running.
            cluster_context: A ClusterContext object. It contains all the
                context information about the cluster for the job.
        """
        pass

    @abc.abstractmethod
    def on_node_succeeded(self, node: Node, cluster_context):
        """
        The handler for the node succeeded event.
        Args:
            node: A PodInfo object. It's the node that just terminates
                in success.
            cluster_context: A ClusterContext object. It contains all the
                context information about the cluster for the job.
        """
        pass

    @abc.abstractmethod
    def on_node_failed(self, node: Node, cluster_context):
        """
        The handler for the node failed event.
        Args:
            node: A PodInfo object. It's the node that just terminates
                in failure.
            cluster_context: A ClusterContext object. It contains all the
                context information about the cluster for the job.
        """
        pass

    @abc.abstractmethod
    def on_node_deleted(self, node: Node, cluster_context):
        """
        The handler for the node deleted event.
        Args:
            node: A PodInfo object. It's the node which is just deleted.
            cluster_context: A ClusterContext object. It contains all the
                context information about the cluster for the job.
        """
        pass


class TaskRescheduleCallback(NodeEventCallback):
    def __init__(self, task_manager):
        super(TaskRescheduleCallback, self).__init__()
        self._task_manager = task_manager

    def on_node_started(self, node, cluster_context):
        pass

    def on_node_succeeded(self, node, cluster_context):
        pass

    @NodeEventCallback.log_callback_exception
    def on_node_failed(self, node, cluster_context):
        if node.id is not None and node.type == NodeType.WORKER:
            self._task_manager.recover_tasks(node.id)

    @NodeEventCallback.log_callback_exception
    def on_node_deleted(self, node, cluster_context):
        if node.id is not None and node.type == NodeType.WORKER:
            self._task_manager.recover_tasks(node.id)
