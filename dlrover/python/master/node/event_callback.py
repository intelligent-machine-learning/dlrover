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
import sys
from datetime import datetime
from typing import Dict

from dlrover.python.common.constants import (
    JobExitReason,
    NodeExitReason,
    NodeType,
    RendezvousName,
    TrainingExceptionLevel,
)
from dlrover.python.common.global_context import Context
from dlrover.python.common.log import default_logger as logger
from dlrover.python.master.elastic_training.rdzv_manager import (
    RendezvousManager,
)
from dlrover.python.master.monitor.speed_monitor import SpeedMonitor
from dlrover.python.master.watcher.base_watcher import Node

_dlrover_ctx = Context.singleton_instance()


class ClusterContext(object):
    def __init__(self, job_manager):
        self.job_manager = job_manager


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
        if node.id is not None:
            self._task_manager.recover_tasks(node.type, node.id)

    @NodeEventCallback.log_callback_exception
    def on_node_deleted(self, node, cluster_context):
        if node.id is not None and node.type == NodeType.WORKER:
            self._task_manager.recover_tasks(node.type, node.id)


class TFPSNodeHandlingCallback(NodeEventCallback):
    def __init__(self, master):
        super(TFPSNodeHandlingCallback, self).__init__()
        self._master = master

    def get_job_exit_reason(self, node: Node):
        if node.type == NodeType.PS and node.exit_reason == NodeExitReason.OOM:
            return JobExitReason.PS_OOM_ERROR
        if self._master.task_manager.training_started():
            if node.type == NodeType.WORKER:
                if node.exit_reason == NodeExitReason.OOM:
                    return JobExitReason.WORKER_OOM
                else:
                    return JobExitReason.WORKER_ERROR
            elif node.type == NodeType.PS:
                return JobExitReason.PS_ERROR
            elif node.type == NodeType.EVALUATOR:
                if node.exit_reason == NodeExitReason.OOM:
                    return JobExitReason.EVALUATOR_OOM
                else:
                    return JobExitReason.EVALUATOR_ERROR
            else:
                return JobExitReason.UNKNOWN_ERROR
        else:
            return JobExitReason.CODE_ERROR

    @NodeEventCallback.log_callback_exception
    def on_node_started(self, node: Node, cluster_context):
        pass

    @NodeEventCallback.log_callback_exception
    def on_node_succeeded(self, node: Node, cluster_context: ClusterContext):
        node.finish_time = datetime.now()  # type: ignore
        job_manager = cluster_context.job_manager
        if node.critical:
            completed = job_manager.all_critical_node_completed()
            if completed:
                self._master.request_stop(
                    success=True,
                    reason=JobExitReason.SUCCEEDED,
                    msg="All critical nodes completed",
                )
        self._master.speed_monitor.reduce_target_worker_num(
            [(node.type, node.id)]
        )
        self._master.speed_monitor.remove_running_worker(node.type, node.id)
        self._master.sync_service.remove_exited_worker_sync(node.type, node.id)

    @NodeEventCallback.log_callback_exception
    def on_node_failed(self, node: Node, cluster_context):
        node.finish_time = datetime.now()  # type: ignore
        self._stop_job_if_needed(node)
        if node.type == NodeType.PS:
            self._master.elastic_ps_service.inc_global_cluster_version()
        if node.is_unrecoverable_failure():
            self._master.speed_monitor.reduce_target_worker_num(
                [(node.type, node.id)]
            )
        self._master.speed_monitor.remove_running_worker(node.type, node.id)
        self._master.sync_service.remove_exited_worker_sync(node.type, node.id)

    @NodeEventCallback.log_callback_exception
    def on_node_deleted(self, node, cluster_context):
        node.finish_time = datetime.now()  # type: ignore
        self._stop_job_if_needed(node)
        if node.type == NodeType.PS:
            self._master.elastic_ps_service.inc_global_cluster_version()
        self._master.speed_monitor.remove_running_worker(node.type, node.id)
        self._master.sync_service.remove_exited_worker_sync(node.type, node.id)

    def _stop_job_if_needed(self, node: Node):
        if node.critical and node.is_unrecoverable_failure():
            job_exit_reason = self.get_job_exit_reason(node)
            self._master.request_stop(
                success=False,
                reason=job_exit_reason,
                msg=(
                    "Critical node (type={}, id={}) is failed "
                    "and {}.".format(
                        node.type, node.id, node.unrecoverable_failure_msg
                    )
                ),
            )


class AllReduceNodeHandlingCallback(NodeEventCallback):
    def __init__(self, master):
        super(AllReduceNodeHandlingCallback, self).__init__()
        self._master = master
        self._speed_monitor: SpeedMonitor = self._master.speed_monitor
        self._rdzv_managers: Dict[
            str, RendezvousManager
        ] = self._master.rdzv_managers
        rdzv_manager = self._rdzv_managers.get(
            RendezvousName.ELASTIC_TRAINING, None
        )
        if rdzv_manager:
            self._min_node = rdzv_manager.get_min_nodes()
        else:
            self._min_node = sys.maxsize
        self._failed_worker_count = 0
        self._total_worker_num = self._master.job_manager.get_worker_num()
        self._available_worker_num = self._total_worker_num

    def get_job_exit_reason(self, node: Node):
        if self._master.task_manager.training_started():
            if node.type == NodeType.WORKER:
                if node.exit_reason == NodeExitReason.OOM:
                    return JobExitReason.WORKER_OOM
                else:
                    return JobExitReason.WORKER_ERROR
            else:
                return JobExitReason.UNKNOWN_ERROR
        else:
            return JobExitReason.CODE_ERROR

    @NodeEventCallback.log_callback_exception
    def on_node_started(self, node: Node, cluster_context):
        if node.type == NodeType.WORKER and node.id == 0:
            self._master.job_manager.start_auto_scaling()
        for manager in self._rdzv_managers.values():
            manager.add_alive_node(node)

    @NodeEventCallback.log_callback_exception
    def on_node_succeeded(self, node: Node, cluster_context: ClusterContext):
        node.finish_time = datetime.now()  # type: ignore
        job_manager = self._master.job_manager
        if node.critical:
            completed = job_manager.all_critical_node_completed()
            if completed:
                self._master.request_stop(
                    success=True,
                    reason=JobExitReason.SUCCEEDED,
                    msg="All critical nodes completed",
                )
        self._speed_monitor.remove_running_worker(node.type, node.id)
        self._remove_node_from_rdzv(node)

    @NodeEventCallback.log_callback_exception
    def on_node_failed(self, node: Node, cluster_context):
        node.finish_time = datetime.now()  # type: ignore
        self._failed_worker_count += 1
        self._stop_job_if_needed(node)
        if node.is_unrecoverable_failure():
            self._master.speed_monitor.reduce_target_worker_num(
                [(node.type, node.id)]
            )
        if node.exit_reason == NodeExitReason.HARDWARE_ERROR:
            self._master.job_manager.handle_training_failure(
                node.type,
                node.id,
                error_data=NodeExitReason.HARDWARE_ERROR,
                level=TrainingExceptionLevel.NODE_ERROR,
            )
        self._remove_node_from_rdzv(node)

    @NodeEventCallback.log_callback_exception
    def on_node_deleted(self, node, cluster_context):
        node.finish_time = datetime.now()  # type: ignore
        self._stop_job_if_needed(node)
        self._remove_node_from_rdzv(node)

    def _remove_node_from_rdzv(self, node):
        for manager in self._rdzv_managers.values():
            manager.remove_alive_node(node)

    def _stop_job_if_needed(self, node: Node):
        stop_node = False
        if node.exit_reason == NodeExitReason.FATAL_ERROR:
            if not _dlrover_ctx.relaunch_always:
                logger.info(
                    f"Need to stop job for node: {node.name} "
                    "has fatal error."
                )
                stop_node = True
        if node.relaunch_count >= node.max_relaunch_count:
            logger.info(
                "Need to stop job for node relaunching "
                f"count: {node.relaunch_count} "
                f"over limit: {node.max_relaunch_count}."
            )
            self._available_worker_num -= 1
            stop_node = True

        job_exit_reason = self.get_job_exit_reason(node)
        max_failure_num = max(self._total_worker_num, node.max_relaunch_count)
        if node.critical and stop_node:
            self._master.request_stop(
                success=False,
                reason=job_exit_reason,
                msg=(
                    "Critical node (type={}, id={}) is failed "
                    "and {}.".format(
                        node.type, node.id, node.unrecoverable_failure_msg
                    )
                ),
            )
        elif self._failed_worker_count >= max_failure_num:
            # The job early stops if there are a lot of failed workers.
            self._master.request_stop(
                success=False,
                reason=job_exit_reason,
                msg=(
                    "The number of worker failure exceeds the "
                    f"worker count {self._total_worker_num} "
                ),
            )
        elif self._available_worker_num < self._min_node:
            self._master.request_stop(
                success=False,
                reason=job_exit_reason,
                msg=(
                    "The available number of worker is less than the minimum"
                    f"number {self._min_node} of redzv  "
                ),
            )
