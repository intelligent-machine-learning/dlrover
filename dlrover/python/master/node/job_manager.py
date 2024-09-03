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

from abc import ABCMeta, abstractmethod
from typing import Dict

from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.node import Node
from dlrover.python.master.hyperparams.simple_strategy_generator import (
    SimpleStrategyGenerator,
)
from dlrover.python.master.monitor.error_monitor import ErrorMonitor
from dlrover.python.master.monitor.speed_monitor import SpeedMonitor
from dlrover.python.master.node.training_node import (
    SyncNodeTrainingPorts,
    TrainingNodeConfigure,
)
from dlrover.python.master.resource.job import JobResource
from dlrover.python.scheduler.job import JobArgs


class JobManager(metaclass=ABCMeta):
    """The manager manages the status of a job including the running
    nodes and training hyper-parameters.
    """

    def __init__(
        self,
        job_args: JobArgs,
        speed_monitor=None,
        error_monitor=None,
    ):
        self._job_resource = JobResource()
        self._job_args = job_args
        self._job_strategy_generator: SimpleStrategyGenerator = (
            SimpleStrategyGenerator(self._job_args.job_uuid)
        )

        self._stopped = False
        self._speed_monitor: SpeedMonitor = speed_monitor
        self._error_monitor: ErrorMonitor = error_monitor

        self._job_nodes: Dict[str, Dict[int, Node]] = {}
        self._nodes_required = (0, 0, 0)

        self._training_node_configure = TrainingNodeConfigure()

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def add_node_event_callback(self, node_event_callback):
        pass

    @abstractmethod
    def update_node_resource_usage(
        self, node_type, node_id, cpu, memory, gpu_stats=[]
    ):
        pass

    @abstractmethod
    def close_job(self):
        pass

    @abstractmethod
    def all_workers_exited(self):
        return False

    @abstractmethod
    def all_workers_failed(self):
        return False

    @abstractmethod
    def all_workers_deleted(self):
        return False

    @abstractmethod
    def all_critical_node_completed(self):
        return False

    @abstractmethod
    def remove_worker(self, worker_id):
        pass

    @abstractmethod
    def get_running_nodes(self):
        pass

    @abstractmethod
    def get_running_workers(self):
        pass

    @abstractmethod
    def post_ps_ready(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    def update_node_service_addr(self, node_type, node_id, service_addr):
        pass

    @abstractmethod
    def get_cur_cluster_ps(self):
        pass

    @abstractmethod
    def get_next_cluster_ps(self):
        pass

    @abstractmethod
    def ready_for_new_ps_cluster(self):
        pass

    @abstractmethod
    def has_ps_failure(self):
        pass

    @abstractmethod
    def remove_training_nodes(self):
        """Remove all PS and workers"""
        pass

    @abstractmethod
    def start_auto_scaling(self):
        pass

    @abstractmethod
    def all_running_node_hanged(self):
        pass

    @abstractmethod
    def remove_not_joined_rdzv_workers(self, worker_ranks):
        pass

    @abstractmethod
    def pend_without_workers(self):
        pass

    @abstractmethod
    def update_allreduce_node_unit(self, node_unit):
        pass

    @abstractmethod
    def should_early_stop(self):
        """
        Should the job be stopped early?

        Returns:
            result(bool): True if the job should be stopped early.
            reason: The short reason(constant) of the job being stopped.
            msg: The long reason of the job being stopped.
        """
        pass

    @abstractmethod
    def get_opt_strategy(self):
        pass

    @abstractmethod
    def update_node_paral_config(self, node_type, node_id, paral_config):
        pass

    @abstractmethod
    def verify_restarting_worker_training(self, node_type, node_id):
        """
        Verify the necessity of restarting the training process
        on the worker nodes.

        Returns:
            bool
        """
        pass

    @abstractmethod
    def handle_training_failure(
        self, node_type, node_id, restart_count=-1, error_data="", level=""
    ):
        """Process the training failure reported by the node."""
        pass

    @abstractmethod
    def collect_node_heart_beat(self, node_type, node_id, timestamp):
        """Collect the heart beat message of nodes."""
        pass

    def sync_node_training_port(self, node_id, port) -> SyncNodeTrainingPorts:
        return self._training_node_configure.sync_node_training_port(
            node_id, port
        )

    def update_node_required_info(self, min_required, max_required, timeout):
        """
        Update the nodes min/max requirements.

        Args:
            min_required(int): Minimum number of nodes for training.
            max_required(int): Maximum number of nodes for training.
            timeout(int): Required timeout in seconds.
        """

        if 0 < min_required <= max_required and max_required > 0:
            self._nodes_required = (min_required, max_required, timeout)
            self.update_node_required_info_callback()
        else:
            logger.warning(
                f"Invalid required info, min_required: {min_required}, "
                f"max_required: {max_required}, "
                f"required_timeout: {timeout}."
            )

    def update_node_required_info_callback(self):
        """Callback when 'update_node_required_info' is invoked."""

        pass
