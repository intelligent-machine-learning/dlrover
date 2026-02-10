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

from dlrover.python.common.constants import TrainingExceptionLevel
from dlrover.python.common.event.reporter import get_event_reporter
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.node import Node, NodeEvent
from dlrover.python.diagnosis.common.diagnosis_action import DiagnosisAction
from dlrover.python.master.hyperparams.simple_strategy_generator import (
    SimpleStrategyGenerator,
)
from dlrover.python.master.monitor.perf_monitor import PerfMonitor
from dlrover.python.master.node.job_context import get_job_context
from dlrover.python.master.node.training_node import (
    SyncNodeTrainingPorts,
    TrainingNodeConfig,
)
from dlrover.python.master.resource.job import JobResource
from dlrover.python.scheduler.job import JobArgs
from dlrover.python.scheduler.kubernetes import k8sClient


class JobManager(metaclass=ABCMeta):
    """The manager manages the status of a job including the running
    nodes and training hyper-parameters.
    """

    def __init__(
        self,
        job_args: JobArgs,
        perf_monitor=None,
        external_config=None,
    ):
        self._job_resource = JobResource()
        self._job_args = job_args
        self._k8s_client = k8sClient.singleton_instance(job_args.namespace)
        self._job_strategy_generator: SimpleStrategyGenerator = (
            SimpleStrategyGenerator(self._job_args.job_uuid)
        )
        self._perf_monitor: PerfMonitor = perf_monitor
        self._event_reporter = get_event_reporter()

        self._stopped = False
        self._restart_errors: Dict[int, str] = {}
        self._nodes_required = (0, 0, 0)

        self._training_node_config = TrainingNodeConfig(external_config)
        self._job_context = get_job_context()

    @property
    def job_uid(self):
        return self._job_args.job_uuid

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def restart(self):
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
    def collect_node_heart_beat(
        self, node_type, node_id, timestamp
    ) -> DiagnosisAction:
        """Collect the heart beat message of nodes."""
        pass

    def get_job_nodes(self, node_type=""):
        if node_type == "":
            return self._job_context.job_nodes()
        return self._job_context.job_nodes_by_type(node_type)

    def get_job_node_groups(self):
        return self._job_context.job_node_groups()

    def get_job_node_group(self, group):
        return self._job_context.job_node_group(group)

    def sync_node_training_port(self, node_id, port) -> SyncNodeTrainingPorts:
        return self._training_node_config.sync_node_training_port(
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

    def get_elastic_run_configs(self) -> Dict[str, str]:
        return self._training_node_config.get_elastic_run_configs()

    def process_reported_node_event(self, node_event: NodeEvent):
        """
        The node events here is reported from training agent.

        Args:
            node_event: The event from training agent.
        """

        pass

    def process_diagnosis_action(self, action: DiagnosisAction):
        """
        Procedure for diagnosis action.
        """

        pass

    def process_error(
        self, node: Node, restart_count: int, error_data: str, level: str
    ) -> bool:
        """
        Handle the error of training.

        Args:
            node: The Node instance.
            restart_count: The restart count of training on the node.
            error_data: The error message data.
            level: The error level.

        Returns:
            bool: whether to relaunch the node.
        """

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

    def _handle_process_error(
        self, node: Node, restart_count: int, error_data: str
    ):
        """
        Handle the process error of training.

        Args:
            node: The Node instance.
            error_data: The error message data.

        Returns:
            bool: whether to relaunch the node.
        """

        if restart_count not in self._restart_errors:
            self._restart_errors[restart_count] = error_data
            logger.error(
                f"{node.type}-{node.id} restart {restart_count} fails: {error_data}"
            )
        return False

    def _handle_node_error(self, node: Node, error_data: str):
        """
        Handle the node error of training.

        Args:
            node: The Node instance.
            error_data: The error message data.
        Returns:
            bool: whether to relaunch the node.
        """

        logger.info(f"{node.type}-{node.id} is down. Reason: {error_data}")
        return True
