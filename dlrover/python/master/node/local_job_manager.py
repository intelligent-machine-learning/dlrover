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

from dlrover.python.common.constants import NodeStatus, NodeType
from dlrover.python.common.grpc import ParallelConfig
from dlrover.python.common.node import Node
from dlrover.python.master.monitor.error_monitor import SimpleErrorMonitor
from dlrover.python.master.node.job_manager import JobManager
from dlrover.python.scheduler.job import JobArgs


class LocalJobManager(JobManager):
    """The manager manages a local job on a single node. It can:
    - collect the running metrics of a local job.
    - optimize the hyper-parameters of a local job, like batch size and
        `num_workers` of the dataloader.
    """

    def __init__(
        self,
        job_args: JobArgs,
        speed_monitor=None,
        error_monitor=None,
    ):
        super().__init__(job_args, speed_monitor, error_monitor)
        self._job_resource_optimizer = None

    def start(self):
        self._job_nodes[NodeType.WORKER] = {}
        worker = self._job_args.node_args[NodeType.WORKER].group_resource
        self._training_node_configure.set_node_num(worker.count)
        for i in range(worker.count):
            self._job_nodes[NodeType.WORKER][i] = Node(
                name=NodeType.WORKER + f"-{i}",
                node_type=NodeType.WORKER,
                node_id=i,
                status=NodeStatus.RUNNING,
            )

    def should_early_stop(self):
        return False

    def add_node_event_callback(self, node_event_callback):
        pass

    def update_node_resource_usage(
        self, node_type, node_id, cpu, memory, gpu_stats=[]
    ):
        node = self._job_nodes[node_type][node_id]
        node.update_resource_usage(cpu, memory, gpu_stats)

    def handle_training_failure(
        self, node_type, node_id, restart_count=-1, error_data="", level=""
    ):
        """Process the training failure reported by the node."""
        node = self._job_nodes[node_type][node_id]
        self._error_monitor.process_error(
            node, restart_count, error_data, level
        )

    def collect_node_heart_beat(self, node_type, node_id, timestamp):
        node = self._job_nodes[node_type][node_id]
        node.heartbeat_time = timestamp

    def close_job(self):
        pass

    def all_workers_exited(self):
        return False

    def all_workers_failed(self):
        return False

    def all_workers_deleted(self):
        return False

    def all_critical_node_completed(self):
        return False

    def remove_worker(self, worker_id):
        pass

    def get_running_nodes(self):
        nodes = list(self._job_nodes[NodeType.WORKER].values())
        return nodes

    def get_running_workers(self):
        workers = list(self._job_nodes[NodeType.WORKER].values())
        return workers

    def post_ps_ready(self):
        pass

    def stop(self):
        self._stopped = True

    def update_node_service_addr(self, node_type, node_id, service_addr):
        pass

    def get_cur_cluster_ps(self):
        return []

    def get_next_cluster_ps(self):
        return []

    def ready_for_new_ps_cluster(self):
        return True

    def has_ps_failure(self):
        return False

    def remove_training_nodes(self):
        """Remove all PS and workers"""
        pass

    def start_auto_scaling(self):
        pass

    def all_running_node_hanged(self):
        return False

    def remove_not_joined_rdzv_workers(self, worker_ranks):
        pass

    def pend_without_workers(self):
        return False

    def update_allreduce_node_unit(self, node_unit):
        pass

    def verify_restarting_worker_training(self, node_type, node_id):
        return False

    def get_opt_strategy(self) -> ParallelConfig:
        strategy = self._job_strategy_generator.generate_opt_strategy()
        return strategy

    def update_node_paral_config(self, node_type, node_id, paral_config):
        node = self._job_nodes[node_type][node_id]
        node.update_paral_config(paral_config)


def create_job_manager(args: JobArgs, speed_monitor) -> LocalJobManager:
    return LocalJobManager(
        job_args=args,
        speed_monitor=speed_monitor,
        error_monitor=SimpleErrorMonitor(),
    )
