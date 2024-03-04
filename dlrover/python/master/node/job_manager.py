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
from abc import ABCMeta, abstractclassmethod


class JobManager(metaclass=ABCMeta):
    """The manager manages the status of a job including the running
    nodes and training hyper-parameters.
    """

    @abstractclassmethod
    def start(self):
        pass

    @abstractclassmethod
    def add_node_event_callback(self, node_event_callback):
        pass

    @abstractclassmethod
    def update_node_resource_usage(
        self, node_type, node_id, cpu, memory, gpu_stats=[]
    ):
        pass

    @abstractclassmethod
    def close_job(self):
        pass

    @abstractclassmethod
    def all_workers_exited(self):
        return False

    @abstractclassmethod
    def all_workers_failed(self):
        return False

    @abstractclassmethod
    def all_workers_deleted(self):
        return False

    @abstractclassmethod
    def all_critical_node_completed(self):
        return False

    @abstractclassmethod
    def remove_worker(self, worker_id):
        pass

    @abstractclassmethod
    def get_running_nodes(self):
        pass

    @abstractclassmethod
    def get_running_workers(self):
        pass

    @abstractclassmethod
    def post_ps_ready(self):
        pass

    @abstractclassmethod
    def stop(self):
        pass

    def update_node_service_addr(self, node_type, node_id, service_addr):
        pass

    @abstractclassmethod
    def get_cur_cluster_ps(self):
        pass

    @abstractclassmethod
    def get_next_cluster_ps(self):
        pass

    @abstractclassmethod
    def ready_for_new_ps_cluster(self):
        pass

    @abstractclassmethod
    def has_ps_failure(self):
        pass

    @abstractclassmethod
    def remove_training_nodes(self):
        """Remove all PS and workers"""
        pass

    @abstractclassmethod
    def start_auto_scaling(self):
        pass

    @abstractclassmethod
    def all_running_node_hanged(self):
        pass

    @abstractclassmethod
    def remove_not_joined_rdzv_workers(self, worker_ranks):
        pass

    @abstractclassmethod
    def pend_without_workers(self):
        pass

    @abstractclassmethod
    def update_allreduce_node_unit(self, node_unit):
        pass

    @abstractclassmethod
    def early_stop(self):
        pass

    @abstractclassmethod
    def get_opt_strategy(self):
        pass

    @abstractclassmethod
    def update_node_paral_config(self, node_type, node_id, paral_config):
        pass

    @abstractclassmethod
    def verify_restarting_worker_training(self, node_type, node_id):
        """
        Verify the necessity of restarting the training process
        on the worker nodes.

        Returns:
            bool
        """
        pass

    @abstractclassmethod
    def handle_training_failure(
        self, node_type, node_id, restart_count=-1, error_data="", level=""
    ):
        """Process the training failure reported by the node."""
        pass

    @abstractclassmethod
    def collect_node_heart_beat(self, node_type, node_id, timestamp):
        """Collect the heart beat message of nodes."""
        pass
