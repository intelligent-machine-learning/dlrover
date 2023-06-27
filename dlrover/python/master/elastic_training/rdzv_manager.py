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

import time
from threading import Lock
from typing import Dict

from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.node import Node


class RendezvousParameters(object):
    """Holds the parameters to construct rendezvous.
    Args:
        min_nodes:
            The minimum number of nodes to admit to the rendezvous.
        max_nodes:
            The maximum number of nodes to admit to the rendezvous.
        waiting_timeout:
            An additional wait amount before completing the rendezvous once
            the rendezvous has the minimum number of required participants.
            Default 30s,
    """

    def __init__(
        self,
        min_nodes: int,
        max_nodes: int,
        waiting_timeout=30,
    ):
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.waiting_timeout = waiting_timeout


class RendezvousManager(object):
    """RendezvousManager runs on the DLRover master. The manager
    add workers into a waiting list and completes a rendezvous
    if the number of workers in the wait list is beyond the mininum
    nodes.
    """

    def __init__(self):
        self._lock = Lock()
        self._alive_nodes = set()
        self._scale_down_ts = 0
        self._released_workers = []
        self._waiting_nodes: Dict[int, int] = {}
        self._rdzv_nodes = {}
        self._lastcall_time = 0
        self._rdzv_params = RendezvousParameters(0, 0)
        self._rdzv_round = 0

    def update_rdzv_params(self, min_nodes, max_ndoes, waiting_timeout):
        self._rdzv_params.min_nodes = min_nodes
        self._rdzv_params.max_nodes = max_ndoes
        self._rdzv_params.waiting_timeout = waiting_timeout

    def add_alive_worker(self, worker: Node):
        self._alive_nodes.add(worker.id)
        logger.info(f"Add alive worker {worker.name} to Rendezvous.")

    def remove_alive_worker(self, worker: Node):
        if worker.name in self._alive_nodes:
            self._alive_nodes.remove(worker.id)
            logger.info(f"Remove exited worker {worker.name} from Rendezvous.")

    def get_released_workers(self):
        return []

    def get_comm_world(self):
        with self._lock:
            rdzv_completed = False
            if self._rdzv_nodes:
                return self._rdzv_nodes
            if len(self._waiting_nodes) == self._rdzv_params.max_nodes:
                rdzv_completed = True
            else:
                waiting_num = len(self._waiting_nodes)
                alive_num = len(self._alive_nodes)
                waiting_time = time.time() - self._lastcall_time
                rdzv_completed = (
                    waiting_num == self._rdzv_params.min_nodes
                    and waiting_num == alive_num
                    and waiting_time >= self._rdzv_params.waiting_timeout
                )

            if rdzv_completed:
                self._rdzv_nodes = dict(sorted(self._waiting_nodes.items()))
                self._waiting_nodes = dict()
                self._lastcall_time = 0
                logger.info(
                    f"Completed round {self._rdzv_round}"
                    f"rendezvous {self._rdzv_nodes}"
                )
                self._rdzv_round += 1

            return self._rdzv_nodes

    def join_rendezvous(self, node_id, worker_num):
        with self._lock:
            if node_id in self._waiting_nodes:
                return
            self._waiting_nodes[node_id] = worker_num
            self._rdzv_nodes = {}
            if len(self._waiting_nodes) >= self._rdzv_params.min_nodes:
                if self._lastcall_time == 0:
                    self._lastcall_time = time.time()
        return self._rdzv_round

    def num_nodes_waiting(self):
        with self._lock:
            return len(self._waiting_nodes)
