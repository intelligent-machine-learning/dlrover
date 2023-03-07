# Copyright 2020 The ElasticDL Authors. All rights reserved.
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


import threading
import time
from typing import Dict, List, Set, Tuple

from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.node import Node
from dlrover.python.master.node.job_manager import JobManager

_WAIT_SYNC_TINEOUT = 3600


class SyncService(object):
    def __init__(self, job_manager):
        self._job_manager: JobManager = job_manager
        self._sync_objs_target: Dict[str, List[Tuple[str, int]]] = {}
        self._finished_barriers: Set[str] = set()
        self._lock = threading.Lock()
        self._sync_start_time: Dict[str, float] = {}
        self.timeout = _WAIT_SYNC_TINEOUT
        threading.Thread(
            target=self.delete_sync_timeout_worker,
            name="sync timeout worker monitor",
            daemon=True,
        ).start()

    def join_sync(self, sync_name, worker_type, worker_id):
        """Worker joins a synchronized group. The service will
        add all running workers into a synchronized group when
        the first worker reaches the synchronized point.
        Then, the service will remove the worker from from the group
        if a worker reaches the point."""
        with self._lock:
            worker = (worker_type, worker_id)
            if sync_name not in self._sync_objs_target:
                nodes: List[Node] = self._job_manager.get_running_workers()
                workers = [(n.type, n.id) for n in nodes]
                self._sync_objs_target[sync_name] = workers
                logger.info(
                    "New worker sync {} added for worker {}".format(
                        sync_name, self._sync_objs_target[sync_name]
                    )
                )
                self._sync_start_time[sync_name] = time.time()
            if worker in self._sync_objs_target[sync_name]:
                self._sync_objs_target[sync_name].remove(worker)
                logger.info(
                    "{}: {} synced. Remaining {}".format(
                        sync_name, worker, self._sync_objs_target[sync_name]
                    )
                )
                if len(self._sync_objs_target[sync_name]) == 0:
                    self._sync_start_time.pop(sync_name)
                    logger.info("Worker sync {} done.".format(sync_name))
        return True

    def sync_finished(self, sync_name):
        with self._lock:
            if len(self._sync_objs_target[sync_name]) == 0:
                return True
            return False

    def barrier(self, barrier_name):
        with self._lock:
            if barrier_name in self._finished_barriers:
                return True
            return False

    def notify_barrier(self, barrier_name):
        with self._lock:
            self._finished_barriers.add(barrier_name)
            logger.info("Worker barrier {} notified".format(barrier_name))
        return True

    def remove_exited_worker_sync(self, worker_type, worker_id):
        worker = (worker_type, worker_id)
        for sync_name in self._sync_objs_target:
            if worker in self._sync_objs_target[sync_name]:
                self._sync_objs_target[sync_name].remove(worker)
                logger.info(
                    "{} not running, removed from {}. "
                    "Remaining {}".format(
                        worker,
                        sync_name,
                        self._sync_objs_target[sync_name],
                    )
                )
                if len(self._sync_objs_target[sync_name]) == 0:
                    logger.info("Worker sync {} done.".format(sync_name))

    def delete_sync_timeout_worker(self):
        while True:
            timeout_syncs = []
            timeout_workers = set()
            with self._lock:
                for sync_name, start_time in self._sync_start_time.items():
                    if time.time() - start_time > self.timeout:
                        timeout_syncs.append(sync_name)
                        for worker in self._sync_objs_target[sync_name]:
                            self._sync_objs_target[sync_name].remove(worker)
                            timeout_workers.add(worker)

            for worker in timeout_workers:
                logger.info("Remove timeout worker {}".format(worker))
                self._job_manager.remove_worker(worker)
            time.sleep(15)
