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
from typing import Any, Dict, List, Optional, Tuple

from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.node import Node


def base2(n):
    return (n & (n - 1) == 0) and n != 0


class RendezvousState(object):
    def __init__(self) -> None:
        self.latest_state_bits = b""
        self.completed_state_bits = b""
        self.participants: Dict[str, int] = {}
        self.wait_list: List[str] = []


class TorchRendezvousService(object):
    """TorchRendezvousService runs on the DLRover master.
    The service can update the rendezvous states according to
    the node status.
    """

    def __init__(self):
        self._lock = Lock()
        self._rdzv_states: Dict[str, RendezvousState] = {}
        self._token = -1
        self._alive_workers = []
        self._participants = []
        self._scale_down_ts = 0

    def add_alive_worker(self, worker: Node):
        self._alive_workers.append(worker.name)
        if base2(self._alive_workers):
            self._participants = self._alive_workers

    def remove_alive_worker(self, worker: Node):
        self._alive_workers.remove(worker.name)
        self._scale_down_ts = int(time.time())
        self._participants = []

    def get_participants(self):
        return self._participants

    def start(self):
        pass

    def set_state(
        self,
        key,
        state_bits: bytes,
        token: Optional[Any],
        participants,
        wait_list,
        host_name,
    ):
        """Set the _RendezvousState into the store in the master.
        Returns:
            A tuple of the serialized rendezvous state, its fencing token, and
            a boolean value indicating whether our set attempt succeeded.
        """
        if host_name not in self._participants:
            return False
        with self._lock:
            self._scale_down_worker_base2()
            self._rdzv_states.setdefault(key, RendezvousState())
            if self._rdzv_states[key].latest_state_bits == state_bits:
                return False
            logger.info(
                "wait list = %s, participants = %s",
                wait_list,
                participants,
            )
            rdzv_state = self._rdzv_states[key]
            rdzv_state.latest_state_bits = state_bits
            rdzv_state.participants = participants
            rdzv_state.wait_list = wait_list
            self._token += 1
            return True

    def get_state(self, worker_name, key) -> Optional[Tuple[bytes, Any]]:
        """Return a new state only if len(_RendezvousState.participants)
        + len(_RendezvousState.wait_list) is base 2. Then, we can
        keep the fixed batch size by setting backward_passes_per_step
        in the worker.
        Returns:
            A tuple of the encoded rendezvous state and its fencing token or
            `None` if no state is found in the backend.
        """
        with self._lock:
            self._scale_down_worker_base2()
            completed_state_bits = b""
            if (
                key not in self._rdzv_states
                or worker_name not in self._participants
            ):
                return completed_state_bits, self._token

            rdzv_state = self._rdzv_states[key]
            participant_num = len(rdzv_state.participants)
            wait_num = len(rdzv_state.wait_list)

            num_nodes_ready = participant_num + wait_num

            if base2(num_nodes_ready):
                rdzv_state.completed_state_bits = rdzv_state.latest_state_bits
                completed_state_bits = rdzv_state.completed_state_bits
            return completed_state_bits, self._token

    def _scale_down_worker_base2(self):
        """Scale down the number of worker base2 like 1,2,4,8,...."""
        if not self._participants and self._scale_down_ts > 0:
            now = int(time.time())
            worker_num = len(self._alive_workers)
            n = 0
            while worker_num > 0:
                worker_num = worker_num >> 1
                n += 1
            target_worker_num = pow(2, n - 1)
            if now - self._scale_down_ts > 300:
                self._participants = self._alive_workers[:target_worker_num]
                self._scale_down_ts = 0
