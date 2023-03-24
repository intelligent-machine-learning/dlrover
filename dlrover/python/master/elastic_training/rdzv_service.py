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

from threading import Lock
from typing import Any, Dict, Optional, Tuple

from dlrover.python.common.log import default_logger as logger


def base2(n):
    return (n & (n - 1) == 0) and n != 0


class RendezvousState(object):
    def __init__(self) -> None:
        self.latest_state_bits = b""
        self.completed_state_bits = b""
        self.participant_num = 0
        self.wait_num = 0


class TorchRendezvousService(object):
    """TorchRendezvousService runs on the DLRover master.
    The service can update the rendezvous states according to
    the node status.
    """

    def __init__(self):
        self._lock = Lock()
        self._rdzv_states: Dict[str, RendezvousState] = {}
        self._token = -1

    def start(self):
        pass

    def set_state(
        self,
        key,
        state_bits: bytes,
        token: Optional[Any],
        participant_num,
        wait_num,
    ):
        """Set the _RendezvousState into the store in the master.
        Returns:
            A tuple of the serialized rendezvous state, its fencing token, and
            a boolean value indicating whether our set attempt succeeded.
        """
        with self._lock:
            self._rdzv_states.setdefault(key, RendezvousState())
            if self._rdzv_states[key].latest_state_bits == state_bits:
                return False
            logger.info(
                "wait list = %s, participants = %s",
                wait_num,
                participant_num,
            )
            rdzv_state = self._rdzv_states[key]
            rdzv_state.latest_state_bits = state_bits
            rdzv_state.participant_num = participant_num
            rdzv_state.wait_num = wait_num
            self._token += 1
            return True

    def get_state(self, key) -> Optional[Tuple[bytes, Any]]:
        """Return a new state only if len(_RendezvousState.participants)
        + len(_RendezvousState.wait_list) is base 2. Then, we can
        keep the fixed batch size by setting backward_passes_per_step
        in the worker.
        Returns:
            A tuple of the encoded rendezvous state and its fencing token or
            ``None`` if no state is found in the backend.
        """
        with self._lock:
            completed_state_bits = b""
            if key not in self._rdzv_states:
                return completed_state_bits, self._token

            rdzv_state = self._rdzv_states[key]

            num_nodes_ready = rdzv_state.participant_num + rdzv_state.wait_num

            if base2(num_nodes_ready):
                rdzv_state.completed_state_bits = rdzv_state.latest_state_bits
                completed_state_bits = rdzv_state.completed_state_bits
            return completed_state_bits, self._token
