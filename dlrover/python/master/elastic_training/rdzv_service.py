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

import pickle
from datetime import datetime
from threading import Lock
from typing import Any, Dict, Optional, Set, Tuple


def base2(n):
    return (n & (n - 1) == 0) and n != 0


class _NodeDesc:
    """Describes a node in the rendezvous.

    Attributes:
        addr:
            The FQDN of the node or user specified local node address.
        pid:
            The id of the process in which the rendezvous handler runs.
        local_id:
            A process-wide unique id.
    """

    addr: str
    pid: int
    local_id: int

    def __repr__(self) -> str:
        return f"{self.addr}_{self.pid}_{self.local_id}"


class _RendezvousState:
    """Holds the state of a rendezvous.

    Attributes:
        round:
            The current round of the rendezvous.
        complete:
            A boolean value indicating whether the current round of the
            rendezvous is complete.
        deadline:
            The time at which the current round of the rendezvous will be
            considered complete if it is still waiting for nodes to join.
        closed:
            A boolean value indicating whether the rendezvous is closed.
        participants:
            A dictionary of the participants and their corresponding ranks.
        wait_list:
            A set of nodes that are waiting to participate in the next round of
            the rendezvous.
        last_heartbeats:
            A dictionary containing each node's last heartbeat time.
    """

    round: int
    complete: bool
    deadline: Optional[datetime]
    closed: bool
    participants: Dict[_NodeDesc, int]
    wait_list: Set[_NodeDesc]
    last_heartbeats: Dict[_NodeDesc, datetime]

    def __init__(self) -> None:
        self.round = 0
        self.complete = False
        self.deadline = None
        self.closed = False
        self.participants = {}
        self.wait_list = set()
        self.last_heartbeats = {}


class TorchRendezvousService(object):
    """TorchRendezvousService runs on the DLRover master.
    The service can update the rendezvous states according to
    the node status.
    """

    def __init__(self):
        self._lock = Lock()
        self._latest_state = _RendezvousState()
        self._completed_state = _RendezvousState()

    def set_state(self, state_bits: bytes, token: Optional[Any]):
        """Set the _RendezvousState into the store in the master.
        Returns:
            A tuple of the serialized rendezvous state, its fencing token, and
            a boolean value indicating whether our set attempt succeeded.
        """
        with self._lock:
            self._latest_state = pickle.loads(state_bits)
            self._token = token

    def get_state(self) -> Optional[Tuple[bytes, Any]]:
        """Return a new state only if len(_RendezvousState.participants)
        + len(_RendezvousState.wait_list) is base 2. Then, we can
        keep the fixed batch size by setting backward_passes_per_step
        in the worker.
        Returns:
            A tuple of the encoded rendezvous state and its fencing token or
            ``None`` if no state is found in the backend.
        """
        num_nodes_ready = len(self._latest_state.wait_list) + len(
            self._latest_state.participants
        )
        if base2(num_nodes_ready):
            self._completed_state = self._latest_state
        return pickle.dumps(self._completed_state), self._token
