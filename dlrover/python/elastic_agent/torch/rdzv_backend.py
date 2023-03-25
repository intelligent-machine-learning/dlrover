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
import socket
from typing import Optional, Tuple

import grpc
from torch.distributed import Store
from torch.distributed.elastic.rendezvous.api import (
    RendezvousConnectionError,
    RendezvousParameters,
)
from torch.distributed.elastic.rendezvous.dynamic_rendezvous import (
    RendezvousBackend,
    Token,
    _RendezvousState,
)

from dlrover.python.common.log import default_logger as logger
from dlrover.python.elastic_agent.master_client import (
    GlobalMasterClient,
    MasterClient,
)
from dlrover.python.elastic_agent.torch.master_kv_store import MasterKVStore

_MASTER_ADDR_KEY = "MASTER_ADDR"


class DlroverRendezvousBackend(RendezvousBackend):
    """Represents an etcd-based rendezvous backend.

    Args:
        client:
            The ``master_client.MasterClient`` instance to use
            to communicate with the master server.
        run_id:
            The run id of the rendezvous.
    """

    _client: MasterClient
    _key: str

    def __init__(self, run_id: str, key_prefix) -> None:
        if not run_id:
            raise ValueError("The run id must be a non-empty string.")

        self._client = GlobalMasterClient.MASTER_CLIENT
        self._key = key_prefix + run_id
        self._has_set_master = False

    @property
    def name(self) -> str:
        """See base class."""
        return "dlrover-master"

    def get_state(self) -> Optional[Tuple[bytes, Token]]:
        """See base class."""
        try:
            result = self._client.get_rdzv_state(self._key)
        except grpc.RpcError as exc:
            raise RendezvousConnectionError(
                "The connection to job master has failed."
                "See inner exception for details."
            ) from exc

        new_state_bits = result[0]
        token = result[1]
        if new_state_bits == b"":
            return None
        new_state = pickle.loads(new_state_bits)
        self._set_master_addr(new_state)
        return new_state_bits, token

    def _set_master_addr(self, rendezvous_state: _RendezvousState):
        for node, rank in rendezvous_state.participants.items():
            host_name = socket.gethostname()
            if (
                host_name == node.fqdn
                and rank == 0
                and not self._has_set_master
            ):
                self._has_set_master = True
                local_ip = socket.gethostbyname(host_name)
                self._client.kv_store_set(_MASTER_ADDR_KEY, local_ip.encode())
                logger.info("Set master addr %s", local_ip)
                break

    def set_state(
        self, state: bytes, token: Optional[Token] = None
    ) -> Optional[Tuple[bytes, Token, bool]]:
        """See base class."""

        def get_state():
            result = self.get_state()
            if result is not None:
                tmp = *result, False
                return tmp
            return None

        if token:
            try:
                token = int(token)
            except ValueError:
                return get_state()
        else:
            token = 0
        try:
            rdzv_state = pickle.loads(state)
            succeed = self._client.set_rdzv_state(
                self._key,
                state,
                token,
                rdzv_state.participants,
                rdzv_state.wait_list,
            )

        except grpc.RpcError as exc:
            succeed = False
            raise RendezvousConnectionError(
                "The connection to job master has failed. "
                "See inner exception for details."
            ) from exc

        if not succeed:
            return get_state()

        return state, token, succeed


def create_backend(
    params: RendezvousParameters,
) -> Tuple[DlroverRendezvousBackend, Store]:
    """Creates a new :py:class:`DlroverRendezvousBackend` from the specified
    parameters.
    """

    backend = DlroverRendezvousBackend(
        params.run_id, key_prefix="torch.elastic.rendezvous."
    )

    store = MasterKVStore("/torch/elastic/store")

    return backend, store
