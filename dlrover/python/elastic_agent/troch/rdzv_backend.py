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

import binascii
from base64 import b64decode, b64encode
from typing import Optional, Tuple

import grpc
from torch.distributed import Store
from torch.distributed.elastic.rendezvous.api import (
    RendezvousConnectionError,
    RendezvousParameters,
    RendezvousStateError,
)
from torch.distributed.elastic.rendezvous.dynamic_rendezvous import (
    RendezvousBackend,
    Token,
)

from dlrover.python.elastic_agent.master_client import (
    GlobalMasterClient,
    MasterClient,
)
from dlrover.python.elastic_agent.troch.master_kv_store import MasterKVStore


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

    def __init__(
        self,
        run_id: str,
    ) -> None:
        if not run_id:
            raise ValueError("The run id must be a non-empty string.")

        self._client = GlobalMasterClient.MASTER_CLIENT
        self._key = "torch.rendezvous." + run_id

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

        token = result[1]
        new_state = self._decode_state(result[0])
        return new_state, token

    def set_state(
        self, state: bytes, token: Optional[Token] = None
    ) -> Optional[Tuple[bytes, Token, bool]]:
        """See base class."""
        base64_state = b64encode(state).decode()

        def get_state():
            result = self.get_state()
            if result is not None:
                tmp = *result, False
                # Python 3.6 does not support tuple unpacking in return
                # statements.
                return tmp
            return None

        if token:
            try:
                token = int(token)
            except ValueError:
                return get_state()
        try:
            succeed = self._client.set_rdzv_state(
                self._key, base64_state, token
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

    def _decode_state(self, result) -> Tuple[bytes, Token]:
        base64_state = result.encode()

        try:
            state = b64decode(base64_state)
        except binascii.Error as exc:
            raise RendezvousStateError(
                "The state object is corrupt. See inner exception for details."
            ) from exc

        return state


def create_backend(
    params: RendezvousParameters,
) -> Tuple[DlroverRendezvousBackend, Store]:
    """Creates a new :py:class:`EtcdRendezvousBackend` from the specified
    parameters.

    +--------------+-----------------------------------------------------------+
    | Parameter    | Description                                               |
    +==============+===========================================================+
    | read_timeout | The read timeout, in seconds, for etcd operations.        |
    |              | Defaults to 60 seconds.                                   |
    +--------------+-----------------------------------------------------------+
    | protocol     | The protocol to use to communicate with etcd. Valid       |
    |              | values are "http" and "https". Defaults to "http".        |
    +--------------+-----------------------------------------------------------+
    | ssl_cert     | The path to the SSL client certificate to use along with  |
    |              | HTTPS. Defaults to ``None``.                              |
    +--------------+-----------------------------------------------------------+
    | ssl_cert_key | The path to the private key of the SSL client certificate |
    |              | to use along with HTTPS. Defaults to ``None``.            |
    +--------------+-----------------------------------------------------------+
    | ca_cert      | The path to the rool SSL authority certificate. Defaults  |
    |              | to ``None``.                                              |
    +--------------+-----------------------------------------------------------+
    """

    backend = DlroverRendezvousBackend(
        params.run_id, key_prefix="/torch/elastic/rendezvous"
    )

    store = MasterKVStore("/torch/elastic/store")

    return backend, store
