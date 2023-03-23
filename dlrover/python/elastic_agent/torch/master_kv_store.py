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

import datetime
import time
from base64 import b64decode, b64encode
from typing import Dict, Optional

from torch.distributed import Store

from dlrover.python.elastic_agent.master_client import GlobalMasterClient


class MasterKVStore(Store):
    """
    Implements a c10 Store interface by piggybacking on the rendezvous
    instance of DLRover job master. This is the store object
    returned by ``EtcdRendezvous``
    """

    def __init__(
        self,
        prefix,
        # Default timeout same as in c10d/Store.hpp
        timeout: Optional[datetime.timedelta] = None,
    ):
        super().__init__()

        self.client = GlobalMasterClient.MASTER_CLIENT
        self.prefix = prefix

        if timeout is not None:
            self.set_timeout(timeout)

        if not self.prefix.endswith("/"):
            self.prefix += "/"

    def set(self, key, value):
        """
        Write a key/value pair into ``MasterKVStore``.
        Both key and value may be either Python ``str`` or ``bytes``.
        """
        key = self.prefix + self._encode(key)
        value = self._encode(value)
        self.client.kv_store_set(key, value)

    def get(self, key) -> bytes:
        """
        Get a value by key, possibly doing a blocking wait.

        If key is not immediately present, will do a blocking wait
        for at most ``timeout`` duration or until the key is published.

        Returns:
            value ``(bytes)``

        Raises:
            LookupError - If key still not published after timeout
        """
        b64_key = self.prefix + self._encode(key)
        value = self.client.kv_store_get(b64_key)

        if value is None:
            raise LookupError(
                f"Key {key} not found in the store of job master"
            )

        return self._decode(value)

    def add(self, key, num: int) -> int:
        """
        Atomically increment a value by an integer amount. The integer is
        represented as a string using base 10. If key is not present,
        a default value of ``0`` will be assumed.

        Returns:
             the new (incremented) value
        """
        b64_key = self.prefix + self._encode(key)
        value = self.client.kv_store_get(b64_key)
        if value:
            value = int(self._decode(value)) + num
        else:
            value = num

        value_str = self._encode(str(value))
        self.client.kv_store_set(b64_key, value_str)
        return value

    def wait(
        self, keys, override_timeout: Optional[datetime.timedelta] = None
    ):
        """
        Waits until all of the keys are published, or until timeout.

        Raises:
            LookupError - if timeout occurs
        """
        b64_keys = [self.prefix + self._encode(key) for key in keys]
        kvs = self._try_wait_get(b64_keys, override_timeout)
        if kvs is None:
            raise LookupError(
                "Timeout while waiting for keys in KVStore of master"
            )

    def check(self, keys) -> bool:
        """
        Check if all of the keys are immediately present (without waiting).
        """
        b64_keys = [self.prefix + self._encode(key) for key in keys]
        kvs = self._try_wait_get(
            b64_keys,
            override_timeout=datetime.timedelta(microseconds=1),
        )
        return kvs is not None

    def _encode(self, value) -> str:
        """
        Encode key/value data in base64, so we can store arbitrary binary data
        in MasterKVStore. Input can be `str` or `bytes`.
        In case of `str`, utf-8 encoding is assumed.
        """
        if type(value) == bytes:
            return b64encode(value).decode()
        elif type(value) == str:
            return b64encode(value.encode()).decode()
        raise ValueError("Value must be of type str or bytes")

    def _decode(self, value) -> bytes:
        """
        Decode a base64 string (of type `str` or `bytes`).
        Return type is `bytes`, which is more convenient with
        the Store interface.
        """
        if type(value) == bytes:
            return b64decode(value)
        elif type(value) == str:
            return b64decode(value.encode())
        raise ValueError("Value must be of type str or bytes")

    def _try_wait_get(self, b64_keys, override_timeout=None):
        """
        Get all of the (base64-encoded) etcd keys at once, or wait until
        all the keys are published or timeout occurs.
        """
        timeout = override_timeout if override_timeout else self.timeout
        print(timeout)
        deadline = time.time() + timeout.total_seconds()

        kvs: Dict[str, str] = {}
        while True:
            for b64_key in b64_keys:
                if b64_key not in kvs:
                    value = self.client.kv_store_get(b64_key)
                    if value:
                        kvs[b64_key] = value

            if len(kvs) == len(b64_keys):
                return kvs

            watch_timeout = deadline - time.time()
            if watch_timeout <= 0:
                return None
            time.sleep(2)
