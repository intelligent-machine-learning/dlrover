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
from typing import Optional

from torch.distributed import Store

from dlrover.python.common.log import default_logger as logger
from dlrover.python.elastic_agent.master_client import MasterClient


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

        self.client = MasterClient.singleton_instance()
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
        key = self.prefix + key
        self.client.kv_store_set(key, value)

    def multi_set(self, keys, values):
        """
        Write many key/value pairs into ``MasterKVStore``.
        Both key and value may be either Python ``str`` or ``bytes``.
        """
        new_keys = [self.prefix + key for key in keys]
        self.client.kv_store_multi_set(new_keys, values)

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
        key = self.prefix + key
        kvs = self._try_wait_get([key])
        if not kvs:
            raise LookupError(
                f"Key {key} not found in the store of job master"
            )

        value = kvs[key]
        if value == b"":
            raise ValueError(f"Key {key} is emtpy in the store of job master")

        return value

    def multi_get(self, keys):
        new_keys = [self.prefix + key for key in keys]
        kvs = self._try_wait_get(new_keys)
        if not kvs:
            raise LookupError(
                f"Keys {new_keys} not found in the store of job master"
            )

        values = []
        try:
            for key in new_keys:
                value = kvs[key]
                if value == b"":
                    raise ValueError(
                        f"Key {key} is empty in the store of job master"
                    )
                values.append(value)
        except Exception:
            raise KeyError(f"Not all {new_keys} in the store of job master")

        return values

    def add(self, key, num: int) -> int:
        """
        Atomically increment a value by an integer amount. The integer is
        represented as a string using base 10. If key is not present,
        a default value of ``0`` will be assumed.

        Returns:
             the new (incremented) value
        """
        key = self.prefix + key
        return self.client.kv_store_add(key, num)

    def wait(
        self, keys, override_timeout: Optional[datetime.timedelta] = None
    ):
        """
        Waits until all of the keys are published, or until timeout.

        Raises:
            LookupError - if timeout occurs
        """
        keys = [self.prefix + key for key in keys]
        kvs = self._try_wait_get(keys, override_timeout)
        if not kvs:
            raise LookupError(
                "Timeout while waiting for keys in KVStore of master"
            )

    def check(self, keys) -> bool:
        """
        Check if all of the keys are immediately present (without waiting).
        """
        keys = [self.prefix + key for key in keys]
        kvs = self._try_wait_get(
            keys,
            override_timeout=datetime.timedelta(microseconds=1),
        )
        return kvs is not None

    def _try_wait_get(self, keys, override_timeout=None):
        """
        Get all of the (base64-encoded) etcd keys at once, or wait until
        all the keys are published or timeout occurs.
        """
        timeout = override_timeout if override_timeout else self.timeout
        deadline = time.time() + timeout.total_seconds()

        while True:
            try:
                kvs = self.client.kv_store_multi_get(keys)
                if kvs:
                    logger.debug(f"_try_wait_get {keys}: {kvs}")
                    return kvs
            except Exception as e:
                logger.warning(
                    f"_try_wait_get {keys} exception: {e}, try again!"
                )

            watch_timeout = deadline - time.time()
            if watch_timeout <= 0:
                logger.debug(
                    f"_try_wait_get {keys} timeout: {timeout.total_seconds()}"
                )
                return None
            time.sleep(2)
