# Copyright 2025 The DLRover Authors. All rights reserved.
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

import asyncio
from concurrent.futures import Future
from typing import Generic, List, Sequence, TypeVar

import ray

from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.controller.api import PrimeMasterApi
from dlrover.python.unified.util.async_helper import as_future, wait

from .rpc_helper import (
    create_rpc_proxy,
    export_rpc_instance,
    rpc,
)
from .worker import current_worker

T = TypeVar("T")


class DataQueueImpl(Generic[T]):
    """Core implementation of the data queue. Runs on the owner actor."""

    def __init__(self, name: str, size: int):
        self.name = name
        self._queue = asyncio.Queue[T](size)
        self._fifo_lock = asyncio.Lock()

    @rpc()
    def qsize(self) -> int:
        """Get the current size of the queue."""
        return self._queue.qsize()

    @rpc()
    async def put(self, objs: Sequence[T]):
        """Put an object into the queue. Waits if the queue is full."""
        if self._queue.full():
            logger.debug(
                f"Queue is full, {self.qsize()}/{self._queue.maxsize}"
            )
        for obj in objs:
            await self._queue.put(obj)

    @rpc()
    async def get(self, batch_size: int) -> List[T]:
        """Get a batch of objects from the queue."""
        # Let the first user get all needed objects first
        async with self._fifo_lock:
            return [
                await self._queue.get()
                for _ in range(min(batch_size, self.qsize()))
            ]

    @rpc()
    def get_nowait(self, max_size: int = 1) -> List[T]:
        """
        Get a batch of objects from the queue without waiting.
        If the queue is empty, it returns an empty list.
        :param max_size: Maximum number of objects to get
        :return: List of ObjectRefs
        """
        return [
            self._queue.get_nowait()
            for _ in range(min(max_size, self.qsize()))
        ]


class DataQueue(Generic[T]):
    """
    Distributed data queue interface.
    """

    def __init__(self, name: str, is_master: bool, size: int = 1000):
        self._is_master = is_master
        if self._is_master:
            self._impl = DataQueueImpl["ray.ObjectRef[T]"](name, size)
            export_rpc_instance(f"{DataQueue.__name__}.{name}", self._impl)
            PrimeMasterApi.register_data_queue(
                name, current_worker().actor_info.name, size
            )
        else:
            owner = PrimeMasterApi.get_data_queue_owner(name)
            self._impl = create_rpc_proxy(
                owner,
                f"{DataQueue.__name__}.{name}",
                DataQueueImpl["ray.ObjectRef[T]"],
            )

    def qsize(self) -> int:
        """Get the current size of the queue."""
        return self._impl.qsize()

    def put(self, *obj: T) -> None:
        """Put an object into the queue."""
        self.put_async(*obj).result()

    def put_async(self, *obj: T) -> Future[None]:
        """Put an object into the queue."""
        ref = [ray.put(o) for o in obj]
        return as_future(self._impl.put(ref))

    def get(self, batch_size: int) -> List[T]:
        """Get a batch of objects from the queue."""
        refs = wait(self._impl.get(batch_size))
        return ray.get(refs)

    def get_nowait(self, max_size: int = 1) -> List[T]:
        """Get a batch of objects from the queue without waiting."""
        refs = self._impl.get_nowait(max_size)
        return ray.get(refs) if refs else []
