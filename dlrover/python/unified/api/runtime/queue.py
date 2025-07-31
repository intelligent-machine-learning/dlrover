import asyncio
from concurrent.futures import Future
from functools import partial
from typing import Dict, Generic, List, Sequence

import ray
from gguf import ClassVar, TypeVar
from ray import ObjectRef

from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.api.runtime.rpc import rpc, rpc_call_t
from dlrover.python.unified.api.runtime.worker import current_worker
from dlrover.python.unified.controller.api import PrimeMasterApi
from dlrover.python.unified.util.async_helper import as_future, wait


class DataQueueImpl:
    """Core implementation of the data queue."""

    def __init__(self, name: str, size: int):
        self.name = name
        self._queue = asyncio.Queue[ObjectRef](size)
        self._fifo_lock = asyncio.Lock()

    def qsize(self) -> int:
        """Get the current size of the queue."""
        return self._queue.qsize()

    async def put(self, objs: Sequence[ObjectRef]):
        """Put an object into the queue. Waits if the queue is full."""
        if self._queue.full():
            logger.debug(
                f"Queue is full, {self.qsize()}/{self._queue.maxsize}"
            )
        for obj in objs:
            await self._queue.put(obj)

    async def get(self, batch_size: int) -> List[ObjectRef]:
        """Get a batch of objects from the queue."""
        # Let the first user get all needed objects first
        async with self._fifo_lock:
            return [
                await self._queue.get()
                for _ in range(min(batch_size, self.qsize()))
            ]

    def get_nowait(self, max_size: int = 1) -> List[ObjectRef]:
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


@rpc()
def _handle_data_queue_call(queue_name: str, op: str, *args, **kwargs):
    """Handle data queue calls."""
    if queue_name not in DataQueue.REGISTRY:
        raise ValueError(f"Data queue {queue_name} not registered.")
    queue = DataQueue.REGISTRY[queue_name]
    func = getattr(queue, op, None)
    if func is None:
        raise ValueError(f"Operate {op} not found for DataQueue")
    return func(*args, **kwargs)


class DataQueueRemoteImpl:
    """Remote proxy for DataQueueImpl. see DataQueueImpl for details."""

    # TODO 使用Proxy对象替换

    def __init__(self, name: str, owner_actor: str):
        self.name = name
        self._owner = owner_actor
        self._call = partial(rpc_call_t, self._owner, _handle_data_queue_call)

    def qsize(self) -> int:
        return self._call(self.name, "qsize").wait()

    async def put(self, objs: Sequence[ObjectRef]):
        return await self._call(self.name, "put", objs)

    async def get(self, batch_size: int) -> List[ObjectRef]:
        return await self._call(self.name, "get", batch_size)

    def get_nowait(self, max_size: int = 1) -> List[ObjectRef]:
        return self._call(self.name, "get_nowait", max_size).wait()


T = TypeVar("T")


class DataQueue(Generic[T]):
    """
    Distributed data queue interface.
    """

    REGISTRY: ClassVar[Dict[str, "DataQueueImpl"]] = {}

    def __init__(self, name: str, size: int = 1000, master_rank: int = 0):
        self._is_master = current_worker().actor_info.rank == master_rank
        if self._is_master:
            self._impl = DataQueueImpl(name, size)
            DataQueue.REGISTRY[name] = self._impl
            PrimeMasterApi.register_data_queue(
                name, current_worker().actor_info.name, size
            )
        else:
            owner = PrimeMasterApi.get_data_queue_owner(name)
            self._impl = DataQueueRemoteImpl(name, owner)

    def qsize(self) -> int:
        """Get the current size of the queue."""
        if self._is_master:
            return self._impl.qsize()
        else:
            pass

        return self._impl.qsize()

    def put(self, obj: T) -> None:
        """Put an object into the queue."""
        self.put_async(obj).result()

    def put_async(self, obj: T) -> Future[None]:
        """Put an object into the queue."""
        ref = ray.put(obj)
        return as_future(self._impl.put(ref))

    def get(self, batch_size: int) -> List[T]:
        """Get a batch of objects from the queue."""
        refs = wait(self._impl.get(batch_size))
        return ray.get(refs)

    def get_nowait(self, max_size: int = 1) -> List[T]:
        """Get a batch of objects from the queue without waiting."""
        refs = self._impl.get_nowait(max_size)
        return ray.get(refs) if refs else []
