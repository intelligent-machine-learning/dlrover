import asyncio
from dataclasses import dataclass
from typing import Dict

from dlrover.python.common.log import default_logger as logger


@dataclass
class DataQueueInfo:
    name: str
    size: int
    owner_actor: str


class SyncManager:
    """Manager for Job-related sync operations.
    1. Manage data queues registration.
    """

    def __init__(self) -> None:
        self._queues: Dict[str, DataQueueInfo] = {}
        self._wait_queues: Dict[str, asyncio.Event] = {}

    def register_data_queue(self, name: str, owner_actor: str, size: int):
        logger.info(
            f"Registering data queue: {name}, size: {size}, master: {owner_actor}"
        )
        if name in self._queues:
            logger.warning(f"Data queue {name} already registered, updating.")
        self._queues[name] = DataQueueInfo(name, size, owner_actor)
        if name in self._wait_queues:
            self._wait_queues.pop(name).set()
        # RESET when actor is restarted

    async def get_data_queue_owner(self, name: str) -> str:
        if name not in self._queues:
            if name not in self._wait_queues:
                self._wait_queues[name] = asyncio.Event()
            await self._wait_queues[name].wait()
        return self._queues[name].owner_actor
