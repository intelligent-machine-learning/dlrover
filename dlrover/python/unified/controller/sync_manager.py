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
