#  Copyright 2025 The DLRover Authors. All rights reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import time
from typing import Dict, List, Union

from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.common.node_defines import NodeInfo

# TODO unused class, just for reference


class ElasticExecutor:
    def __init__(self, nodes: List[NodeInfo]):
        self.nodes = nodes
        self._train_result: Dict[str, Union[bool, None]] = {
            vertex.name: None for vertex in nodes
        }

    def execute(self):
        # loop = asyncio.get_event_loop()
        # asyncio.run_coroutine_threadsafe(self._async_execute(), loop)
        asyncio.create_task(self._async_execute())

    def _update_train_result(self, name, result):
        self._train_result[name] = result
        return name, result

    async def _async_execute(self):
        logger.info("Start elastic training execution...")

        tasks = {
            vertex.name: vertex.actor_handle.run.remote()
            for vertex in self.nodes
        }

        async def run_per_vertex(name, coro):
            try:
                start = time.time()
                await coro
                logger.info(
                    f"Node {name} elastic training completed in "
                    f"{time.time() - start} seconds."
                )
                return self._update_train_result(name, True)
            except Exception as e:
                logger.error(f"{name} run elastic training failed: {e}")
                return self._update_train_result(name, False)

        run_tasks = [
            run_per_vertex(name, coro) for name, coro in tasks.items()
        ]
        for completed_task in asyncio.as_completed(run_tasks):
            vertex_name, result = await completed_task
            if result:
                logger.info(
                    f"Node {vertex_name} elastic training completed: "
                    f"{self._train_result}."
                )
            else:
                logger.warning(
                    f"Node {vertex_name} elastic training failed: "
                    f"{self._train_result}."
                )
                # TODO: trigger failover

    def is_finished(self):
        logger.debug(f"Current elastic training result: {self._train_result}")
        return all(result for result in list(self._train_result.values()))

    def get_error(self):
        # return the failed vertices name
        errors = []
        for vertex_name, result in self._train_result.items():
            if result is not None and not result:
                errors.append(vertex_name)
        return errors
