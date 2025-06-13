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

from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.common.constant import InternalDLWorkloadRole
from dlrover.python.unified.master.executor import Executor
from dlrover.python.unified.master.graph import DLExecutionGraph


class ElasticExecutor(Executor):
    def __init__(self, execution_graph: DLExecutionGraph):
        super().__init__(execution_graph)

    def execute(self):
        loop = asyncio.get_event_loop()
        asyncio.run_coroutine_threadsafe(self._async_execute(), loop)

    async def _async_execute(self):
        logger.info("Start elastic training execution...")

        elastic_vertices = self.graph.execution_vertices[
            InternalDLWorkloadRole.ELASTIC_ROLE
        ]
        tasks = {
            vertex.name: vertex.actor_handle.run.remote()
            for vertex in elastic_vertices
        }

        async def run_per_vertex(vertex_name, coro):
            try:
                start = time.time()
                await coro
                logger.info(
                    f"Node {vertex_name} elastic training completed in "
                    f"{time.time() - start} seconds."
                )
                return vertex_name, True
            except Exception as e:
                logger.error(f"{vertex_name} run elastic training failed: {e}")
                return vertex_name, False

        run_tasks = [
            run_per_vertex(name, coro) for name, coro in tasks.items()
        ]
        for completed_task in asyncio.as_completed(run_tasks):
            vertex_name, result = await completed_task
            if result:
                logger.info(f"Node {vertex_name} elastic training completed.")
            else:
                logger.warning(f"Node {vertex_name} elastic training failed.")
                # TODO: trigger failover
