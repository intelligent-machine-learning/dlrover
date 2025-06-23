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
from unittest.mock import AsyncMock, MagicMock, patch

from dlrover.python.unified.common.constant import InternalDLWorkloadRole
from dlrover.python.unified.master.elastic.executor import ElasticExecutor
from dlrover.python.unified.master.elastic.job_manager import ElasticJobManager
from dlrover.python.unified.master.graph import (
    DLExecutionGraph,
    DLExecutionVertex,
)
from dlrover.python.unified.tests.base import AsyncBaseTest
from dlrover.python.unified.tests.master.elastic.base import ElasticBaseTest


class ElasticExecutorTest(ElasticBaseTest):
    def setUp(self):
        super().setUp()
        job_manager = ElasticJobManager()
        self.executor = job_manager.executor

    def test_basic(self):
        self.assertEqual(
            self.executor._update_train_result("1", True), ("1", True)
        )
        self.assertFalse(self.executor.is_finished())

    @patch("asyncio.get_event_loop")
    @patch("asyncio.run_coroutine_threadsafe")
    def test_execute(self, mock_run_coroutine, mock_get_loop):
        mock_loop = MagicMock()
        mock_get_loop.return_value = mock_loop

        self.executor.execute()

        mock_get_loop.assert_called_once()

        mock_run_coroutine.assert_called_once()
        called_coro = mock_run_coroutine.call_args[0][0]
        self.assertEqual(
            called_coro.__qualname__, "ElasticExecutor._async_execute"
        )


class ElasticExecutorAsyncTest(AsyncBaseTest):
    def setUp(self):
        super().setUp()
        self.mock_graph = MagicMock(spec=DLExecutionGraph)
        self.mock_vertices = {
            InternalDLWorkloadRole.ELASTIC_ROLE: [
                DLExecutionVertex(
                    InternalDLWorkloadRole.ELASTIC_ROLE,
                    module_name="test",
                    class_name="test",
                    resource=None,
                    world_size=2,
                    rank=0,
                    local_world_size=2,
                    local_rank=0,
                ),
                DLExecutionVertex(
                    InternalDLWorkloadRole.ELASTIC_ROLE,
                    module_name="test",
                    class_name="test",
                    resource=None,
                    world_size=2,
                    rank=1,
                    local_world_size=2,
                    local_rank=1,
                ),
            ]
        }
        for vertex in self.mock_vertices[InternalDLWorkloadRole.ELASTIC_ROLE]:
            vertex._actor_handle = MagicMock()
        self.mock_graph.execution_vertices = self.mock_vertices
        self.executor = ElasticExecutor(self.mock_graph)

    async def test_async_execute_success(self):
        for vertex in self.mock_vertices[InternalDLWorkloadRole.ELASTIC_ROLE]:
            vertex.actor_handle.run.remote = AsyncMock()

        await self.executor._async_execute()

        for vertex in self.mock_vertices[InternalDLWorkloadRole.ELASTIC_ROLE]:
            self.assertTrue(self.executor._train_result[vertex.name])

    async def test_async_execute_single(self):
        for vertex in self.mock_vertices[InternalDLWorkloadRole.ELASTIC_ROLE]:
            vertex.actor_handle.run.remote = AsyncMock()

        await self.executor._async_execute("ELASTIC_2-0_2-0")

        for vertex in self.mock_vertices[InternalDLWorkloadRole.ELASTIC_ROLE]:
            if vertex.name == "ELASTIC_2-0_2-0":
                self.assertTrue(self.executor._train_result[vertex.name])

    async def test_async_execute_failure(self):
        for vertex in self.mock_vertices[InternalDLWorkloadRole.ELASTIC_ROLE]:
            if vertex.name == "ELASTIC_2-0_2-0":
                vertex.actor_handle.run.remote = AsyncMock(
                    side_effect=RuntimeError
                )
            else:
                vertex.actor_handle.run.remote = AsyncMock()

        await self.executor._async_execute()

        for vertex in self.mock_vertices[InternalDLWorkloadRole.ELASTIC_ROLE]:
            if vertex.name == "ELASTIC_2-0_2-0":
                self.assertFalse(self.executor._train_result[vertex.name])
            else:
                self.assertTrue(self.executor._train_result[vertex.name])
