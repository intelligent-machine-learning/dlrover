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
from unittest.mock import MagicMock

from dlrover.python.unified.common.constant import InternalDLWorkloadRole
from dlrover.python.unified.master.elastic.executor import ElasticExecutor
from dlrover.python.unified.master.elastic.job_manager import ElasticJobManager
from dlrover.python.unified.master.graph import (
    DLExecutionGraph,
    DLExecutionVertex,
)
from dlrover.python.unified.tests.master.elastic.base import ElasticBaseTest


class ElasticExecutorTest(ElasticBaseTest):
    def setUp(self):
        super().setUp()
        job_manager = ElasticJobManager()
        self.executor = job_manager.executor

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

    def test_basic(self):
        self.assertEqual(
            self.executor._update_train_result("1", True), ("1", True)
        )
        self.assertFalse(self.executor.is_finished())

    def test_execute(self):
        self.assertEqual(
            self.executor._train_result,
            {"ELASTIC_2-0_2-0": None, "ELASTIC_2-1_2-1": None},
        )
        for vertex in self.mock_vertices[InternalDLWorkloadRole.ELASTIC_ROLE]:
            vertex.actor_handle.run.remote = MagicMock()
        self.executor.execute()

        self.assertEqual(
            self.executor._train_result,
            {"ELASTIC_2-0_2-0": False, "ELASTIC_2-1_2-1": False},
        )
