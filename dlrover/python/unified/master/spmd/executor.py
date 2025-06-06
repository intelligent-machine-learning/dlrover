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

import ray

from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.common.constant import InternalDLWorkloadRole
from dlrover.python.unified.master.executor import Executor
from dlrover.python.unified.master.graph import DLExecutionGraph


class ElasticExecutor(Executor):
    def __init__(self, execution_graph: DLExecutionGraph):
        super().__init__(execution_graph)

    def execute(self):
        logger.info("Start elastic execution")

        elastic_vertices = self.graph.execution_vertices[
            InternalDLWorkloadRole.ELASTIC_ROLE
        ]

        start_refs = [
            vertex.actor_handle.start.remote()
            for vertex in elastic_vertices
        ]
        ready, not_ready = ray.wait(
            start_refs,
            num_returns=len(start_refs),
            timeout=Executor.CALL_TIMEOUT_DEFAULT,
        )
        if len(not_ready) > 0:
            raise TimeoutError(
                f"{len(not_ready)} elastic workload actor "
                f"start timeout: {Executor.CALL_TIMEOUT_DEFAULT}s."
            )
