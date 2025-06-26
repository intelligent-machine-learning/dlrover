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
from typing import Dict, Union

import ray

from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.common.constant import InternalDLWorkloadRole
from dlrover.python.unified.master.executor import Executor
from dlrover.python.unified.master.graph import DLExecutionGraph


class ElasticExecutor(Executor):
    def __init__(self, execution_graph: DLExecutionGraph):
        super().__init__(execution_graph)

        self.__loop = asyncio.get_event_loop()
        self._train_result: Dict[str, Union[bool, None]] = {
            vertex.name: None
            for vertex in self.graph.execution_vertices[
                InternalDLWorkloadRole.ELASTIC_ROLE
            ]
        }

    def _update_train_result(self, name, result):
        self._train_result[name] = result
        return name, result

    def execute(self, target=None):
        def run_per_vertex(name, run_ref):
            try:
                start = time.time()
                ray.get(run_ref)
                logger.info(
                    f"Node {name} elastic training completed in "
                    f"{time.time() - start} seconds."
                )
                return self._update_train_result(name, True)
            except Exception as e:
                logger.error(f"{name} run elastic training failed: {e}")
                return self._update_train_result(name, False)

        elastic_vertices = self.graph.execution_vertices[
            InternalDLWorkloadRole.ELASTIC_ROLE
        ]

        if target:
            logger.info(
                f"Start elastic training execution on target: {target}"
            )
            tasks = {
                vertex.name: vertex.actor_handle.run.remote()
                for vertex in elastic_vertices
                if vertex.name == target
            }
        else:
            logger.info("Start elastic training execution...")
            tasks = {
                vertex.name: vertex.actor_handle.run.remote()
                for vertex in elastic_vertices
            }

        run_tasks = [
            run_per_vertex(name, task_ref) for name, task_ref in tasks.items()
        ]

        logger.info(f"Elastic training execution results: {run_tasks}")

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

    def reset_error(self, target=None):
        if target and target in self._train_result:
            self._train_result[target] = None
        else:
            self._train_result = {
                key: None if value is False else value
                for key, value in self._train_result.items()
            }
        logger.info(
            f"Reset elastic training error result: {self._train_result}"
        )
