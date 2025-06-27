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
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Union

import ray
from ray import ObjectRef
from ray.exceptions import RayTaskError

from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.common.constant import InternalDLWorkloadRole
from dlrover.python.unified.master.executor import Executor
from dlrover.python.unified.master.graph import DLExecutionGraph


class ElasticExecutor(Executor):
    def __init__(self, execution_graph: DLExecutionGraph):
        super().__init__(execution_graph)

        self._executor = ThreadPoolExecutor(max_workers=1)
        self._lock = threading.Lock()
        self._tasks: Dict[str, ObjectRef] = {}
        self._train_result: Dict[str, Union[bool, None]] = {
            vertex.name: None
            for vertex in self.graph.execution_vertices[
                InternalDLWorkloadRole.ELASTIC_ROLE
            ]
        }

    def _update_train_result(self, name, result):
        self._train_result[name] = result
        return name, result

    def add_execution(self, vertex_name: str):
        logger.info(f"Re execute elastic training on: {vertex_name}")
        with self._lock:
            self._tasks[vertex_name] = self.graph.name_vertex_mapping[
                vertex_name
            ].actor_handle.run.remote()

    def _execute_all(self):
        elastic_vertices = self.graph.execution_vertices[
            InternalDLWorkloadRole.ELASTIC_ROLE
        ]

        logger.info("Start elastic training execution...")
        with self._lock:
            self._tasks = {
                vertex.name: vertex.actor_handle.run.remote()
                for vertex in elastic_vertices
            }

    def _wait_execution(self):
        while True:
            if self.is_finished():
                logger.info("Elastic training execution all finished.")
                break

            with self._lock:
                tasks_mapping = self._tasks
                task_refs = list(self._tasks.values())

            ready, not_ready = ray.wait(task_refs, num_returns=len(task_refs))

            for task_ref in ready:
                name = next(
                    key
                    for key, value in tasks_mapping.items()
                    if value == task_ref
                )
                try:
                    if not self._train_result[name]:
                        ray.get(task_ref)
                        logger.info(
                            "Update succeeded elastic training result "
                            f"for: {name}"
                        )
                        self._update_train_result(name, True)
                except RayTaskError as e:
                    if self._train_result[name] is None:
                        logger.info(
                            "Update failed elastic training result "
                            f"for: {name}, error: {e}"
                        )
                        self._update_train_result(name, False)

            logger.debug(
                "Current elastic training execution "
                f"results: {self._train_result}"
            )
            time.sleep(1)

    def execute(self):
        self._execute_all()
        self._executor.submit(self._wait_execution)

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
