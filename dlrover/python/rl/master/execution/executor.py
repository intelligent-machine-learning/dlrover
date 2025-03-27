# Copyright 2025 The EasyDL Authors. All rights reserved.
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
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Union

from dlrover.python.common.log import default_logger as logger
from dlrover.python.rl.master.execution.graph import RLExecutionGraph
from dlrover.python.rl.master.execution.scheduling_strategy import (
    SchedulingStrategy,
)
from dlrover.python.rl.trainer.trainer import BaseTrainer


class Executor(object):
    def __init__(
        self,
        execution_graph: RLExecutionGraph,
        scheduling_strategy: SchedulingStrategy,
    ):
        self.__execution_graph = execution_graph
        self._scheduling_strategy = scheduling_strategy

        self.__trainer: Union[BaseTrainer, None] = None
        self.__trainer_result: Union[Future, None] = None

    def create_workloads(self):
        """Sync operation."""
        logger.info("Create all workloads.")

        self._scheduling_strategy.schedule(self.__execution_graph)

    def init_trainer(self):
        trainer_cls = self.__execution_graph.get_trainer_cls()
        actor_handles = self.__execution_graph.get_actor_handles()
        config = self.__execution_graph.get_rl_config()

        self.__trainer = trainer_cls(actor_handles, config)

    def destroy_workloads(self):
        """Sync operation."""

        logger.info("Destroy all workloads.")
        self._scheduling_strategy.cleanup(self.__execution_graph)

    def execute(self):
        self.create_workloads()
        self.init_trainer()

        logger.info(
            "Start trainer execution, "
            f"trainer: {({self.__trainer.__class__.__name__})}"
        )

        with ThreadPoolExecutor(max_workers=1) as executor:
            self.__trainer_result = executor.submit(self._trainer_async)
            self.__trainer_result.add_done_callback(
                self._trainer_done_callback
            )

    def cleanup(self):
        self.destroy_workloads()

    def is_trainer_finished(self):
        if self.__trainer_result is None:
            return False
        return self.__trainer_result.done()

    def _trainer_async(self):
        try:
            logger.info("Trainer init invocation.")
            self.__trainer.init()
            logger.info("Trainer fit invocation.")
            self.__trainer.fit()
        except Exception as e:
            logger.error(f"Trainer execution got error: {e}")
            traceback.print_exc()
            return False, e
        return True, None

    def _trainer_done_callback(self, future):
        result = future.result()
        if result[0]:
            logger.info("Trainer execution finished.")
        else:
            logger.error(f"Trainer execution failed by: {result[1]}.")
