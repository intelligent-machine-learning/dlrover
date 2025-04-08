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
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Union

from dlrover.python.common.log import default_logger as logger
from dlrover.python.rl.common.job_context import get_job_context
from dlrover.python.rl.master.graph import RLExecutionGraph
from dlrover.python.rl.trainer.trainer import BaseTrainer


class Executor(object):
    def __init__(self, execution_graph: RLExecutionGraph):
        self._graph = execution_graph

        self.__trainer: Union[BaseTrainer, None] = None
        self.__trainer_result: Union[Future, None] = None
        self.__trainer_error = None

    @property
    def graph(self):
        return self._graph

    def init_trainer(self):
        trainer_cls = self.graph.get_trainer_cls()
        actor_handles = self.graph.get_actor_handles()
        actor_cls = self.graph.get_actor_cls()
        config = self.graph.rl_config

        self.__trainer = trainer_cls(actor_handles, actor_cls, config)
        self.__trainer_error = None
        if self.__trainer.is_recoverable():
            get_job_context().set_trainer_recoverable()

    def execute(self):
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

    def is_trainer_finished(self):
        if self.__trainer_result is None:
            return False
        return (
            self.__trainer_result.done() and self.__trainer_result.result()[0]
        )

    def is_trainer_error(self):
        return self.__trainer_error is not None

    def get_trainer_error(self):
        return self.__trainer_error

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
            self.__trainer_error = int(time.time()), result[1]
            logger.error(
                f"Trainer execution failed by: {self.__trainer_error}."
            )
