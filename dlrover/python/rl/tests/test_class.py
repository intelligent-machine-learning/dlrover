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

import ray

from dlrover.python.common.log import default_logger as logger
from dlrover.python.rl.common.enums import ModelParallelismArcType
from dlrover.python.rl.trainer.trainer import BaseTrainer
from dlrover.python.rl.trainer.workload import BaseWorkload


class TestTrainer(BaseTrainer):
    def init(self):
        logger.info("TestTrainer init called")
        time.sleep(1)

    def fit(self):
        logger.info("TestTrainer fit called")
        time.sleep(1)


class TestInteractiveTrainer(BaseTrainer):
    def init(self):
        pass

    def fit(self):
        pass


class TestWorkload(BaseWorkload):
    def get_model_arc(self) -> ModelParallelismArcType:
        return ModelParallelismArcType.MEGATRON


@ray.remote
class TestActor(TestWorkload):
    def init(self):
        time.sleep(2)


@ray.remote
class TestGenerator(TestWorkload):
    def init(self):
        time.sleep(1)


@ray.remote
class TestReference(TestWorkload):
    def init(self):
        time.sleep(1)


@ray.remote
class TestReward(TestWorkload):
    def init(self):
        time.sleep(1)
