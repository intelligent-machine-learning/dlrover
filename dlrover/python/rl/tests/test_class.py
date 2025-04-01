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
from dlrover.python.rl.trainer.workload import BaseWorkload, trainer_invocation


class TestTrainer(BaseTrainer):
    def init(self):
        logger.info("TestTrainer init called")
        time.sleep(0.1)

    def fit(self):
        logger.info("TestTrainer fit called")
        time.sleep(0.1)


class TestInteractiveTrainer(BaseTrainer):
    def init(self):
        self.RG_ACTOR.init()
        self.RG_ROLLOUT.init()
        self.RG_REFERENCE.init()
        self.RG_REWARD.init()
        time.sleep(0.1)

    def fit(self):
        self.RG_ACTOR.compute(1)
        self.RG_ROLLOUT.generate(2)
        self.RG_REFERENCE.compute(3)
        self.RG_REWARD.reward()
        time.sleep(1)


class TestWorkload(BaseWorkload):
    def get_model_arc(self) -> ModelParallelismArcType:
        return ModelParallelismArcType.MEGATRON


@ray.remote
class TestActor(TestWorkload):
    @trainer_invocation()
    def init(self):
        logger.info("TestActor init called")
        time.sleep(0.1)

    @trainer_invocation(is_async=True, timeout=5)
    def compute(self, value=0):
        logger.info(f"TestActor compute called: {value}")


@ray.remote
class TestRollout(TestWorkload):
    @trainer_invocation()
    def init(self):
        logger.info("TestRollout init called")
        time.sleep(0.1)

    @trainer_invocation()
    def generate(self, value=0):
        logger.info(f"TestRollout generate called: {value}")


@ray.remote
class TestReference(TestWorkload):
    @trainer_invocation()
    def init(self):
        logger.info("TestReference init called")
        time.sleep(0.1)

    def compute(self, value=0):
        logger.info(f"TestReference compute called: {value}")


@ray.remote
class TestReward(TestWorkload):
    @trainer_invocation()
    def init(self):
        logger.info("TestReward init called")
        time.sleep(0.1)

    @trainer_invocation()
    def reward(self):
        logger.info("TestReward reward called")
