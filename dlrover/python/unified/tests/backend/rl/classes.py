# Copyright 2025 The DLRover Authors. All rights reserved.
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
import os
import time

from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.backend.rl.trainer import (
    BaseRLTrainer,
    trainer_invocation,
)
from dlrover.python.unified.backend.rl.worker import BaseRLWorker


class SimpleInteractiveTrainer(BaseRLTrainer):
    def trainer_run(self):
        self.init()
        self.fit()

    def init(self):
        logger.info("TestInteractiveTrainer init called")
        self.RG_ACTOR.init()
        self.RG_ROLLOUT.init()
        self.RG_REFERENCE.init()
        time.sleep(0.1)

    def fit(self):
        logger.info("TestInteractiveTrainer fit called")
        self.RG_ACTOR.compute(1)
        self.RG_ROLLOUT.generate(2)
        self.RG_REFERENCE.compute(3)
        time.sleep(1)


class SimpleActor(BaseRLWorker):
    @trainer_invocation()
    def init(self):
        logger.info("TestActor init called")
        if "e1" in os.environ:
            e1_value = os.environ["e1"]
            logger.info(f"e1: {e1_value}")

        if "e2" in os.environ:
            e2_value = os.environ["e2"]
            logger.info(f"e2: {e2_value}")

        if "e3" in os.environ:
            e3_value = os.environ["e3"]
            logger.info(f"e3: {e3_value}")

        time.sleep(0.1)

    @trainer_invocation(is_async=True, timeout=5)
    def compute(self, value=0):
        logger.info(f"TestActor compute called: {value}")

    @trainer_invocation()
    def test0(self):
        pass

    @trainer_invocation(blocking=False)
    def test1(self):
        pass

    @trainer_invocation(is_async=True, timeout=5)
    def test2(self):
        pass

    @trainer_invocation(target="RANK0")
    def test3(self):
        pass

    def pre_func(self, *args, **kwargs):
        return args, kwargs

    def post_func(self, result):
        return result

    @trainer_invocation(
        is_async=True, timeout=5, pre_func=pre_func, post_func=post_func
    )
    def test4(self):
        pass
