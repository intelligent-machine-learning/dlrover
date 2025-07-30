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
import threading
import time

import ray

from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.backend.rl.trainer import (
    BaseRLTrainer,
    trainer_invocation,
)
from dlrover.python.unified.backend.rl.worker import BaseRLWorker


class TestInteractiveTrainer(BaseRLTrainer):
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


class TestInteractiveErrorTrainer(BaseRLTrainer):
    def init(self):
        self.RG_ACTOR.init()
        time.sleep(0.1)

    def fit(self):
        self.RG_ACTOR.compute(1)
        raise RuntimeError("Failover testing...")


class TestInteractiveActorErrorTrainer(BaseRLTrainer):
    def init(self):
        self.RG_ACTOR.init()
        # the actor will restart at this point
        time.sleep(3)

    def fit(self):
        time.sleep(10)


@ray.remote
class TestActor(BaseRLWorker):
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


@ray.remote
class TestErrorActor(BaseRLWorker):
    @trainer_invocation()
    def init(self):
        logger.info("TestErrorActor init called")
        thread = threading.Thread(target=self.raise_error, daemon=True)
        thread.start()
        logger.info("TestErrorActor init done")

    def raise_error(self):
        time.sleep(0.5)
        os._exit(1)

    @trainer_invocation(is_async=True, timeout=5)
    def compute(self, value=0):
        logger.info(f"TestErrorActor compute called: {value}")
        self.raise_error()


@ray.remote
class TestRollout(BaseRLWorker):
    @trainer_invocation()
    def init(self):
        logger.info("TestRollout init called")
        time.sleep(0.1)

    @trainer_invocation()
    def generate(self, value=0):
        logger.info(f"TestRollout generate called: {value}")


@ray.remote
class TestReference(BaseRLWorker):
    @trainer_invocation()
    def init(self):
        logger.info("TestReference init called")
        time.sleep(0.1)

    def compute(self, value=0):
        logger.info(f"TestReference compute called: {value}")


@ray.remote
class TestReward(BaseRLWorker):
    @trainer_invocation()
    def init(self):
        logger.info("TestReward init called")
        time.sleep(0.1)

    @trainer_invocation()
    def reward(self):
        logger.info("TestReward reward called")


class TestInteractiveTorchTrainer(BaseRLTrainer):
    def init(self):
        self.RG_ACTOR.init()
        self.RG_ROLLOUT.init()
        self.RG_REFERENCE.init()
        self.RG_REWARD.init()

        logger.info("TestInteractiveTorchTrainer init done")

    def fit(self):
        self.RG_ACTOR.compute()
        self.RG_ROLLOUT.generate(2)
        self.RG_REFERENCE.compute(3)
        self.RG_REWARD.reward()

        time.sleep(0.5)

        self.RG_ACTOR.close()


@ray.remote
class TestElasticWorkload(BaseRLWorker):
    def run(self):
        time.sleep(1)
        logger.info(f"TestElasticWorkload-{self.name} run called")
        return


def elastic_workload_run():
    time.sleep(1)
    logger.info("elastic_workload_run run called")


def elastic_workload_run_error():
    raise Exception("elastic_workload_run_error run failed")
