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
import torch
import torch.distributed as dist

from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.trainer.rl_trainer import BaseRLTrainer
from dlrover.python.unified.trainer.trainer import BaseTrainer
from dlrover.python.unified.trainer.workload import (
    BaseWorkload,
    trainer_invocation,
)


class TestTrainer(BaseTrainer):
    def init(self):
        logger.info("TestTrainer init called")
        time.sleep(0.1)

    def fit(self):
        logger.info("TestTrainer fit called")
        time.sleep(0.1)


class TestInteractiveTrainer(BaseRLTrainer):
    def init(self):
        self.RG_ACTOR.init()
        self.RG_ROLLOUT.init()
        self.RG_REFERENCE.init()
        time.sleep(0.1)

    def fit(self):
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
class TestActor(BaseWorkload):
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

    def pre_func(self):
        pass

    def post_func(self):
        pass

    @trainer_invocation(
        is_async=True, timeout=5, pre_func=pre_func, post_func=post_func
    )
    def test4(self):
        pass


@ray.remote
class TestErrorActor(BaseWorkload):
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
class TestRollout(BaseWorkload):
    @trainer_invocation()
    def init(self):
        logger.info("TestRollout init called")
        time.sleep(0.1)

    @trainer_invocation()
    def generate(self, value=0):
        logger.info(f"TestRollout generate called: {value}")


@ray.remote
class TestReference(BaseWorkload):
    @trainer_invocation()
    def init(self):
        logger.info("TestReference init called")
        time.sleep(0.1)

    def compute(self, value=0):
        logger.info(f"TestReference compute called: {value}")


@ray.remote
class TestReward(BaseWorkload):
    @trainer_invocation()
    def init(self):
        logger.info("TestReward init called")
        time.sleep(0.1)

    @trainer_invocation()
    def reward(self):
        logger.info("TestReward reward called")


class TestInteractiveTorchTrainer(BaseTrainer):
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
class TestTorchActor(BaseWorkload):
    @trainer_invocation()
    def init(self):
        logger.info("TestTorchActor init called")

        dist.init_process_group(
            backend="gloo",
            init_method="env://",
            rank=self.rank,
            world_size=self.world_size,
        )
        self.matrix = torch.randn(1024, 1024)

    @trainer_invocation()
    def compute(self):
        logger.info("TestTorchActor compute called")

        local_result = torch.mm(self.matrix, self.matrix.t())
        torch.distributed.all_reduce(
            local_result, op=torch.distributed.ReduceOp.SUM
        )

        logger.info("TestTorchActor compute done")
        return local_result

    @trainer_invocation()
    def close(self):
        logger.info("TestTorchActor close called")
        dist.destroy_process_group()


@ray.remote
class TestElasticWorkload(BaseWorkload):
    def run(self):
        time.sleep(1)
        logger.info(f"TestElasticWorkload-{self.name} run called")
        return


@ray.remote
class TestErrorElasticWorkload(BaseWorkload):
    def run(self):
        if self.rank == 0:
            raise Exception(f"TestErrorElasticWorkload-{self.name} run failed")
        time.sleep(1)
        logger.info(f"TestErrorElasticWorkload-{self.name} run called")
        return
