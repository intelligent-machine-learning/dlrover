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
import os
import time

import ray
import torch
import torch.distributed as dist

from dlrover.python.common.log import default_logger as logger
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


class TestInteractiveErrorTrainer(BaseTrainer):
    def init(self):
        self.RG_ACTOR.init()
        self.RG_ROLLOUT.init()
        self.RG_REFERENCE.init()
        self.RG_REWARD.init()
        time.sleep(0.1)

    def fit(self):
        self.RG_ACTOR.compute(1)
        self.RG_ROLLOUT.generate(2)
        raise RuntimeError("Failover testing...")
        self.RG_REFERENCE.compute(3)
        self.RG_REWARD.reward()
        time.sleep(1)


@ray.remote
class TestActor(BaseWorkload):
    @trainer_invocation()
    def init(self):
        logger.info("TestActor init called")
        time.sleep(0.1)

    @trainer_invocation(is_async=True, timeout=5)
    def compute(self, value=0):
        logger.info(f"TestActor compute called: {value}")


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
