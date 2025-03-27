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
from dlrover.python.rl.common.enums import ModelParallelismArcType, RLRoleType
from dlrover.python.rl.trainer.trainer import BaseTrainer
from dlrover.python.rl.trainer.workload import BaseWorkload


class TestTrainer(BaseTrainer):
    def init(self):
        logger.info("TestTrainer init called")
        time.sleep(0.1)

    def fit(self):
        logger.info("TestTrainer fit called")
        time.sleep(0.1)


class TestInteractiveTrainer(BaseTrainer):
    def init(self):
        for role, actor_handles in self.actor_handles.items():
            now = time.time() * 1000
            init_refs = [
                actor_handle.init.remote() for actor_handle in actor_handles
            ]
            ray.get(init_refs)
            logger.info(f"Done {role} init, cost {time.time()*1000 - now}ms")
        time.sleep(0.1)

    def fit(self):
        for role, actor_handles in self.actor_handles.items():
            now = time.time() * 1000
            if role == RLRoleType.ACTOR or role == RLRoleType.REFERENCE:
                method_refs = [
                    actor_handle.compute.remote()
                    for actor_handle in actor_handles
                ]
                ray.get(method_refs)
                logger.info(
                    f"Done {role} compute, cost {time.time() * 1000 - now}ms"
                )
            elif role == RLRoleType.REWARD:
                method_refs = [
                    actor_handle.reward.remote()
                    for actor_handle in actor_handles
                ]
                ray.get(method_refs)
                logger.info(
                    f"Done {role} reward, cost {time.time() * 1000 - now}ms"
                )
            elif role == RLRoleType.ROLLOUT:
                method_refs = [
                    actor_handle.generate.remote()
                    for actor_handle in actor_handles
                ]
                ray.get(method_refs)
                logger.info(
                    f"Done {role} generate, cost {time.time() * 1000 - now}ms"
                )
        time.sleep(0.1)


class TestWorkload(BaseWorkload):
    def get_model_arc(self) -> ModelParallelismArcType:
        return ModelParallelismArcType.MEGATRON


@ray.remote
class TestActor(TestWorkload):
    def init(self):
        time.sleep(0.1)

    def compute(self):
        logger.info("TestActor compute called")


@ray.remote
class TestRollout(TestWorkload):
    def init(self):
        time.sleep(0.1)

    def generate(self):
        logger.info("TestRollout generate called")


@ray.remote
class TestReference(TestWorkload):
    def init(self):
        time.sleep(0.1)

    def compute(self):
        logger.info("TestReference compute called")


@ray.remote
class TestReward(TestWorkload):
    def init(self):
        time.sleep(0.1)

    def reward(self):
        logger.info("TestReward reward called")
