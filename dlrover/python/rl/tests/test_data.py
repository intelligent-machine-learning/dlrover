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


class TestData(object):

    UD_SIMPLE_MOCK_RL_CONF = {
        "type": "USER_DEFINED",
        "arc_type": "MEGATRON",
        "module": "test_trainer",
        "class": "TestTrainer",
        "algorithm_type": "GRPO",
        "config": {"c1": "v1"},
        "workload": {
            "actor": {"num": 2, "module": "test_actor", "class": "TestActor"},
            "generator": {
                "num": 1,
                "module": "test_generator",
                "class": "TestGenerator",
            },
            "reference": {"num": 2, "module": "test_ref", "class": "TestRef"},
            "reward": {
                "num": 1,
                "module": "test_reward",
                "class": "testReward",
            },
        },
    }

    UD_INVALID_RESOURCE_RL_CONF = {
        "type": "USER_DEFINED",
        "arc_type": "MEGATRON",
        "module": "test_trainer",
        "class": "TestTrainer",
        "algorithm_type": "GRPO",
        "config": {"c1": "v1"},
        "workload": {
            "actor": {
                "num": 2,
                "module": "test_actor",
                "class": "TestActor",
                "resource": {"gpu": -1},
            },
            "generator": {
                "num": 1,
                "module": "test_generator",
                "class": "TestGenerator",
            },
            "reference": {"num": 2, "module": "test_ref", "class": "TestRef"},
            "reward": {
                "num": 1,
                "module": "test_reward",
                "class": "testReward",
            },
        },
    }

    UD_SIMPLE_TEST_RL_CONF = {
        "type": "USER_DEFINED",
        "arc_type": "MEGATRON",
        "module": "dlrover.python.rl.tests.test_class",
        "class": "TestTrainer",
        "algorithm_type": "GRPO",
        "config": {"c1": "v1"},
        "workload": {
            "actor": {
                "num": 2,
                "module": "dlrover.python.rl.tests.test_class",
                "class": "TestActor",
            },
            "generator": {
                "num": 1,
                "module": "dlrover.python.rl.tests.test_class",
                "class": "TestGenerator",
            },
            "reference": {
                "num": 2,
                "module": "dlrover.python.rl.tests.test_class",
                "class": "TestReference",
            },
            "reward": {
                "num": 1,
                "module": "dlrover.python.rl.tests.test_class",
                "class": "TestReward",
            },
        },
    }

    UD_DPO_MOCK_RL_CONF = {
        "type": "USER_DEFINED",
        "arc_type": "MEGATRON",
        "module": "test_trainer",
        "class": "TestTrainer",
        "algorithm_type": "DPO",
        "config": {"c1": "v1"},
        "workload": {
            "actor": {"num": 2, "module": "test_actor", "class": "TestActor"},
            "generator": {
                "num": 1,
                "module": "test_generator",
                "class": "TestGenerator",
            },
            "reference": {"num": 2, "module": "test_ref", "class": "TestRef"},
            "reward": {
                "num": 1,
                "module": "test_reward",
                "class": "testReward",
            },
        },
    }
