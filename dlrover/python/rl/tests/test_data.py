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
        "algorithm_type": "GRPO",
        "config": {"c1": "v1"},
        "trainer": {
            "type": "USER_DEFINED",
            "module": "test_trainer",
            "class": "TestTrainer",
        },
        "workload": {
            "actor": {"num": 2, "module": "test_actor", "class": "TestActor"},
            "rollout": {
                "num": 1,
                "module": "test_rollout",
                "class": "TestRollout",
            },
            "reference": {"num": 2, "module": "test_ref", "class": "TestRef"},
            "reward": {
                "num": 1,
                "module": "test_reward",
                "class": "testReward",
            },
        },
    }

    UD_SIMPLE_MOCK_HOST_GROUPED_RL_CONF = {
        "algorithm_type": "GRPO",
        "config": {"c1": "v1"},
        "trainer": {
            "type": "USER_DEFINED",
            "module": "test_trainer",
            "class": "TestTrainer",
        },
        "workload_group": {"host_group": [{"actor": 2, "rollout": 2}]},
        "workload": {
            "actor": {"num": 2, "module": "test_actor", "class": "TestActor"},
            "rollout": {
                "num": 2,
                "module": "test_rollout",
                "class": "TestRollout",
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
        "algorithm_type": "GRPO",
        "config": {"c1": "v1"},
        "trainer": {
            "type": "USER_DEFINED",
            "module": "test_trainer",
            "class": "TestTrainer",
        },
        "workload": {
            "actor": {
                "num": 2,
                "module": "test_actor",
                "class": "TestActor",
                "resource": {"gpu": -1},
            },
            "rollout": {
                "num": 1,
                "module": "test_rollout",
                "class": "TestRollout",
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
        "algorithm_type": "GRPO",
        "config": {"c1": "v1"},
        "trainer": {
            "type": "USER_DEFINED",
            "module": "dlrover.python.rl.tests.test_class",
            "class": "TestTrainer",
        },
        "workload": {
            "actor": {
                "num": 2,
                "module": "dlrover.python.rl.tests.test_class",
                "class": "TestActor",
                "resource": {"cpu": 0.1},
            },
            "rollout": {
                "num": 1,
                "module": "dlrover.python.rl.tests.test_class",
                "class": "TestRollout",
                "resource": {"cpu": 0.1},
            },
            "reference": {
                "num": 2,
                "module": "dlrover.python.rl.tests.test_class",
                "class": "TestReference",
                "resource": {"cpu": 0.1},
            },
            "reward": {
                "num": 1,
                "module": "dlrover.python.rl.tests.test_class",
                "class": "TestReward",
                "resource": {"cpu": 0.1},
            },
        },
    }

    UD_SIMPLE_TEST_WITH_INTERACTIVE_RL_CONF = {
        "algorithm_type": "GRPO",
        "config": {"c1": "v1"},
        "trainer": {
            "type": "USER_DEFINED",
            "module": "dlrover.python.rl.tests.test_class",
            "class": "TestInteractiveTrainer",
        },
        "workload": {
            "actor": {
                "num": 2,
                "module": "dlrover.python.rl.tests.test_class",
                "class": "TestActor",
                "resource": {"cpu": 0.1},
            },
            "rollout": {
                "num": 1,
                "module": "dlrover.python.rl.tests.test_class",
                "class": "TestRollout",
                "resource": {"cpu": 0.1},
            },
            "reference": {
                "num": 2,
                "module": "dlrover.python.rl.tests.test_class",
                "class": "TestReference",
                "resource": {"cpu": 0.1},
            },
            "reward": {
                "num": 1,
                "module": "dlrover.python.rl.tests.test_class",
                "class": "TestReward",
                "resource": {"cpu": 0.1},
            },
        },
    }

    UD_DPO_MOCK_RL_CONF = {
        "algorithm_type": "DPO",
        "config": {"c1": "v1"},
        "trainer": {
            "module": "test_trainer",
            "class": "TestTrainer",
            "type": "USER_DEFINED",
        },
        "workload": {
            "actor": {"num": 2, "module": "test_actor", "class": "TestActor"},
            "rollout": {
                "num": 1,
                "module": "test_rollout",
                "class": "TestRollout",
            },
            "reference": {"num": 2, "module": "test_ref", "class": "TestRef"},
            "reward": {
                "num": 1,
                "module": "test_reward",
                "class": "testReward",
            },
        },
    }
