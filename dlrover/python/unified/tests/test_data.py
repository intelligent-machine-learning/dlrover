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


class TestData(object):

    UD_SIMPLE_MOCK_RL_CONF = {
        "config": {"c1": "v1"},
        "trainer": {
            "type": "USER_DEFINED",
            "module": "test_trainer",
            "class": "TestTrainer",
            "node_number": 4,
            "device_type": "CPU",
            "device_per_node": 1,
        },
        "workload": {
            "actor": {"num": 1, "module": "test_actor", "class": "TestActor"},
            "rollout": {
                "num": 1,
                "module": "test_rollout",
                "class": "TestRollout",
            },
            "reference": {"num": 1, "module": "test_ref", "class": "TestRef"},
            "reward": {
                "num": 1,
                "module": "test_reward",
                "class": "testReward",
            },
        },
    }

    UD_SIMPLE_HOST_INVALID_GROUPED_RL_CONF_0 = {
        "config": {"c1": "v1"},
        "trainer": {
            "type": "USER_DEFINED",
            "module": "dlrover.python.unified.tests.test_class",
            "class": "TestInteractiveTrainer",
            "node_number": 1,
            "device_type": "CPU",
            "device_per_node": 1,
        },
        "workload_group": [
            {"actor": 2, "rollout": 2},
            {"actor": 2, "reference": 2},
        ],
        "workload": {
            "actor": {
                "num": 2,
                "module": "dlrover.python.unified.tests.test_class",
                "class": "TestActor",
            },
            "rollout": {
                "num": 2,
                "module": "dlrover.python.unified.tests.test_class",
                "class": "TestRollout",
            },
            "reference": {
                "num": 2,
                "module": "dlrover.python.unified.tests.test_class",
                "class": "TestReference",
            },
            "reward": {
                "num": 1,
                "module": "dlrover.python.unified.tests.test_class",
                "class": "TestReward",
            },
        },
    }

    UD_SIMPLE_HOST_INVALID_GROUPED_RL_CONF_1 = {
        "config": {"c1": "v1"},
        "trainer": {
            "type": "USER_DEFINED",
            "module": "dlrover.python.unified.tests.test_class",
            "class": "TestInteractiveTrainer",
            "node_number": 1,
            "device_type": "CPU",
            "device_per_node": 1,
        },
        "workload_group": [
            {"actor": 2, "rollout": 2},
        ],
        "workload": {
            "actor": {
                "num": 2,
                "module": "dlrover.python.unified.tests.test_class",
                "class": "TestActor",
            },
            "rollout": {
                "num": 2,
                "module": "dlrover.python.unified.tests.test_class",
                "class": "TestRollout",
            },
        },
    }

    UD_INVALID_RESOURCE_RL_CONF_0 = {
        "config": {"c1": "v1"},
        "trainer": {
            "type": "USER_DEFINED",
            "module": "dlrover.python.unified.tests.test_class",
            "class": "TestInteractiveTrainer",
            "node_number": 1,
            "device_type": "CPU",
            "device_per_node": 1,
        },
        "workload": {
            "actor": {
                "num": 2,
                "module": "dlrover.python.unified.tests.test_class",
                "class": "TestActor",
                "resource": {"gpu": -1},
            },
            "rollout": {
                "num": 1,
                "module": "dlrover.python.unified.tests.test_class",
                "class": "TestRollout",
            },
            "reference": {
                "num": 2,
                "module": "dlrover.python.unified.tests.test_class",
                "class": "TestReference",
            },
            "reward": {
                "num": 1,
                "module": "dlrover.python.unified.tests.test_class",
                "class": "TestReward",
            },
        },
    }

    UD_SIMPLE_TEST_SFT_CONF_0 = {
        "config": {"c1": "v1"},
        "trainer": {
            "type": "ELASTIC_TRAINING",
            "module": "dlrover.python.unified.tests.test_class",
            "class": "TestTrainer",
            "node_number": 2,
            "device_type": "CPU",
            "device_per_node": 1,
        },
        "workload": {
            "ELASTIC": {
                "num": 2,
                "module": "dlrover.python.unified.tests.test_class",
                "class": "TestActor",
                "resource": {"cpu": 1},
            }
        },
    }

    UD_SIMPLE_TEST_RL_CONF_0 = {
        "config": {"c1": "v1"},
        "trainer": {
            "type": "USER_DEFINED",
            "module": "dlrover.python.unified.tests.test_class",
            "class": "TestTrainer",
            "node_number": 1,
            "device_type": "CPU",
            "device_per_node": 1,
        },
        "workload_group": [
            {"actor": 1, "rollout": 1, "reference": 1, "reward": 1}
        ],
        "workload": {
            "actor": {
                "num": 1,
                "module": "dlrover.python.unified.tests.test_class",
                "class": "TestActor",
                "resource": {"cpu": 0.25},
            },
            "rollout": {
                "num": 1,
                "module": "dlrover.python.unified.tests.test_class",
                "class": "TestRollout",
                "resource": {"cpu": 0.25},
            },
            "reference": {
                "num": 1,
                "module": "dlrover.python.unified.tests.test_class",
                "class": "TestReference",
                "resource": {"cpu": 0.25},
            },
            "reward": {
                "num": 1,
                "module": "dlrover.python.unified.tests.test_class",
                "class": "TestReward",
                "resource": {"cpu": 0.25},
            },
        },
    }

    UD_SIMPLE_TEST_RL_CONF_1 = {
        "config": {"c1": "v1"},
        "trainer": {
            "type": "USER_DEFINED",
            "module": "dlrover.python.unified.tests.test_class",
            "class": "TestTrainer",
            "node_number": 4,
            "device_type": "CPU",
            "device_per_node": 1,
        },
        "workload_group": [{"actor": 1, "rollout": 1}],
        "workload": {
            "actor": {
                "num": 2,
                "module": "dlrover.python.unified.tests.test_class",
                "class": "TestActor",
                "resource": {"cpu": 0.5},
            },
            "rollout": {
                "num": 2,
                "module": "dlrover.python.unified.tests.test_class",
                "class": "TestRollout",
                "resource": {"cpu": 0.5},
            },
            "reference": {
                "num": 1,
                "module": "dlrover.python.unified.tests.test_class",
                "class": "TestReference",
            },
            "reward": {
                "num": 1,
                "module": "dlrover.python.unified.tests.test_class",
                "class": "TestReward",
            },
        },
    }

    UD_SIMPLE_TEST_WITH_INTERACTIVE_RL_CONF = {
        "config": {"c1": "v1"},
        "env": {"e1": "v1", "e2": "v2"},
        "trainer": {
            "type": "USER_DEFINED",
            "module": "dlrover.python.unified.tests.test_class",
            "class": "TestInteractiveTrainer",
            "node_number": 3,
            "device_type": "CPU",
            "device_per_node": 2,
        },
        "workload": {
            "actor": {
                "num": 2,
                "module": "dlrover.python.unified.tests.test_class",
                "class": "TestActor",
                "env": {"e2": "v22", "e3": "v3"},
            },
            "rollout": {
                "num": 2,
                "module": "dlrover.python.unified.tests.test_class",
                "class": "TestRollout",
            },
            "reference": {
                "num": 2,
                "module": "dlrover.python.unified.tests.test_class",
                "class": "TestReference",
            },
        },
    }

    UD_SIMPLE_TEST_WITH_ERROR_TRAINER_RL_CONF = {
        "config": {"c1": "v1"},
        "trainer": {
            "type": "USER_DEFINED",
            "module": "dlrover.python.unified.tests.test_class",
            "class": "TestInteractiveErrorTrainer",
            "node_number": 1,
            "device_type": "CPU",
            "device_per_node": 1,
        },
        "workload": {
            "actor": {
                "num": 1,
                "module": "dlrover.python.unified.tests.test_class",
                "class": "TestActor",
            },
        },
    }

    UD_SIMPLE_TEST_WITH_ERROR_ACTOR_RL_CONF = {
        "config": {"c1": "v1"},
        "trainer": {
            "type": "USER_DEFINED",
            "module": "dlrover.python.unified.tests.test_class",
            "class": "TestInteractiveActorErrorTrainer",
            "node_number": 1,
            "device_type": "CPU",
            "device_per_node": 1,
        },
        "workload": {
            "actor": {
                "num": 1,
                "module": "dlrover.python.unified.tests.test_class",
                "class": "TestErrorActor",
            },
        },
    }

    UD_SIMPLE_TEST_WITH_ERROR_TRAINER_ACTOR_RL_CONF = {
        "config": {"c1": "v1"},
        "trainer": {
            "type": "USER_DEFINED",
            "module": "dlrover.python.unified.tests.test_class",
            "class": "TestInteractiveErrorTrainer",
            "node_number": 1,
            "device_type": "CPU",
            "device_per_node": 1,
        },
        "workload": {
            "actor": {
                "num": 1,
                "module": "dlrover.python.unified.tests.test_class",
                "class": "TestErrorActor",
            },
        },
    }

    UD_SIMPLE_TEST_WITH_INTERACTIVE_GROUPED_RL_CONF = {
        "config": {"c1": "v1"},
        "trainer": {
            "type": "USER_DEFINED",
            "module": "dlrover.python.unified.tests.test_class",
            "class": "TestInteractiveTrainer",
            "node_number": 4,
            "device_type": "CPU",
            "device_per_node": 2,
        },
        "workload_group": [{"actor": 2, "rollout": 2}],
        "workload": {
            "actor": {
                "num": 4,
                "module": "dlrover.python.unified.tests.test_class",
                "class": "TestActor",
                "resource": {"cpu": 0.5},
            },
            "reference": {
                "num": 2,
                "module": "dlrover.python.unified.tests.test_class",
                "class": "TestReference",
            },
            "rollout": {
                "num": 4,
                "module": "dlrover.python.unified.tests.test_class",
                "class": "TestRollout",
                "resource": {"cpu": 0.5},
            },
            "reward": {
                "num": 2,
                "module": "dlrover.python.unified.tests.test_class",
                "class": "TestReward",
            },
        },
    }

    UD_SIMPLE_TEST_WITH_INTERACTIVE_GROUPED_RL_CONF_1 = {
        "config": {"c1": "v1"},
        "trainer": {
            "type": "USER_DEFINED",
            "module": "dlrover.python.unified.tests.test_class",
            "class": "TestInteractiveTrainer",
            "node_number": 1,
            "device_type": "CPU",
            "device_per_node": 4,
        },
        "workload_group": [{"actor": 4, "rollout": 4, "reference": 4}],
        "workload": {
            "actor": {
                "num": 4,
                "module": "dlrover.python.unified.tests.test_class",
                "class": "TestActor",
                "resource": {"cpu": 0.33},
            },
            "reference": {
                "num": 4,
                "module": "dlrover.python.unified.tests.test_class",
                "class": "TestReference",
                "resource": {"cpu": 0.33},
            },
            "rollout": {
                "num": 4,
                "module": "dlrover.python.unified.tests.test_class",
                "class": "TestRollout",
                "resource": {"cpu": 0.33},
            },
        },
    }

    UD_SIMPLE_TEST_NONE_COLOCATE_HOST_GROUPED_RL_CONF = {
        "config": {"c1": "v1"},
        "trainer": {
            "type": "USER_DEFINED",
            "module": "dlrover.python.unified.tests.test_class",
            "class": "TestInteractiveTrainer",
            "node_number": 4,
            "device_type": "CPU",
            "device_per_node": 2,
        },
        "workload_group": [{"actor": 1, "rollout": 1}],
        "workload": {
            "actor": {
                "num": 2,
                "module": "dlrover.python.unified.tests.test_class",
                "class": "TestActor",
            },
            "reference": {
                "num": 2,
                "module": "dlrover.python.unified.tests.test_class",
                "class": "TestReference",
            },
            "rollout": {
                "num": 2,
                "module": "dlrover.python.unified.tests.test_class",
                "class": "TestRollout",
            },
            "reward": {
                "num": 2,
                "module": "dlrover.python.unified.tests.test_class",
                "class": "TestReward",
            },
        },
    }

    UD_DPO_MOCK_RL_CONF = {
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
