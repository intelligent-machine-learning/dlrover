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
import unittest

from omegaconf import OmegaConf

from dlrover.python.rl.api.api import RLJob, RLJobBuilder
from dlrover.python.rl.common.exception import InvalidRLConfiguration


class ApiTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_basic(self):
        conf = OmegaConf.create({"k1": "v1"})

        rl_job = (
            RLJobBuilder()
            .node_num(2)
            .device_per_node(4)
            .config(conf)
            .global_env({"e0": "v0"})
            .trainer("m0", "c0")
            .actor("m1", "c1")
            .total(4)
            .per_node(2)
            .env({"e1": "v1"})
            .rollout("m2", "c2")
            .total(4)
            .per_node(2)
            .env({"e2": "v2"})
            .reward("m3", "c3")
            .total(4)
            .per_node(2)
            .reference("m4", "c4")
            .total(4)
            .per_node(2)
            .critic("m5", "c5")
            .total(4)
            .per_node(2)
            .with_collocation("Actor", "rollout")
            .build()
        )

        self.assertIsNotNone(rl_job)
        self.assertTrue(isinstance(rl_job, RLJob))
        self.assertEqual(rl_job.node_num, 2)
        self.assertEqual(rl_job.device_per_node, 4)
        self.assertEqual(rl_job.device_type, "GPU")
        self.assertEqual(rl_job.config, {"k1": "v1"})
        self.assertEqual(rl_job.env, {"e0": "v0"})
        self.assertEqual(rl_job.trainer.module_class, ("m0", "c0"))
        self.assertEqual(rl_job.trainer.total, 1)
        self.assertEqual(rl_job.actor.module_class, ("m1", "c1"))
        self.assertEqual(rl_job.actor.total, 4)
        self.assertEqual(rl_job.actor.per_node, 2)
        self.assertEqual(rl_job.actor.env, {"e1": "v1"})
        self.assertEqual(rl_job.actor.sub_stage, [])
        self.assertEqual(rl_job.rollout.module_class, ("m2", "c2"))
        self.assertEqual(rl_job.rollout.total, 4)
        self.assertEqual(rl_job.rollout.per_node, 2)
        self.assertEqual(rl_job.rollout.env, {"e2": "v2"})
        self.assertEqual(rl_job.reward.module_class, ("m3", "c3"))
        self.assertEqual(rl_job.reference.module_class, ("m4", "c4"))
        self.assertEqual(rl_job.critic.module_class, ("m5", "c5"))
        self.assertEqual(len(rl_job.collocations), 1)
        self.assertTrue("actor" in rl_job.collocations[0])
        self.assertTrue("rollout" in rl_job.collocations[0])

        rl_config = rl_job._to_rl_config()
        self.assertTrue(len(rl_config) > 1)
        self.assertEqual(rl_config["trainer"]["class"], "c0")
        self.assertEqual(len(rl_config["workload"]), 5)
        self.assertEqual(len(rl_config["workload_group"]), 1)

    def test_validation(self):
        with self.assertRaises(InvalidRLConfiguration):
            RLJobBuilder().build()
            RLJobBuilder().node_num(1).build()
            RLJobBuilder().node_num(1).device_per_node(1).device_type(
                "mem"
            ).build()
            RLJobBuilder().node_num(1).device_per_node(1).build()
            RLJobBuilder().node_num(1).device_per_node(1).config(
                {"k1": "v1"}
            ).build()
            RLJobBuilder().node_num(1).device_per_node(1).config(
                {"k1": "v1"}
            ).trainer("m0", "c0").build()

        RLJobBuilder().node_num(1).device_per_node(1).config(
            {"k1": "v1"}
        ).trainer("m0", "c0").actor("m1", "c1").build()

    def test_collocation_all(self):
        rl_job = (
            RLJobBuilder()
            .node_num(1)
            .device_per_node(1)
            .config({"k1": "v1"})
            .trainer("m0", "c0")
            .actor("m1", "c1")
            .rollout("m2", "c2")
            .with_collocation_all()
            .build()
        )
        self.assertEqual(rl_job._collocations[0], {"actor", "rollout"})

    def test_to_rl_config(self):
        # collocation: 4: 4+4
        rl_job = (
            RLJobBuilder()
            .node_num(2)
            .device_per_node(4)
            .config({"k1": "v1"})
            .global_env({"e0": "v0"})
            .trainer("m0", "c0")
            .actor("m1", "c1")
            .total(4)
            .per_node(4)
            .env({"e1": "v1"})
            .rollout("m2", "c2")
            .total(4)
            .per_node(4)
            .env({"e2": "v2"})
            .with_collocation("Actor", "rollout")
            .build()
        )
        rl_config = rl_job._to_rl_config()
        self.assertEqual(
            rl_config["workload_group"], [{"actor": 4, "rollout": 4}]
        )
        self.assertEqual(rl_config["workload"]["actor"]["num"], 4)
        self.assertEqual(
            rl_config["workload"]["actor"]["resource"]["GPU"], 0.5
        )
        self.assertEqual(rl_config["workload"]["rollout"]["num"], 4)
        self.assertEqual(
            rl_config["workload"]["rollout"]["resource"]["GPU"], 0.5
        )

        # collocation: 4: 4+4+4
        rl_job = (
            RLJobBuilder()
            .node_num(2)
            .device_per_node(4)
            .config({"k1": "v1"})
            .global_env({"e0": "v0"})
            .trainer("m0", "c0")
            .actor("m1", "c1")
            .total(4)
            .per_node(4)
            .env({"e1": "v1"})
            .rollout("m2", "c2")
            .total(4)
            .per_node(4)
            .env({"e2": "v2"})
            .reference("m3", "c3")
            .total(4)
            .per_node(4)
            .with_collocation("actor", "rollout", "reference")
            .build()
        )
        rl_config = rl_job._to_rl_config()
        self.assertEqual(
            rl_config["workload_group"],
            [{"actor": 4, "rollout": 4, "reference": 4}],
        )
        self.assertEqual(rl_config["workload"]["actor"]["num"], 4)
        self.assertEqual(
            rl_config["workload"]["actor"]["resource"]["GPU"], 0.33
        )
        self.assertEqual(rl_config["workload"]["rollout"]["num"], 4)
        self.assertEqual(
            rl_config["workload"]["rollout"]["resource"]["GPU"], 0.33
        )
        self.assertEqual(rl_config["workload"]["reference"]["num"], 4)
        self.assertEqual(
            rl_config["workload"]["reference"]["resource"]["GPU"], 0.33
        )

        # collocation: 4: 4+4 2+2
        rl_job = (
            RLJobBuilder()
            .node_num(2)
            .device_per_node(4)
            .config({"k1": "v1"})
            .global_env({"e0": "v0"})
            .trainer("m0", "c0")
            .actor("m1", "c1")
            .total(4)
            .per_node(2)
            .env({"e1": "v1"})
            .rollout("m2", "c2")
            .total(4)
            .per_node(2)
            .env({"e2": "v2"})
            .reference("m3", "c3")
            .total(12)
            .per_node(6)
            .reward("m4", "c4")
            .total(4)
            .per_node(2)
            .with_collocation("actor", "rollout")
            .with_collocation("reward", "reference")
            .build()
        )
        rl_config = rl_job._to_rl_config()
        self.assertEqual(
            rl_config["workload_group"],
            [{"actor": 2, "rollout": 2}, {"reward": 2, "reference": 6}],
        )
        self.assertEqual(rl_config["workload"]["actor"]["num"], 4)
        self.assertEqual(rl_config["workload"]["actor"]["resource"]["GPU"], 1)
        self.assertEqual(rl_config["workload"]["rollout"]["num"], 4)
        self.assertEqual(
            rl_config["workload"]["rollout"]["resource"]["GPU"], 1
        )
        self.assertEqual(rl_config["workload"]["reference"]["num"], 12)
        self.assertEqual(
            rl_config["workload"]["reference"]["resource"]["GPU"], 0.5
        )
        self.assertEqual(rl_config["workload"]["reward"]["num"], 4)
        self.assertEqual(
            rl_config["workload"]["reward"]["resource"]["GPU"], 0.5
        )

    def test_enable_ray_auto_visible_device(self):
        rl_job = (
            RLJobBuilder()
            .node_num(2)
            .device_per_node(4)
            .config({"k1": "v1"})
            .global_env({"e0": "v0"})
            .trainer("m0", "c0")
            .actor("m1", "c1")
            .total(4)
            .per_node(4)
            .env({"e1": "v1"})
            .rollout("m2", "c2")
            .total(4)
            .enable_ray_auto_visible_device()
            .env({"e2": "v2"})
            .per_node(4)
            .with_collocation("Actor", "rollout")
            .build()
        )
        rl_config = rl_job._to_rl_config()
        self.assertEqual(
            rl_config["workload"]["rollout"]["env"][
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"
            ],
            "false",
        )
