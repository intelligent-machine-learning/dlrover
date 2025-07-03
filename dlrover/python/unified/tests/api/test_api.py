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

from omegaconf import OmegaConf

from dlrover.python.common.constants import CommunicationType, NodeEnv
from dlrover.python.unified.api.base import (
    DLJob,
    DLJobBuilder,
    DLRoverRunBuilder,
    TrainerBuilder,
    WorkloadBuilder,
)
from dlrover.python.unified.api.rl import RLJobBuilder
from dlrover.python.unified.common.constant import InternalDLConfig
from dlrover.python.unified.common.enums import (
    DLStreamType,
    DLType,
    TrainerType,
)
from dlrover.python.unified.common.exception import InvalidDLConfiguration
from dlrover.python.unified.tests.base import BaseTest


class ApiTest(BaseTest):
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
        self.assertTrue(isinstance(rl_job, DLJob))
        self.assertEqual(rl_job.dl_type, "RL")
        self.assertEqual(rl_job.stream_type, DLStreamType.TASK_STREAM)
        self.assertEqual(rl_job.node_num, 2)
        self.assertEqual(rl_job.device_per_node, 4)
        self.assertEqual(rl_job.device_type, "GPU")
        self.assertEqual(rl_job.config, {"k1": "v1"})
        self.assertEqual(rl_job.env, {"e0": "v0"})
        self.assertEqual(rl_job.trainer.module_class, ("m0", "c0"))
        self.assertEqual(rl_job.trainer.trainer_type, TrainerType.USER_DEFINED)
        self.assertEqual(
            rl_job.get_workload("actor").module_class, ("m1", "c1")
        )
        self.assertEqual(rl_job._components["actor"].role_name, "actor")
        self.assertEqual(rl_job.get_workload("actor").total, 4)
        self.assertEqual(rl_job.get_workload("actor").per_node, 2)
        self.assertEqual(rl_job.get_workload("actor").env, {"e1": "v1"})
        self.assertEqual(rl_job.get_workload("actor").sub_stage, [])
        self.assertEqual(
            rl_job.get_workload("rollout").module_class, ("m2", "c2")
        )
        self.assertEqual(rl_job.get_workload("rollout").total, 4)
        self.assertEqual(rl_job.get_workload("rollout").per_node, 2)
        self.assertEqual(rl_job.get_workload("rollout").env, {"e2": "v2"})
        self.assertEqual(
            rl_job.get_workload("reward").module_class, ("m3", "c3")
        )
        self.assertEqual(
            rl_job.get_workload("reference").module_class, ("m4", "c4")
        )
        self.assertEqual(
            rl_job.get_workload("critic").module_class, ("m5", "c5")
        )
        self.assertEqual(len(rl_job.collocations), 1)
        self.assertTrue("actor" in rl_job.collocations[0])
        self.assertTrue("rollout" in rl_job.collocations[0])

        rl_config = rl_job._to_dl_config()
        self.assertTrue(len(rl_config) > 1)
        self.assertEqual(rl_config["trainer"]["class"], "c0")
        self.assertEqual(len(rl_config["workload"]), 5)
        self.assertEqual(len(rl_config["workload_group"]), 1)

    def test_dl_type(self):
        dl_job = (
            DLJobBuilder()
            .as_data_stream()
            .SFT_type()
            .node_num(2)
            .device_per_node(4)
            .config({"k1": "v1"})
            .build()
        )

        self.assertEqual(dl_job.dl_type, DLType.SFT.name)

        dl_job = (
            DLJobBuilder()
            .as_data_stream()
            .PRE_type()
            .node_num(2)
            .device_per_node(4)
            .config({"k1": "v1"})
            .build()
        )

        self.assertEqual(dl_job.dl_type, DLType.PRE.name)

        dl_job = (
            DLJobBuilder()
            .as_data_stream()
            .MULTIMODAL_type()
            .node_num(2)
            .device_per_node(4)
            .config({"k1": "v1"})
            .build()
        )

        self.assertEqual(dl_job.dl_type, DLType.MULTIMODAL.name)

    def test_validation(self):
        # without dl type
        with self.assertRaises(InvalidDLConfiguration):
            DLJobBuilder().build()

        # without node num
        with self.assertRaises(InvalidDLConfiguration):
            DLJobBuilder().dl_type().build()

        # without device per node
        with self.assertRaises(InvalidDLConfiguration):
            DLJobBuilder().dl_type().node_num(1).build()

        # invalid device type
        with self.assertRaises(InvalidDLConfiguration):
            DLJobBuilder().dl_type().node_num(1).device_per_node(
                1
            ).device_type("mem").build()

        # without config
        with self.assertRaises(InvalidDLConfiguration):
            DLJobBuilder().dl_type().node_num(1).device_per_node(1).build()

        # without trainer
        with self.assertRaises(InvalidDLConfiguration):
            DLJobBuilder().dl_type().node_num(1).device_per_node(1).config(
                {"k1": "v1"}
            ).build()

        # without actor
        with self.assertRaises(InvalidDLConfiguration):
            RLJobBuilder().node_num(1).device_per_node(1).config(
                {"k1": "v1"}
            ).trainer("m0", "c0").build()

        # invalid trainer
        with self.assertRaises(InvalidDLConfiguration):
            RLJobBuilder().node_num(1).device_per_node(1).config(
                {"k1": "v1"}
            ).trainer("", "").build()

        # invalid collocation
        with self.assertRaises(InvalidDLConfiguration):
            RLJobBuilder().node_num(1).device_per_node(1).config(
                {"k1": "v1"}
            ).trainer("m0", "c0").actor("m1", "c1").rollout("m2", "c2").reward(
                "m3", "c3"
            ).with_collocation(
                "actor", "rollout"
            ).with_collocation(
                "rollout", "reward"
            ).build()

        # a minimum valid rl
        RLJobBuilder().node_num(1).device_per_node(1).config(
            {"k1": "v1"}
        ).trainer("m0", "c0").actor("m1", "c1").total(1).per_node(1).build()

    def test_collocation_all(self):
        rl_job = (
            RLJobBuilder()
            .node_num(1)
            .device_per_node(1)
            .config({"k1": "v1"})
            .trainer("m0", "c0")
            .actor("m1", "c1")
            .total(1)
            .per_node(1)
            .rollout("m2", "c2")
            .total(1)
            .per_node(1)
            .with_collocation_all()
            .build()
        )
        self.assertEqual(rl_job._collocations[0], {"actor", "rollout"})

    def test_to_dl_config(self):
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
        rl_config = rl_job._to_dl_config()
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
        rl_config = rl_job._to_dl_config()
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
        rl_config = rl_job._to_dl_config()
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
        rl_config = rl_job._to_dl_config()
        self.assertEqual(
            rl_config["workload"]["rollout"]["env"][
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"
            ],
            "false",
        )

    def test_elastic(self):
        cmd = "dlrover-run --nnodes=2 --nproc_per_node=2 test.py"
        dl_job = (
            DLJobBuilder()
            .SFT_type()
            .node_num(3)
            .device_per_node(2)
            .device_type("CPU")
            .config({"c1": "v1"})
            .global_env({"e0": "v0"})
            .dlrover_run(cmd)
            .build()
        )

        self.assertIsNotNone(dl_job)
        dl_config = dl_job._to_dl_config()
        self.assertEqual(
            dl_config["trainer"]["type"],
            "ELASTIC_TRAINING",
        )
        self.assertEqual(
            dl_config["trainer"]["class"],
            "DefaultTrainer",
        )
        self.assertEqual(
            dl_config["config"][InternalDLConfig.ELASTIC_RUN_CMD],
            cmd,
        )
        self.assertEqual(
            dl_config["env"][NodeEnv.DLROVER_MASTER_SERVICE_TYPE],
            CommunicationType.COMM_SERVICE_RAY,
        )
        self.assertEqual(
            dl_config["workload"]["ELASTIC"]["class"],
            "ElasticWorkload",
        )
        self.assertEqual(
            dl_config["workload"]["ELASTIC"]["num"],
            2,
        )
        self.assertEqual(
            dl_config["workload"]["ELASTIC"]["resource"]["CPU"],
            2,
        )

    def test_role_builder(self):
        trainer_builder = TrainerBuilder(DLJobBuilder(), "test", "m0", "c0")
        self.assertIsNotNone(trainer_builder)
        self.assertEqual(trainer_builder._get_total(), 1)
        self.assertEqual(trainer_builder._get_per_node(), 0)
        with self.assertRaises(AttributeError):
            trainer_builder.test()

        workload_builder = WorkloadBuilder(DLJobBuilder(), "test", "m0", "c0")
        workload_builder.env(None)
        self.assertEqual(workload_builder._env, {})
        workload_builder.sub_stage(None)
        self.assertEqual(workload_builder._sub_stage, [])
        workload_builder.sub_stage([1])
        self.assertEqual(workload_builder._sub_stage, [1])
        self.assertFalse(workload_builder._validate())
        workload_builder.total(1)
        workload_builder.per_node(1)
        self.assertTrue(workload_builder._validate())

        dlrover_run_builder = DLRoverRunBuilder(DLJobBuilder(), "test")
        self.assertFalse(dlrover_run_builder._validate())
        dlrover_run_builder._cmd = "dlrover-run tset.py"
        self.assertFalse(dlrover_run_builder._validate())
        dlrover_run_builder.per_node(2)
        dlrover_run_builder.total(1)
        self.assertTrue(dlrover_run_builder._validate())

        dlrover_run_builder = DLRoverRunBuilder(
            DLJobBuilder(),
            "dlrover-run --nnodes=1:2 --nproc_per_node=2 "
            "--rdzv_conf join_timeout=600 --network_check "
            "--max-restarts=1 test.py",
            "dlrover.python.unified.tests.test_class",
            "TestElasticWorkload",
        )
        self.assertTrue(dlrover_run_builder._validate())
