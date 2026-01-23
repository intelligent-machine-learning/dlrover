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

from unittest.mock import patch

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from dlrover.python.unified.api.builder import (
    DLJob,
    DLJobBuilder,
    RLJobBuilder,
)
from dlrover.python.unified.api.builder.rl import RLJob
from dlrover.python.unified.common.config import DLConfig, JobConfig
from dlrover.python.unified.common.enums import (
    DLStreamType,
    RLRoleType,
)
from dlrover.python.unified.tests.base import BaseTest
import os


class ApiTest(BaseTest):
    def test_basic(self):
        conf = OmegaConf.create({"k1": "v1"})

        rl_job: DLJob = (
            RLJobBuilder()
            .node_num(2)
            .device_per_node(4)
            .config(conf)
            .global_env({"e0": "v0"})
            .trainer("m0.c0")
            .end()
            .actor("m1.c1")
            .total(4)
            .per_group(2)
            .env({"e1": "v1"})
            .end()
            .rollout("m2.c2")
            .total(4)
            .per_group(2)
            .env({"e2": "v2"})
            .end()
            .reward("m3.c3")
            .total(4)
            .per_group(2)
            .env({"e3": "v3"})
            .end()
            .reference("m4.c4")
            .total(4)
            .per_group(2)
            .end()
            .critic("m5.c5")
            .total(4)
            .per_group(2)
            .end()
            .with_collocation(RLRoleType.ACTOR.name, RLRoleType.ROLLOUT.name)
            .build()
        )

        assert isinstance(rl_job, RLJob)
        self.assertEqual(rl_job.stream_type, DLStreamType.TASK_STREAM)
        self.assertEqual(rl_job.node_num, 2)
        self.assertEqual(rl_job.device_per_node, 4)
        self.assertEqual(rl_job.device_type, "GPU")
        self.assertEqual(rl_job.config, {"k1": "v1"})
        self.assertEqual(rl_job.env, {"e0": "v0"})
        self.assertEqual(
            rl_job.get_workload(RLRoleType.TRAINER.name).entry_point, "m0.c0"
        )
        self.assertEqual(
            rl_job.get_workload(RLRoleType.ACTOR.name).entry_point, "m1.c1"
        )
        self.assertEqual(rl_job.get_workload(RLRoleType.ACTOR.name).total, 4)
        self.assertEqual(
            rl_job.get_workload(RLRoleType.ACTOR.name).per_group, 2
        )
        self.assertEqual(
            rl_job.get_workload(RLRoleType.ACTOR.name).envs, {"e1": "v1"}
        )
        self.assertEqual(
            rl_job.get_workload(RLRoleType.ROLLOUT.name).entry_point, "m2.c2"
        )
        self.assertEqual(rl_job.get_workload(RLRoleType.ROLLOUT.name).total, 4)
        self.assertEqual(
            rl_job.get_workload(RLRoleType.ROLLOUT.name).per_group, 2
        )
        self.assertEqual(
            rl_job.get_workload(RLRoleType.ROLLOUT.name).envs, {"e2": "v2"}
        )
        self.assertEqual(
            rl_job.get_workload(RLRoleType.REWARD.name).entry_point,
            "m3.c3",
        )
        self.assertEqual(
            rl_job.get_workload(RLRoleType.REFERENCE.name).entry_point,
            "m4.c4",
        )
        self.assertEqual(
            rl_job.get_workload(RLRoleType.CRITIC.name).entry_point,
            "m5.c5",
        )
        self.assertEqual(len(rl_job.collocations), 1)
        self.assertTrue(RLRoleType.ACTOR.name in rl_job.collocations[0])
        self.assertTrue(RLRoleType.ROLLOUT.name in rl_job.collocations[0])

        self.assertEqual(len(rl_job.workloads), 6)

    def test_by_dlrover_run_cmd(self):
        root_dir = os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                )
            )
        )
        cmd = f"dlrover-run --nnodes=2 --nproc_per_node=2 --node_check {root_dir}/dlrover/python/unified/tests/integration_test/dummy_run.py --test 0"

        dl_job = DLJobBuilder().by_dlrover_run_cmd(cmd).build()

        for workload in dl_job.workloads.values():
            if workload.backend == "elastic":
                self.assertEqual(workload.comm_pre_check, True)

        self.assertEqual(dl_job.node_num, 2)
        self.assertEqual(dl_job.device_per_node, 2)
        workload = dl_job.workloads["ELASTIC"]
        self.assertEqual(
            workload.entry_point,
            f"{root_dir}/dlrover/python/unified/tests/integration_test/dummy_run.py --test 0",
        )
        self.assertEqual(workload.total, 4)  # nnodes * nproc_per_node

        # test unspported cases
        with self.assertRaises(ValueError):
            DLJobBuilder().by_dlrover_run_cmd(
                "unsupported-run --nnodes=1 train.py"
            )

    def test_by_torchrun_cmd(self):
        root_dir = os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                )
            )
        )
        cmd = f"torchrun --nnodes=2 --nproc_per_node=2  {root_dir}/dlrover/python/unified/tests/integration_test/dummy_run.py --test 0"

        dl_job = DLJobBuilder().by_dlrover_run_cmd(cmd).build()

        self.assertEqual(dl_job.node_num, 2)
        self.assertEqual(dl_job.device_per_node, 2)
        workload = dl_job.workloads["ELASTIC"]
        self.assertEqual(
            workload.entry_point,
            f"{root_dir}/dlrover/python/unified/tests/integration_test/dummy_run.py --test 0",
        )
        self.assertEqual(workload.total, 4)  # nnodes * nproc_per_node

    def test_extra_flag(self):
        job = (
            DLJobBuilder()
            .train("a.b")
            .end()
            .role("r2")
            .run("a.b")
            .end()
            .no_setup_process_group()
            .skip_node_check()
            .build()
        )
        assert all(
            it.backend != "elastic" or it.comm_pre_check is False
            for it in job.workloads.values()
        )
        assert all(
            it.backend != "elastic"
            or it.comm_auto_setup_process_group is False
            for it in job.workloads.values()
        )

    def test_validation(self):
        # without dl type
        with self.assertRaises(ValidationError):
            DLJobBuilder().build()

        # without node num
        with self.assertRaises(ValidationError):
            DLJobBuilder().build()

        # without device per node
        with self.assertRaises(ValidationError):
            DLJobBuilder().node_num(1).build()

        # invalid device type
        with self.assertRaises(ValidationError):
            DLJobBuilder().node_num(1).device_per_node(1).device_type(
                "mem"
            ).build()

        # without config
        with self.assertRaises(ValidationError):
            DLJobBuilder().node_num(1).device_per_node(1).build()

        # rl without trainer
        with self.assertRaises(ValidationError):
            RLJobBuilder().node_num(1).device_per_node(1).config(
                {"k1": "v1"}
            ).build()

        # rl without actor
        with self.assertRaises(ValidationError):
            RLJobBuilder().node_num(1).device_per_node(1).config(
                {"k1": "v1"}
            ).trainer("m0.c0").end().build()

        # invalid trainer
        with self.assertRaises(ValidationError):
            RLJobBuilder().node_num(1).device_per_node(1).config(
                {"k1": "v1"}
            ).trainer("").end().build()

        # a minimum valid rl
        (
            RLJobBuilder()
            .node_num(1)
            .device_per_node(1)
            .config({"k1": "v1"})
            .trainer("m0.c0")
            .end()
            .actor("m1.c1")
            .total(1)
            .per_group(1)
            .end()
            .build()
        )

    def test_invalid_collocation(self):
        # invalid collocation
        with self.assertRaises(ValidationError):
            (
                RLJobBuilder()
                .node_num(1)
                .device_per_node(1)
                .config({"k1": "v1"})
                .trainer("m0.c0")
                .end()
                .actor("m1.c1")
                .end()
                .rollout("m2.c2")
                .end()
                .reward("m3.c3")
                .end()
                .with_collocation(
                    RLRoleType.ACTOR.name, RLRoleType.ROLLOUT.name
                )
                .with_collocation(
                    RLRoleType.ROLLOUT.name, RLRoleType.REWARD.name
                )
                .build()
            )
        # not defined role
        with self.assertRaises(ValidationError) as e:
            (
                RLJobBuilder()
                .trainer("m0.c0")
                .end()
                .actor("m1.c1")
                .end()
                .with_collocation(
                    RLRoleType.ACTOR.name, RLRoleType.ROLLOUT.name
                )
                .build()
            )
        self.assertIn("not defined", str(e.exception))
        # collocation twice
        with self.assertRaises(ValidationError) as e:
            (
                RLJobBuilder()
                .trainer("m0.c0")
                .end()
                .actor("m1.c1")
                .end()
                .rollout("m1.c1")
                .end()
                .reward("m1.c1")
                .end()
                .with_collocation(
                    RLRoleType.ACTOR.name, RLRoleType.ROLLOUT.name
                )
                .with_collocation(
                    RLRoleType.ACTOR.name, RLRoleType.REWARD.name
                )
                .build()
            )
        self.assertIn("already been assigned", str(e.exception))

    def test_collocation_all(self):
        rl_job: DLJob = (
            RLJobBuilder()
            .node_num(1)
            .device_per_node(2)
            .config({"k1": "v1"})
            .trainer("m0.c0")
            .end()
            .actor("m1.c1")
            .total(1)
            .per_group(1)
            .end()
            .rollout("m2.c2")
            .total(1)
            .per_group(1)
            .end()
            .with_collocation_all()
            .build()
        )
        self.assertEqual(
            rl_job.collocations[0],
            {RLRoleType.ACTOR.name, RLRoleType.ROLLOUT.name},
        )

    def test_to_dl_config(self):
        # collocation: 4: 4+4
        rl_job: DLJob = (
            RLJobBuilder()
            .node_num(2)
            .device_per_node(4)
            .config({"k1": "v1"})
            .global_env({"e0": "v0"})
            .trainer("m0.c0")
            .end()
            .actor("m1.c1")
            .total(4)
            .per_group(4)
            .env({"e1": "v1"})
            .end()
            .rollout("m2.c2")
            .total(4)
            .per_group(4)
            .env({"e2": "v2"})
            .end()
            .with_collocation(RLRoleType.ACTOR.name, RLRoleType.ROLLOUT.name)
            .build()
        )
        assert isinstance(rl_job, DLConfig)
        assert len(rl_job.workloads.keys()) == 3
        assert (
            rl_job.workloads[RLRoleType.ACTOR.name].group
            == rl_job.workloads[RLRoleType.ROLLOUT.name].group
        )
        assert rl_job.workloads[RLRoleType.ACTOR.name].per_group == 4
        assert rl_job.workloads[RLRoleType.ROLLOUT.name].per_group == 4
        assert rl_job.workloads[RLRoleType.ACTOR.name].total == 4
        assert (
            rl_job.workloads[RLRoleType.ACTOR.name].resource.accelerator == 0.5
        )
        assert rl_job.workloads[RLRoleType.ROLLOUT.name].total == 4
        assert (
            rl_job.workloads[RLRoleType.ROLLOUT.name].resource.accelerator
            == 0.5
        )
        # collocation: 4: 4+4+4
        rl_job = (
            RLJobBuilder()
            .node_num(2)
            .device_per_node(4)
            .config({"k1": "v1"})
            .global_env({"e0": "v0"})
            .trainer("m0.c0")
            .end()
            .actor("m1.c1")
            .total(4)
            .per_group(4)
            .env({"e1": "v1"})
            .end()
            .rollout("m2.c2")
            .total(4)
            .per_group(4)
            .env({"e2": "v2"})
            .end()
            .reference("m3.c3")
            .total(4)
            .per_group(4)
            .end()
            .with_collocation(
                RLRoleType.ACTOR.name,
                RLRoleType.ROLLOUT.name,
                RLRoleType.REFERENCE.name,
            )
            .build()
        )
        assert isinstance(rl_job, DLConfig)
        assert len(rl_job.workloads.keys()) == 4
        assert (
            rl_job.workloads[RLRoleType.ACTOR.name].group
            == rl_job.workloads[RLRoleType.ROLLOUT.name].group
            == rl_job.workloads[RLRoleType.REFERENCE.name].group
        )
        assert rl_job.workloads[RLRoleType.ACTOR.name].per_group == 4
        assert rl_job.workloads[RLRoleType.ROLLOUT.name].per_group == 4
        assert rl_job.workloads[RLRoleType.REFERENCE.name].per_group == 4

        assert rl_job.workloads[RLRoleType.ACTOR.name].total == 4
        assert (
            rl_job.workloads[RLRoleType.ACTOR.name].resource.accelerator
            == 0.33
        )
        assert rl_job.workloads[RLRoleType.ROLLOUT.name].total == 4
        assert (
            rl_job.workloads[RLRoleType.ROLLOUT.name].resource.accelerator
            == 0.33
        )
        assert rl_job.workloads[RLRoleType.REFERENCE.name].total == 4
        assert (
            rl_job.workloads[RLRoleType.REFERENCE.name].resource.accelerator
            == 0.33
        )

        # collocation: 4: 4+4 2+2
        rl_job = (
            RLJobBuilder()
            .node_num(2)
            .device_per_node(6)
            .config({"k1": "v1"})
            .global_env({"e0": "v0"})
            .trainer("m0.c0")
            .end()
            .actor("m1.c1")
            .total(4)
            .per_group(2)
            .env({"e1": "v1"})
            .end()
            .rollout("m2.c2")
            .total(4)
            .per_group(2)
            .env({"e2": "v2"})
            .end()
            .reference("m3.c3")
            .total(12)
            .per_group(6)
            .end()
            .reward("m4.c4")
            .total(4)
            .per_group(2)
            .end()
            .with_collocation(RLRoleType.ACTOR.name, RLRoleType.ROLLOUT.name)
            .with_collocation(
                RLRoleType.REWARD.name, RLRoleType.REFERENCE.name
            )
            .build()
        )
        assert isinstance(rl_job, DLConfig)
        assert len(rl_job.workloads.keys()) == 5
        assert (
            rl_job.workloads[RLRoleType.ACTOR.name].group
            == rl_job.workloads[RLRoleType.ROLLOUT.name].group
        )
        assert (
            rl_job.workloads[RLRoleType.REWARD.name].group
            == rl_job.workloads[RLRoleType.REFERENCE.name].group
        )
        assert rl_job.workloads[RLRoleType.ACTOR.name].per_group == 2
        assert rl_job.workloads[RLRoleType.ROLLOUT.name].per_group == 2
        assert rl_job.workloads[RLRoleType.REWARD.name].per_group == 2
        assert rl_job.workloads[RLRoleType.REFERENCE.name].per_group == 6

        assert rl_job.workloads[RLRoleType.ACTOR.name].total == 4
        assert (
            rl_job.workloads[RLRoleType.ACTOR.name].resource.accelerator == 0.5
        )
        assert rl_job.workloads[RLRoleType.ROLLOUT.name].total == 4
        assert (
            rl_job.workloads[RLRoleType.ROLLOUT.name].resource.accelerator
            == 0.5
        )
        assert rl_job.workloads[RLRoleType.REFERENCE.name].total == 12
        assert (
            rl_job.workloads[RLRoleType.REFERENCE.name].resource.accelerator
            == 0.5
        )
        assert rl_job.workloads[RLRoleType.REWARD.name].total == 4
        assert (
            rl_job.workloads[RLRoleType.REWARD.name].resource.accelerator
            == 0.5
        )

    def test_disable_ray_auto_visible_device(self):
        rl_job = (
            RLJobBuilder()
            .node_num(2)
            .device_per_node(4)
            .config({"k1": "v1"})
            .global_env({"e0": "v0"})
            .trainer("m0.c0")
            .end()
            .actor("m1.c1")
            .total(4)
            .per_group(4)
            .env({"e1": "v1"})
            .end()
            .rollout("m2.c2")
            .total(4)
            .disable_ray_auto_visible_device()
            .env({"e2": "v2"})
            .per_group(4)
            .end()
            .with_collocation("Actor", RLRoleType.ROLLOUT.name)
            .build()
        )
        assert isinstance(rl_job, DLConfig)
        assert (
            rl_job.workloads[RLRoleType.ROLLOUT.name].envs[
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"
            ]
            == "true"
        )

    def test_elastic(self):
        dl_job = (
            DLJobBuilder()
            .node_num(2)
            .device_per_node(2)
            .device_type("CPU")
            .config({"c1": "v1"})
            .global_env({"e0": "v0"})
            .train("test::main")
            .nnodes(2)
            .nproc_per_node(2)
            .end()
            .build()
        )

        self.assertIsNotNone(dl_job)
        assert isinstance(dl_job, DLConfig)
        assert len(dl_job.workloads) == 1
        assert "ELASTIC" in dl_job.workloads
        workload = dl_job.workloads["ELASTIC"]
        assert workload.backend == "elastic"
        assert workload.entry_point == "test.main"
        assert workload.total == 4
        assert workload.resource.accelerator == 1

        with pytest.raises(ValidationError):
            (
                DLJobBuilder()
                .node_num(2)
                .device_per_node(2)
                .device_type("CPU")
                .config({"c1": "v1"})
                .global_env({"e0": "v0"})
                .train("test")
                .nnodes(2)
                .nproc_per_node(2)
                .end()
                .build()
            )

    def test_role_builder(self):
        workload_builder = DLJobBuilder().role("test").run("m0.c0")
        workload_builder.env(None)
        self.assertEqual(workload_builder._env, {})
        workload_builder.sub_stage(None)
        self.assertEqual(workload_builder._sub_stage, [])
        workload_builder.sub_stage([1])
        self.assertEqual(workload_builder._sub_stage, [1])
        workload_builder = DLJobBuilder().workload("test", "m0.c0")
        workload_builder.end().build()  # success

        # Bad entrypoint
        with self.assertRaises(ValidationError):
            DLJobBuilder().role("test").run("main").end().build()

        # Normalize old format
        res = (
            DLJobBuilder().role("test").run("tset::main").end().build()
        )  # success
        assert res.workloads["test"].entry_point == "tset.main"

        # Bad total
        with self.assertRaises(ValidationError):
            (
                DLJobBuilder()
                .role("test")
                .run("tset.main")
                .total(-1)
                .end()
                .build()
            )

        DLJobBuilder().role("test").run("tset.main").total(
            1
        ).build()  # success


def test_submit(monkeypatch):
    # mock submit, just return the config
    dl_job = (
        DLJobBuilder()
        .node_num(1)
        .device_per_node(1)
        .config({"c1": "v1"})
        .global_env({"e0": "v0"})
        .train("test::main")
        .nnodes(1)
        .nproc_per_node(1)
        .end()
        .build()
    )

    with patch(
        "dlrover.python.unified.api.builder.base.submit"
    ) as mock_submit:
        ret = dl_job.submit()
        assert ret is mock_submit.return_value
        assert mock_submit.called
        job = mock_submit.call_args.args[0]
        assert isinstance(job, JobConfig)
        assert job.job_name.startswith("dlrover-")

    monkeypatch.setenv("DLROVER_UNIFIED_JOB_NAME", "test_env_name")
    with patch(
        "dlrover.python.unified.api.builder.base.submit"
    ) as mock_submit:
        ret = dl_job.submit()
        job = mock_submit.call_args.args[0]
        assert isinstance(job, JobConfig)
        assert job.job_name == "test_env_name"

    with patch(
        "dlrover.python.unified.api.builder.base.submit"
    ) as mock_submit:
        ret = dl_job.submit("set_name", master_cpu=8)
        job = mock_submit.call_args.args[0]
        assert isinstance(job, JobConfig)
        assert job.job_name == "set_name"
        assert job.master_cpu == 8
