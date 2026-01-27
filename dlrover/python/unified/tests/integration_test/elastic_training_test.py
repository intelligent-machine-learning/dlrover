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
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from dlrover.python.unified.api.builder import DLJobBuilder
from dlrover.python.unified.controller.master import PrimeMaster
from dlrover.python.unified.tests.fixtures._ray_setup_hooks import inject_hook
from dlrover.python.unified.tests.fixtures.example_jobs import (
    elastic_training_job,
)


def elastic_workload_run():
    time.sleep(0.1)
    print("elastic_workload_run run called")


def elastic_workload_run_error():
    raise Exception("elastic_workload_run_error run failed")


@pytest.mark.timeout(40, func_only=True)  # 20s in ci
def test_elastic_training(tmp_ray):
    job = elastic_training_job()
    master = PrimeMaster.create(job)
    assert master.get_status().stage == "INIT"
    master.start()
    assert master.get_status().stage == "RUNNING"
    master.wait()
    master.stop()  # Noop
    assert master.get_status().stage == "STOPPED"
    master.shutdown()


@pytest.mark.timeout(40, func_only=True)  # 25s in ci
def test_api_full(tmp_ray):
    dl_job = (
        DLJobBuilder()
        .node_num(2)
        .device_per_node(2)
        .device_type("CPU")
        .config({"c1": "v1"})
        .global_env({"e0": "v0", "DLROVER_LOG_LEVEL": "DEBUG"})
        .train(f"{__name__}.elastic_workload_run")
        .nnodes(2)
        .nproc_per_node(2)
        .end()
        .build()
    )

    ret = dl_job.submit("test", master_cpu=1, master_memory=128)
    assert ret == 0, "Job should succeed"


@pytest.mark.timeout(40, func_only=True)
def test_api_full_by_dlrover_run_cmd(tmp_ray):
    root_dir = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            )
        )
    )
    cmd = f"dlrover-run --nnodes=2 --nproc_per_node=2 {root_dir}/dlrover/python/unified/tests/integration_test/dummy_run.py --test 0"

    dl_job = DLJobBuilder().by_dlrover_run_cmd(cmd).build()

    ret = dl_job.submit("test_cmd_api", master_cpu=1, master_memory=128)
    assert ret == 0, "Job submitted via by_dlrover_run_cmd should succeed"


@pytest.mark.timeout(40, func_only=True)
def test_api_full_by_torchrun_cmd(tmp_ray):
    root_dir = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            )
        )
    )
    cmd = f"torchrun --nnodes=2 --nproc_per_node=2 {root_dir}/dlrover/python/unified/tests/integration_test/dummy_run.py --test 0"

    dl_job = DLJobBuilder().by_dlrover_run_cmd(cmd).build()

    ret = dl_job.submit("test_cmd_api", master_cpu=1, master_memory=128)
    assert ret == 0, "Job submitted via by_dlrover_run_cmd should succeed"


@pytest.mark.timeout(40, func_only=True)  # 25s in ci
def test_api_full_with_cmd(tmp_ray):
    root_dir = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            )
        )
    )
    dl_job = (
        DLJobBuilder()
        .node_num(2)
        .device_per_node(2)
        .device_type("CPU")
        .config({"c1": "v1"})
        .global_env({"e0": "v0", "DLROVER_LOG_LEVEL": "DEBUG"})
        .train(
            f"{root_dir}/dlrover/python/unified/tests/integration_test/dummy_run.py --test 0"
        )
        .nnodes(2)
        .nproc_per_node(2)
        .end()
        .build()
    )

    ret = dl_job.submit("test", master_cpu=1, master_memory=128)
    assert ret == 0, "Job should succeed"


@pytest.mark.timeout(40, func_only=True)  # 25s in ci
def test_api_full_with_error_with_no_failover(tmp_ray):
    dl_job = (
        DLJobBuilder()
        .node_num(3)
        .device_per_node(2)
        .device_type("CPU")
        .config({"c1": "v1"})
        .train(f"{__name__}.elastic_workload_run_error")
        .nproc_per_node(2)
        .end()
        .build()
    )

    ret = dl_job.submit(
        "test",
        master_cpu=1,
        master_memory=128,
        failover_exec_strategy=0,
    )
    assert ret != 0, "Job should fail due to error in workload"


@pytest.mark.timeout(40, func_only=True)  # 25s in ci
def test_api_full_with_error_with_job_failover(tmp_ray):
    dl_job = (
        DLJobBuilder()
        .node_num(3)
        .device_per_node(2)
        .device_type("CPU")
        .config({"c1": "v1"})
        .train(f"{__name__}.elastic_workload_run_error")
        .nproc_per_node(2)
        .end()
        .build()
    )

    ret = dl_job.submit(
        "test",
        master_cpu=1,
        master_memory=128,
        job_max_restart=1,
    )
    assert ret != 0, "Job should fail due to error in workload"


@pytest.mark.timeout(40, func_only=True)  # 25s in ci
def test_api_full_with_error_with_role_failover(tmp_ray):
    dl_job = (
        DLJobBuilder()
        .node_num(3)
        .device_per_node(2)
        .device_type("CPU")
        .config({"c1": "v1"})
        .train(f"{__name__}.elastic_workload_run_error")
        .nproc_per_node(2)
        .end()
        .build()
    )

    ret = dl_job.submit(
        "test",
        master_cpu=1,
        master_memory=128,
        job_max_restart=1,
        failover_exec_strategy=2,
    )
    assert ret != 0, "Job should fail due to error in workload"


def _inject_comm_fault():
    import ray

    from dlrover.python.unified.backend.elastic.worker.node_check import (
        run_comm_check as raw_comm_check,
    )

    def run_comm_check():
        """
        Mocked function to simulate communication check failure.
        """
        if ray.get_runtime_context().was_current_actor_reconstructed:
            patcher.stop()
            return raw_comm_check()
        res_file = Path(os.environ["TMP_PATH"]) / "comm_fault"
        res_file.write_text("triggered")
        raise RuntimeError("Mocked comm_check_error")

    patcher = patch(
        "dlrover.python.unified.backend.elastic.worker.node_check.run_comm_check",
        side_effect=run_comm_check,
    )
    patcher.start()


# __module__ is not accurate in pytest, so we set it manually
MODULE_NAME = (
    "dlrover.python.unified.tests.integration_test.elastic_training_test"
)


@pytest.mark.timeout(40, func_only=True)  # 20s in ci
def test_comm_fault(tmp_ray, tmp_path: Path):
    job = elastic_training_job()
    job.dl_config.workloads["training"].envs.update(
        {
            **inject_hook(f"{MODULE_NAME}._inject_comm_fault"),
            "TMP_PATH": tmp_path.as_posix(),
        }
    )
    master = PrimeMaster.create(job)
    assert master.get_status().stage == "INIT"
    master.start()
    assert (tmp_path / "comm_fault").exists(), (
        "Subprocess triggered comm fault"
    )
    master.wait()
    assert master.get_status().stage == "STOPPED"
    assert master.get_status().exit_code == 0, "Success expected"


def _mock_node_crash_when_training():
    import time

    import ray

    time.sleep(1)
    if ray.get_runtime_context().was_current_actor_reconstructed:
        return

    res_file = Path(os.environ["TMP_PATH"]) / "training_crash"
    res_file.write_text("triggered")
    ray.kill(ray.get_runtime_context().current_actor, no_restart=False)


@pytest.mark.timeout(60, func_only=True)  # 26s in ci
def test_failover_training(tmp_ray, tmp_path: Path):
    job = elastic_training_job()
    workload = job.dl_config.workloads["training"]
    assert workload.backend == "elastic"
    workload.comm_pre_check = False  # Make test faster
    workload.envs["TMP_PATH"] = tmp_path.as_posix()
    workload.entry_point = f"{MODULE_NAME}._mock_node_crash_when_training"

    job.failover_trigger_strategy = 2  # Enable failover on failure
    job.failover_exec_strategy = 1  # Enable job-level failover
    job.job_max_restart = 3  # Allow max 3 restarts

    master = PrimeMaster.create(job)
    assert master.get_status().stage == "INIT"
    master.start()
    assert master.get_status().stage == "RUNNING"
    master.wait()
    assert master.get_status().stage == "STOPPED"

    assert (tmp_path / "training_crash").exists(), (
        "Subprocess triggered training crash"
    )
    assert master.get_status().exit_code == 0, "Success expected"


@pytest.mark.timeout(60)
def test_failover_entire_job(tmp_ray):
    job = elastic_training_job()
    workload = job.dl_config.workloads["training"]
    assert workload.backend == "elastic"
    workload.comm_pre_check = False  # Make test faster

    master = PrimeMaster.create(job)
    assert master.get_status().stage == "INIT"
    master.start()
    assert master.get_status().stage == "RUNNING"
    master.restart()

    master.wait()
    assert master.get_status().stage == "STOPPED"

    assert master.get_status().job_restart_count == 1
    assert master.get_status().exit_code == 0, "Success expected"


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "tmp_ray", [{"resources": {"MR": 999, "WR": 999}}], indirect=True
)
def test_master_worker_isolation_scheduling(tmp_ray, monkeypatch):
    monkeypatch.setenv(
        "DLROVER_UNIFIED_MASTER_ISOLATION_SCHEDULE_RESOURCE", "MR"
    )
    monkeypatch.setenv(
        "DLROVER_UNIFIED_WORKER_ISOLATION_SCHEDULE_RESOURCE", "WR"
    )

    dl_job = (
        DLJobBuilder()
        .node_num(2)
        .device_per_node(2)
        .device_type("CPU")
        .config({"c1": "v1"})
        .train(f"{__name__}.elastic_workload_run")
        .nnodes(2)
        .nproc_per_node(2)
        .end()
        .build()
    )

    ret = dl_job.submit("test", master_cpu=1, master_memory=128)
    assert ret == 0, "Job should succeed"
