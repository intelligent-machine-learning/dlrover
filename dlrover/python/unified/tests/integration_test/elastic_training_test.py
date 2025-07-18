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


from dlrover.python.unified.api.base import DLJobBuilder
from dlrover.python.unified.controller.master import PrimeMaster
from dlrover.python.unified.tests.fixtures.example_jobs import (
    elastic_training_job,
)
from dlrover.python.util.function_util import timeout


@timeout(30)
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


@timeout(30)
def test_elastic_training_api_full(tmp_ray):
    dl_job = (
        DLJobBuilder()
        .SFT_type()
        .node_num(2)
        .device_per_node(2)
        .device_type("CPU")
        .config({"c1": "v1"})
        .global_env({"e0": "v0", "DLROVER_LOG_LEVEL": "DEBUG"})
        .dlrover_run(
            "dlrover.python.unified.tests.test_class::elastic_workload_run",
            nnodes=2,
            nproc_per_node=2,
        )
        .build()
    )

    ret = dl_job.submit("test", master_cpu=1, master_memory=128)
    assert ret == 0, "Job should succeed"


@timeout(30)
def test_elastic_training_with_error(tmp_ray):
    dl_job = (
        DLJobBuilder()
        .SFT_type()
        .node_num(3)
        .device_per_node(2)
        .device_type("CPU")
        .config({"c1": "v1"})
        .global_env({"e0": "v0", "DLROVER_LOG_LEVEL": "DEBUG"})
        .dlrover_run(
            "dlrover.python.unified.tests.test_class::elastic_workload_run_error",
            nnodes=2,
            nproc_per_node=2,
        )
        .build()
    )

    ret = dl_job.submit(
        "test",
        master_cpu=1,
        master_memory=128,
        workload_max_restart={"ELASTIC": 1},
    )
    assert ret != 0, "Job should fail due to error in workload"


# TODO abnormal test cases
