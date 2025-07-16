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

import time

import pytest

from dlrover.python.unified.controller.master import PrimeMaster
from dlrover.python.unified.tests.fixtures.example_jobs import (
    elastic_training_job,
)


@pytest.mark.usefixtures("tmp_ray")
def test_elastic_training():
    job = elastic_training_job()
    master = PrimeMaster.create(job)
    assert master.get_status().stage == "INIT"
    master.start()
    assert master.get_status().stage == "RUNNING"
    while master.get_status().stage != "STOPPED":
        time.sleep(1)
    master.stop()  # Noop
    assert master.get_status().stage == "STOPPED"
    master.shutdown()


# TODO abnormal test cases
