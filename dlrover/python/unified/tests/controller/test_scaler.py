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

from dlrover.python.unified.common.actor_base import NodeInfo
from dlrover.python.unified.controller.schedule.scaler import (
    DefaultRayNodeScaler,
)
from dlrover.python.unified.tests.fixtures.example_jobs import (
    elastic_training_job,
)


def test_default_scaler():
    scaler = DefaultRayNodeScaler(elastic_training_job())
    assert not scaler.relaunch([NodeInfo(id="node0")])
    assert not scaler.scale_up(count=2)
    assert not scaler.scale_down([NodeInfo(id="node0")])
