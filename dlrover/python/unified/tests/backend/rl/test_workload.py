#  Copyright 2025 The DLRover Authors. All rights reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

from dlrover.python.unified.backend.rl.worker import BaseRLWorker
from dlrover.python.unified.common.workload_base import ActorInfo, JobInfo
from dlrover.python.unified.common.workload_desc import CustomWorkloadDesc
from dlrover.python.unified.tests.base import BaseTest


class RLWorkerTest(BaseTest):
    # ActorBase.__init__ expects an event loop to be running.
    async def test_basic(self):
        os.environ["NAME"] = "test"
        os.environ["ROLE"] = "ACTOR"
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"

        spec = CustomWorkloadDesc(module_name="test", class_name="test")
        workload = BaseRLWorker(
            JobInfo(name="test", job_id="test", user_config={}),
            ActorInfo(name="test", role="ACTOR", spec=spec),
        )
        self.assertIsNotNone(workload)
        self.assertEqual(workload.name, "test")
        self.assertEqual(workload.local_rank, 0)

        workload.setup_rl_workload({"k2": "v2"})
        self.assertEqual(os.environ["k2"], "v2")
