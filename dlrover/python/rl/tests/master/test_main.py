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

import ray

from dlrover.python.rl.common.args import parse_job_args
from dlrover.python.rl.common.rl_context import RLContext
from dlrover.python.rl.master.main import DLRoverRLMaster
from dlrover.python.rl.tests.master.base import BaseMasterTest
from dlrover.python.rl.tests.test_data import TestData
from dlrover.python.util.function_util import timeout


class RLMasterTest(BaseMasterTest):
    def setUp(self):
        super().setUp()
        args = [
            "--job_name",
            "test",
            "--rl_config",
            f"{TestData.UD_SIMPLE_TEST_WITH_INTERACTIVE_RL_CONF}",
        ]
        parsed_args = parse_job_args(args)
        rl_context = RLContext.build_from_args(parsed_args)
        self._job_context._rl_context = rl_context
        ray.init()

    def tearDown(self):
        super().tearDown()
        ray.shutdown()

    @timeout(90)
    def test_main(self):
        master_name = "test"

        master_actor = DLRoverRLMaster.options(
            name=master_name, lifetime="detached"
        ).remote(
            self._job_context.job_config.serialize(),
            self._job_context.rl_context.serialize(),
        )

        ray.get(master_actor.ping.remote())
        master_actor.run.remote()
        time.sleep(5)

        # wait master done
        while True:
            try:
                ready, unready = ray.wait(
                    [master_actor.ping.remote()], timeout=1
                )
                if len(ready) > 0:
                    time.sleep(1)
                else:
                    break
            except Exception as e:
                print(e)
                break
