# Copyright 2023 The DLRover Authors. All rights reserved.
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
import unittest

from dlrover.client.platform.ray.ray_job_submitter import create_scheduler


class RayJobSubmitter(unittest.TestCase):
    def setUp(self) -> None:
        r = os.system("ray start --head --port=5001  --dashboard-port=5000")
        self.assertEqual(r, 0)

    def tearDown(self) -> None:
        r = os.system("ray stop")
        self.assertEqual(r, 0)

    def test_submit_job(self):
        submitter = create_scheduler()
        job_id = submitter.submit()
        self.assertNotEqual(job_id, None)
        submitter.wait_until_finish(job_id)