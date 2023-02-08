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

from dlrover.client.platform.ray.ray_job_submitter import (
    create_scheduler,
    load_conf,
)


class RayJobSubmitter(unittest.TestCase):
    def setUp(self):
        os.system("ray stop")
        r = os.system("ray start --head --port=5001  --dashboard-port=5000")
        self.assertEqual(r, 0)

    def test_parse_conf(self):
        current_directory = os.path.dirname(os.path.abspath(__file__))
        conf_path = os.path.join(current_directory, "demo.yaml")
        all_data = load_conf(conf_path)
        self.assertEqual(all_data["dashboardUrl"], "localhost:5000")

    def test_submit_job(self):
        current_directory = os.path.dirname(os.path.abspath(__file__))
        conf_path = os.path.join(current_directory, "demo.yaml")
        submitter = create_scheduler(conf_path)
        job_id = submitter.submit()
        self.assertNotEqual(job_id, None)
        submitter.wait_until_finish(job_id)

    def tearDown(self):
        r = os.system("ray stop")
        self.assertEqual(r, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
