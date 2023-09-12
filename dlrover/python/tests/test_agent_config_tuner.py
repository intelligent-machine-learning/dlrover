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

import json
import os
import unittest

from dlrover.python.elastic_agent.config.paral_config_tuner import (
    ParalConfigTuner,
)
from dlrover.python.common.constants import ConfigPath

MOCKED_CONFIG = {
    "dataloader": {
        "batch_size": 3,
        "dataloader_name": 2,
        "num_workers": 4,
        "pin_memory": 0,
        "version": 1,
    },
    "optimizer": {"learning_rate": 0.0, "optimizer_name": 6, "version": 5},
}


class TestParalConfigTuner(unittest.TestCase):
    def setUp(self):
        self.tuner = ParalConfigTuner()

    def test_set_paral_config(self):
        self.tuner._set_paral_config()
        self.assertTrue(os.path.exists(self.tuner.config_dir))
        self.assertEqual(
            os.environ[ConfigPath.ENV_PARAL_CONFIG],
            ConfigPath.PARAL_CONFIG,
        )

    def test_read_paral_config(self):
        with open(self.tuner.config_path, "w") as json_file:
            json.dump(MOCKED_CONFIG, json_file)
        config = self.tuner._read_paral_config(self.tuner.config_path)
        self.assertEqual(config, MOCKED_CONFIG)

    def test_read_paral_config_file_not_found(self):
        os.remove(self.tuner.config_path)
        config = self.tuner._read_paral_config(self.tuner.config_path)
        self.assertIsNone(config)


if __name__ == "__main__":
    unittest.main()
