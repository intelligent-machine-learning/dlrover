# Copyright 2026 The DLRover Authors. All rights reserved.
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
from unittest.mock import patch

from dlrover.python.common.constants import Accelerators
from dlrover.python.elastic_agent.torch.training import ElasticLaunchConfig


class MetaXGPUAutoConfigureTest(unittest.TestCase):
    @patch("torch.cuda.is_available")
    @patch("torch.cuda.get_device_name")
    def test_metaxgpu_auto_configure(
        self, mock_get_device_name, mock_is_available
    ):
        mock_get_device_name.return_value = "MetaX"
        mock_is_available.return_value = True
        config = ElasticLaunchConfig(
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=8,
            run_id="test",
            auto_config=True,
        )
        os.environ["NODE_NUM"] = "4"
        config.auto_configure_params()
        self.assertEqual(config.max_nodes, 4)
        self.assertEqual(config.min_nodes, 4)
        self.assertEqual(config.accelerator, Accelerators.METAX_GPU)


if __name__ == "__main__":
    unittest.main()
