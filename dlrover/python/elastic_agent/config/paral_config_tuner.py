# Copyright 2022 The DLRover Authors. All rights reserved.
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
import threading
import time

from dlrover.python.common.constants import ConfigPath
from dlrover.python.common.grpc import ParallelConfig
from dlrover.python.common.singleton import singleton
from dlrover.python.elastic_agent.master_client import GlobalMasterClient


@singleton
class ParalConfigTuner(object):
    def __init__(self):
        """
        Parallelism config tuner for updating parallelism config file.
        """
        self._master_client = GlobalMasterClient.MASTER_CLIENT
        self.config_dir = os.path.dirname(
            os.environ[ConfigPath.ENV_PARAL_CONFIG]
        )
        self.config_path = os.environ[ConfigPath.ENV_PARAL_CONFIG]

    def start(self):
        threading.Thread(
            target=self._periodically_update_paral_config,
            name="config-updater",
            daemon=True,
        ).start()

    def _periodically_update_paral_config(self):
        """
        Updates the parallelism configuration every 30 seconds. This method is
        intended to run on a separate thread started by `self.start`.
        """
        while True:
            config: ParallelConfig = self._master_client.get_paral_config()
            with open(self.config_path, "w") as f:
                f.write(config.to_json())
            time.sleep(30)

    def _read_paral_config(self, config_path):
        """
        Read the parallelism configuration from a JSON file.
        """
        try:
            with open(config_path, "r") as json_file:
                self.config = json.load(json_file)
            return self.config
        except FileNotFoundError:
            print(f"Error: Config file '{config_path}' not found.")
            return None
        except json.JSONDecodeError as e:
            print(f"Error: Failed to decode JSON file '{config_path}': {e}")
            return None
