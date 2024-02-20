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
from dlrover.python.common.grpc import (
    DataLoaderConfig,
    OptimizerConfig,
    ParallelConfig,
)
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.singleton import Singleton
from dlrover.python.elastic_agent.master_client import MasterClient


class ParalConfigTuner(Singleton):
    def __init__(self):
        """
        Parallelism config tuner for updating parallelism config file.
        """
        self._master_client = MasterClient.singleton_instance()
        self.config_dir = os.path.dirname(
            os.environ[ConfigPath.ENV_PARAL_CONFIG]
        )
        self.config_path = os.environ[ConfigPath.ENV_PARAL_CONFIG]
        self._create_paral_config_file()
        self._started = False

    def start(self):
        if self._started:
            return
        threading.Thread(
            target=self._periodically_update_paral_config,
            name="config-updater",
            daemon=True,
        ).start()
        logger.info("Started parallelism config tuner.")
        self._started = True

    def _periodically_update_paral_config(self):
        """
        Updates the parallelism configuration every 30 seconds. This method is
        intended to run on a separate thread started by `self.start`.
        """
        while True:
            local_config = self._read_paral_config(self.config_path)
            if not self._master_client:
                break
            self._master_client.report_paral_config(local_config)
            time.sleep(30)
            config: ParallelConfig = self._master_client.get_paral_config()
            if config is not None:
                with open(self.config_path, "w") as f:
                    f.write(config.to_json())

    def _create_paral_config_file(self):
        """
        Create a parallelism configuration file.
        """
        config = ParallelConfig()
        with open(self.config_path, "w") as f:
            f.write(config.to_json())

    def _read_paral_config(self, config_path):
        """
        Read the parallelism configuration from a JSON file.
        """
        try:
            with open(config_path, "r") as json_file:
                content = json_file.read()
                json_data = json.loads(content)
            # if we find the instance_name in the JSON file, create it
            for key, value in json_data.items():
                if key == "dataloader":
                    dataloader = DataLoaderConfig(**(value))
                elif key == "optimizer":
                    optimizer = OptimizerConfig(**(value))
            self.config = ParallelConfig(
                dataloader=dataloader, optimizer=optimizer
            )
            return self.config
        except FileNotFoundError:
            print(f"Error: Config file '{config_path}' not found.")
            return None
        except json.JSONDecodeError as e:
            print(f"Error: Failed to decode JSON file '{config_path}': {e}")
            return None
