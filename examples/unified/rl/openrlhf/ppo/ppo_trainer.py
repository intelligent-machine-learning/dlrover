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
#
# This package includes code from [https://github.com/OpenRLHF/OpenRLHF]
# licensed under the Apache License 2.0. See [https://github.com/OpenRLHF/
# OpenRLHF] for details.

from omegaconf import DictConfig

from dlrover.python.unified.api.runtime.worker import current_worker
from examples.unified.rl.openrlhf.ppo.trainer import BasePPOTrainer


class PPOTrainerActor(BasePPOTrainer):
    def __init__(self) -> None:
        config = DictConfig(current_worker().job_info.user_config)
        self.config = config
        super().__init__(config)

    def run(self):
        self.prepare_datasets()
        self.init_workers()
        self._init_wandb()

        self.fit()
