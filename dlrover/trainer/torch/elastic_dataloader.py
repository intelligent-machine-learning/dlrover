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
from torch.utils.data import DataLoader, BatchSampler


class ElasticDataLoader(DataLoader):
    def __init__(self, *args, config_file=None, **kwargs):
        super(ElasticDataLoader, self).__init__(*args, **kwargs)
        self.current_batch_size = self.batch_size
        self.config_file = config_file

        if self.config_file:
            self.load_config()

    def load_config(self):
        if self.config_file:
            with open(self.config_file, "r") as f:
                config = json.load(f)
                if "batch_size" in config:
                    self.set_batch_size(config["batch_size"])

    def __iter__(self):
        batch_sampler = BatchSampler(
            self.sampler, batch_size=self.current_batch_size, drop_last=False
        )
        for batch_indices in batch_sampler:
            yield self.collate_fn([self.dataset[i] for i in batch_indices])

    def set_batch_size(self, batch_size):
        self.current_batch_size = batch_size
