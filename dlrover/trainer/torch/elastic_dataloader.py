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

from torch.utils.data import BatchSampler, DataLoader


class ElasticDataLoader(DataLoader):
    """
    A DataLoader class based on PyTorch's DataLoader that allows dynamic
    adjustment of batch size and optionally loads configuration settings from a
    file.

    This DataLoader inherits from PyTorch's DataLoader and extends its
    functionality by enabling the user to change the batch size during runtime.
    Additionally, it provides an option to load configuration settings from a
    JSON file to initialize the batch size.

    Args:
        constructor. config_file (str, optional): The path to a JSON
        configuration file that specifies
            the initial batch size (default: None).

    Attributes:
        current_batch_size (int): The current batch size used by the
        DataLoader. config_file (str): The path to the configuration file if
        provided.

    Methods:
        load_config(): Load the batch size configuration from the specified
        JSON file. set_batch_size(batch_size): Dynamically set the batch size.

    Usage Example:
        >>> # create a elastic dataloader with config.json
        >>> loader = ElasticDataLoader(dataset, shuffle=True,
        >>> config_file="config.json")
        >>> # Dynamically change the batch size to 64.
        >>> loader.set_batch_size(64)
        >>> for batch in loader:
        ...     # Training loop

    See Also:
        - PyTorch DataLoader:
          https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    """
    def __init__(self, *args, config_file=None, **kwargs):
        super(ElasticDataLoader, self).__init__(*args, **kwargs)
        self.current_batch_size = self.batch_size
        self.config_file = config_file

        if self.config_file:
            self.load_config(self.config_file)

    def load_config(self, config_file=None):
        """
        Load the batch size configuration from a JSON file specified by
        `config_file`.

        If the configuration file contains a 'batch_size' key, it will be used
        to set the initial batch size for the DataLoader.

        Note:
            This method is automatically called during DataLoader
            initialization if `config_file` is provided.
        """
        if not config_file:
            return
        with open(config_file, "r") as f:
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
        """
        Dynamically set the batch size to the specified value.

        Args:
            batch_size (int): The new batch size to be used.

        Example:
            >>> loader = ElasticDataLoader(dataset, batch_size=32)
            >>> loader.set_batch_size(64)  # Change batch size to 64.
        """
        self.current_batch_size = batch_size
