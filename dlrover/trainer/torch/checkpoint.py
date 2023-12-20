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

from abc import ABCMeta, abstractmethod
from enum import Enum, auto


class StorageType(Enum):
    MEMORY = auto()
    DISK = auto()


class CheckpointManger(metaclass=ABCMeta):
    """CheckpontManager can save and load checkpoint states.
    """

    @abstractmethod
    def save_checkpoint(self, *args, **kwargs):
        """
        Save the checkpoint of model, optimizer and sampler.
        """
        pass

    @abstractmethod
    def load_checkpoint(self, resuming_path=None):
        """
        The manager loads the states from the files in the
        checkpoint direcotry to the model, optimizer and sampler.

        Args:
            resuming_path (str, optinoal): The manager will load checkpoint
                from the path. If the path is None, the manager will load
                the state checkpoint from the file with the maximum step.

        Return:
            step (int): the iteration step.
            A dict: a state dict.
        """
        pass
