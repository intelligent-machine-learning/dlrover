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


class Checkpointer(metaclass=ABCMeta):
    """
    Checkpointer can save and load PyTorch module states efficiently.

    It begins by duplicating the state dictionary to shared memory,
    then proceeds to save it to storage asynchronously. This process ensures
    that the Checkpointer's save operation minimally blocks training time.
    If the node does not fail, the Checkpointer prioritizes restoring the
    checkpointed state dictionary directly from the shared memory upon load
    requests. However, if the node has restarted, the Checkpointer reverts
    to loading the state dictionary from the designated storage instead.
    """

    @abstractmethod
    def save_checkpoint(
        self, step, state_dict, path, storage_type=StorageType.DISK
    ):
        """
        Save the checkpoint of model, optimizer and sampler.

        Args:
            step (int): the global iteration step.
            state_dict (dict): the state dict of model and optimizer to save.
            path (str): the storage path to save the state dict.
                Note, the path is used to save the state dict to storage
                only if the training process fails.
            storage_type (StorageType): StorageType.MEMORY or StorageType.DISK.
        """
        pass

    @abstractmethod
    def load_checkpoint(self, resuming_path=None):
        """
        The manager loads the states from the files in the
        checkpoint directory to the model, optimizer and sampler.
        Args:
            resuming_path (str, optional): The manager will load checkpoint
                from the path. If the path is None, the manager will load
                the state checkpoint from the file with the maximum step.
        Return:
            A dict: a state dict.
        """
        pass
