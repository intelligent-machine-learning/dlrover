# Copyright 2024 The DLRover Authors. All rights reserved.
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
import shutil
from abc import ABCMeta, abstractmethod
from typing import Callable, List

from .constants import CheckpointConstant
from .log import default_logger as logger
from .serialize import ClassMeta


class CheckpointStorage(metaclass=ABCMeta):
    """
    We can implement interfaces of CheckpointStorage to
    write/read the data.
    """

    @abstractmethod
    def write(self, content, path):
        """
        Write the content into the path of storage.

        Args:
            content (str/bytes): the content to write.
            path (str): the path of storage.
        """
        pass

    @abstractmethod
    def write_state_dict(self, state_dict, path, write_func):
        """
        Write the state dict of PyTorch into the path of storage.

        Args:
            state_dict (dict): a state dict.
            path (str): the path of storage.
        """
        pass

    @abstractmethod
    def read(self, path):
        """
        Read string from  the path.

        Args:
            path(str): the file path of storage.
        """
        pass

    @abstractmethod
    def read_state_dict(self, path, read_func):
        """
        Read state dict from the path.

        Args:
            path(str): the file path of storage.
        """
        pass

    @abstractmethod
    def safe_rmtree(self, dir):
        pass

    @abstractmethod
    def safe_remove(self, path):
        pass

    @abstractmethod
    def safe_makedirs(self, dir):
        pass

    @abstractmethod
    def safe_move(self, src_path, dst_path):
        pass

    @abstractmethod
    def commit(self, step: int, success: bool):
        """
        We can implement the method to commit the checkpoint step.

        Args:
            step (int): the iteration step.
            success (bool): whether to persist the checkpoint of step.
        """
        pass

    @abstractmethod
    def exists(self, path: str):
        """
        The method checks whether the path exists.

        Args:
            path (str): a path.
        """
        pass

    @abstractmethod
    def listdir(self, path: str):
        """
        The method list all objects in the path.

        Args:
            path (str): a path.
        """
        pass

    @abstractmethod
    def get_class_meta(self):
        """
        The method returns a ClassMeta instance which can be used to
        initialize a new instance of the class.
        """
        pass


class PosixDiskStorage(CheckpointStorage):
    def write(self, content, path):
        mode = "w"
        if isinstance(content, bytes) or isinstance(content, memoryview):
            mode = "wb"
        try:
            with open(path, mode) as stream:
                stream.write(content)
                os.fsync(stream.fileno())
        except OSError as e:
            logger.error(
                f"Failed to write file with path: {path}, " f"mode: {mode}"
            )
            raise e

    def write_state_dict(self, state_dict, path, write_func=None):
        dir = os.path.dirname(path)
        os.makedirs(dir, exist_ok=True)
        if write_func:
            write_func(state_dict, path)

    def read(self, path, mode="r"):
        if not os.path.exists(path):
            return ""
        with open(path, mode) as stream:
            content = stream.read()
        return content

    def read_state_dict(self, path, read_func):
        if not os.path.exists(path) or not read_func:
            return {}
        return read_func(path)

    def safe_rmtree(self, dir):
        if os.path.exists(dir):
            shutil.rmtree(dir, ignore_errors=True)

    def safe_remove(self, path):
        if os.path.exists(path):
            os.remove(path)

    def safe_makedirs(self, dir):
        os.makedirs(dir, exist_ok=True)

    def safe_move(self, src_path, dst_path):
        if os.path.exists(src_path) and not os.path.exists(dst_path):
            shutil.move(src_path, dst_path)

    def commit(self, step, success):
        logger.info(
            f"Succeed {success} in persisting the checkpoint of step {step}."
        )

    def exists(self, path: str):
        return os.path.exists(path)

    def listdir(self, path: str):
        return os.listdir(path)

    def get_class_meta(self):
        class_mata = ClassMeta(
            module_path=self.__class__.__module__,
            class_name=self.__class__.__name__,
        )
        return class_mata


class CheckpointDeletionStrategy(metaclass=ABCMeta):
    @abstractmethod
    def clean_up(self, step: int, delete_func: Callable):
        """
        Clean up the checkpoint of step.

        Arguments:
            step (int): the iteration step of a checkpoint.
            delete_func: A function to remove a directory, the argument
                is a directory of a folder.
        """
        pass


class KeepStepIntervalStrategy(CheckpointDeletionStrategy):
    """
    The strategy only keeps the step which is a multiple of the keep_interval
    and clean the other previous step after saving a new step checkpoint.

    Arguments:
        keep_interval (int): The step interval to keep. If the step is not
            a multiple of it, the strategy will clean up the step checkpoint
            after saving a new step checkpoint.
        checkpoint_dir (str): The common folder directory of checkpoint of
            all steps.
    """

    def __init__(self, keep_interval: int, checkpoint_dir: str):
        self._keep_interval = keep_interval
        self._checkpoint_dir = checkpoint_dir

    def clean_up(self, step, delete_func):
        if step % self._keep_interval == 0:
            return
        rm_dir = os.path.join(self._checkpoint_dir, str(step))
        try:
            logger.info(f"Clean path {rm_dir}")
            delete_func(rm_dir)
        except Exception:
            logger.warning(f"Fail to clean path {rm_dir}!")


class KeepLatestStepStrategy(CheckpointDeletionStrategy):
    """
    The strategy only remains the latest steps and delete the outdated
    checkpoints.
    Arguments:
        max_to_keep (int): An integer, the number of checkpoints to keep.
        checkpoint_dir (str): The common folder directory of checkpoint of
            all steps.
    """

    def __init__(self, max_to_keep: int, checkpoint_dir: str):
        self._max_to_keep = max(max_to_keep, 1)
        self._checkpoint_dir = checkpoint_dir
        self._steps: List[int] = []

    def clean_up(self, step, delete_func):
        self._steps.append(step)
        if len(self._steps) == self._max_to_keep:
            rm_step = self._steps.pop(0)
            rm_dir = os.path.join(self._checkpoint_dir, str(rm_step))
            try:
                logger.info(f"Clean path {rm_dir}")
                delete_func(rm_dir)
            except Exception:
                logger.warning(f"Fail to clean path {rm_dir}!")


class PosixStorageWithDeletion(PosixDiskStorage):
    """
    The storage will call a CheckpointDeletionStrategy to
    delete the outdated checkpoints.

    Arguments:
        tracker_file (str): the file name to store the latest checkpoint step.
        deletion_strategy (str): the strategy to clean outdated checkpoints.

    Example::
        from dlrover.python.common.storage import KeepStepIntervalStrategy
        from dlrover.trainer.torch.flash_checkpoint.ddp import DdpCheckpointer

        # Only keep the checkpoint
        keep_strategy = KeepStepIntervalStrategy(
            keep_interval=250,
            checkpoint_dir="./checkpoint/",
        )
        storage = PosixStorageWithDeletion(
            deletion_strategy=keep_strategy,
        )
        checkpointer = DdpCheckpointer(
            checkpoint_dir="./checkpoint/",
            storage=storage,
        )
    """

    def __init__(
        self, tracker_file: str, deletion_strategy: CheckpointDeletionStrategy
    ):
        super().__init__()
        self._deletion_strategy = deletion_strategy
        self._tracker_file = tracker_file
        self._pre_step = 0

    def write(self, content, path: str):
        path = str(path)  # The path maybe a PosixPath.
        if path.endswith(self._tracker_file):
            pre_step = self.read(path)
            if pre_step:
                self._pre_step = int(pre_step)
        super().write(content, path)

    def commit(self, step, success):
        super().commit(step, success)
        if not success or self._pre_step == step:
            return
        self._deletion_strategy.clean_up(self._pre_step, shutil.rmtree)

    def get_class_meta(self):
        kwargs = {
            "tracker_file": self._tracker_file,
            "deletion_strategy": self._deletion_strategy,
        }
        class_mata = ClassMeta(
            module_path=self.__class__.__module__,
            class_name=self.__class__.__name__,
            kwargs=kwargs,
        )
        return class_mata


def get_checkpoint_storage(deletion_strategy=None):
    if deletion_strategy:
        storage = PosixStorageWithDeletion(
            tracker_file=CheckpointConstant.TRACER_FILE_NAME,
            deletion_strategy=deletion_strategy,
        )
    else:
        storage = PosixDiskStorage()
    return storage
