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
            succeed (bool): whether to persist the checkpoint of step.
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
    def __init__(self):
        self._latest_path = ""

    def write(self, content, path):
        mode = "w"
        if isinstance(content, bytes) or isinstance(content, memoryview):
            mode = "wb"
        with open(path, mode) as stream:
            stream.write(content)
            os.fsync(stream.fileno())

    def write_state_dict(self, state_dict, path, write_func=None):
        dir = os.path.dirname(path)
        os.makedirs(dir, exist_ok=True)
        if write_func:
            write_func(state_dict, path)
        self._latest_path = path

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
            f"Succeed {success} in persisting the checkpoint to "
            f"{self._latest_path} for step {step}"
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
