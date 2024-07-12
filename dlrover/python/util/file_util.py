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
from pathlib import Path


def is_same_path(path1: str, path2: str):
    """
    Is target path the same.

    Args:
        path1 (str): Path 1.
        path2 (str): Path 2.

    Returns:
        result
    """

    return Path(path1).resolve() == Path(path2).resolve()


def find_file_in_parents(filename, start_dir=None):
    """
    Find target file in parent directories.

    Args:
        filename (str): Target filename.
        start_dir (str, optional): Target directory. Defaults to None(from current directory).

    Returns:
        result: The target file path.
    """

    if start_dir is None:
        start_dir = os.path.abspath(os.curdir)

    current_dir = start_dir

    while True:
        file_path = os.path.join(current_dir, filename)

        if os.path.isfile(file_path):
            return file_path

        parent_dir = os.path.dirname(current_dir)

        if parent_dir == current_dir:
            return None

        current_dir = parent_dir
