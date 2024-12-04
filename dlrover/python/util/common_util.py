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

import importlib.metadata
import re
import socket

import dlrover.python.util.file_util as fu


def get_dlrover_version():
    """Get the installed dlrover version."""

    version = get_installed_version("dlrover")
    if not version:
        # get from setup.py
        setup_file = fu.find_file_in_parents("setup.py")
        if setup_file:
            return get_version_from_setup(setup_file)
        return "Unknown"
    return version


def get_version_from_setup(setup_file):
    with open(setup_file, "r") as f:
        setup_content = f.read()

    version_match = re.search(r"version=['\"]([^'\"]+)['\"]", setup_content)
    if version_match:
        return version_match.group(1)
    else:
        raise ValueError("Version not found in setup.py.")


def get_installed_version(package_name):
    """
    Get the installed version of a package.

    Args:
        package_name (str): The name of the package.

    Return:
        result(str): Version of the package.
    """

    try:
        version = importlib.metadata.version(package_name)
        return version
    except importlib.metadata.PackageNotFoundError:
        return None


def is_port_in_use(port=0) -> bool:
    """
    Check if the port is in use.
    """

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        result = sock.connect_ex(("localhost", int(port)))
        return result == 0
