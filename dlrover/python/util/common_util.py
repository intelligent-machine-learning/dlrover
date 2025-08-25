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
import inspect
import os
import random
import re
import socket
from contextlib import closing

import dlrover.python.util.file_util as fu
from dlrover.python.common.constants import AscendConstants
from dlrover.python.common.log import default_logger as logger


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


def find_free_port_from_env_and_bind(addr: str, **kwargs):
    """Find a free port by find_free_port_from_env and create a socket to bind it.
    Returns:
        (port, socket): A tuple containing the free port and the bound socket.
    """
    tries = 10
    while True:
        port = find_free_port_from_env(**kwargs)
        try:
            # Bind a free port
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind((addr, port))
            break
        except OSError:
            # Already in use, try next port
            if tries > 0:
                tries -= 1
                continue
            raise RuntimeError(
                f"Failed to bind a free port on {addr} after multiple attempts."
            )
    return port, s


def find_free_port_from_env(
    default_env_ports="", default_range_start=20000
) -> int:
    """
    Find a free port from the HOST_PORTS in env.

    Will return port from default_range_start~default_range_start+10000 if
    there is no port env.
    """
    free_port = None
    host_ports = os.getenv("HOST_PORTS", default_env_ports)
    if host_ports:
        ports = []
        for port in host_ports.split(","):
            ports.append(int(port))
        try:
            free_port = find_free_port_in_set(ports)
        except RuntimeError as e:
            logger.warning(e)
    if not free_port:
        free_port = find_free_port_in_range(
            default_range_start, default_range_start + 10000
        )
    return free_port


def find_free_port(port=0):
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", port))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def find_free_port_in_range(start=0, end=65535, random_port=True):
    """Find a free port from a range."""
    bind_ports = set()
    while True:
        if random_port:
            port = random.randint(start, end)
        else:
            port = start + len(bind_ports)
        if port in bind_ports:
            continue
        try:
            return find_free_port(port)
        except OSError:
            logger.warning(f"Socket creation attempt failed with {port}.")
            bind_ports.add(port)
        if len(bind_ports) == end - start + 1:
            break
    raise RuntimeError(f"Fail to find a free port in [{start}, {end})")


def find_free_port_in_set(ports):
    for port in ports:
        try:
            return find_free_port(port)
        except OSError:
            logger.warning(f"Socket creation attempt failed with {port}.")
    raise RuntimeError(f"Fail to find a free port in {ports}")


def find_free_port_for_hccl(
    start=AscendConstants.HCCL_PORT_START_DEFAULT,
) -> int:
    max_port = 65500
    cur_start = start
    end = start + 10000
    if end > max_port:
        end = max_port
    logger.info(f"Try to find available port for hccl from {start}")
    checking_port = 0
    while True:
        try:
            cur_end = cur_start + AscendConstants.NPU_PER_NODE
            for port in range(cur_start, cur_end):
                checking_port = port
                find_free_port(port)
            logger.info(f"Find available port start from: {cur_start}")
            break
        except OSError:
            logger.warning(
                f"Target port has already been used: {checking_port}."
            )
            if checking_port > 0:
                cur_start = checking_port + 1
            else:
                cur_start = cur_start + AscendConstants.NPU_PER_NODE
            if cur_start > end:
                cur_start = 0
                break
    return cur_start


def print_args(args, exclude_args=[], groups=None):
    """
    Args:
        args: parsing results returned from `parser.parse_args`
        exclude_args: the arguments which won't be printed.
        groups: It is a list of a list. It controls which options should be
        printed together. For example, we expect all model specifications such
        as `optimizer`, `loss` are better printed together.
        groups = [["optimizer", "loss"]]
    """

    def _get_attr(instance, attribute):
        try:
            return getattr(instance, attribute)
        except AttributeError:
            return None

    dedup = set()
    if groups:
        for group in groups:
            for element in group:
                dedup.add(element)
                logger.info("%s = %s", element, _get_attr(args, element))
    other_options = [
        (key, value)
        for (key, value) in args.__dict__.items()
        if key not in dedup and key not in exclude_args
    ]
    for key, value in other_options:
        logger.info("%s = %s", key, value)


def get_class_by_module_and_class_name(module_name, class_name):
    """
    Get class object by the given module name and a class name.
    """

    module = importlib.import_module(module_name)
    if hasattr(module, class_name):
        return getattr(module, class_name)
    return None


def get_methods_by_class(clz: type, with_protect=False):
    """
    Get all (method_name, method) in list format by a give class.
    """

    result = []
    for name, method in inspect.getmembers(clz):
        if not inspect.ismethod(method) and not inspect.isfunction(method):
            continue
        if not with_protect and name.startswith("_"):
            continue
        result.append((name, method))

    return result
