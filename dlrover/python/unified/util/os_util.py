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

import os

from dlrover.python.common.log import default_logger as logger
from dlrover.python.util.common_util import (
    find_free_port_in_range,
    find_free_port_in_set,
)


def get_free_port() -> int:
    """Find a free port from the HOST_PORTS in env."""
    free_port = None
    host_ports = os.getenv("HOST_PORTS", "")
    if host_ports:
        ports = []
        for port in host_ports.split(","):
            ports.append(int(port))
        try:
            free_port = find_free_port_in_set(ports)
        except RuntimeError as e:
            logger.warning(e)
    if not free_port:
        free_port = find_free_port_in_range(20000, 30000)
    return free_port
