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

import subprocess

from dlrover.python.common.log import default_logger as logger


def get_npu_pci_bus(rank):
    """
    get NPU pci bus info in BDF format

    Args:
        rank: NPU rank number

    Returns:
        NPU pci bus info, None if failed
    """
    """ e.g.
    NPU ID                         : 0
        Product Name                   : xxxxxxxxxx
        Model                          : NA
        Manufacturer                   : xxxxxx
        Serial Number                  : xxxxxxxxxxx
        Software Version               : xxxxxxx
        Firmware Version               : xxxxxxxxxx
        Compatibility                  : OK
        Board ID                       : 0x65
        PCB ID                         : A
        BOM ID                         : 1
        PCIe Bus Info                  : 0000:5A:00.0
        Slot ID                        : 0
        Class ID                       : NA
        PCI Vendor ID                  : xxxx
        PCI Device ID                  : xxxx
        Subsystem Vendor ID            : xxxx
        Subsystem Device ID            : xxxx
        Chip Count                     : 1
    """
    cmd = ["npu-smi", "info", "-t", "board", "-i", "{}".format(rank)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            if "PCIe Bus Info" in line:
                return line.split(":", 1)[1].strip().lower()


def get_npu_numa_node(rank):
    """
    get NPU device nearest numa node

    Args:
        rank: NPU rank number

    Returns:
        numa node number, None if failed
    """
    bus = get_npu_pci_bus(rank)
    if bus is not None:
        path = r"/sys/bus/pci/devices/{}/numa_node".format(bus)
        with open(path, "r") as f:
            return int(f.read().strip())


def get_gpu_pci_bus(rank):
    """
    get GPU pci bus info in BDF format

    Args:
        rank: GPU rank number

    Returns:
        GPU pci bus info, None if failed
    """
    """ e.g.
    00000000:18:00.0
    """
    cmd = [
        "nvidia-smi",
        "--query-gpu=pci.bus_id",
        "-i",
        "{}".format(rank),
        "--format=csv,noheader",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.lower().strip()[4:]


def get_gpu_numa_node(rank):
    """
    get GPU device nearest numa node

    Args:
        rank: GPU rank number

    Returns:
        numa node number, None if failed
    """
    bus = get_gpu_pci_bus(rank)
    if bus is not None:
        path = r"/sys/bus/pci/devices/{}/numa_node".format(bus)
        with open(path, "r") as f:
            return int(f.read().strip())


def parse_cpulist(cpulist):
    """

    Args:
        cpulist: e.g. 0-47,96-143

    Returns:
        cpu number set, e.g. {0, 1, 2, ... 46, 47, 96, 97, ... 142, 143}
    """
    all_cpus = set()
    for slice in cpulist.split(","):
        start = int(slice.split("-")[0])
        end = int(slice.split("-")[1]) + 1
        all_cpus = all_cpus.union(set([i for i in range(start, end)]))

    return all_cpus


def get_gpu_affinity(rank):
    """
    get gpu numa-affinity cpuset

    Args:
        rank: gpu rank

    Returns:
        cpu set
    """
    try:
        node = get_gpu_numa_node(rank)
        if node is not None:
            with open(f"/sys/devices/system/node/node{node}/cpulist") as f:
                return parse_cpulist(f.read().strip())
    except ValueError as e:
        logger.warning(f"get numa affinity value error: {e}")
    except OSError as e:
        logger.warning(f"get numa affinity os error: {e}")
    except Exception as e:
        logger.warning(f"get numa affinity unexpected error: {e}")


def get_npu_affinity(rank):
    """
    get npu numa-affinity cpuset

    Args:
        rank: npu rank

    Returns:
        cpu set
    """
    try:
        node = get_npu_numa_node(rank)
        if node is not None:
            with open(f"/sys/devices/system/node/node{node}/cpulist") as f:
                return parse_cpulist(f.read().strip())
    except ValueError as e:
        logger.warning(f"get numa affinity value error: {e}")
    except OSError as e:
        logger.warning(f"get numa affinity os error: {e}")
    except Exception as e:
        logger.warning(f"get numa affinity unexpected error: {e}")
