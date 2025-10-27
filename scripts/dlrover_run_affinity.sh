#!/usr/bin/bash
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



# Query the bus ID for device LOCAL_RANK
BUS_ID=$(nvidia-smi --query-gpu=pci.bus_id -i "${LOCAL_RANK}" --format=csv,noheader)
BUS_ID=${BUS_ID,,}

# Find the numa node for device LOCAL_RANK
NODE=$(cat /sys/bus/pci/devices/"${BUS_ID:4}"/numa_node)

echo "Starting local rank $RANK on numa node $NODE"
echo -n "Cmd: numactl --cpunodebind=${NODE} --membind=${NODE}"
echo "$@"
numactl --cpunodebind="${NODE}" --membind="${NODE}" "$@"