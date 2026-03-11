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
if [ "${DLROVER_ACCELERATOR_TYPE}" = "metax-tech.com/gpu" ]; then
  BUS_ID=$(mx-smi -i  "${LOCAL_RANK}"  --show-pcie | grep "GPU#" | awk '{print $3}')
  BUS_ID=${BUS_ID,,}
elif [ "${DLROVER_ACCELERATOR_TYPE}" = "nvidia.com/gpu" ]; then
  BUS_ID=$(nvidia-smi --query-gpu=pci.bus_id -i "${LOCAL_RANK}" --format=csv,noheader)
  BUS_ID=${BUS_ID,,}
  BUS_ID=${BUS_ID:4}
fi

# Find the numa node for device LOCAL_RANK
NODE=$(cat /sys/bus/pci/devices/"${BUS_ID}"/numa_node)

echo "Starting local rank $RANK on numa node $NODE"
echo "mempolicy: ${DLROVER_MEMBIND_POLICY}"

if [ "${DLROVER_MEMBIND_POLICY}" = "bind" ]; then
  echo -n "Cmd: numactl --cpunodebind=${NODE} --membind=${NODE} "
  echo "$@"
  numactl --cpunodebind="${NODE}" --membind="${NODE}" "$@"
elif [ "${DLROVER_MEMBIND_POLICY}" = "preferred" ]; then
  echo -n "Cmd: numactl --cpunodebind=${NODE} --preferred=${NODE} "
  echo "$@"
  numactl --cpunodebind="${NODE}" --preferred="${NODE}" "$@"
else
  echo -n "Cmd: numactl --cpunodebind=${NODE} "
  echo "$@"
  numactl --cpunodebind="${NODE}" "$@"
fi