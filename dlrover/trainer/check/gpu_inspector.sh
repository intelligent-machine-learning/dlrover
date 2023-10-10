#!/bin/bash
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


# The script can check where the GPU is avaliable.

set -e

# If there is no gpu(nvidia-smi does not exists), exit 0.
if ! command -v nvidia-smi &> /dev/null
then
    exit 0
fi

# If the execution of nvidia-smi times out, it is assumed that the GPU driver has an error
status=$(timeout -s 9 5s nvidia-smi || echo "nvidia_smi_timeout")

if [ "$status" == "nvidia_smi_timeout" ];
then
    echo "execute nvidia-smi timeout. There maybe a exeception on GPU driver."
    exit 201  # exit code for DLRover job Master.
fi

memory_occupied_by_residual_process=$(echo "$status" | grep "Default" | awk '{print $9}' | awk -F "M" '{print $1}' | awk '{ SUM += $1} END {print SUM}')

# Sum the memory usage of all GPUs. If it exceeds 512MB, it is considered to have Pod residue.
# Note: the check may misjudge there is the Pod residue on the machine.
if [ "$memory_occupied_by_residual_process" -ge 512 ];
then
    echo "Found residual process"
    exit 202  # exit code for DLRover job Master.
fi