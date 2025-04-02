#!/bin/bash
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

echo "Start installing CI python packages."
start_time=$(date +%s)

echo "Upgrading pip tools"
pip install --upgrade pip
pip install -q kubernetes
pip install -q grpcio-tools
pip install -q psutil
pip install -q deprecated
pip install -q tornado

if [ "$1" = "basic" ]; then
  echo ""
  end_time=$(date +%s)
  cost_time=$((end_time-start_time))
  echo "'Basic' dependencies only, cost time: $((cost_time/60))min $((cost_time%60))s"
  exit 0
fi

pip install -q 'ray[default]'
pip install -q pyhocon
pip install -q pytest-cov
pip install -q pytest-ordering
pip install -q packaging
pip install -q tensorflow==2.13.0
pip install -q torch==2.0.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install -q deepspeed==0.12.6
pip install -q accelerate==0.29.2
pip install -q transformers==4.37.2
pip install -q peft==0.10.0

end_time=$(date +%s)
cost_time=$((end_time-start_time))
echo "All dependencies cost time: $((cost_time/60))min $((cost_time%60))s"
