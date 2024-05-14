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

pip install kubernetes
pip install grpcio-tools
pip install psutil
pip install deprecated
pip install 'ray[default]'
pip install pyhocon
pip install pytest-cov
pip install tensorflow==2.13.0
pip install deepspeed==0.12.6
pip install accelerate==0.29.2
pip install transformers==4.37.2
pip install torch==2.0.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install peft==0.10.0
pip install botorch==0.8.5
