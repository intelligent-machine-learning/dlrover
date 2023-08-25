#!/bin/sh
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
# export PYTHONPATH=`pwd`
rm -rf log* eval/ export/  checkpoint  model.ckpt* events.out* -rf graph.pbtxt ps_address* 
pkill python
python -m dlrover.python.master.main --platform=local --job_name=train-test --port 12348 &
export DLROVER_MASTER_ADDR=127.0.0.1:12348
ray stop --force
ray start --head --port=5001  --dashboard-port=5000
python -m dlrover.python.master.main --namespace dlrover --platform ray --job_name elasticjob-sample-24 --port 50001