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
rm -rf log* eval/ export/  checkpoint  model.ckpt* events.out* -rf graph.pbtxt
python -m dlrover.python.master.main --platform=local --job_name=train-test --port 12346 &
export DLROVER_MASTER_ADDR=127.0.0.1:12346
python -m dlrover.trainer --platform=local_kubernetes --conf=conf.TrainConf --ps_num=1 --worker_num=1  
