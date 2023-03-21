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

import json

import yaml

p = "/home/dlrover/dlrover/python/tests/test.json"


def parse_yaml_file(file_path):
    data = None
    with open(file_path, "r", encoding="utf-8") as file:
        file_data = file.read()
        data = yaml.safe_load(file_data)
    return data


data = parse_yaml_file("/home/dlrover/dlrover/python/tests/data/demo.yaml")
with open(p, "w") as f:
    json.dump(data, f)
