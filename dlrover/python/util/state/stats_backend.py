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


def parse_json_file(file_path):
    data = None
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def parse_yaml_file(file_path):
    data = None
    with open(file_path, "r", encoding="utf-8") as file:
        file_data = file.read()
        data = yaml.safe_load(file_data)
    return data


class LocalFileStateBackend:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = {}

    def load(self):
        data = {}
        if self.file_path.endswith("json"):
            data = parse_json_file(self.file_path)
        elif self.file_path.endswith("yaml"):
            data = parse_yaml_file(self.file_path)
        else:
            raise Exception("fails to parse file %s" % self.file_path)
        self.data = data
        return data

    def get(self, key, default_value=None):
        return self.data.get(key, default_value)

    def put(self, key, value):
        self.data[key] = value
