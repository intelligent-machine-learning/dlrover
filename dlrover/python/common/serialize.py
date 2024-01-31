# Copyright 2022 The DLRover Authors. All rights reserved.
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
from dataclasses import dataclass, field
from typing import Dict


def to_dict(o):
    if hasattr(o, "__dict__"):
        return o.__dict__
    else:
        return {}


class JsonSerializable(object):
    def to_json(self, indent=None):
        return json.dumps(
            self,
            default=to_dict,
            sort_keys=True,
            indent=indent,
        )


@dataclass
class ClassMeta:
    module_path: str = ""
    class_name: str = ""
    kwargs: Dict[str, str] = field(default_factory=dict)
