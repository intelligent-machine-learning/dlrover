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

import io
import pickle

whitelist = ["dlrover.python.common.comm"]


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module not in whitelist:
            raise pickle.UnpicklingError(
                f"Unpickle illegal module: {module} or class: {name}"
            )

        return pickle.Unpickler.find_class(self, module, name)


def loads(s):
    if isinstance(s, str):
        raise TypeError("Can't load pickle from unicode string")
    return RestrictedUnpickler(io.BytesIO(s)).load()


dumps = pickle.dumps
