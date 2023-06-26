# Copyright 2022 The ElasticDL Authors. All rights reserved.
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


class AnalyserResult(object):
    """Store analyzer's result"""

    def __init__(self):
        self._res = {}

    def get(self, key):
        """Return the value of the key"""
        _, res = self._search_recursively(self._res, key)
        return res

    def _search_recursively(self, res, key):
        """DepthFirstSearch the key in res and return the corresponding value"""
        if not isinstance(res, dict):
            # reach the leaf node
            return False, None

        if key in res:
            return True, res[key]

        for k in res:
            found, current_res = self._search_recursively(res[k], key)
            if found:
                return True, current_res

        return False, None

    def put(self, key, val):
        self._res[key] = val

    def update(self, res):
        """Update all result
        Args:
            res(dict): use res to update value
        """
        self._res.update(res)
