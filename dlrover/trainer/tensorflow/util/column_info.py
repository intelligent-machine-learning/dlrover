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

from dlrover.trainer.tensorflow.util.common_util import add_prop


class Property(object):
    """Property descriptor"""

    def __init__(self, name, property_name, property_doc, default=None):
        self._name = name
        self._names = property_name
        self._default = default
        self._doc = property_doc

    def __get__(self, obj, objtype):
        return obj.__dict__.get(self._name, self._default)

    def __set__(self, obj, value):
        if obj is None:
            return self
        for name in self._names:
            obj.__dict__[name] = value

    @property
    def __doc__(self):
        return self._doc


@add_prop(
    ("dtype", "data type"),
    ("name", "feature_name"),
    ("is_sparse", "whether it is a sparse or dense feature"),
    ("is_label", "whether it is a label or a feature"),
)
class Column(object):
    @property
    def keys(self):
        """Get keys of a `Column`"""
        return self.added_prop

    def __str__(self):
        result = [k + "=" + str(getattr(self, k)) for k in self.keys]
        return "{" + ";".join(result) + "}"

    def set_default(self):
        """Set default value of column fields"""
        pass

    __repr__ = __str__
