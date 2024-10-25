# Copyright 2020 The EasyDL Authors. All rights reserved.
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

try:
    from importlib import metadata
except ImportError:
    import importlib_metadata as metadata

from packaging.version import Version


def package_version_bigger_than(pkg_name, version):
    pkg_v = metadata.version(pkg_name)
    return Version(pkg_v) > Version(version)


if package_version_bigger_than("protobuf", "3.20.3"):
    from .protobuf_4_25_3 import (
        brain_pb2,
        brain_pb2_grpc,
        elastic_training_pb2,
        elastic_training_pb2_grpc,
    )
else:
    from .protobuf_3_20_3 import (
        brain_pb2,
        brain_pb2_grpc,
        elastic_training_pb2,
        elastic_training_pb2_grpc,
    )

__all__ = [
    "elastic_training_pb2",
    "elastic_training_pb2_grpc",
    "brain_pb2",
    "brain_pb2_grpc",
]
