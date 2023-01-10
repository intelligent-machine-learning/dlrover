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

from dlrover.trainer.constants.tf_constants import TFConstants


def parse_uri(path):
    """Parse a path into a schema"""
    path = path.strip()
    scheme = None
    if path.startswith(TFConstants.FILE_SCHEME()):
        scheme = TFConstants.FILE_SCHEME()
    elif path.startswith(TFConstants.FAKE_SCHEME()):
        scheme = TFConstants.FAKE_SCHEME()
    else:
        raise ValueError("Wrong path provided: %s" % path)
    scheme_prefix = scheme
    start_pos = len(scheme_prefix)
    return scheme, path[start_pos:]
