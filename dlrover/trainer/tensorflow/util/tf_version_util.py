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

import tensorflow as tf


def get_tf_version():
    version = tf.__version__
    version = version.split(".")
    return version


def is_tf_1():
    version = get_tf_version()
    return int(version[0]) == 1


def is_tf_2():
    version = get_tf_version()
    return int(version[0]) > 1


def is_tf_113():
    version = get_tf_version()
    return int(version[0]) == 1 and int(version[1]) == 13


def is_tf_115():
    version = get_tf_version()
    return int(version[0]) == 1 and int(version[1]) == 15
