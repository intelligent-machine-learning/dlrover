# Copyright 2024 The DLRover Authors. All rights reserved.
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


def gen_k8s_label_selector_from_dict(label_dict):
    """
    Generate a kubernetes label selector from dict.
    e.g. {key1: value1, key2: value2} -> "key1=value1,key2=value2"
    """

    return ",".join(f"{key}={value}" for key, value in label_dict.items())


def gen_dict_from_k8s_label_selector(label_selector):
    """
    Generate a dict from kubernetes label selector format.
    e.g. "key1=value1,key2=value2" -> {key1: value1, key2: value2}
    """

    return {
        k: v
        for k, v in (item.split("=") for item in label_selector.split(","))
    }


def is_target_labels_equal(target_labels: dict, source_labels: dict):
    """
    Whether the target labels exist in the labels source and are all equal.

    Args:
        target_labels (dict): Target labels.
        source_labels (dict): Source labels.

    e.g.
    Return true if:
    target_labels: {key1: value1, key2: value2}
    source_labels: {key1: value1, key2: value2, key3: value3}

    Return false if:
    target_labels: {key1: value1, key2: value3}
    source_labels: {key1: value1, key2: value2, key3: value3}
    """

    # all values to str for compatible
    target_labels = {k: str(v) for k, v in target_labels.items()}
    source_labels = {k: str(v) for k, v in source_labels.items()}

    if all(target_labels[x] == source_labels[x] for x in target_labels):
        return True
    return False
