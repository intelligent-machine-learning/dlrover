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

from importlib.metadata import EntryPoint

import pytest
from pytest_mock import MockerFixture

from dlrover.python.unified.util.extension_util import (
    Extensible,
    load_entrypoints,
)


class SimpleExtension(Extensible):
    def method(self):
        return "original"


class ExtensionA(SimpleExtension):
    def method(self):
        return "A"


class ExtensionA2(SimpleExtension):
    def method2(self):
        return "A2"


class ExtensionB(SimpleExtension):
    def method(self):
        return "B"


def test_register_extension():
    assert SimpleExtension.extensions() == []
    SimpleExtension.register_extension(ExtensionA)
    assert SimpleExtension.extensions() == [ExtensionA]
    SimpleExtension.register_extension(ExtensionB)
    assert SimpleExtension.extensions() == [ExtensionA, ExtensionB]

    with pytest.raises(AssertionError):
        ExtensionA.extensions()  # Not directly derived from Extensible


def test_build_mixed_class():
    MixedClass = SimpleExtension.build_mixed_class()
    assert MixedClass is SimpleExtension  # No extensions registered yet
    assert MixedClass().method() == "original"

    # basic
    SimpleExtension.register_extension(ExtensionA)
    MixedClass = SimpleExtension.build_mixed_class()
    assert MixedClass().method() == "A"

    # A and A2 do not conflict
    SimpleExtension.register_extension(ExtensionA2)
    assert SimpleExtension.extensions() == [ExtensionA, ExtensionA2]
    MixedClass = SimpleExtension.build_mixed_class()
    assert MixedClass().method2() == "A2"

    # Conflict between A and B
    SimpleExtension.register_extension(ExtensionB)
    with pytest.raises(RuntimeError):
        _ = SimpleExtension.build_mixed_class()  # Conflict between A and B


def test_load_entrypoints(mocker: MockerFixture):
    mock_entry = mocker.Mock(EntryPoint)
    mock_entry.name = "plugin_a"
    mock_entry.value = "module:PluginA"
    mock_entry_points = mocker.patch(
        "importlib.metadata.entry_points", return_value=[mock_entry]
    )

    load_entrypoints("my_extension")

    mock_entry_points.assert_called_once_with(group="my_extension")
    mock_entry.load.assert_called_once()
