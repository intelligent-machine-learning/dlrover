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

from unittest.mock import Mock

import pytest

from dlrover.python.training_event.emitter import (
    DurationSpan,
    EventEmitter,
    Process,
    generate_event_id,
)
from dlrover.python.training_event.event import EventType
from dlrover.python.training_event.exporter import EventExporter


def test_event_emitter_instant():
    mock_exporter = Mock(spec=EventExporter)
    emitter = EventEmitter("test", mock_exporter)
    emitter.instant("foo", {"a": 1})
    mock_exporter.close()

    mock_exporter.export.assert_called_once()
    event = mock_exporter.export.call_args.args[0]
    assert event.content["a"] == 1
    assert event.name == "foo"
    assert event.target == "test"
    assert event.event_type == EventType.INSTANT
    assert event.event_time is not None


def test_event_emitter_begin():
    mock_exporter = Mock(spec=EventExporter)
    emitter = EventEmitter("test", mock_exporter)
    emitter.begin("test_id", "foo", {"a": 1})
    mock_exporter.close()

    mock_exporter.export.assert_called_once()
    event = mock_exporter.export.call_args.args[0]
    assert event.content["a"] == 1
    assert event.name == "foo"
    assert event.target == "test"
    assert event.event_type == EventType.BEGIN
    assert event.event_time is not None


def test_event_emitter_end():
    mock_exporter = Mock(spec=EventExporter)
    emitter = EventEmitter("test", mock_exporter)
    emitter.end("test_id", "foo", {"a": 1})
    mock_exporter.close()

    mock_exporter.export.assert_called_once()
    event = mock_exporter.export.call_args.args[0]
    assert event.content["a"] == 1
    assert event.name == "foo"
    assert event.target == "test"
    assert event.event_type == EventType.END
    assert event.event_time is not None


def test_duration_context_manager():
    mock_exporter = Mock(spec=EventExporter)
    emitter = EventEmitter("test", mock_exporter)
    with DurationSpan(emitter, "foo", {"a": 1}):
        pass
    mock_exporter.close()

    assert mock_exporter.export.call_count == 2
    begin_event = mock_exporter.export.call_args_list[0].args[0]
    assert begin_event.content["a"] == 1
    assert begin_event.name == "foo"
    assert begin_event.target == "test"
    assert begin_event.event_type == EventType.BEGIN
    assert begin_event.event_time is not None

    end_event = mock_exporter.export.call_args_list[1].args[0]
    assert end_event.content["a"] == 1
    assert end_event.name == "foo"
    assert end_event.target == "test"
    assert end_event.event_type == EventType.END
    assert end_event.event_time is not None


def test_duration_context_manager_exception():
    mock_exporter = Mock(spec=EventExporter)
    emitter = EventEmitter("test", mock_exporter)
    with pytest.raises(RuntimeError):
        with DurationSpan(emitter, "foo", {"a": 1}):
            raise RuntimeError("test exception")
    mock_exporter.close()
    assert mock_exporter.export.call_count == 2
    end_event = mock_exporter.export.call_args_list[1].args[0]
    assert end_event.content["a"] == 1
    assert end_event.name == "foo"
    assert end_event.content["success"] is False
    assert end_event.content["error"] != ""


def test_duration_begin_end():
    mock_exporter = Mock(spec=EventExporter)
    emitter = EventEmitter("test", mock_exporter)
    duration = DurationSpan(emitter, "foo", {"a": 1})
    duration.begin()
    duration.end()
    mock_exporter.close()

    assert mock_exporter.export.call_count == 2
    begin_event = mock_exporter.export.call_args_list[0].args[0]
    assert begin_event.content["a"] == 1
    assert begin_event.name == "foo"
    assert begin_event.target == "test"
    assert begin_event.event_type == EventType.BEGIN
    assert begin_event.event_time is not None

    end_event = mock_exporter.export.call_args_list[1].args[0]
    assert end_event.content["a"] == 1
    assert end_event.name == "foo"
    assert end_event.target == "test"
    assert end_event.event_type == EventType.END
    assert end_event.event_time is not None


def test_duration_multiple_begin():
    mock_exporter = Mock(spec=EventExporter)
    emitter = EventEmitter("test", mock_exporter)
    duration = DurationSpan(emitter, "foo", {"a": 1})
    duration.begin()
    duration.begin()
    mock_exporter.close()

    assert mock_exporter.export.call_count == 1
    begin_event = mock_exporter.export.call_args_list[0].args[0]
    assert begin_event.content["a"] == 1
    assert begin_event.name == "foo"
    assert begin_event.target == "test"
    assert begin_event.event_type == EventType.BEGIN
    assert begin_event.event_time is not None


def test_duration_multiple_end():
    mock_exporter = Mock(spec=EventExporter)
    emitter = EventEmitter("test", mock_exporter)
    duration = DurationSpan(emitter, "foo", {"a": 1})
    duration.begin()
    duration.end()
    duration.end()
    mock_exporter.close()

    assert mock_exporter.export.call_count == 2


def test_duration_end_without_begin():
    mock_exporter = Mock(spec=EventExporter)
    emitter = EventEmitter("test", mock_exporter)
    duration = DurationSpan(emitter, "foo", {"a": 1})
    duration.end()
    mock_exporter.close()

    assert mock_exporter.export.call_count == 0


def test_duration_success():
    mock_exporter = Mock(spec=EventExporter)
    emitter = EventEmitter("test", mock_exporter)
    duration = DurationSpan(emitter, "foo", {"a": 1})
    duration.begin()
    duration.success()
    mock_exporter.close()

    assert mock_exporter.export.call_count == 2
    begin_event = mock_exporter.export.call_args_list[0].args[0]
    assert begin_event.content["a"] == 1
    assert begin_event.name == "foo"
    assert begin_event.target == "test"
    assert begin_event.event_type == EventType.BEGIN
    assert begin_event.event_time is not None

    end_event = mock_exporter.export.call_args_list[1].args[0]
    assert end_event.content["a"] == 1
    assert end_event.content["success"] is True
    assert end_event.name == "foo"
    assert end_event.target == "test"
    assert end_event.event_type == EventType.END
    assert end_event.event_time is not None


def test_duration_fail():
    mock_exporter = Mock(spec=EventExporter)
    emitter = EventEmitter("test", mock_exporter)
    duration = DurationSpan(emitter, "foo", {"a": 1})
    duration.begin()
    duration.fail(error="test error")
    mock_exporter.close()

    assert mock_exporter.export.call_count == 2
    begin_event = mock_exporter.export.call_args_list[0].args[0]
    assert begin_event.content["a"] == 1
    assert begin_event.name == "foo"
    assert begin_event.target == "test"
    assert begin_event.event_type == EventType.BEGIN
    assert begin_event.event_time is not None

    end_event = mock_exporter.export.call_args_list[1].args[0]
    assert end_event.content["a"] == 1
    assert end_event.content["success"] is False
    assert end_event.content["error"] == "test error"
    assert end_event.name == "foo"
    assert end_event.target == "test"
    assert end_event.event_type == EventType.END
    assert end_event.event_time is not None


def test_duration_stage():
    mock_exporter = Mock(spec=EventExporter)
    emitter = EventEmitter("test", mock_exporter)
    duration = DurationSpan(emitter, "foo", {"a": 1})
    duration.begin()
    with duration.stage("stage1", {"b": 2}):
        pass
    duration.end()
    mock_exporter.close()

    assert mock_exporter.export.call_count == 4
    begin_event = mock_exporter.export.call_args_list[0].args[0]
    assert begin_event.content["a"] == 1
    assert begin_event.name == "foo"
    assert begin_event.target == "test"
    assert begin_event.event_type == EventType.BEGIN
    assert begin_event.event_time is not None

    stage_event = mock_exporter.export.call_args_list[1].args[0]
    assert stage_event.content["a"] == 1
    assert stage_event.content["b"] == 2
    assert stage_event.name == "foo#stage1"
    assert stage_event.target == "test"
    assert stage_event.event_type == EventType.BEGIN
    assert stage_event.event_time is not None

    stage_end_event = mock_exporter.export.call_args_list[2].args[0]
    assert stage_end_event.content["a"] == 1
    assert stage_end_event.content["b"] == 2
    assert stage_end_event.name == "foo#stage1"
    assert stage_end_event.target == "test"
    assert stage_end_event.event_type == EventType.END
    assert stage_end_event.event_time is not None

    end_event = mock_exporter.export.call_args_list[3].args[0]
    assert end_event.content["a"] == 1
    assert end_event.name == "foo"
    assert end_event.target == "test"
    assert end_event.event_type == EventType.END
    assert end_event.event_time is not None


def test_duration_stage_end_without_begin():
    mock_exporter = Mock(spec=EventExporter)
    emitter = EventEmitter("test", mock_exporter)
    duration = DurationSpan(emitter, "foo", {"a": 1})
    with duration.stage("stage1", {"b": 2}):
        pass
    mock_exporter.close()

    assert mock_exporter.export.call_count == 0


def test_duration_stage_end_after_end():
    mock_exporter = Mock(spec=EventExporter)
    emitter = EventEmitter("test", mock_exporter)
    duration = DurationSpan(emitter, "foo", {"a": 1})
    duration.begin()
    duration.end()
    with duration.stage("stage1", {"b": 2}):
        pass
    mock_exporter.close()

    assert mock_exporter.export.call_count == 2


def test_duration_extra_args_dict():
    mock_exporter = Mock(spec=EventExporter)
    emitter = EventEmitter("test", mock_exporter)
    duration = DurationSpan(emitter, "foo", {"a": 1})
    duration.extra_args(b=2)
    duration.begin()
    duration.extra_dict({"c": 3})
    duration.end()
    duration.extra_dict({"d": 4})
    duration.extra_args(e=5)
    mock_exporter.close()

    assert mock_exporter.export.call_count == 2
    begin_event = mock_exporter.export.call_args_list[0].args[0]
    assert begin_event.content["a"] == 1
    assert begin_event.content["b"] == 2
    assert begin_event.name == "foo"
    assert begin_event.target == "test"
    assert begin_event.event_type == EventType.BEGIN
    assert begin_event.event_time is not None

    end_event = mock_exporter.export.call_args_list[1].args[0]
    assert end_event.content["a"] == 1
    assert end_event.content["b"] == 2
    assert end_event.content["c"] == 3
    assert end_event.name == "foo"
    assert end_event.target == "test"
    assert end_event.event_type == EventType.END
    assert end_event.event_time is not None


class CustomDuration(DurationSpan):
    def custom_stage(self):
        return self.stage("custom_stage")

    def custom_exception(self):
        raise RuntimeError("test exception")


def test_custom_duration():
    mock_exporter = Mock(spec=EventExporter)
    emitter = EventEmitter("test", mock_exporter)
    duration = CustomDuration(emitter, "foo", {"a": 1})
    with duration:
        with duration.custom_stage():
            pass

        with duration.custom_exception():
            pass

    with pytest.raises(RuntimeError):
        with duration.custom_exception():
            raise RuntimeError("test exception")

    custom = duration.custom_exception().begin()
    custom.extra_args(b=2)
    custom.end()

    mock_exporter.close()

    assert mock_exporter.export.call_count == 4


class CustomProcess(Process):
    def custom(self):
        return self.custom_duration(CustomDuration, "custom_duration")

    def my_event(self):
        return self.instant("my_event")


def test_process_instant():
    mock_exporter = Mock(spec=EventExporter)
    process = Process("test", mock_exporter)
    process.instant("foo", {"a": 1})
    mock_exporter.close()

    mock_exporter.export.assert_called_once()
    event = mock_exporter.export.call_args.args[0]
    assert event.content["a"] == 1
    assert event.name == "foo"
    assert event.target == "test"
    assert event.event_type == EventType.INSTANT
    assert event.event_time is not None


def test_process_duration():
    mock_exporter = Mock(spec=EventExporter)
    process = Process("test", mock_exporter)
    with process.duration("foo", {"a": 1}):
        pass
    mock_exporter.close()

    assert mock_exporter.export.call_count == 2


def test_process_custom_duration():
    mock_exporter = Mock(spec=EventExporter)
    process = CustomProcess("test", mock_exporter)
    with process.custom():
        pass
    mock_exporter.close()

    assert mock_exporter.export.call_count == 2


def test_process_custom_event():
    mock_exporter = Mock(spec=EventExporter)
    process = CustomProcess("test", mock_exporter)
    process.my_event()
    mock_exporter.close()

    assert mock_exporter.export.call_count == 1


def test_generate_event_id():
    assert generate_event_id() is not None
    assert len(generate_event_id()) == 8
    assert generate_event_id() != generate_event_id()
