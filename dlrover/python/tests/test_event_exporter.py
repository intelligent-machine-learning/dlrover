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

import json
import time
from datetime import datetime
from unittest.mock import Mock

from dlrover.python.training_event.event import Event, EventType
from dlrover.python.training_event.exporter import (
    AsyncExporter,
    Config,
    ConsoleExporter,
    EventExporter,
    JsonFormatter,
    LogFormatter,
    TextFileExporter,
    close_default_exporter,
    get_default_exporter,
    init_default_exporter,
)


def test_log_formatter():
    formatter = LogFormatter()
    event = Event(
        event_id="test_id",
        event_time=datetime(2025, 1, 1, 0, 0, 0),
        target="test",
        name="foo",
        event_type=EventType.INSTANT,
        content={"a": 1},
        pid=1,
    )
    assert formatter.format(event) == (
        '[2025-01-01T00:00:00] [1] [test_id] [test] [foo] [INSTANT] {"a": 1}'
    )


def test_json_formatter():
    formatter = JsonFormatter()
    event = Event(
        event_id="test_id",
        event_time=datetime(2025, 1, 1, 0, 0, 0),
        target="test",
        name="foo",
        event_type=EventType.INSTANT,
        content={"a": 1},
        pid=1,
    )

    event_dict = json.loads(formatter.format(event))
    assert event_dict["pid"] == 1
    assert event_dict["event_id"] == "test_id"
    assert event_dict["event_time"] == "2025-01-01T00:00:00"
    assert event_dict["target"] == "test"
    assert event_dict["name"] == "foo"
    assert event_dict["event_type"] == "INSTANT"
    assert event_dict["content"] == {"a": 1}


def test_text_file_exporter(tmpdir):
    event = Event(
        event_id="test_id",
        event_time=datetime(2025, 1, 1, 0, 0, 0),
        target="test",
        name="foo",
        event_type=EventType.INSTANT,
        content={"a": 1},
        pid=1,
    )
    exporter = TextFileExporter(tmpdir.strpath, LogFormatter())
    exporter.export(event)
    exporter.close()

    # list file in tmpdir
    assert len(tmpdir.listdir()) == 1

    # file content
    assert tmpdir.listdir()[0].read() == (
        '[2025-01-01T00:00:00] [1] [test_id] [test] [foo] [INSTANT] {"a": 1}\n'
    )


def test_console_exporter(capsys):
    formatter = LogFormatter()
    exporter = ConsoleExporter(formatter)
    event = Event(
        event_id="test_id",
        event_time=datetime(2025, 1, 1, 0, 0, 0),
        target="test",
        name="foo",
        event_type=EventType.INSTANT,
        content={"a": 1},
        pid=1,
    )
    exporter.export(event)
    exporter.close()
    # check if the event is printed
    captured = capsys.readouterr()
    assert captured.out == (
        '[2025-01-01T00:00:00] [1] [test_id] [test] [foo] [INSTANT] {"a": 1}\n'
    )


def test_async_exporter():
    mock_exporter = Mock(spec=EventExporter)
    exporter = AsyncExporter(mock_exporter)
    exporter.start()
    event = Event(
        event_id="test_id",
        event_time=datetime(2025, 1, 1, 0, 0, 0),
        target="test",
        name="foo",
        event_type=EventType.INSTANT,
        content={"a": 1},
        pid=1,
    )
    exporter.export(event)
    exporter.close()
    mock_exporter.export.assert_called_once_with(event)


def test_async_exporter_close_timeout():
    mock_exporter = Mock(spec=EventExporter)
    mock_exporter.export.side_effect = lambda _: time.sleep(1)
    exporter = AsyncExporter(mock_exporter)
    exporter.start()
    event = Event(
        event_id="test_id",
        event_time=datetime(2025, 1, 1, 0, 0, 0),
        target="test",
        name="foo",
        event_type=EventType.INSTANT,
        content={"a": 1},
        pid=1,
    )
    start_time = time.time()
    exporter.export(event)
    exporter.export(event)
    exporter.export(event)

    # only wait for 1s, so the event is not exported
    exporter.close(timeout=1)

    assert exporter.get_metrics()["dropped_events"] == 2
    assert time.time() - start_time < 2


def test_async_exporter_close_error():
    mock_exporter = Mock(spec=EventExporter)
    mock_exporter.export.side_effect = Exception("test")
    exporter = AsyncExporter(mock_exporter)
    exporter.start()
    exporter.export(Event.instant("test", "foo", {"a": 1}))
    exporter.close()
    assert mock_exporter.export.call_count == 1
    assert exporter.get_metrics()["error_events"] == 1


def test_async_exporter_full():
    mock_exporter = Mock(spec=EventExporter)
    mock_exporter.export.side_effect = lambda _: time.sleep(1)
    exporter = AsyncExporter(mock_exporter, max_queue_size=1)
    exporter.start()
    exporter.export(Event.instant("test", "foo", {"a": 1}))
    exporter.export(Event.instant("test", "foo", {"a": 1}))
    exporter.close()
    assert exporter.get_metrics()["dropped_events"] == 1


def test_default_exporter():
    config = Config.singleton_instance()
    config.enable = True
    init_default_exporter()
    default_exporter = get_default_exporter()
    assert default_exporter is not None
    close_default_exporter()
    assert get_default_exporter() is None
    config.enable = False
