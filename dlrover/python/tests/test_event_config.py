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

import os
import tempfile

from dlrover.python.training_event.config import (
    ENV_PREFIX,
    Config,
    parse_bool,
    parse_int,
    parse_str,
)


def test_parse_bool():
    assert parse_bool("true") is True
    assert parse_bool("false") is False
    assert parse_bool("True") is True
    assert parse_bool("False") is False
    assert parse_bool("1") is True
    assert parse_bool("0") is False
    assert parse_bool("yes") is True
    assert parse_bool("no") is False


def test_parse_int():
    assert parse_int("1") == 1
    assert parse_int("0") == 0
    assert parse_int("100") == 100
    assert parse_int("-1") == -1
    assert parse_int("1000000") == 1000000


def test_parse_str():
    assert parse_str("hello") == "hello"
    assert parse_str("world") == "world"
    assert parse_str("123") == "123"
    assert parse_str("True") == "True"
    assert parse_str("False") == "False"
    assert parse_str(123) is None
    assert parse_str(None) is None
    assert parse_str(True) is None
    assert parse_str(False) is None
    assert parse_str([]) is None
    assert parse_str({}) is None
    assert parse_str(()) is None


def test_init_from_env():
    os.environ[f"{ENV_PREFIX}_EVENT_EXPORTER"] = "TEXT_FILE"
    os.environ[f"{ENV_PREFIX}_ASYNC_EXPORTER"] = "True"
    os.environ[f"{ENV_PREFIX}_QUEUE_SIZE"] = "1024"
    os.environ[f"{ENV_PREFIX}_FILE_DIR"] = "/tmp"
    os.environ[f"{ENV_PREFIX}_TEXT_FORMATTER"] = "TEXT"
    os.environ[f"{ENV_PREFIX}_ENABLE"] = "True"
    os.environ[f"{ENV_PREFIX}_DEBUG_MODE"] = "True"
    os.environ[f"{ENV_PREFIX}_HOOK_ERROR"] = "True"

    config = Config()
    assert config.event_exporter == "TEXT_FILE"
    assert config.async_exporter is True
    assert config.queue_size == 1024
    assert config.file_dir == "/tmp"
    assert config.text_formatter == "TEXT"
    assert config.enable is True
    assert config.debug_mode is True
    assert config.hook_error is True


def test_init_from_file():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(
            b'{"event_exporter": "TEXT_FILE", '
            b'"async_exporter": true, '
            b'"queue_size": 1024, '
            b'"file_dir": "/tmp", '
            b'"text_formatter": "TEXT", '
            b'"enable": true, '
            b'"debug_mode": true, '
            b'"hook_error": true}'
        )
        f.flush()
        f.seek(0)

        config = Config(f.name)
        assert config.event_exporter == "TEXT_FILE"
        assert config.async_exporter is True
        assert config.queue_size == 1024
        assert config.file_dir == "/tmp"
        assert config.text_formatter == "TEXT"
        assert config.enable is True
        assert config.debug_mode is True
        assert config.hook_error is True
