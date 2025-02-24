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

import signal
import sys
import threading
from unittest.mock import Mock

import pytest

from dlrover.python.training_event.emitter import Process
from dlrover.python.training_event.error_handler import ErrorHandler


@pytest.fixture
def exception_handler():
    handler = ErrorHandler()
    handler._process = Mock(spec=Process)
    yield handler


def test_singleton():
    handler1 = ErrorHandler()
    handler2 = ErrorHandler()
    assert handler1 is handler2


def test_register_unregister(exception_handler):

    exception_handler.unregister()

    original_excepthook = sys.excepthook

    exception_handler.register()
    assert sys.excepthook != original_excepthook
    assert len(exception_handler._original_handlers) > 0

    exception_handler.unregister()
    assert sys.excepthook is original_excepthook  # 使用 'is' 而不是 '=='
    assert len(exception_handler._original_handlers) == 0


def test_handle_exception(exception_handler):
    try:
        raise ValueError("test")
    except Exception:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        exception_handler._handle_exception(exc_type, exc_value, exc_traceback)

    mock_process = exception_handler._process
    assert mock_process.instant.call_count == 1
    assert mock_process.instant.call_args.args[0] == "exception"
    assert mock_process.instant.call_args.args[1]["stack"] is not None


def test_handle_signal(exception_handler):
    exception_handler.register()
    mock_frame = None
    with pytest.raises(KeyboardInterrupt):
        exception_handler._handle_signal(signal.SIGINT, mock_frame)
    mock_process = exception_handler._process
    assert mock_process.instant.call_count == 1
    assert mock_process.instant.call_args.args[0] == "exit_sig"
    assert mock_process.instant.call_args.args[1]["sig"] is not None
    exception_handler.unregister()


def test_thread_safety():
    def create_handler():
        return ErrorHandler()

    lock = threading.Lock()
    handlers = []
    threads = []

    def append_handler():
        handler = create_handler()
        with lock:
            handlers.append(handler)

    for _ in range(10):
        thread = threading.Thread(target=append_handler)
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    first_handler = handlers[0]
    for handler in handlers[1:]:
        assert handler is first_handler
