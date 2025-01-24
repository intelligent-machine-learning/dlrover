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
import signal
import sys
import threading
import traceback

from .config import get_default_config, get_default_logger
from .emitter import Process
from .exporter import close_default_exporter

_LOGGER = get_default_logger()
_CONFIG = get_default_config()


class ErrorHandler:
    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def __init__(self):
        with self._lock:
            if not self._initialized:
                self._original_excepthook = None
                self._original_handlers = {}
                self._initialized = True
                self._process = Process("ErrorReporter")

    def _handle_exception(self, exc_type, exc_value, exc_traceback):
        """handle exception"""
        if exc_traceback is not None:
            # get exception stack
            stack_info = traceback.format_exception(
                exc_type, exc_value, exc_traceback
            )
            self._process.instant(
                "exception",
                {
                    "stack": stack_info,
                    "pid": os.getpid(),
                },
            )

        if self._original_excepthook:
            self._original_excepthook(exc_type, exc_value, exc_traceback)

    def _handle_signal(self, signum, frame):
        """handle signal"""

        try:
            content = {
                "sig": signum,
                "sig_name": signal.Signals(signum).name,
                "pid": os.getpid(),
            }

            try:
                if frame:
                    stack = traceback.extract_stack(frame)
                    content["stack"] = traceback.format_list(stack)
            except Exception as e:
                content["stack"] = f"获取堆栈信息失败: {str(e)}"

            self._process.instant("exit_sig", content)
            close_default_exporter()

        except Exception as e:
            _LOGGER.error(f"处理信号 {signum} 时发生错误: {str(e)}")

        finally:
            for sig, handler in self._original_handlers.items():
                if callable(handler):
                    handler(sig, frame)
                else:
                    if handler == signal.SIG_DFL:
                        signal.signal(sig, signal.SIG_DFL)
                        os.kill(os.getpid(), signum)
                    elif handler == signal.SIG_IGN:
                        pass

    def _register(self):
        """register exception handler"""
        self._original_excepthook = sys.excepthook
        sys.excepthook = self._handle_exception

        signals = [
            signal.SIGINT,
            signal.SIGTERM,
            signal.SIGABRT,
            signal.SIGFPE,
            signal.SIGBUS,
            signal.SIGPIPE,
            signal.SIGSEGV,
            signal.SIGHUP,
        ]
        for sig in signals:
            self._original_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, self._handle_signal)

        _LOGGER.info("Exception handler registered")

    def _unregister(self):
        """unregister exception handler"""
        if self._original_excepthook:
            sys.excepthook = self._original_excepthook
            self._original_excepthook = None

        for sig, handler in self._original_handlers.items():
            if handler:
                signal.signal(sig, handler)
        self._original_handlers.clear()

        _LOGGER.info("Exception handler unregistered")

    @classmethod
    def register(cls):
        """register exception handler"""
        instance = cls()
        instance._register()

    @classmethod
    def unregister(cls):
        """unregister exception handler"""
        instance = cls()
        instance._unregister()


def init_error_handler():
    if _CONFIG.hook_error:
        ErrorHandler.register()
