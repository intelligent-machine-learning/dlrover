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
import traceback

from dlrover.python.common.singleton import Singleton
from dlrover.python.training_event.config import Config, get_default_logger
from dlrover.python.training_event.emitter import Process

logger = get_default_logger()


class ErrorHandler(Singleton):
    def __init__(self):
        self._original_excepthook = None
        self._original_handlers = {}
        self._process = Process("ErrorReporter")
        self._registered = False

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
                content["stack"] = f"get stack failed: {str(e)}"

            self._process.instant("exit_sig", content)

        except Exception as e:
            logger.error(f"process signal {signum} error: {str(e)}")

        finally:
            self._call_original_handler(signum, frame)

    def _call_original_handler(self, signum, frame):
        """call original handler with signal"""

        # if the handler is callable, call it
        if callable(self._original_handlers[signum]):
            self._original_handlers[signum](signum, frame)
        # if the handler is SIG_IGN or signal is SIGCHLD, do nothing
        elif (
            self._original_handlers[signum] == signal.SIG_IGN
            or signum == signal.SIGCHLD
        ):
            return
        else:
            if self._registered:
                self.unregister()
            # call original handler with signal
            os.kill(os.getpid(), signum)

    def register(self):
        """register exception handler"""
        with self._lock:
            if self._registered:
                return
            self._original_excepthook = sys.excepthook
            sys.excepthook = self._handle_exception

            # only catch exit signals
            # non exit signals like sigchld, will cause reentrance/deadlock
            signals = [
                signal.SIGINT,
                signal.SIGTERM,
                signal.SIGABRT,
                signal.SIGFPE,
                signal.SIGBUS,
                signal.SIGPIPE,
                signal.SIGSEGV,
                signal.SIGILL,
                signal.SIGHUP,
                signal.SIGQUIT,
            ]
            for sig in signals:
                self._original_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, self._handle_signal)

            self._registered = True
            logger.info("Exception handler registered")

    def unregister(self):
        """unregister exception handler"""
        with self._lock:
            if not self._registered:
                return

            if self._original_excepthook:
                sys.excepthook = self._original_excepthook
                self._original_excepthook = None

            for sig, handler in self._original_handlers.items():
                signal.signal(sig, handler)
            self._original_handlers.clear()

            self._registered = False

            logger.info("Exception handler unregistered")


def init_error_handler():
    config = Config.singleton_instance()
    if config.enable and config.hook_error:
        ErrorHandler.singleton_instance().register()
