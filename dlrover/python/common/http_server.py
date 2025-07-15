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

import abc
import asyncio
import threading
import time

import tornado

from dlrover.python.common.log import default_logger as logger


def is_asyncio_loop_running():
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


class CustomHTTPServer(abc.ABC):
    """Self designed http server."""

    def __init__(self, address, port, handler_classes):
        self._address = address
        self._port = port
        self._handler_classes = handler_classes

    @property
    def address(self):
        return self._address

    @property
    def port(self):
        return self._port

    @property
    def handler_classes(self):
        return self._handler_classes

    @abc.abstractmethod
    def start(self):
        """Start the server."""
        pass

    @abc.abstractmethod
    def stop(self, grace=None):
        """
        Stop the server.

        Arg:
            grace (Optional[float]): Grace period.
        """
        pass


class TornadoHTTPServer(CustomHTTPServer):
    SERVING_THREAD_NAME = "http-server-serving-thread"

    def __init__(self, address, port, handler_class):
        super().__init__(address, port, handler_class)

        self._io_loop = None
        self._server = None
        self._serving_started = False

    def start(self):
        if not self.is_serving():
            self._serving_started = True

            server_thread = threading.Thread(
                target=self._start_server,
                name=TornadoHTTPServer.SERVING_THREAD_NAME,
            )
            server_thread.start()

            while not self._io_loop or is_asyncio_loop_running():
                time.sleep(0.1)

    def _start_server(self):
        try:
            self._server = tornado.httpserver.HTTPServer(
                tornado.web.Application(self._handler_classes)
            )
            self._server.listen(self._port)
            self._io_loop = tornado.ioloop.IOLoop.current()
            self._io_loop.start()
        except Exception as e:
            logger.error(f"Http server start with error: {e}")

    def stop(self, grace=None):
        if self._server:
            self._server.stop()
            if self._io_loop:
                self._io_loop.add_callback(self._io_loop.stop)

        self._serving_started = False

    def is_serving(self):
        return self._serving_started
