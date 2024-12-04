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
import threading
import time

import tornado

from dlrover.python.common.log import default_logger as logger


class CustomHTTPServer(abc.ABC):
    def __init__(self, address, port, handler_class):
        self._address = address
        self._port = port
        self._handler_class = handler_class

    @property
    def address(self):
        return self._address

    @property
    def port(self):
        return self._port

    @property
    def handler_class(self):
        return self._handler_class

    @abc.abstractmethod
    def start(self):
        """Start the server."""
        pass

    @abc.abstractmethod
    def stop(self):
        """Stop the server."""
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

            # wait 3s for sever start
            time.sleep(3)

    def _start_server(self):
        try:
            self._server = tornado.httpserver.HTTPServer(
                tornado.web.Application([(r"/", self._handler_class)])
            )
            self._server.listen(self._port)
            self._io_loop = tornado.ioloop.IOLoop.current()
            self._io_loop.start()
        except Exception as e:
            logger.error(f"Http server start with error: {e}")

    def stop(self):
        if self._server:
            self._server.stop()
            self._io_loop.add_callback(self._io_loop.stop)

        self._serving_started = False

    def is_serving(self):
        return self._serving_started
