# Copyright 2022 The DLRover Authors. All rights reserved.
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

import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import tornado

from dlrover.python.common.http_server import TornadoHTTPServer
from dlrover.python.util.common_util import is_port_in_use

TEST_SERVER_ADDR = "localhost"
TEST_SERVER_PORT = 8000


class HttpServerClientTest(unittest.TestCase):
    def setUp(self):
        self.server = None

    def tearDown(self):
        if self.server is not None:
            self.server.stop()
            self.server = None

    def test_tornado_server_basic(self):
        self.server = TornadoHTTPServer(
            TEST_SERVER_ADDR,
            TEST_SERVER_PORT,
            [(r"/", TestRequestHandler), (r"/report", TestRequestHandler)],
        )
        self.assertIsNotNone(self.server)
        self.assertFalse(is_port_in_use(TEST_SERVER_PORT))

        self.assertFalse(self.server.is_serving())
        self.server.start()
        self.assertTrue(self.server.is_serving())
        self.assertTrue(is_port_in_use(TEST_SERVER_PORT))
        self.server.start()
        self.assertTrue(self.server.is_serving())

        active_threads_name = [t.name for t in threading.enumerate()]
        self.assertIn(
            TornadoHTTPServer.SERVING_THREAD_NAME, active_threads_name
        )
        time.sleep(1)

        # test get and post request
        self._test_get_request()
        self._test_post_request()

        self.server.stop()
        self.assertFalse(self.server.is_serving())

    def _test_get_request(self):
        try:
            with requests.get("http://localhost:8000") as response:
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.text, "Hello, world!")
                return response
        except Exception as e:
            raise e

    def _test_post_request(self):
        try:
            with requests.post("http://localhost:8000/report") as response:
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.text, "Hello, world!!")
                return response
        except Exception as e:
            raise e

    def test_server_concurrency(self):
        self.server = TornadoHTTPServer(
            TEST_SERVER_ADDR,
            TEST_SERVER_PORT,
            [(r"/", TestRequestHandler), (r"/report", TestRequestHandler)],
        )
        self.server.start()

        futures = []
        result_num = 0
        client_size = 100
        with ThreadPoolExecutor(max_workers=client_size) as executor:
            for i in range(client_size):
                futures.append(executor.submit(self._test_get_request))

            for future in as_completed(futures):
                if future.result().status_code == 200:
                    result_num += 1
        self.assertEqual(len(futures), client_size)
        self.assertEqual(result_num, client_size)

        self.server.stop()


class TestRequestHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, world!")

    def post(self):
        self.write("Hello, world!!")
