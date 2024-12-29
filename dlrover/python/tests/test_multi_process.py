# Copyright 2023 The DLRover Authors. All rights reserved.
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
import time
import unittest
from unittest import mock

from dlrover.python.common.multi_process import (
    ERROR_CODE,
    SOCKET_TMP_DIR,
    SharedDict,
    SharedLock,
    SharedMemory,
    SharedQueue,
    SocketResponse,
    clear_sock_dir,
    retry_socket,
)


class SocketTest(object):
    @retry_socket
    def test_retry(self, retry):
        raise FileNotFoundError("test")


class SharedObjectTest(unittest.TestCase):
    def tearDown(self) -> None:
        clear_sock_dir()

    def test_retry(self):
        t = SocketTest()
        with self.assertRaises(FileNotFoundError):
            t.test_retry(retry=1)

    def test_shared_lock(self):
        name = "test"
        os.environ["TORCHELASTIC_RUN_ID"] = "test_job"
        server_lock = SharedLock(name, create=True)
        self.assertTrue(
            os.path.exists(f"{SOCKET_TMP_DIR}/test_job/sharedlock_test.sock")
        )
        client_lock = SharedLock(name, create=False)
        acquired = server_lock.acquire()
        self.assertTrue(acquired)
        self.assertTrue(server_lock.locked())
        acquired = client_lock.acquire(blocking=False)
        self.assertFalse(acquired)
        server_lock.release()
        acquired = client_lock.acquire(blocking=False)
        self.assertTrue(acquired)
        self.assertTrue(server_lock.locked())
        client_lock.release()

    def test_shared_queue(self):
        name = "test"
        server_queue = SharedQueue(name, create=True)
        client_queue = SharedQueue(name, create=False)
        server_queue.put(2)
        qsize = server_queue.qsize()
        self.assertEqual(qsize, 1)
        value = server_queue.get()
        self.assertEqual(value, 2)
        client_queue.put(3)
        qsize = client_queue.qsize()
        self.assertEqual(qsize, 1)
        qsize = client_queue.qsize()
        self.assertEqual(qsize, 1)
        value = client_queue.get()
        self.assertEqual(value, 3)
        time.sleep(1)
        self.assertTrue(client_queue.is_available())

    def test_shared_dict(self):
        name = "test"
        server_dict = SharedDict(name=name, create=True)
        client_dict = SharedDict(name=name, create=False)
        new_dict = {"a": 1, "b": 2}
        client_dict.set(new_dict)
        new_dict["a"] = 4
        client_dict.set(new_dict)
        d = server_dict.get()
        self.assertDictEqual(d, new_dict)
        d = client_dict.get()
        self.assertDictEqual(d, new_dict)
        response = SocketResponse(status=ERROR_CODE)
        client_dict._request = mock.MagicMock(return_value=response)
        with self.assertRaises(RuntimeError):
            client_dict.set(new_dict)
        server_dict.unlink()


class SharedMemoryTest(unittest.TestCase):
    def test_unlink(self):
        fanme = "test-shm"
        with self.assertRaises(ValueError):
            shm = SharedMemory(name=fanme, create=True, size=-1)
        with self.assertRaises(ValueError):
            shm = SharedMemory(name=fanme, create=True, size=0)
        shm = SharedMemory(name=fanme, create=True, size=1024)
        shm.buf[0:4] = b"abcd"
        shm.close()
        shm.unlink()
        with self.assertRaises(FileNotFoundError):
            shm = SharedMemory(name=fanme, create=False)
