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
import pickle
import queue
import socket
import threading
import time
import mmap
import _posixshmem
from multiprocessing import shared_memory
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Dict

from .log import default_logger as logger

TMP_DIR = "/tmp"

SUCCESS_CODE = "OK"
ERROR_CODE = "ERROR"


def _create_socket_server(path):
    """
    Create a socket server.

    Args:
        path (str): a file path.
    """
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    path_dir = os.path.dirname(path)
    os.makedirs(path_dir, exist_ok=True)
    if os.path.exists(path):
        os.unlink(path)
    server.bind(path)
    server.listen(0)
    return server


def _create_socket_client(path):
    """
    Create a socket client.

    Args:
        path (str): a file path.

    """
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(path)
    return client


@dataclass
class SocketRequest(object):
    """
    A socket request.

    Attributes:
        method (str): the method name to call.
        args (dict): the arguments of the method.
    """

    method: str = ""
    args: Dict[str, object] = None  # type: ignore


@dataclass
class SocketResponse(object):
    """
    A socket response.

    Attributes:
        status (str): the return code which may be "OK" or "ERROR".
    """

    status: str = ""


@dataclass
class LockAcquireResponse(SocketResponse):
    """
    A response to acquire a shared lock using local socket.

    Attributes:
        acquired (bool): Ture if the lock is acquired.
    """

    acquired: bool = False


class LocalSocketComm(metaclass=ABCMeta):
    """
    Local socket for processes to communicate.

    Args:
        name (str): the instance name which must be unique if multiple
            process share a common object using the local socket.
        create (bool): If ture, the instance creates a socket server
            Otherwise, the instance creates a socket client to access
            the shared object.
    """

    def __init__(self, name="", create=False):
        self._name = name
        self._file = self._create_socket_path()
        self._create = create
        self._server = None
        self._init_socket()

    def __del__(self):
        os.unlink(self._file)

    def _create_socket_path(self):
        """Create a file path for the local socket."""
        fname = self.__class__.__name__.lower() + "_" + self._name + ".sock"
        return os.path.join(TMP_DIR, fname)

    def _init_socket(self):
        """Initialze a socket server."""
        if self._create:
            self._server = _create_socket_server(self._file)
            t = threading.Thread(
                target=self._sync,
                daemon=True,
            )
            t.start()

    @abstractmethod
    def _sync(self):
        """Synchronize the obj between processes."""
        pass

    def _request(self, request: SocketRequest):
        """Create a socket client to requet the shared object."""
        client = _create_socket_client(self._file)
        send_data = pickle.dumps(request)
        client.send(send_data)
        recv_data = client.recv(256)
        client.close()
        response: LockAcquireResponse = pickle.loads(recv_data)
        return response


class SharedLock(LocalSocketComm):
    """
    On a single node, processes can share a lock with an identical name
    via socket-based communication.

    Args:
        name (str): the lock name, processes can share a lock with an
            identical name on a single node.
        create (bool): If ture, the lock creates a socket server and a lock.
            Otherwise, the lock need to create a socket client to access
            the lock.
    """

    def __init__(self, name="", create=False):
        super().__init__(name, create)
        if self._create:
            self._lock = threading.Lock()
        else:
            self._lock = None

    def _sync(self):
        while True:
            connection, _ = self._server.accept()
            try:
                recv_data = connection.recv(256)
                msg: SocketRequest = pickle.loads(recv_data)
                response = LockAcquireResponse()
                if msg.method == "acquire":
                    response.acquired = self.acquire(**msg.args)
                elif msg.method == "release":
                    self.release()
                response.status = SUCCESS_CODE
            except Exception:
                response = LockAcquireResponse()
                response.status = ERROR_CODE
            send_data = pickle.dumps(response)
            connection.send(send_data)

    def acquire(self, blocking=True):
        """
        Acquire a lock shared by multiple process, blocking or non-blocking.

        Args:
            blocking (bool): blocking or non-blocking.
        """
        if self._server:
            return self._lock.acquire(blocking=blocking)
        else:
            request = SocketRequest(
                method="acquire",
                args={"blocking": blocking},
            )
            response = self._request(request)
            if response:
                return response.acquired
            return False

    def release(self):
        """
        Release a lock shared by multiple processes.
        """
        if self._server:
            if self._lock.locked():
                self._lock.release()
        else:
            request = SocketRequest(
                method="release",
                args={},
            )
            self._request(request)


@dataclass
class QueueGetResponse(SocketResponse):
    """
    The response to get an obj from a shared queue using local socket.

    Attributes:
        obj (object): the return value to get an obj from a shared queue.
    """

    obj: object = None


@dataclass
class QueueSizeResponse(SocketResponse):
    """
    The response to get the size of a shared queue using local socket.

    Attributes:
        size (int): the size of a queue.
    """

    size: int = 0


@dataclass
class QueueEmptyResponse(SocketResponse):
    """
    The response to verify a shared queue is empty.

    Attributes:
        empty (bool): True if the queue is empty.
    """

    empty: bool = False


class SharedQueue(LocalSocketComm):
    """
    On a single node, processes can share a queue with an identical name
    via local socket communication.

    Args:
        name (str): the queue name, processes can share the queue with an
            identical name on a single node.
        create (bool): If ture, the instance creates a socket server and a
            queue. Otherwise, the instance need to create a local socket
            client to access the queue.
    """

    def __init__(self, name="", create=False, maxsize=1):
        super().__init__(name, create)
        if self._create:
            self._queue = queue.Queue(maxsize)
        else:
            self._queue = None

    def _sync(self):
        while True:
            connection, _ = self._server.accept()
            try:
                recv_data = connection.recv(256)
                msg: SocketRequest = pickle.loads(recv_data)
                response = SocketResponse()
                if msg.method == "put":
                    self.put(**msg.args)
                elif msg.method == "get":
                    response = QueueGetResponse()
                    response.obj = self.get(**msg.args)
                elif msg.method == "qsize":
                    response = QueueSizeResponse()
                    response.size = self.qsize()
                elif msg.method == "empty":
                    response = QueueEmptyResponse()
                    response.empty = self.empty()
                response.status = SUCCESS_CODE
            except Exception:
                response = SocketResponse()
                response.status = ERROR_CODE
            send_data = pickle.dumps(response)
            connection.send(send_data)

    def put(self, obj, block=True, timeout=None):
        """Put an object into the queue."""
        if self._server:
            self._queue.put(obj, block=block, timeout=timeout)
        else:
            args = {}
            args["obj"] = obj
            args["block"] = block
            args["timeout"] = timeout
            request = SocketRequest(method="put", args=args)
            self._request(request)

    def get(self, block=True, timeout=None):
        """Get an object from the queue."""
        if self._server:
            obj = self._queue.get(block=block, timeout=timeout)
            return obj
        else:
            args = {}
            args["block"] = block
            args["timeout"] = timeout
            request = SocketRequest(method="get", args=args)
            response: QueueGetResponse = self._request(request)
            if response.status == SUCCESS_CODE:
                return response.obj
            return None

    def qsize(self):
        """Get the size of the queue."""
        if self._server:
            return self._queue.qsize()
        else:
            request = SocketRequest(method="qsize", args={})
            response: QueueSizeResponse = self._request(request)
            if response.status == SUCCESS_CODE:
                return response.size
            return -1

    def empty(self):
        """Verify the queue is empty."""
        if self._server:
            return self._queue.empty()
        else:
            request = SocketRequest(method="empty", args={})
            response: QueueEmptyResponse = self._request(request)
            if response.status == SUCCESS_CODE:
                return response.empty
            return False


# The process uses FIFO pipe not the local socket to transfer
# the tensor meta dict. Because, the local socket needs buffers
# at both the sending end and receiving end. The FIFO only need
# one buffer. The size of tensor meta dict may be large. Local socket
# may need double memory buffer size to transfer the dict.
class SharedDict(object):
    """
    A shared dict is used in two processes. One process updates the dict
    and another uses the dict.

    Args:
        name (str): the shared dictionary name, one process can update the
            dict with the same name  of another process by fifo pipe.
        create (bool): If ture, the instance reads the dict from the fifo.
            Otherwist, the instance writes the dict into the fifo.
    """

    def __init__(self, name="", create=False):
        self._name = name
        self._create = create
        fname = self.__class__.__name__.lower() + "_" + self._name + ".fifo"
        self._file = os.path.join(TMP_DIR, fname)
        self._fd = None

        if not os.path.exists(self._file):
            os.mkfifo(self._file, 0o666)
        if self._create:
            self._dict = {}
            self._shared_queue = SharedQueue(
                name=f"shard_dict_{name}", create=self._create
            )
            threading.Thread(
                target=self._sync, daemon=True, name=f"{name}-receiver"
            ).start()
        else:
            self._dict = None
            self._shared_queue = SharedQueue(
                name=f"shard_dict_{name}", create=self._create
            )

    def __del__(self):
        os.unlink(self._file)

    def _sync(self):
        if self._create:
            self._fd = os.open(self._file, os.O_RDONLY)
        while True:
            recv_bytes = os.read(self._fd, 4)
            msg_size = int.from_bytes(recv_bytes, "big")
            total_bytes = b""
            while True:
                buffer_size = 1024 * 1024
                recv_bytes = os.read(self._fd, buffer_size)
                total_bytes += recv_bytes
                if len(total_bytes) == msg_size:
                    break
            d = pickle.loads(total_bytes)
            self._dict.update(d)
            self._shared_queue.get()

    def update(self, new_dict):
        """
        Update the shared Dict with a new Dict.

        Args:
            new_dict (dict): a new dict to update.
        """
        if self._create:
            self._dict.update(new_dict)
        else:
            if not self._fd:
                self._fd = os.open(self._file, os.O_WRONLY)
            bs = pickle.dumps(new_dict)
            bs_size = len(bs)
            try:
                self._shared_queue.put(1)
                # Firstly send the size of the message.
                os.write(self._fd, bs_size.to_bytes(4, "big"))
                os.write(self._fd, bs)
            except Exception:
                logger.info("The recv processs has breakdown.")

    def get(self):
        """
        Returns a Python Dict from the shared Dict.

        If the writing instance sends the dict into the FIFO, the get method
        should wait for the sync thread to update the dict.
        """
        while not self._shared_queue.empty():
            time.sleep(0.1)
        return self._dict


class SharedMemory(shared_memory.SharedMemory):
    """
    Customization of the SharedMemory is necessary, as the
    'resource_tracker.ResourceTracker' in Python will unlink and remove the
    file if one process fails. Our objective is to ensure that the training
    process does not unlink the shared memory upon failure, 
    hereby allowing a new training process to commence utilizing
    the existing shared memory to load checkpoint.

    Note:: We must explicitly unlink the SharedMemory to avoid memory leak.
    """

    # Defaults; enables close() and unlink() to run without errors.
    _name = None
    _fd = -1
    _mmap = None
    _buf = None
    _flags = os.O_RDWR
    _mode = 0o600
    _prepend_leading_slash = True

    def __init__(self, name=None, create=False, size=0):
        if not size >= 0:
            raise ValueError("'size' must be a positive integer")
        if create:
            self._flags = os.O_CREAT | os.O_EXCL | os.O_RDWR
            if size == 0:
                raise ValueError("'size' must be a positive number different from zero")
        if name is None and not self._flags & os.O_EXCL:
            raise ValueError("'name' can only be None if create=True")

        if name is None:
            while True:
                name = shared_memory._make_filename()
                try:
                    self._fd = _posixshmem.shm_open(
                        name,
                        self._flags,
                        mode=self._mode
                    )
                except FileExistsError:
                    continue
                self._name = name
                break
        else:
            name = "/" + name if self._prepend_leading_slash else name
            self._fd = _posixshmem.shm_open(
                name,
                self._flags,
                mode=self._mode
            )
            self._name = name
        try:
            if create and size:
                os.ftruncate(self._fd, size)
            stats = os.fstat(self._fd)
            size = stats.st_size
            self._mmap = mmap.mmap(self._fd, size)
        except OSError:
            self.unlink()
            raise

        self._size = size
        self._buf = memoryview(self._mmap)

    def unlink(self):
        """Requests that the underlying shared memory block be destroyed.

        In order to ensure proper cleanup of resources, unlink should be
        called once (and only once) across all processes which have access
        to the shared memory block."""
        if self._name:
            _posixshmem.shm_unlink(self._name)
