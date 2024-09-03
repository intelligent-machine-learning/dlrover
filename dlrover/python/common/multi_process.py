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

import mmap
import os
import pickle
import queue
import shutil
import socket
import threading
import time
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from multiprocessing import shared_memory
from typing import Dict

import _posixshmem

from .constants import NodeEnv
from .log import default_logger as logger

SOCKET_TMP_DIR = "/tmp/ckpt_sock/"

SUCCESS_CODE = "OK"
ERROR_CODE = "ERROR"


def retry_socket(func):
    def wrapper(self, *args, **kwargs):
        retry = kwargs.get("retry", 30)
        succeed = False
        for i in range(retry):
            try:
                result = func(self, *args, **kwargs)
                succeed = True
                return result
            except (FileNotFoundError, ConnectionRefusedError):
                time.sleep(1)
        if not succeed:
            return func(self, *args, **kwargs)

    return wrapper


def clear_sock_dir():
    shutil.rmtree(SOCKET_TMP_DIR, ignore_errors=True)


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

    logger.info(f"Creating socket server at {path}.")
    server.bind(path)
    server.listen(0)
    return server


def _create_socket_client(path):
    """
    Create a socket client.

    Args:
        path (str): a file path.

    """

    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(path)
    except Exception as e:
        logger.warning(
            "Unexpected error when creating socket client by "
            f"path: {path}, error: {e}"
        )
        raise e
    return client


def _socket_send(socket: socket.socket, message):
    """
    In the protocol, the first 4 bytes is the size of message.
    """
    head = len(message).to_bytes(4, "big")
    send_data = head + message
    socket.send(send_data)


def _socket_recv(socket: socket.socket):
    """
    In the protocol, the first 4 bytes is the size of message.
    """
    recv_data = socket.recv(1024)
    head = recv_data[0:4]
    message = recv_data[4:]
    message_len = int.from_bytes(head, "big")
    while len(message) < message_len:
        recv_data = socket.recv(1024)
        message += recv_data
    return message


@dataclass
class SocketRequest(object):
    """
    A socket request.

    Attributes:
        method (str): the method name to call.
        args (dict): the arguments of the method.
    """

    method: str = ""
    args: Dict[str, object] = field(default_factory=dict)


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


@dataclass
class LockedResponse(SocketResponse):
    """
    A response to acquire the status of a lock.

    Attributes:
        locked (bool): Ture if the lock is locked.
    """

    locked: bool = False


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
        logger.info(
            f"Initialize(create:{create}) {self.__class__.__name__.lower()} "
            f"for {name}"
        )
        self._name = name
        self._socket_file = self._create_socket_path()
        self._create = create
        self._server = None
        self._init_socket()

    def unlink(self):
        try:
            os.unlink(self._socket_file)
        except FileNotFoundError:
            pass

    def _create_socket_path(self):
        """Create a file path for the local socket."""
        fname = self.__class__.__name__.lower() + "_" + self._name + ".sock"
        job_name = os.getenv(NodeEnv.TORCHELASTIC_RUN_ID, "")
        if job_name:
            root_dir = os.path.join(SOCKET_TMP_DIR, job_name)
        else:
            root_dir = SOCKET_TMP_DIR
        os.makedirs(root_dir, exist_ok=True)
        return os.path.join(root_dir, fname)

    def _init_socket(self):
        """Initialize a socket server."""
        if self._create:
            self._server = _create_socket_server(self._socket_file)
            t = threading.Thread(
                target=self._sync,
                daemon=True,
            )
            t.start()

    @abstractmethod
    def _sync(self):
        """Synchronize the obj between processes."""
        pass

    @retry_socket
    def _request(self, request: SocketRequest):
        """Create a socket client to request the shared object."""
        client = _create_socket_client(self._socket_file)
        message = pickle.dumps(request)
        _socket_send(client, message)
        recv_data = _socket_recv(client)
        client.close()
        response: LockAcquireResponse = pickle.loads(recv_data)
        return response

    def is_available(self):
        try:
            return os.path.exists(self._socket_file)
        except Exception:
            return False


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
                recv_data = _socket_recv(connection)
                msg: SocketRequest = pickle.loads(recv_data)
                if msg.method == "acquire":
                    response = LockAcquireResponse()
                    response.acquired = self.acquire(**msg.args)
                elif msg.method == "locked":
                    response = LockedResponse()
                    response.locked = self.locked()
                elif msg.method == "release":
                    self.release()
                response.status = SUCCESS_CODE
            except Exception:
                response = SocketResponse()
                response.status = ERROR_CODE
            send_data = pickle.dumps(response)
            _socket_send(connection, send_data)

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
            try:
                response = self._request(request)
                if response.status == SUCCESS_CODE:
                    return response.acquired
            except Exception as e:
                logger.warning(
                    f"Failed to acquire lock due to unexpected error: {e}"
                )
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

    def locked(self):
        if self._server:
            return self._lock.locked()
        else:
            request = SocketRequest(
                method="locked",
                args={},
            )
            return self._request(request)


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

    @property
    def queue(self):
        return self._queue

    def _sync(self):
        while True:
            connection, _ = self._server.accept()
            try:
                recv_data = _socket_recv(connection)
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
            message = pickle.dumps(response)
            _socket_send(connection, message)

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


@dataclass
class DictMessage(SocketResponse):
    """
    The response to get the dict of shared dict using local socket.

    Attributes:
        meta_dict (dict): the return value to get an obj from a shared queue.
    """

    meta_dict: object = None


class SharedDict(LocalSocketComm):
    """
    A shared dict between local processes.

    Args:
        name (str): the shared dictionary name, one process can update the
            dict with the same name of another process by local socket.
        create (bool): If ture, the instance will receive the dict from the
            sending process to update its dict.
    """

    def __init__(self, name="", create=False):
        super().__init__(name, create)
        self._dict = {}

        # The queue is used to notify the saver waiting for a new dict.
        self._shared_queue = SharedQueue(
            name=f"shard_dict_{name}", create=self._create
        )

    def _sync(self):
        while True:
            connection, _ = self._server.accept()
            try:
                recv_data = _socket_recv(connection)
                msg: SocketRequest = pickle.loads(recv_data)
                response = DictMessage()
                if msg.method == "set":
                    self.set(**msg.args)
                elif msg.method == "get":
                    response = DictMessage()
                    response.meta_dict = self.get(**msg.args)
                response.status = SUCCESS_CODE
            except Exception as e:
                response = SocketResponse()
                response.status = ERROR_CODE
                logger.error(e)
            finally:
                if not self._shared_queue.empty():
                    self._shared_queue.get(1)
            message = pickle.dumps(response)
            _socket_send(connection, message)

    def set(self, new_dict):
        """
        Set the dict to the remote shared dict.

        Args:
            new_dict (dict): a new dict to set.
        """
        self._dict = new_dict
        if not self._server:
            args = {"new_dict": self._dict}
            request = SocketRequest(method="set", args=args)
            self._shared_queue.put(1)
            response = self._request(request)
            if response.status == ERROR_CODE:
                raise RuntimeError("Fail to set metadata!")

    def get(self, local=False):
        """
        Returns a Python Dict from the remote shared Dict.

        If the writing instance sends the dict into the FIFO, the get method
        should wait for the sync thread to update the dict.

        Args:
            local (bool): If true, returns the local dict.
        """
        if local:
            return self._dict
        if self._server:
            while not self._shared_queue.empty():
                time.sleep(0.1)
            return self._dict
        else:
            request = SocketRequest(method="get", args={})
            response: DictMessage = self._request(request)
            if response.status == SUCCESS_CODE:
                self._dict = response.meta_dict
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
                raise ValueError(
                    "'size' must be a positive number different from zero"
                )
        if name is None and not self._flags & os.O_EXCL:
            raise ValueError("'name' can only be None if create=True")

        if name is None:
            while True:
                name = shared_memory._make_filename()
                try:
                    self._fd = _posixshmem.shm_open(
                        name, self._flags, mode=self._mode
                    )
                except FileExistsError:
                    continue
                self._name = name
                break
        else:
            name = "/" + name if self._prepend_leading_slash else name
            self._fd = _posixshmem.shm_open(name, self._flags, mode=self._mode)
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
            try:
                _posixshmem.shm_unlink(self._name)
            except FileNotFoundError:
                pass
            logger.info(f"Unlink the shared memory {self._name}")
