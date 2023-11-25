import os
import socket
import threading
import time
import pickle
from dataclasses import dataclass
from typing import Dict


SOCKER_TEMP_FILE_DIR = "/tmp/checkpoint/"

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
    method: str = ""
    args: Dict[str, object] = None  # type: ignore


@dataclass
class SocketResponse(object):
    status: str = ""


@dataclass
class LockResponse(SocketResponse):
    acquired: bool = False


class SharedLock(object):
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
        self._name = name
        self._lock = threading.Lock()
        fname = "lock_" + self._name + ".sock"
        self._path = os.path.join(SOCKER_TEMP_FILE_DIR, fname)
        self._create = create
        self._server = None
        self._init_socket()

    def _init_socket(self):
        if self._create:
            self._server = _create_socket_server(self._path)
            t = threading.Thread(
                target=self._sync_lock_status, daemon=True,
            )
            t.start()

    def _sync_lock_status(self):
        while True:
            connection, _ = self._server.accept()
            try:
                recv_data = connection.recv(256)
                print(recv_data)
                msg: SocketRequest = pickle.loads(recv_data)
                response = LockResponse()
                if msg.method == "acquire":
                    response.acquired = self.acquire(**msg.args)
                elif msg.method == "release":
                    self.release()
                response.status = SUCCESS_CODE
            except Exception as e:
                print(e)
                response = LockResponse()
                response.status = ERROR_CODE
            send_data = pickle.dumps(response)
            connection.send(send_data)

    def acquire(self, blocking=True):
        """
        Acquire a lock shared by multiple process, blocking or non-blocking.

        Args:
            blocking (bool): blocking or non-blocking.
        """
        if self._create:
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
        if self._create:
            if self._lock.locked():
                self._lock.release()
        else:
            request = SocketRequest(
                method="release",
                args={},
            )
            self._request(request)

    def _request(self, request: SocketRequest):
        for _ in range(3):
            client = _create_socket_client(self._path)
            send_data = pickle.dumps(request)
            client.send(send_data)
            recv_data = client.recv(256)
            client.close()
            response: LockResponse = pickle.loads(recv_data)
            print(response)
            if response.status == SUCCESS_CODE:
                return response
            else:
                time.sleep(1)
                continue
