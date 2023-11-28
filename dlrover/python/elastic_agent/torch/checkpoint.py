import time
import socket
import os
import pickle
from typing import Dict
from dlrover.python.common.socket import CheckpointMeta


class CheckpointBuffer(object):
    """

    Args:
        num_proc (int): Number of workers on the node.
    """
    def __init__(self, num_proc):
        self._num_proc = num_proc
        self._rank_ckpt_metas: Dict[int, CheckpointMeta] = dict()

    def _create_socket_server(self):
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        if os.path.exists("/tmp/checkpoint.sock"):
            os.unlink("/tmp/checkpoint.sock")
        server.bind("/tmp/checkpoint.sock")
        server.listen(0)
        while True:
            try:
                connection, _ = server.accept()
                recv = connection.recv(1024)
                self._deserialize(recv)
                connection.send(b"OK")
            except Exception:
                connection.close()
                break

    def _deserialize(self, buffer):
        obj = pickle.loads(buffer)
        if isinstance(obj, CheckpointMeta):
            self._rank_ckpt_metas[obj.rank] = obj

    def _wait_training_process_init_model(self):
        while True:
            if len(self._rank_ckpt_metas) == self._num_proc:
                break
            time.sleep(1)

    def _create_shared_memory_buffer(self):
        pass

    def save_to_storage(self):
        pass