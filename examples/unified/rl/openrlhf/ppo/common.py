import threading

from dlrover.python.unified.api.runtime.rpc import rpc

from . import remote_call


class BaseActor:
    def __init__(self) -> None:
        self.end_event = threading.Event()

    @rpc(remote_call.end_job)
    def end_job(self):
        self.end_event.set()

    def run(self):
        self.end_event.wait()
