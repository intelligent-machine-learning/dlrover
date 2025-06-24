import ray

from dlrover.python.hybrid.defines import ActorBase


@ray.remote
class Worker(ActorBase):
    def status(self):
        return "Ready"

    def self_check(self):
        """Check the worker itself."""
        print("Worker self check")
        return "Self check passed"

    def start(self):
        """Start the worker. If already started, do nothing."""
        print("Worker started")
        return "Worker started"
