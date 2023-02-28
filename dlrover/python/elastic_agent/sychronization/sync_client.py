
import time

from dlrover.python.elastic_agent.master_client import GlobalMasterClient


class SyncClient(object):
    def __init__(self):
        self._master_client = GlobalMasterClient.MASTER_CLIENT

    def join_sync(self, sync_name):
        """Join a synchronizationg group."""
        for _ in range(3):
            res = self._master_client.join_sync(sync_name)
            if res.success:
                return res.success
            time.sleep(3)

    def wait_sync_finished(self, sync_name, timeout=0):
        """Wait that the synchronized group finishes when all running
        workers joins the group.
        Input:
            sync_name: a name (string) for this sync
            timeout: if not None, timeout in seconds
        Return:
            True if sync successfully. False if timeout.
        """
        while True:
            res = self._master_client.sync_finished(sync_name)
            if res.success:
                return res.success
            time.sleep(1)
            if timeout > 0:
                timeout -= 1
                if timeout <= 0:
                    return False

    def barrier(self, barrier_name, timeout=0):
        """Wait that a barrier is notifed. For example, workers call
        the method to wait that the chief notifies the barrier.
        Input:
            sync_name: a name (string) for this sync
            timeout: if not None, timeout in seconds if notify=False
        """
        while True:
            res = self._master_client.barrier(barrier_name)
            if res.success:
                return res.success
            time.sleep(1)
            if timeout > 0:
                timeout -= 1
                if timeout <= 0:
                    return False

    def notify_barrier(self, barrier_name):
        """Notify a barrier is finished. All worker will get Ture
        by calling `barrier` after the barrier is notified."""
        for _ in range(3):
            res = self._master_client.barrier(barrier_name, True)
            if res.success:
                return res.success
            time.sleep(3)
