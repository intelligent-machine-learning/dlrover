import unittest

from dlrover.python.common.constants import NodeType, NodeStatus
from dlrover.python.master.monitor.speed_monitor import SpeedMonitor
from dlrover.python.master.elastic_training.sync_service import SyncService
from dlrover.python.tests.test_utils import MockK8sJobArgs, mock_k8s_client
from dlrover.python.master.node.job_manager import create_job_manager


class SyncServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        mock_k8s_client()
        params = MockK8sJobArgs()
        params.node_args
        params.initilize()
        self.job_manager = create_job_manager(params, SpeedMonitor())
        self.job_manager._init_nodes()
        
    def test_sync(self):
        sync_service = SyncService(self.job_manager)
        for node in self.job_manager._job_nodes[NodeType.CHIEF].values():
            node.status = NodeStatus.RUNNING

        for node in self.job_manager._job_nodes[NodeType.WORKER].values():
            node.status = NodeStatus.RUNNING
        
        sync_name = "sync-0"
        for node in self.job_manager._job_nodes[NodeType.CHIEF].values():
            sync_service.join_sync(sync_name, node.type, node.id)
        finished = sync_service.sync_finished(sync_name)
        self.assertFalse(finished)

        for node in self.job_manager._job_nodes[NodeType.WORKER].values():
            sync_service.join_sync(sync_name, node.type, node.id)
        finished = sync_service.sync_finished(sync_name)
        self.assertTrue(finished)

    def test_barrier(self):
        mock_k8s_client()
        sync_service = SyncService(self.job_manager)
        barrier_name = "barrier-0"
        finished = sync_service.barrier(barrier_name)
        self.assertFalse(finished)
        sync_service.notify_barrier(barrier_name)
        finished = sync_service.barrier(barrier_name)
        self.assertTrue(finished)
