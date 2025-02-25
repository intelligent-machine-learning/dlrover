import unittest
from dlrover.python.elastic_agent.tensorflow.hooks import (
    ReportModelInfoHook,
)
from dlrover.python.elastic_agent.master_client import (
    MasterClient,
    build_master_client,
)
from dlrover.python.tests.test_utils import start_local_master


class TestHooks(unittest.TestCase):
    def setUp(self):
        self._master, self.addr = start_local_master()
        MasterClient._instance = build_master_client(self.addr, 1)

    def tearDown(self):
        self._master.stop()

    def test_ReportModelInfoHook(self):
        hook = ReportModelInfoHook()

        hook._is_chief = True