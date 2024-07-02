import unittest
import os
from dlrover.python.elastic_agent.datacollector.cuda_log_collector import (
    CudaLogCollector,
)
from dlrover.python.common.diagnosis import DiagnosisDataType
from dlrover.python.tests.test_utils import generate_path


class WorkerDiagnosisTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_cuda_log_collector(self):
        path = generate_path("data")
        collector = CudaLogCollector(path)
        self.assertFalse(collector.to_collect_data())

        path = generate_path("data/cuda_logs/")
        collector = CudaLogCollector(path)
        self.assertTrue(collector.to_collect_data())

        cuda_log = collector.collect_data()
        self.assertEqual(cuda_log.get_type(), DiagnosisDataType.CUDALOG)
        traces = cuda_log.get_traces()
        self.assertEqual(len(traces), 7)
        rank_str = cuda_log.format_rank_trace()
        self.assertTrue("0-1" in rank_str[0])


