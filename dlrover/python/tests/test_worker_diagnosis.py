import unittest
import os
from dlrover.python.elastic_agent.datacollector.cuda_log_collector import (
    CudaLogCollector,
)


def generate_path(path: str) -> str:
    cur_path = os.path.dirname(__file__)
    dir_path = os.path.join(cur_path, path)
    print(f"log dir: {str(dir_path)}\n")
    return str(dir_path)


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

        # path = "data/cuda_logs"
        # collector = CudaLogCollector(path)
        # collector.collect_data()
        # # main_traces = log.get_main_traces()
        # # for rank, trace in main_traces.items():
        # #     print(f"{rank}: {trace}\n")
        # complete_traces = collector.get_all_traces()
        # for trace in complete_traces:
        #     print(f"{trace}\n")
        #     print("-----------------------\n")
