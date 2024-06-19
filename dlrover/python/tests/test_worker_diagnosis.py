import unittest
from dlrover.python.elastic_agent.datacollector.cuda_log_collector import (
    CudaLogCollector,
)


class WorkerDiagnosisTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_worker(self):
        path = "/dlrover/cuda_logs"
        collector = CudaLogCollector(path)
        collector.collect_data()
        # main_traces = log.get_main_traces()
        # for rank, trace in main_traces.items():
        #     print(f"{rank}: {trace}\n")
        complete_traces = collector.get_all_traces()
        for trace in complete_traces:
            print(f"{trace}\n")
            print("-----------------------\n")
