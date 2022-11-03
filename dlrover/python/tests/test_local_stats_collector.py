import unittest

from dlrover.python.common.resource import NodeResource
from dlrover.python.master.stats_collector.local_collector import (
    LocalStatsCollector,
)


class ALocalStatsCollectorTest(unittest.TestCase):
    def test_report_resource_usage(self):
        collector = LocalStatsCollector("123")
        resource = NodeResource(cpu=1, memory=1024)
        worker0_name = "worker-0"
        worker1_name = "worker-1"
        collector.collect_node_resource_usage(worker0_name, resource)
        collector.report_resource_usage()
        collector.collect_node_resource_usage(worker0_name, resource)
        collector.collect_node_resource_usage(worker1_name, resource)
        collector.report_resource_usage()
        self.assertEqual(len(collector._all_node_resources), 2)
        self.assertEqual(len(collector._all_node_resources[worker0_name]), 2)
        self.assertEqual(len(collector._all_node_resources[worker1_name]), 1)
