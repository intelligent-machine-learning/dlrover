import unittest
import json 
from dlrover.python.common.constants import (
    PlatformType,
)
from dlrover.python.scheduler.ray import RayJobArgs
from dlrover.python.master.stats.stats_backend import parse_yaml_file
import os

from dlrover.python.common.constants import (
    DistributionStrategy,
    NodeType,
    PlatformType,
)


class RayJobArgsTest(unittest.TestCase):
    def test_initialize_params(self):
        file = "data/demo.yaml"
        path = os.path.dirname(__file__)
        file_path = os.path.join(path, file)
        data = parse_yaml_file(file_path)
        with open("test.json",'w') as f:
            json.dump(data, f)

        params = RayJobArgs(PlatformType.RAY, "default", "test")
        self.assertEqual(params.job_name, "test")
        self.assertEqual(params.namespace, "default")
        params.initilize()
        self.assertTrue(NodeType.WORKER in params.node_args)
        self.assertTrue(NodeType.PS in params.node_args)
        self.assertTrue(
            params.distribution_strategy
            == DistributionStrategy.PARAMETER_SERVER
        )
        worker_params = params.node_args[NodeType.WORKER]
        self.assertEqual(worker_params.restart_count, 3)
        self.assertEqual(worker_params.restart_timeout, 0)
        self.assertEqual(worker_params.group_resource.count, 0)
 