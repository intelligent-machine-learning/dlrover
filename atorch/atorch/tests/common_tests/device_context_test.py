import unittest

import torch

from atorch.auto.device_context import DeviceContext


class DeviceContextTest(unittest.TestCase):
    def test_device_cpu(self):
        dc = DeviceContext()
        context = dc.detect()
        self.assertGreater(context["cpu_num_per_node"], 0)
        self.assertGreater(context["cpu_memory_per_node"], 0)

    @unittest.skipIf(not torch.cuda.is_available(), "Skip for non gpu device.")
    def test_device_gpu(self):
        dc = DeviceContext()
        context = dc.detect()
        self.assertGreater(context["total_gpu"], 0)
        self.assertGreater(context["gpu_memory"], 0)
