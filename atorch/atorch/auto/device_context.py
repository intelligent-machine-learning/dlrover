import os

import psutil
import torch

from atorch.common.log_utils import default_logger as logger
from atorch.distributed.distributed import _DistributedContext as dc


class DeviceContext(object):
    """
    Device context contains compute resources information below.
        number of nodes
        number of logical cpu cores per node
        gpu model
        gpu memory(B)
        number of gpus per node
        total gpus of the training job
    """

    def __init__(self):
        self._node_num = 1
        self._nproc_per_node = 1
        self._cpu_num_per_node = 0
        self._cpu_memory_per_node = 0
        self._gpu_num_per_node = 0
        self._total_gpu = 0
        self._gpu_model = None
        self._gpu_memory = 0
        self._gpu_compute_capability = None
        self._gpu_multi_processor_count = 0
        self._context = None

    @property
    def context(self):
        """Returns Device Context"""
        if self._context is None:
            logger.warning("Call `detect()` to detect Device Info.")
        return self._context

    # TODO assign bandwidth and fp32_flops according to detected device type
    @property
    def intra_node_bandwidth(self):
        return 1.2e12

    @property
    def inter_node_bandwidth(self):
        return 2.5e9

    @property
    def fp32_flops(self):
        return 19.5e12

    def detect(self):
        self._node_num = self._detect_node_num()
        self._nproc_per_node = self._detect_nproc_per_node()
        self._cpu_num_per_node = self._detect_cpu_num()
        self._cpu_memory_per_node = self._detect_cpu_memory()
        self._gpu_num_per_node = self._detect_gpu_num()
        self._total_gpu = self._gpu_num_per_node * self._node_num
        (
            self._gpu_model,
            self._gpu_compute_capability,
            self._gpu_memory,
            self._gpu_multi_processor_count,
        ) = self._detect_gpu()
        self._context = {
            "node_num": self._node_num,
            "nproc_per_node": self._nproc_per_node,
            "cpu_num_per_node": self._cpu_num_per_node,
            "cpu_memory_per_node": self._cpu_memory_per_node,
            "gpu_num_per_node": self._gpu_num_per_node,
            "total_gpu": self._total_gpu,
            "gpu_model": self._gpu_model,
            "gpu_memory": self._gpu_memory,
            "gpu_compute_capability": self._gpu_compute_capability,
            "gpu_multi_processor_count": self._gpu_multi_processor_count,
        }
        return self._context

    @staticmethod
    def _detect_node_num():
        """Detect the number of nodes of the training job."""
        if dc.INITIALIZED:
            # If job is launched by atorch
            return dc.NODE_SIZE
        elif os.environ.get("WORLD_SIZE", None):
            # get the number nodes from Environment Variable
            return int(os.environ["WORLD_SIZE"])
        else:
            return 1

    @staticmethod
    def _detect_nproc_per_node():
        """Get nproc per node"""
        if dc.INITIALIZED:
            return dc.NPROC_PER_NODE
        else:
            return 1

    @staticmethod
    def _detect_cpu_num():
        """
        Detect the number of logical CPU cores of current node.

        Empirically, os.cpu_count(), psutil.cpu_count() and multiprocessing.cpu_count() may return incorrect cpu number
        when in some Kubernetes containers. So environment variable `LEGACY_CONTAINER_SIZE_CPU_COUNT` is used
        here and os.cpu_count() is used as a backup.
        """
        num_cpu = os.environ.get("LEGACY_CONTAINER_SIZE_CPU_COUNT", None)
        if num_cpu:
            return int(num_cpu)
        else:
            return os.cpu_count()

    @staticmethod
    def _detect_cpu_memory():
        """Detect CPU Memory of current node."""
        cpu_memory = psutil.virtual_memory().total
        return cpu_memory

    @staticmethod
    def _detect_gpu():
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_properties(0)
            compute_capability = "{}.{}".format(gpu.major, gpu.minor)
            return gpu.name, compute_capability, gpu.total_memory, gpu.multi_processor_count
        else:
            return None, None, 0, 0

    @staticmethod
    def _detect_gpu_num():
        """Detect the number of gpus of current node."""
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        else:
            return 0

    def __repr__(self):
        dc_str = ""
        if self._context is None:
            logger.warning("Call `detect()` to detect device info.")
            return dc_str
        for k, v in self._context.items():
            if "memory" in k:
                mb = v / 1024 / 1024
                gb = mb / 1024
                dc_str += f"{k}: {v} B ({mb:.3f} MB, {gb:.3f} GB)\n"
            else:
                dc_str += f"{k}: {v}\n"
        return dc_str[:-1]

    @property
    def node_num(self):
        return self._node_num

    @property
    def nproc_per_node(self):
        return self._nproc_per_node

    @property
    def cpu_num_per_node(self):
        return self._cpu_num_per_node

    @property
    def cpu_memory_per_node(self):
        return self._cpu_memory_per_node

    @property
    def gpu_model(self):
        return self._gpu_model

    @property
    def gpu_num_per_node(self):
        return self._gpu_num_per_node

    @property
    def total_gpu(self):
        return self._total_gpu

    @property
    def gpu_memory(self):
        return self._gpu_memory

    @property
    def gpu_compute_capability(self):
        return self._gpu_compute_capability

    @property
    def gpu_multi_processor_count(self):
        return self._gpu_multi_processor_count


_DEVICE_CONTEXT = None


def get_device_context():
    global _DEVICE_CONTEXT
    if _DEVICE_CONTEXT is None:
        device_context = DeviceContext()
        device_context.detect()
        _DEVICE_CONTEXT = device_context
    return _DEVICE_CONTEXT
