import os
import subprocess
import sys
import tempfile
import unittest
from string import Template

import numpy as np
import torch

from atorch.distributed.distributed import (
    get_pg_ranks,
    init_distributed,
    local_rank,
    rank,
    reset_distributed,
    world_size,
)
from atorch.distributed.launch import parse_args
from atorch.tests.test_utils import run_multi_process_init_distributed

code_template = Template(
    """
import atorch
if __name__ == "__main__":
  res = atorch.init_distributed("$backend", $kargs)
  if not res:
      raise Exception("init failed")
  atorch.reset_distributed()
"""
)

pg_code_template = Template(
    """
import atorch
import torch
import os
from atorch.distributed.distributed import create_parallel_group, destroy_parallel_group, rank
from atorch.distributed.distributed import parallel_group, ParallelGroupContextManager
from atorch.distributed.distributed import parallel_group_and_ranks, parallel_rank, parallel_group_size
from atorch.distributed.distributed import _DistributedContext as dc
if __name__ == "__main__":
  if $use_atorch_init:
    res = atorch.init_distributed("$backend", $kargs)
    if not res:
        raise Exception("init failed")
  else:
    torch.distributed.init_process_group(
        "$backend",
        init_method="env://",
        world_size=int(os.getenv("WORLD_SIZE")),
        rank=int(os.getenv("RANK")),
    )
    if not torch.distributed.is_initialized():
        raise Exception("init failed")
  parallel_config = ($config1, None)
  create_parallel_group(parallel_config)
  p_names = [p[0] for p in parallel_config[0]]
  def _test_allreduce(data, names):
      for name in names:
          assert parallel_group_and_ranks(name) is not None
          assert parallel_rank(name) is not None
          assert parallel_group_size(name) is not None
          group=parallel_group(name)
          assert group is not None
          torch.distributed.all_reduce(data, group=group)
  data = torch.ones(1)
  _test_allreduce(data, p_names)
  if $use_atorch_init:
    destroy_parallel_group()
  parallel_config = [$config2, None]
  prefix_name = "pg_p_"
  with ParallelGroupContextManager(prefix_name):
    create_parallel_group(parallel_config)
    assert(dc.PG_NAME_PREFIX == prefix_name)
    for name in dc.PARALLEL_RANK:
      assert(name.startswith(prefix_name))
    for name in dc.PARALLEL_GROUP:
      assert(name.startswith(prefix_name))
    p_names = [p[0] for p in parallel_config[0]]
    _test_allreduce(data, p_names)
  if $use_atorch_init:
    atorch.reset_distributed()
  else:
    destroy_parallel_group()
    torch.distributed.destroy_process_group()
"""
)

coworker_code_template = Template(
    """
import atorch
if __name__ == "__main__":
  for _ in range(2):
    res = atorch.init_distributed("$backend", coworker_num_per_node=1, $kargs)
    if not res:
        raise Exception("init failed")
    assert atorch.distributed.world_size() == 1
    assert atorch.distributed.rank() == 0
    assert atorch.distributed.is_coworker() or atorch.distributed.local_rank() == 1
    assert atorch.distributed.coworker_local_rank() == 0 or atorch.distributed.local_rank() == 1
    assert atorch.distributed.worker_local_rank() == 0 or atorch.distributed.local_rank() == 0
    assert atorch.distributed.coworker_num_per_node() == 1
    assert atorch.distributed.worker_num_per_node() == 1
    atorch.reset_distributed()
    assert atorch.distributed.is_coworker() is False
"""
)


def get_test_code(backend, kargs=""):
    return code_template.substitute(backend=backend, kargs=kargs)


def get_pg_test_code(use_atorch_init, backend, config1, config2, kargs=""):
    return pg_code_template.substitute(
        use_atorch_init=use_atorch_init,
        backend=backend,
        kargs=kargs,
        config1=config1,
        config2=config2,
    )


def get_coworker_test_code(backend, kargs=""):
    return coworker_code_template.substitute(backend=backend, kargs=kargs)


class ParseArgsTest(unittest.TestCase):
    def test_parse_args(self):
        args = [
            "--nnodes=1",
            "--node_rank=0",
            "--nproc_per_node=8",
            "mycode.py",
            "params for my code",
        ]
        res = parse_args(args)
        self.assertEqual(res.node_rank, 0)
        self.assertEqual(res.nproc_per_node, 8)
        self.assertEqual(res.nnodes, 1)
        self.assertEqual(res.training_script, args[-2])
        self.assertEqual(res.training_script_args, [args[-1]])


class DistributedTest(unittest.TestCase):
    def test_init_distributed_gloo(self):
        res = init_distributed("gloo")
        self.assertTrue(res)
        self.assertEqual(world_size(), 1)
        self.assertEqual(rank(), 0)
        self.assertEqual(local_rank(), 0)
        reset_distributed()

    def test_two_process_init_distributed(self):
        run_multi_process_init_distributed(get_test_code("gloo"))


@unittest.skipIf(
    not torch.cuda.is_available(),
    "No gpu available for cuda and nccl tests",
)
class DistributedCudaTest(unittest.TestCase):
    def test_init_distributed_nccl(self):
        res = init_distributed("nccl")
        self.assertTrue(res)
        self.assertEqual(world_size(), 1)
        self.assertEqual(rank(), 0)
        self.assertEqual(local_rank(), 0)
        reset_distributed()

    def test_two_process_init_distributed(self):
        run_multi_process_init_distributed(get_test_code("nccl", kargs="set_cuda_device_using_local_rank=True"))


@unittest.skipIf(
    not torch.cuda.is_available(),
    "No gpu available for cuda and nccl tests",
)
class ParallelGroupTest(unittest.TestCase):
    def test_ranks_generation(self):
        dmp_sizes = [
            [("model", 2), ("pipeline", 2), ("data", 2)],
            [("tensor", 4), ("data", 2)],
            [("model", 2), ("sequence", 2), ("pipeline", 2), ("data", 2)],
        ]
        correct_ranks = [
            {
                "model": [[0, 1], [2, 3], [4, 5], [6, 7]],
                "pipeline": [[0, 2], [1, 3], [4, 6], [5, 7]],
                "data": [[0, 4], [1, 5], [2, 6], [3, 7]],
            },
            {
                "tensor": [[0, 1, 2, 3], [4, 5, 6, 7]],
                "data": [[0, 4], [1, 5], [2, 6], [3, 7]],
            },
            {
                "model": [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]],
                "sequence": [[0, 2], [1, 3], [4, 6], [5, 7], [8, 10], [9, 11], [12, 14], [13, 15]],
                "pipeline": [[0, 4], [1, 5], [2, 6], [3, 7], [8, 12], [9, 13], [10, 14], [11, 15]],
                "data": [[0, 8], [1, 9], [2, 10], [3, 11], [4, 12], [5, 13], [6, 14], [7, 15]],
            },
        ]

        for index, slicing_dim in enumerate(dmp_sizes):
            size = np.prod([p[1] for p in slicing_dim])
            rank_order = list(range(size))
            pg_ranks = get_pg_ranks(slicing_dim, rank_order)
            for name in pg_ranks:
                self.assertEqual(pg_ranks[name], correct_ranks[index][name])

    def test_multi_process_create_groups(self):
        codes = get_pg_test_code(True, "gloo", [("model", 2)], [("data", 2)])
        run_multi_process_init_distributed(codes, nproc=2)
        codes = get_pg_test_code(True, "gloo", [("model", 4)], [("model", 2), ("data", 2)])
        run_multi_process_init_distributed(codes, nproc=4)
        codes = get_pg_test_code(False, "gloo", [("model", 4)], [("pipe", 2), ("data", 2)])
        run_multi_process_init_distributed(codes, nproc=4)


rpc_test_codes = """
import atorch
import torch.distributed.rpc as rpc

def mul(a, b):
    return a * b

def call_coworker():
    a, b = 2.0, 3.0
    return rpc.rpc_sync('1', mul, args=(a, b))

if __name__ == "__main__":
    res = atorch.init_distributed()
    if not res:
        raise Exception("init failed")
    res = call_coworker()
    rpc.shutdown()
    assert res == 6.0
"""


class RpcTest(unittest.TestCase):
    def test_one_worker_and_one_coworker(self):
        cmd = [sys.executable, "-u"]
        fd, path = tempfile.mkstemp(suffix=".py")
        print("path:", path)
        with open(fd, "w") as f:
            f.write(rpc_test_codes)
        cmd.append(path)
        current_env = os.environ.copy()
        common_env = {
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "2",
            "COWORKER_SIZE": "1",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29500",
            "MASTER_PORT2": "29501",
        }
        processes = []
        for node_rank in range(2):
            common_env["RANK"] = str(node_rank)
            current_env.update(common_env)
            process = subprocess.Popen(cmd, env=current_env)
            processes.append(process)
