import copy
import os
import subprocess
import sys
import unittest
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist

import atorch
from atorch.common.util_func import find_free_port, get_ip_address
from atorch.distributed.distributed import ParallelGroupContextManager
from atorch.distributed.distributed import _DistributedContext
from atorch.distributed.distributed import _DistributedContext as dc
from atorch.distributed.distributed import (
    create_parallel_group,
    destroy_parallel_group,
    get_pg_ranks,
    get_ranks_in_same_group,
    init_distributed,
    local_rank,
    parallel_group,
    parallel_group_and_ranks,
    parallel_group_size,
    parallel_rank,
    rank,
    reset_distributed,
    world_size,
)
from atorch.distributed.launch import check_process_loop, parse_args
from atorch.tests.utils.test_utils import (
    elastic_run_multi_process,
    run_multi_process_init_distributed,
    start_coverage,
    stop_coverage,
)


def _test_allreduce(data, names):
    for name in names:
        assert parallel_group_and_ranks(name) is not None
        assert parallel_rank(name) is not None
        assert parallel_group_size(name) is not None
        group = parallel_group(name)
        assert group is not None
        torch.distributed.all_reduce(data, group=group)


def dist_code_for_test(backend, use_atorch_init=True, coworker_num_per_node=0, pg_configs=None):
    if use_atorch_init:
        res = atorch.init_distributed(
            backend, coworker_num_per_node=coworker_num_per_node, set_cuda_device_using_local_rank=True
        )
        if not res:
            raise Exception("init failed")
    else:
        torch.distributed.init_process_group(
            backend,
            init_method="env://",
            world_size=int(os.getenv("WORLD_SIZE")),
            rank=int(os.getenv("RANK")),
        )
        if not torch.distributed.is_initialized():
            raise Exception("init failed")

    if coworker_num_per_node > 0:
        assert atorch.distributed.world_size() == 1
        assert atorch.distributed.rank() == 0
        assert atorch.distributed.is_coworker() or atorch.distributed.local_rank() == 1
        assert atorch.distributed.coworker_local_rank() == 0 or atorch.distributed.local_rank() == 1
        assert atorch.distributed.worker_local_rank() == 0 or atorch.distributed.local_rank() == 0
        assert atorch.distributed.coworker_num_per_node() == 1
        assert atorch.distributed.worker_num_per_node() == 1

    device = f"cuda:{local_rank()}" if torch.cuda.is_available() else "cpu"
    if pg_configs is not None:
        data = torch.ones(1).to(device)
        for idx, config in enumerate(pg_configs):
            parallel_config = (config, None)
            p_names = [p[0] for p in parallel_config[0]]
            if idx > 0:
                prefix_name = f"pg_p_{idx}_"
                with ParallelGroupContextManager(prefix_name):
                    create_parallel_group(parallel_config)
                    assert dc.PG_NAME_PREFIX == prefix_name
                    for name in dc.PARALLEL_RANK:
                        assert name.startswith(prefix_name)
                    for name in dc.PARALLEL_GROUP:
                        assert name.startswith(prefix_name)
                    _test_allreduce(data, p_names)
            else:
                create_parallel_group(parallel_config)
                _test_allreduce(data, p_names)
            destroy_parallel_group()
    elif coworker_num_per_node == 0:
        torch.distributed.barrier()

    if use_atorch_init:
        atorch.reset_distributed()
        assert atorch.distributed.is_coworker() is False
    else:
        torch.distributed.destroy_process_group()


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
    def test_init_distributed(self):
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        res = init_distributed(backend)
        self.assertTrue(res)
        self.assertEqual(world_size(), 1)
        self.assertEqual(rank(), 0)
        self.assertEqual(local_rank(), 0)
        reset_distributed()

    def test_two_process_init_distributed(self):
        run_dist_code("test_basic_dist", nproc=2, use_launch=False)

    def test_two_process_init_distributed_with_coworker(self):
        run_dist_code("test_basic_dist_with_coworker", nproc=2)


@unittest.skipIf(True, "Failed on gpu, accl is not installed")
class DistributedCudaAcclTest(unittest.TestCase):
    def test_init_distributed_accl(self):
        res = init_distributed("accl")
        self.assertTrue(res)
        self.assertEqual(world_size(), 1)
        self.assertEqual(rank(), 0)
        self.assertEqual(local_rank(), 0)
        reset_distributed()

    def test_two_process_init_distributed(self):
        run_dist_code("test_basic_dist_with_accl", nproc=2)


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
        correct_same_group_ranks_10 = [
            {
                "model": [10, 11],
                "pipeline": [8, 10],
                "data": [10, 14],
            },
            {
                "tensor": [8, 9, 10, 11],
                "data": [10, 14],
            },
            {
                "model": [10, 11],
                "sequence": [8, 10],
                "pipeline": [10, 14],
                "data": [2, 10],
            },
        ]
        correct_same_group_ranks_0 = [
            {
                "model": [0, 1],
                "pipeline": [0, 2],
                "data": [0, 4],
            },
            {
                "tensor": [0, 1, 2, 3],
                "data": [0, 4],
            },
            {
                "model": [0, 1],
                "sequence": [0, 2],
                "pipeline": [0, 4],
                "data": [0, 8],
            },
        ]

        for index, slicing_dim in enumerate(dmp_sizes):
            size = np.prod([p[1] for p in slicing_dim])
            _DistributedContext.WORLD_SIZE = size
            rank_order = list(range(size))
            pg_ranks = get_pg_ranks(slicing_dim, rank_order)
            for name in pg_ranks[0]:
                self.assertEqual(pg_ranks[0][name], correct_ranks[index][name])
            ranks_in_same_group = get_ranks_in_same_group((slicing_dim, rank_order), 0)
            self.assertEqual(ranks_in_same_group, correct_same_group_ranks_0[index])
            _DistributedContext.WORLD_SIZE = size * 2
            rank_order = list(range(size * 2))
            pg_ranks = get_pg_ranks(slicing_dim, rank_order)
            for name in pg_ranks[0]:
                self.assertEqual(pg_ranks[0][name], correct_ranks[index][name])
                second_instance_correct_ranks = copy.deepcopy(correct_ranks[index][name])
                for d in second_instance_correct_ranks:
                    for i in range(len(d)):
                        d[i] += size
                self.assertEqual(pg_ranks[1][name], second_instance_correct_ranks)
            ranks_in_same_group = get_ranks_in_same_group((slicing_dim, rank_order), 10)
            self.assertEqual(ranks_in_same_group, correct_same_group_ranks_10[index])
        reset_distributed()

    def test_2_process_create_groups(self):
        run_dist_code("test_pg_dist_with_2node", nproc=2)

    @unittest.skipIf(torch.cuda.is_available() and torch.cuda.device_count() < 4, "Requires 4 gpus")
    def test_4_process_create_groups(self):
        run_dist_code("test_pg_dist_with_4node", nproc=4)

    @unittest.skipIf(torch.cuda.is_available() and torch.cuda.device_count() < 4, "Requires 4 gpus")
    def test_4_process_create_groups_without_atorch_init(self):
        run_dist_code("test_pg_dist_with_4node_without_atorch_init", nproc=4)


def rpc_code(backend):
    res = atorch.init_distributed(backend)
    if not res:
        raise Exception("init failed")
    from atorch.distributed.distributed import coworker_addrs

    port = int(os.getenv("MASTER_PORT2", -1))
    co_addr_ip = coworker_addrs()
    store_port = int(co_addr_ip[1].split(":")[1])
    assert port == store_port
    destroy_parallel_group()


class CoworkerTest(unittest.TestCase):
    def test_one_worker_and_one_coworker(self):
        port = find_free_port()
        ip = get_ip_address()
        cmd = [sys.executable, "-u"]
        path = os.path.abspath(__file__)
        cmd.append(path)
        cmd.append("test_rpc")
        current_env = os.environ.copy()
        common_env = {
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "2",
            "COWORKER_SIZE": "1",
            "MASTER_ADDR": ip,
            "MASTER_PORT2": str(port),
        }
        store = dist.TCPStore(
            ip,
            port,
            is_master=True,
            timeout=timedelta(seconds=90),
        )
        ip_and_port = ip + ":" + str(port)
        store.set(str(1), ip_and_port)
        processes = []
        port = find_free_port()
        common_env["MASTER_PORT"] = str(port)
        current_env.update(common_env)
        process = subprocess.Popen(cmd, env=current_env)
        processes.append((process, None, None))
        check_process_loop(processes)


def sp_all2all_code():
    atorch.init_distributed(set_cuda_device_using_local_rank=True)
    sp_size = atorch.world_size()
    r = atorch.rank()
    s = 4
    b = 2
    head = 8
    dim = 3
    data = torch.arange(s * b * head * dim, dtype=torch.float32).reshape(s, b, head, dim)
    split_data = torch.split(data, s // sp_size, dim=0)
    local_data = split_data[r].to(atorch.local_rank())
    local_data.requires_grad = True

    res = atorch.distributed.seq_all_to_all(local_data, 2, 0, None, sp_size)
    assert list(res.shape) == [s, b, head // sp_size, dim]
    loss = torch.sum(res)
    loss.backward()
    assert local_data.grad is not None
    res = atorch.distributed.seq_all_to_all(local_data, 2, 0)
    assert list(res.shape) == [s, b, head // sp_size, dim]

    x = 2
    y = 6
    z = 8
    w = 10
    offset = atorch.local_rank() * 10
    data = torch.arange(offset, offset + x * y * z * w, dtype=torch.float32).reshape(x, y, z, w).to(atorch.local_rank())

    # todo: check value
    def _check_all2all(s_idx, g_idx):
        res = atorch.distributed.seq_all_to_all(
            data, scatter_idx=s_idx, gather_idx=g_idx, group=None, group_size=sp_size
        )
        expect_shape = [x, y, z, w]
        expect_shape[s_idx] //= sp_size
        expect_shape[g_idx] *= sp_size
        assert list(res.shape) == expect_shape

    _check_all2all(0, 1)
    _check_all2all(0, 2)
    _check_all2all(0, 3)
    _check_all2all(1, 2)
    _check_all2all(1, 3)
    _check_all2all(1, 0)
    _check_all2all(2, 0)
    _check_all2all(2, 1)
    _check_all2all(2, 3)
    _check_all2all(3, 0)
    _check_all2all(3, 1)
    _check_all2all(3, 2)

    destroy_parallel_group()


class SeqParallelTest(unittest.TestCase):
    @unittest.skipIf(torch.cuda.device_count() < 2, "requires 2 gpus")
    def test_sp_all_to_all(self):
        run_dist_code("test_sp_all2all_with_2node", nproc=2)


def run_dist_code(name, nproc=2, use_launch=True):
    code_path = os.path.abspath(__file__)
    if use_launch:
        run_multi_process_init_distributed(nproc=nproc, training_script=code_path, training_script_args=(name,))
    else:
        elastic_run_multi_process(nproc=nproc, training_script=code_path, training_script_args=(name,))


if __name__ == "__main__":
    cov_status = start_coverage()
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if sys.argv[1] == "test_basic_dist":
        dist_code_for_test(backend)
    elif sys.argv[1] == "test_basic_dist_with_coworker":
        dist_code_for_test(backend, coworker_num_per_node=1)
    elif sys.argv[1] == "test_basic_dist_with_accl":
        dist_code_for_test("accl")
    elif sys.argv[1] == "test_pg_dist_with_2node":
        configs = [("model", 2)], [("data", 2)]
        dist_code_for_test(backend, pg_configs=configs)
    elif sys.argv[1] == "test_pg_dist_with_4node":
        configs = [("model", 4)], [("model", 2), ("data", 2)]
        dist_code_for_test(backend, pg_configs=configs)
    elif sys.argv[1] == "test_pg_dist_with_4node_without_atorch_init":
        configs = [("model", 4)], [("pipe", 2), ("data", 2)]
        dist_code_for_test(backend, use_atorch_init=False, pg_configs=configs)
    elif sys.argv[1] == "test_rpc":
        rpc_code(backend)
    elif sys.argv[1] == "test_sp_all2all_with_2node":
        sp_all2all_code()
    if cov_status:
        stop_coverage()
