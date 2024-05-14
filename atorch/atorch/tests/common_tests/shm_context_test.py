import collections
import os
import sys
import time
import unittest

import numpy as np
import torch

import atorch
from atorch.data import ShmData
from atorch.data.shm_context import ShmDataContext, create_coworker_shm_context, get_sample_batch
from atorch.tests.toy_modules.toy_module import ToyDataset
from atorch.tests.utils.test_utils import (
    create_sample_batch,
    run_multi_process_init_distributed,
    start_coverage,
    stop_coverage,
)


def context_run_func(task_type):
    if torch.cuda.is_available():
        backend = "nccl"
    else:
        backend = "gloo"
    res = atorch.init_distributed(backend)
    assert res
    rank = atorch.rank()

    sample_batch = create_sample_batch()

    if task_type == "coworker":
        num_read_per_batch = 1
        num_batch_per_step = 1
        num_coworker_per_node = 1
        shm_name_prefix = "atorch_test_mp_O1_"

        shm = ShmDataContext(
            is_master=rank == 0,
            sample_batch=sample_batch,
            rank=0 if rank == 0 else rank - 1,
            num_read_per_batch=num_read_per_batch,
            num_batch_per_step=num_batch_per_step,
            num_coworker_per_node=num_coworker_per_node,
            shm_name_prefix=shm_name_prefix,
            io_timeout=5,
        )
        dsize = 2
        if rank == 0:
            batches = [create_sample_batch(value=i + 10, start_v=i + 1) for i in range(dsize)]
            shm.add_batch(batches)
            shm.add_batch(None)
        else:
            for _ in range(dsize):
                batch = shm.get_data(0)
                assert batch is not None
            batch = shm.get_data(0)
            assert batch is None
    else:
        num_read_per_batch = 1
        num_batch_per_step = 1
        shm_name_prefix = "atorch_test_mp_O2_"
        shm = ShmDataContext(
            is_master=rank == 0,
            sample_batch=sample_batch,
            rank=rank,
            num_read_per_batch=num_read_per_batch,
            num_batch_per_step=num_batch_per_step,
            shm_name_prefix=shm_name_prefix,
            io_timeout=5,
        )
        dsize = 2
        if rank == 0:
            batches = [create_sample_batch(value=i + 10, start_v=i + 1) for i in range(dsize)]
            shm.add_batch(batches)
            shm.add_batch(None)
        else:
            for _ in range(dsize):
                batch = shm.get_data(0)
                assert batch is not None
            batch = shm.get_data(0)
            assert batch is None

    shm.tear_down(master_wait_for_worker=True, wait_timeout=15)
    atorch.reset_distributed()


def api_run_func():
    res = atorch.init_distributed("gloo", coworker_num_per_node=1)
    assert res

    dataset = ToyDataset(50)
    dataloader_args = {"batch_size": 4}

    io_timeout = 5
    initialize_timeout = 15

    shm_context = create_coworker_shm_context(
        dataset=dataset, dataloader_args=dataloader_args, io_timeout=io_timeout, initialize_timeout=initialize_timeout
    )

    dsize = 2
    if atorch.distributed.is_coworker():
        batches = get_sample_batch(dataset, dataloader_args, num=dsize)
        shm_context.add_batch(batches)
        shm_context.add_batch(None)
    else:
        for _ in range(dsize):
            batch = shm_context.get_data(0)
            assert batch is not None
        batch = shm_context.get_data(0)
        assert batch is None

    shm_context.tear_down(master_wait_for_worker=True)
    atorch.reset_distributed()


def shm_data_func():
    res = atorch.init_distributed("gloo")
    assert res
    size = 20
    dtype = np.int64

    def check_value(shm, offset, size, values, local_rank=None):
        for i in range(size):
            shm_value = shm.get(offset + i, local_rank=local_rank)
            if shm_value != values[i]:
                return False
        return True

    def print_value(shm, offset, size, values, local_rank=None):
        shm_values = [shm.get(offset + i, local_rank=local_rank) for i in range(size)]
        print(f"shm data: {shm_values}; values: {values}")

    shm_data = ShmData("test_shm_d", size=size, dtype=dtype, per_rank_data=False, initialize_timeout=30)
    torch.distributed.barrier()  # barrier so that rank1 write happens after rank0 reset.
    values = [0 for i in range(size)]
    values[0] = 11
    values[1] = -1
    values[2] = -2
    values[5] = 20
    values[6] = -10
    values[7] = -20
    if atorch.local_rank() == 0:
        shm_data.put(0, 11)  # set [0] = 11
        shm_data.put([1, 3], [-1, -2])  # set[1,2] = [-1, -2]
    else:
        shm_data.put(5, 20)  # set [5] = 20
        shm_data.put([6, 8], [-10, -20])  # set[6,7] = [-10, -20]
    need_check = True
    sleep_count = 30
    while need_check and sleep_count > 0:
        if check_value(shm_data, 0, size, values):
            need_check = False
            break
        time.sleep(1)
        sleep_count -= 1
    if need_check:
        print_value(shm_data, 0, size, values)
    assert not need_check
    shm_data.tear_down()

    shm_data = ShmData("test_shm_per_rank", size=size, dtype=dtype, per_rank_data=True, initialize_timeout=30)
    torch.distributed.barrier()  # barrier so that rank1 write happens after rank0 reset.
    if atorch.local_rank() == 0:
        shm_data.put(0, 10)  # set [0] = 10
        shm_data.put([1, 3], [-1, -2])  # set[1,2] = [-1, -2]
    else:
        shm_data.put(5, 20)  # set [5] = 20
        shm_data.put([6, 8], [-10, -20])  # set[6,7] = [-10, -20]

    values0 = [0 for i in range(size)]
    values0[0] = 10
    values0[1] = -1
    values0[2] = -2
    values1 = [0 for i in range(size)]
    values1[5] = 20
    values1[6] = -10
    values1[7] = -20

    need_check = True
    sleep_count = 30
    while need_check and sleep_count > 0:
        if check_value(shm_data, 0, size, values0, local_rank=0) and check_value(
            shm_data, 0, size, values1, local_rank=1
        ):
            need_check = False
            break
        time.sleep(1)
        sleep_count -= 1
    if need_check:
        print_value(shm_data, 0, size, values0, local_rank=0)
        print_value(shm_data, 0, size, values1, local_rank=1)
    assert not need_check
    shm_data.tear_down()
    atorch.reset_distributed()


def check_equal(x, y):
    if type(x) is not type(y):
        return False
    if isinstance(x, collections.abc.Sequence):
        if len(x) != len(y):
            return False
        for idx in range(len(x)):
            if not check_equal(x[idx], y[idx]):
                return False
    elif isinstance(x, collections.abc.Mapping):
        for key in x:
            if key not in y:
                return False
            if not check_equal(x[key], y[key]):
                return False
    else:
        return torch.equal(x, y)
    return True


class ShmDataContextTest(unittest.TestCase):
    def test_coworker_case(self):
        num_read_per_batch = 1
        num_batch_per_step = 2
        num_coworker_per_node = 2
        shm_name_prefix = "atorch_test_O1_"

        sample_batch = create_sample_batch()

        master_shms = [
            ShmDataContext(
                is_master=True,
                sample_batch=sample_batch,
                rank=r,
                num_read_per_batch=num_read_per_batch,
                num_batch_per_step=num_batch_per_step,
                num_coworker_per_node=num_coworker_per_node,
                shm_name_prefix=shm_name_prefix,
            )
            for r in range(num_coworker_per_node)
        ]

        worker_shms = [
            ShmDataContext(
                is_master=False,
                sample_batch=sample_batch,
                rank=r,
                num_read_per_batch=num_read_per_batch,
                num_batch_per_step=num_batch_per_step,
                num_coworker_per_node=num_coworker_per_node,
                shm_name_prefix=shm_name_prefix,
            )
            for r in range(num_batch_per_step)
        ]

        batches = [create_sample_batch(value=i + 10, start_v=i + 1) for i in range(num_batch_per_step)]
        for shm in master_shms:
            shm.add_batch(batches)
            shm.add_batch(None)

        for d_idx, shm in enumerate(worker_shms):
            for idx in range(num_coworker_per_node):
                batch = shm.get_data(idx)
                self.assertTrue(check_equal(batch, batches[d_idx]))
                batch = shm.get_data(idx)
                self.assertTrue(batch is None)

        m_write_count = master_shms[0].get_write_count()
        m_read_count = master_shms[0].get_read_count()

        for shm in worker_shms:
            w_write_count = master_shms[0].get_write_count(0)
            w_read_count = master_shms[0].get_read_count(0)
            self.assertEqual(m_write_count, w_write_count)
            self.assertEqual(m_read_count, w_read_count)
            self.assertEqual(m_write_count, m_read_count)

        # reset
        for shm in worker_shms:
            shm.reset()

        for shm in master_shms:
            shm.reset()

        for shm in master_shms:
            shm.add_batch(batches)
            shm.add_batch(None)

        for d_idx, shm in enumerate(worker_shms):
            for idx in range(num_coworker_per_node):
                batch = shm.get_data(idx)
                self.assertTrue(check_equal(batch, batches[d_idx]))
                batch = shm.get_data(idx)
                self.assertTrue(batch is None)

        for shm in worker_shms:
            shm.tear_down()

        for shm in master_shms:
            shm.tear_down(master_wait_for_worker=True)

    def test_model_parallel_case(self):
        num_read_per_batch = 3
        num_batch_per_step = 1
        shm_name_prefix = "atorch_test_O2_"

        sample_batch = create_sample_batch()

        master_shm = ShmDataContext(
            is_master=True,
            sample_batch=sample_batch,
            rank=0,
            num_read_per_batch=num_read_per_batch,
            num_batch_per_step=num_batch_per_step,
            shm_name_prefix=shm_name_prefix,
        )

        worker_shms = [
            ShmDataContext(
                is_master=False,
                sample_batch=sample_batch,
                rank=r + 1,
                num_read_per_batch=num_read_per_batch,
                num_batch_per_step=num_batch_per_step,
                shm_name_prefix=shm_name_prefix,
            )
            for r in range(num_read_per_batch)
        ]

        bs = 4
        batches = [create_sample_batch(value=i + 10, start_v=i + 1) for i in range(bs)]
        master_shm.add_batch(batches)
        state_shm = master_shm.shms[0]["state"][1]
        self.assertEqual(state_shm[1], bs)
        self.assertEqual(state_shm[0], 0)
        master_shm.add_batch(None)
        self.assertEqual(state_shm[0], 1)

        for i in range(bs + 1):
            for shm in worker_shms:
                batch = shm.get_data()
                if i < bs:
                    self.assertTrue(check_equal(batch, batches[i]))
                else:
                    self.assertTrue(batch is None)

        for shm in worker_shms:
            shm.tear_down()

        master_shm.tear_down()

    def test_timeout(self):
        num_read_per_batch = 3
        num_batch_per_step = 1
        shm_name_prefix = "atorch_timeout_"

        sample_batch = create_sample_batch()

        master_shm = ShmDataContext(
            is_master=True,
            sample_batch=sample_batch,
            rank=0,
            num_read_per_batch=num_read_per_batch,
            num_batch_per_step=num_batch_per_step,
            shm_name_prefix=shm_name_prefix,
        )

        worker_shm = ShmDataContext(
            is_master=False,
            sample_batch=sample_batch,
            rank=1,
            num_read_per_batch=num_read_per_batch,
            num_batch_per_step=num_batch_per_step,
            shm_name_prefix=shm_name_prefix,
            io_timeout=2,
        )

        self.assertRaises(TimeoutError, worker_shm.get_data)
        master_shm.tear_down()
        worker_shm.tear_down()

        def create_worker_shm():
            return ShmDataContext(
                is_master=False,
                sample_batch=sample_batch,
                rank=1,
                num_read_per_batch=num_read_per_batch,
                num_batch_per_step=num_batch_per_step,
                shm_name_prefix=shm_name_prefix,
                initialize_timeout=2,
            )

        self.assertRaises(TimeoutError, create_worker_shm)

    def test_coworker_run(self):
        run_dist_code("coworker_test")

    def test_mp_run(self):
        run_dist_code("mp_test")

    @unittest.skipIf(torch.cuda.is_available(), "Skip on gpu as cpu test covers it.")
    def test_api_run(self):
        run_dist_code("api_test")

    @unittest.skipIf(torch.cuda.is_available(), "Skip on gpu as cpu test covers it.")
    def test_shm_data_run(self):
        run_dist_code("shm_data")


def run_dist_code(name, nproc=2):
    code_path = os.path.abspath(__file__)
    run_multi_process_init_distributed(nproc=nproc, training_script=code_path, training_script_args=(name,))


if __name__ == "__main__":
    cov_status = start_coverage()
    if sys.argv[1] == "coworker_test":
        context_run_func("coworker")
    elif sys.argv[1] == "mp_test":
        context_run_func("mp")
    elif sys.argv[1] == "api_test":
        api_run_func()
    elif sys.argv[1] == "shm_data":
        shm_data_func()
    if cov_status:
        stop_coverage()
