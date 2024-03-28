import os
import sys
import unittest

import torch
from torch.utils.data import DataLoader

import atorch
from atorch.data import ShmDataloader, create_coworker_shm_context, create_shm_dataloader
from atorch.tests.toy_modules.toy_module import ToyDataset
from atorch.tests.utils.test_utils import run_multi_process_init_distributed, start_coverage, stop_coverage


def api_run_func():
    if torch.cuda.is_available():
        backend = "nccl"
    else:
        backend = "gloo"
    res = atorch.init_distributed(backend, coworker_num_per_node=1)
    assert res

    data_size = 48
    batch_size = 4
    dataset = ToyDataset(data_size)
    dataloader_args = {"batch_size": batch_size, "drop_last": True}

    io_timeout = 5
    initialize_timeout = 15

    if atorch.distributed.is_coworker():
        shm_context = create_coworker_shm_context(
            dataset=dataset,
            dataloader_args=dataloader_args,
            io_timeout=io_timeout,
            initialize_timeout=initialize_timeout,
        )
        dataloader = DataLoader(dataset, **dataloader_args)
        for n in range(2):
            if n > 0:
                shm_context.reset()
            for batch in dataloader:
                shm_context.add_batch([batch])
            shm_context.add_batch(None)
        shm_context.tear_down(master_wait_for_worker=True)
    else:
        dataloader = ShmDataloader(
            dataset, dataloader_args, io_timeout=io_timeout, initialize_timeout=initialize_timeout
        )
        for n in range(2):
            count = 0
            for data in dataloader:
                count += 1
            assert count == data_size // batch_size

    atorch.reset_distributed()


def advance_api_run_func():
    res = atorch.init_distributed("gloo", coworker_num_per_node=1)
    assert res

    data_size = 48
    batch_size = 4
    dataset = ToyDataset(data_size)
    dataloader_args = {"batch_size": batch_size, "drop_last": True}
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    coworker_dataloader_args = {"sampler": sampler, "batch_size": batch_size * 2, "drop_last": True}

    io_timeout = 5
    initialize_timeout = 15
    epoch_num = 2

    def split_batch(data, batch_size, index):
        if isinstance(data, torch.Tensor):
            res = torch.split(data, batch_size)
            res = res[index]
        elif isinstance(data, dict):
            res = {}
            for key in data:
                res[key] = split_batch(data[key], batch_size, index)
        elif isinstance(data, tuple) or isinstance(data, list):
            res = []
            for d in data:
                res.append(split_batch(d, batch_size, index))
            if isinstance(data, tuple):
                res = tuple(res)
        else:
            return None
        return res

    class MyProcess:
        def __init__(
            self,
            dataset,
            dataloader_args,
            training_batch_size=4,
            epoch_num=2,
        ):
            self.dataset = dataset
            self.dataloader_args = dataloader_args
            self.epoch_num = epoch_num
            self.training_batch_size = training_batch_size

        def process_data(self, shm_context):
            dataloader = DataLoader(self.dataset, **self.dataloader_args)
            num_shms = len(shm_context) if type(shm_context) is list else 1
            if num_shms == 1:
                shm_context = [shm_context]
            should_stop = [False for _ in range(num_shms)]
            for _ in range(self.epoch_num):
                if all(should_stop):
                    break
                for batch in dataloader:
                    # split batch into training batch_size
                    if all(should_stop):
                        break
                    batches = [
                        split_batch(batch, self.training_batch_size, i)
                        for i in range(dataloader.batch_size // self.training_batch_size)
                    ]
                    for i in range(num_shms):
                        if should_stop[i]:
                            continue
                        shm_context[i].add_batch(batches)
                        stop_status = shm_context[i].get_stop_status()
                        if any(stop_status):
                            should_stop[i] = True
                            shm_context[i].add_batch(None)
            for i in range(num_shms):
                if not should_stop[i]:
                    shm_context[i].add_batch(None)

    my_process = MyProcess(dataset, coworker_dataloader_args, epoch_num=epoch_num, training_batch_size=4)
    num_s = 2
    dataset = [dataset] * num_s
    dataloader_args = [dataloader_args] * num_s
    shm_data_size = [4] * num_s
    shm_name_prefix = [f"p_{x}" for x in range(num_s)]

    dataloader = create_shm_dataloader(
        dataset,
        dataloader_args,
        coworker_data_process_func=my_process.process_data,
        io_timeout=io_timeout,
        initialize_timeout=initialize_timeout,
        shm_data_size=shm_data_size,
        shm_name_prefix=shm_name_prefix,
        coworker_wait_worker_read=True,
        coworker_wait_worker_read_timeout=10,
    )
    if type(dataloader) is not list:
        dataloader = [dataloader]
    itt = [iter(dataloader[idx]) for idx in range(num_s)]
    count = [0] * num_s
    total_0 = [0] * num_s
    total_1 = [0] * num_s
    doing = True
    bb = [False] * num_s

    while doing and not all(bb):
        for idx in range(num_s):
            if bb[idx]:
                continue
            try:
                data = next(itt[idx])
                count[idx] += 1
                for i in range(batch_size):
                    total_0[idx] += data[0][i][0].item()
                    total_1[idx] += data[1][i][0].item()
                if count[idx] == data_size // batch_size:
                    dataloader[idx].stop()
                    bb[idx] = True
            except StopIteration:
                doing = False
    for idx in range(num_s):
        assert count[idx] == data_size // batch_size
        assert total_0[idx] == data_size * (data_size - 1) // 2
        assert total_1[idx] == data_size
    atorch.reset_distributed()


def mp_run_func():
    res = atorch.init_distributed("gloo")
    assert res
    data_size = 48
    batch_size = 4
    dataset = ToyDataset(data_size)
    dataloader_args = {"batch_size": batch_size, "num_workers": 1, "drop_last": True}
    io_timeout = 5
    initialize_timeout = 15
    dataloader = ShmDataloader(
        dataset,
        dataloader_args,
        rank=atorch.distributed.rank(),
        group_size=atorch.distributed.world_size(),
        io_timeout=io_timeout,
        initialize_timeout=initialize_timeout,
        need_sync_write=False,
        shm_data_size=2,
    )
    for _ in range(2):
        count = 0
        for data in dataloader:
            count += 1
            assert len(data) == 2
            assert isinstance(data[0], torch.Tensor)
            assert isinstance(data[1], torch.Tensor)
        assert count == data_size // batch_size
    assert dataloader.executor is not None or atorch.distributed.rank() > 0
    atorch.reset_distributed()


def run_dist_code(name, nproc=2):
    code_path = os.path.abspath(__file__)
    run_multi_process_init_distributed(nproc=nproc, training_script=code_path, training_script_args=(name,))


class ShmDataLoaderTest(unittest.TestCase):
    def test_multiprocess_api_run(self):
        run_dist_code("api_run")

    @unittest.skipIf(torch.cuda.is_available(), "Skip on gpu as cpu test covers it.")
    def test_multiprocess_advance_api_run(self):
        run_dist_code("advance_api_run")

    @unittest.skipIf(torch.cuda.is_available(), "Skip on gpu as cpu test covers it.")
    def test_mp_multiprocess_api_run(self):
        run_dist_code("mp_run")


if __name__ == "__main__":
    cov_status = start_coverage()
    if sys.argv[1] == "api_run":
        api_run_func()
    elif sys.argv[1] == "advance_api_run":
        advance_api_run_func()
    elif sys.argv[1] == "mp_run":
        mp_run_func()
    if cov_status:
        stop_coverage()
