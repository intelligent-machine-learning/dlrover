import time
import unittest

import torch
from torch.utils.data import Dataset, IterableDataset

from atorch.data import UnorderedDataLoader


class _TestDataset(Dataset):
    def __init__(self, data_size=32, sleep_time=0):
        self.size = data_size
        self.sleep_time = sleep_time

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        time.sleep(0.001)
        info = torch.utils.data.get_worker_info()
        if info.id == 1 and self.sleep_time > 0:
            time.sleep(self.sleep_time * 2.5)
        if info.id == 2 and self.sleep_time > 0:
            time.sleep(self.sleep_time)
        return info.id


class _TestIterableDataset(IterableDataset):
    def __init__(self, data_size=32, sleep_time=0):
        self.size = data_size
        self.sleep_time = sleep_time

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            worker_id = 0
        else:  # in a worker process
            worker_id = worker_info.id
        if worker_id == 1:
            time.sleep(self.sleep_time)
        for x in range(self.size):
            yield x + worker_id * 100


class UnorderedDataLoaderTest(unittest.TestCase):
    def test_outof_order(self):
        size = 16
        batch_size = 2
        num_workers = 4
        sleep_time = 0.5
        dataset = _TestDataset(size, sleep_time)
        dataloader = UnorderedDataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

        result = [data for data in dataloader]

        self.assertEqual(len(dataloader), len(result))
        # result in the order of other | 2 | 1
        worker_len = len(result) // num_workers
        id_2_index = len(result) - 2 * worker_len
        total_v = 0
        for batch in result[id_2_index:]:
            for d in batch:
                total_v += d
        self.assertEqual(3 * worker_len * batch_size, total_v)
        self.assertEqual(result[id_2_index][0], 2)
        self.assertEqual(result[-1][-1], 1)

        dataloader = UnorderedDataLoader(
            dataset,
            batch_size=1,
            num_workers=num_workers,
            prefetch_factor=1,
        )

        result = [data for data in dataloader]
        self.assertEqual(len(dataloader), len(result))

        worker_1_count = 0
        worker_2_count = 0
        for res in result:
            if res[0] == 1:
                worker_1_count += 1
            if res[0] == 2:
                worker_2_count += 1
        # worker 1 would only process once as they are slow
        self.assertEqual(worker_1_count, 1)
        # worker 2 is also slow and may process once or twice
        self.assertTrue(worker_2_count <= 2)

    def test_iterable(self):
        size = 16
        batch_size = 2
        num_workers = 2
        sleep_time = 0.5
        dataset = _TestIterableDataset(size, sleep_time)
        dataloader = UnorderedDataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

        result = [data for data in dataloader]
        self.assertEqual(len(result), size // batch_size * num_workers)
