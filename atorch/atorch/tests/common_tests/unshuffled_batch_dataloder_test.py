import unittest

from torch.utils.data import Dataset

from atorch.data.unshuffled_batch_dataloader import DistributedUnshuffledBatchSampler


class _TestDataset(Dataset):
    def __init__(self, data_size=32):
        self.size = data_size
        self.data = [i for i in range(data_size)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


class DistributedUnshuffledBatchSamplerTest(unittest.TestCase):
    def test_unshuffled_batch_sampler(self):
        dataset = _TestDataset(data_size=64)
        num_replicas = 8
        rank = 0
        batch_size = 4
        indices = [0, 1, 2, 3, 32, 33, 34, 35]
        sampler = DistributedUnshuffledBatchSampler(
            dataset, num_replicas=num_replicas, rank=rank, batch_size=batch_size
        )
        res = []
        for i in sampler:
            res.append(i)
        self.assertListEqual(res, indices)


if __name__ == "__main__":
    unittest.main()
