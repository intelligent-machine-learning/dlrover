import random
import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from atorch.data import GpuPreLoader


class _TestDataset(Dataset):
    def __init__(self, data_size=32):
        self.size = data_size
        self.a = 0.1
        self.b = 0.3

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = torch.tensor([random.random()])
        return x, self.a * x + self.b


class _TestModel(torch.nn.Module):
    def __init__(self):
        super(_TestModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        out = self.linear(x)
        return out


class PreLoadererTest(unittest.TestCase):
    def setUp(self):
        self.size = 16
        self.batch_size = 2

        dataset = _TestDataset(self.size)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size)
        self.model = _TestModel()

    def test_preloader_cpu(self):
        device = "cpu"

        dataloader = GpuPreLoader(self.dataloader, device=device)
        self.assertEqual(len(dataloader), self.size / self.batch_size)

        for _ in range(2):
            for it, data in enumerate(dataloader):
                for d in data:
                    self.assertEqual(d.device, torch.device(device))
            self.assertEqual(it, len(dataloader) - 1)

    def test_preloader_cpu_with_post_processing(self):
        device = "cpu"
        model = _TestModel()

        def process(data):
            return model(data[0])

        dataloader = GpuPreLoader(self.dataloader, device=device, post_processing=process)
        self.assertEqual(len(dataloader), self.size / self.batch_size)

        for data, process_output in dataloader:
            for d in data:
                self.assertEqual(d.device, torch.device(device))

            self.assertEqual(process_output.device, torch.device(device))

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "No gpu available for cuda tests",
    )
    def test_preloader_cuda(self):
        device = "cuda:0"

        dataloader = GpuPreLoader(self.dataloader, device=device)
        self.assertEqual(len(dataloader), self.size / self.batch_size)

        for _ in range(2):
            for it, data in enumerate(dataloader):
                for d in data:
                    self.assertEqual(d.device, torch.device(device))
            self.assertEqual(it, len(dataloader) - 1)

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "No gpu available for cuda tests",
    )
    def test_preloader_cuda_with_mask(self):
        device = "cuda:0"
        mask = [True, False]

        dataloader = GpuPreLoader(self.dataloader, device=device, mask=mask)
        self.assertEqual(len(dataloader), self.size / self.batch_size)

        for data in dataloader:
            self.assertEqual(data[0].device, torch.device(device))
            self.assertEqual(data[1].device, torch.device("cpu"))

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "No gpu available for cuda tests",
    )
    def test_preloader_cuda_manual_preload(self):
        device = "cuda:0"

        dataloader = GpuPreLoader(self.dataloader, device=device, manual_preload=True)
        self.assertEqual(len(dataloader), self.size / self.batch_size)

        half_size = int(self.size / self.batch_size / 2)
        for idx, data in enumerate(dataloader):
            for d in data:
                self.assertEqual(d.device, torch.device(device))
            if idx >= half_size:
                dataloader.preload()
                self.assertTrue(dataloader.preloaded)
            else:
                self.assertFalse(dataloader.preloaded)

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "No gpu available for cuda tests",
    )
    def test_preloader_gpu_with_post_processing(self):
        device = "cuda:0"
        model = _TestModel()
        model.to(device)

        def process(data):
            return model(data[0])

        dataloader = GpuPreLoader(self.dataloader, device=device, post_processing=process)
        self.assertEqual(len(dataloader), self.size / self.batch_size)

        for data, process_output in dataloader:
            for d in data:
                self.assertEqual(d.device, torch.device(device))

            dataloader.wait_post_processing()
            self.assertEqual(process_output.device, torch.device(device))
