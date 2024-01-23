import json
import os.path
import unittest
from io import BytesIO

import torch
import torchvision.transforms as transforms
from PIL import Image

import atorch
from atorch.common.constants import DataConstants
from atorch.data.dataloader import AtorchDataloader


def transform_item_func(image_bytes, meta_bytes):
    with Image.open(BytesIO(image_bytes)) as img:
        img = transforms.ToTensor()(img)
    return (img, json.load(BytesIO(meta_bytes)))


@unittest.skipIf(not torch.cuda.is_available(), "can not import zdfs in aci image, works in gpu image")
class ChunkDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        import zdfs

        atorch.reset_distributed()
        atorch.init_distributed()
        self.index_file = os.path.join(
            "dfs://unifile_sdk/test-0625",
            "antsys-max-rs_multimodal_TrainJsonl_laion5B_zh_simple_total_pcache_filter_1000000/index.jsonl",
        )
        self.dfs_kwargs = {
            "cluster": "dfs://f-c1f77171wai77.cn-heyuan.dfs.aliyuncs.com:10290",
            "options": zdfs.FileSystemOptions(),
        }

    def test_chunk_dataset(self):
        dataset_options = {
            DataConstants.IS_CHUNK: True,
            DataConstants.JSONL_VERSION: 1,
            DataConstants.DFS_KWARGS: self.dfs_kwargs,
            "buf_size": 10,
            "transform_item_func": transform_item_func,
        }
        dataloader = AtorchDataloader(self.index_file, dataset_options)
        dataloader.dataset._epoch = 0
        num = 0
        for _ in iter(dataloader):
            if num > 10:
                break
            num += 1

    def tearDown(self) -> None:
        atorch.reset_distributed()


if __name__ == "__main__":
    unittest.main()
