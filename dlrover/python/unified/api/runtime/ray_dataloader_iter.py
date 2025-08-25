# Copyright 2025 The DLRover Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import wraps

import ray
from torch.utils.data.dataloader import (
    DataLoader,
    _BaseDataLoaderIter,
    _DatasetKind,
    _utils,
)

from dlrover.python.common.log import default_logger as logger


class RemoteDatasetFetcher:
    def __init__(self, fetcher):
        self._fetcher = fetcher

    def fetch(self, possibly_batched_index):
        # This method should be implemented to fetch data from the remote dataset.
        # For simplicity, we assume it returns a list of data items.
        return self._fetcher.fetch(possibly_batched_index)


class RayDataLoaderIter(_BaseDataLoaderIter):
    """RayDataLoaderIter is a custom DataLoader iterator that uses Ray actors
    to fetch data from the dataset. It is designed to work with Ray's distributed
    data loading capabilities."""

    def __init__(self, loader):
        super().__init__(loader)

        self._dataset_fetcher = _DatasetKind.create_fetcher(
            self._dataset_kind,
            self._dataset,
            self._auto_collation,
            self._collate_fn,
            self._drop_last,
        )
        self._prefetch_factor = loader.prefetch_factor or 0
        logger.info(
            f"Use RayDataLoaderIter with "
            f"dataset_kind={self._dataset_kind}, "
            f"prefetch_factor={self._prefetch_factor}"
        )
        if self._prefetch_factor == 0:
            logger.warning(
                "prefetch_factor is set to 0, which may lead to lower performance. "
                "set it to a higher value for better performance. "
                "(you may need to set num_workers>0 as well)"
            )
        self._prefetch_cache = []
        self._actor = ray.remote(RemoteDatasetFetcher).remote(
            self._dataset_fetcher
        )

    def _next_data(self):
        while len(self._prefetch_cache) <= self._prefetch_factor:
            try:
                index = self._next_index()
            except StopIteration:
                if len(self._prefetch_cache) == 0:
                    raise
                break  # end, but still have prefetched data
            self._prefetch_cache.append(self._actor.fetch.remote(index))
        data = ray.get(self._prefetch_cache.pop(0))
        if self._pin_memory:
            data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)
        return data


def patch_dataloader_ray():
    """Patch the DataLoader to use RayDataLoaderIter.

    This allows DataLoader to use Ray actors for data fetching, enabling
    distributed data loading in Ray clusters.
    """

    @wraps(DataLoader._get_iterator)
    def get_iterator(self, *args, **kwargs):
        return RayDataLoaderIter(self)

    DataLoader._get_iterator = get_iterator
