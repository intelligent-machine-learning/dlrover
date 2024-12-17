# Copyright 2024 The DLRover Authors. All rights reserved.
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

import concurrent.futures
import time
import typing
from typing import Dict, Set

import numpy as np
from tqdm import tqdm


def parallel_job(fn, items, desc, concurrency=32):
    SLEEP = 0.2
    futures: Set[concurrent.futures.Future] = set()
    items: Set[concurrent.futures.Future] = set(items)
    objs = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=concurrency) as e:
        with tqdm(total=len(items), desc=desc) as bar:
            while futures or items:
                done = set()
                added = set()
                for item in items:
                    futures.add(e.submit(fn, item))
                    added.add(item)
                    if len(futures) > concurrency:
                        break
                for future in futures:
                    if future.done():
                        obj = future.result()
                        objs.append(obj)
                        done.add(future)
                        bar.update(1)
                futures -= done
                items -= added
                time.sleep(SLEEP)
    return objs


class GetRankHelper:
    def __init__(self, groups: typing.OrderedDict[str, int]):
        self.world_size = 1
        self.name_to_axis: Dict[str, int] = {}
        order = []
        total_dim = len(groups)
        for index, (group_name, value) in enumerate(groups.items()):
            self.world_size = self.world_size * value
            self.name_to_axis[group_name] = total_dim - index - 1
            order.append(value)
        order = order[::-1]
        self.ranks = np.arange(self.world_size)
        self.ranks = self.ranks.reshape(order)

    def get_ranks(self, group, group_0=False):
        axis = self.name_to_axis[group]
        result = []
        strides = np.array(self.ranks.strides) // self.ranks.itemsize
        shape = self.ranks.shape[axis]
        skip = strides[axis]
        index = [slice(None)] * self.ranks.ndim
        index[axis] = 0
        first = self.ranks[tuple(index)].reshape(-1)
        if group_0:
            return np.array(range(first[0], first[0] + skip * shape, skip))
        for start in first:
            result.append(list(range(start, start + skip * shape, skip)))
        return np.array(result)
