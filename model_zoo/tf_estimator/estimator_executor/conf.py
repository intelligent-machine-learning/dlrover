# Copyright 2023 The DLRover Authors. All rights reserved.
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


import numpy as np
from MyEstimator import MyEstimator

from dlrover.python.elastic_agent.sharding.client import ShardingClient
from dlrover.trainer.tensorflow.util.column_info import Column
from dlrover.trainer.tensorflow.util.reader_util import reader_registery
from dlrover.trainer.tensorflow.reader.base_reader import (   
    ElasticReader,
)

#  define your own custome reader
def build_data_shard_service(
    batch_size=1,
    num_epochs=1,
    dataset_size=1,
    num_minibatches_per_shard=1,
    dataset_name="iris_training_data",
):
    sharding_client = ShardingClient(
        dataset_name=dataset_name,
        batch_size=batch_size,
        num_epochs=num_epochs,
        dataset_size=dataset_size,
        num_minibatches_per_shard=num_minibatches_per_shard,
    )
    return sharding_client


class FakeReader(ElasticReader):
    def __init__(
        self,
        path=None,
        num_epochs=1,
        batch_size=64,
        enable_dynamic_sharding=True,
    ):
 
        super().__init__(
            path=path,
            num_epochs=num_epochs,
            batch_size=batch_size,
            enable_dynamic_sharding=enable_dynamic_sharding,
        )

    def count_data(self):
        self._data_nums = 10000

    def get_default_shard(self):
        return 1

    def _read_data(self):
        shard = None
        if self.data_shard_client is not None:
            shard = self.data_shard_client.fetch_shard()
        data = self.get_data_by_shard(shard)
        if shard is None:
            data = None
        return data

    def get_data_by_shard(self, shard):
        x = np.random.randint(1, 1000)
        y = 2 * x + np.random.randint(1, 5)
        return "{},{}".format(x, y)

    def iterator(self):
        while True:
            data = self._read_data()
            if data is None:
                break
            yield data


# register reader
reader_registery.register_reader("fake", FakeReader)


def compare_fn(prev_eval_result, cur_eval_result):
    return True, {"fake_metric": 0.9}


class TrainConf(object):
    classifier_class = MyEstimator
    batch_size = 3
    epoch = 20
    log_steps = 10
    eval_steps = 10
    save_steps = 10

    model_dir = "/nas"
    params = {
        "deep_embedding_dim": 8,
        "learning_rate": 0.0001,
        "l1": 0.0,
        "l21": 0.0,
        "l2": 0.0,
        "optimizer": "group_adam",
        "log_steps": 100,
    }

    train_set = {
        "path": "fake://test.data",
        "epoch": 1000,
        "batch_size": 200,
        "columns": (
            Column.create(  # type: ignore
                name="x",
                dtype="float32",
                is_label=False,
            ),
            Column.create(  # type: ignore
                name="y",
                dtype="float32",
                is_label=True,
            ),
        ),
    }

    eval_set = {"path": "fake://eval.data", "columns": train_set["columns"]}
