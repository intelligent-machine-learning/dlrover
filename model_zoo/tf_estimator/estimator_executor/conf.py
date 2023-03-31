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

from dlrover.trainer.tensorflow.reader.base_reader import ElasticReader
from dlrover.trainer.tensorflow.util.column_info import Column

#  define your own custome reader


class FakeReader(ElasticReader):
    def __init__(self, path=None):
        self.count = 1
        super().__init__(path=path)

    def count_data(self):
        self._data_nums = 10

    def read_data_by_index_range(self, start_index, end_index):
        data = []
        for i in range(start_index, end_index):
            x = np.random.randint(1, 1000)
            y = 2 * x + np.random.randint(1, 5)
            d = "{}\t{}".format(x, y)
            data.append(d)
        return data


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
        "reader": FakeReader("fake://test.data"),
        "epoch": 10,
        "field_delim": "\t",
        "batch_size": 20,
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
    eval_set = {
        "reader": FakeReader("fake://eval.data"),
        "columns": train_set["columns"],
        "field_delim": "\t",
    }
