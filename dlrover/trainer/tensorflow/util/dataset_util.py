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

from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import parsing_ops

from dlrover.trainer.tensorflow.reader.fake_reader import FakeReader
from dlrover.trainer.tensorflow.util import path_util
from dlrover.trainer.tensorflow.util.column_info import Column
from dlrover.trainer.util.log_util import default_logger as logger


class DatasetUtil(object):
    """Prepare dataset from generator and parse them"""

    def __init__(
        self,
        path=None,
        columns: List[Column] = [],
        reader_fn=None,
        schema=None,
        batch_size=128,
        epoch=10,
    ):

        self.path = path
        self.columns = columns
        self.reader = reader_fn or FakeReader()

    def make_dataset(self):
        def reader_fn():
            for data in self.reader.iterator():
                yield data

        self._reader_fn = reader_fn
        if self._reader_fn is None:
            if isinstance(self.path, (list, tuple)):
                scheme, path = path_util.parse_uri(self.path[0])
            else:
                scheme, path = path_util.parse_uri(self.path)
            if scheme == path_util.FILE_SCHEME:
                if isinstance(path, (list, tuple)):
                    path_without_scheme = [
                        path_util.parse_uri(p)[1] for p in path
                    ]
                else:
                    path_without_scheme = path
                dataset = tf.data.TextLineDataset(path_without_scheme)
        else:
            logger.info("building dataset with reader")

        if self._reader_fn is not None:
            logger.info("Make dataset from reader_fn")
            dataset = tf.data.Dataset.from_generator(
                self._reader_fn, output_types=(tf.string)
            )
        return self.process_dataset(dataset)

    def parse_features(self, dataset):
        default_columns_types = []
        default_columns_names = []
        for i in self.columns:
            float_types = ["float", "float32", "float64", "double"]
            int_types = ["int", "int8", "int16", "int32", "int64"]
            uint_types = ["uint8", "uint16", "uint32", "uint64"]
            all_types = float_types + int_types + uint_types
            dtype = i.dtype
            if dtype == "string":
                default_val = ""
            elif dtype in all_types:
                default_val = np.dtype(dtype).type(0)
            default_columns_types.append(default_val)
            default_columns_names.append(i.name)

        def parse_csv(value):
            columns = parsing_ops.decode_csv(
                value, record_defaults=default_columns_types, field_delim=","
            )
            features = dict(zip(default_columns_names, columns))
            labels = features.pop("y")
            return features, labels

        return dataset.map(parse_csv, num_parallel_calls=10)

    def process_dataset(self, dataset):
        return self.parse_features(dataset)

    def input_fn(self):
        return lambda: self.make_dataset()

    @staticmethod
    def get_dataset_from_uri(path):
        """Get dataset from URI

        Args:
            path: odps://, sls:// or file://
            columns: [str], names of columns
            data_source_info: credential information
            slice_id: for reading data part by part
            slice_cout: for reading data part by part
            start_hour: sls related
            days_before: sls related

        Return:
            dataset
        """
        scheme, path = path_util.parse_uri(path)
        if scheme == path_util.FILE_SCHEME:
            if isinstance(path, (list, tuple)):
                path_without_scheme = [path_util.parse_uri(p)[1] for p in path]
            else:
                path_without_scheme = path
            dataset = tf.data.TextLineDataset(path_without_scheme)
            return dataset
        else:
            raise ValueError("Unknown path: %s" % path)

    @staticmethod
    def create(
        data_set,
    ):
        """
        Args:
            task_conf: dlrover.trainer task conf
            data_name: current data name train_set/test_set..
            initializable: whether reinitialize dataset
            wrapped_reader: for online learning
            job_name: ps/worker/chief/master
            slice_id: used for sharding
            slice_count: used for sharding
            project_dir: estimator project directory
            options: other keyword arguments
        """

        path = data_set.get("path")
        columns = data_set.get("columns")
        reader_fn = data_set.get("reader_fn")
        dataset_util_kwargs = {
            "path": path,
            "columns": columns,
            "reader_fn": reader_fn,
        }
        return DatasetUtil(**dataset_util_kwargs)
