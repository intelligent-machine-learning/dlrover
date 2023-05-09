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
from dlrover.trainer.tensorflow.reader.base_reader import (  # noqa: F401
    ElasticReader,
)


class FileReader(ElasticReader):
    def __init__(self, path=None, skip_header=True):
        self._skip_header = skip_header
        self._file_handler = open(path, "r")
        self.data = self._file_handler.readlines()
        self._file_name = path
        super().__init__(
            path=path,
        )
        self._data_nums = None

    def count_data(self):
        if self._data_nums is None:
            if self._skip_header:
                self._data_nums = len(self.data) - 1
                self.data = self.data[1:]
            else:
                self._data_nums = len(self.data)

    def read_data_by_index_range(self, start_index, end_index):
        for i in range(start_index, end_index):
            d = self.data[i]
            d = d.strip()
            yield d

    def __del__(self):
        if self._file_handler is not None:
            self._file_handler.close()
