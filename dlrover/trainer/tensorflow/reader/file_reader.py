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
from dlrover.trainer.tensorflow.util.reader_util import reader_registery
from dlrover.trainer.util.log_util import default_logger as logger


class FileReader(ElasticReader):
    def __init__(
        self,
        path=None,
        num_epochs=1,
        batch_size=64,
        enable_dynamic_sharding=True,
        skip_header=True,
    ):
        self._skip_header = skip_header
        self._file_handler = open(path, "r")
        self._file_name = path
        self._consumed_data = 0
        super().__init__(
            path=path,
            num_epochs=num_epochs,
            batch_size=batch_size,
            enable_dynamic_sharding=enable_dynamic_sharding,
        )

    def count_data(self):
        self.data = self._file_handler.readlines()
        if self._skip_header:
            self._data_nums = len(self.data) - 1
            self.data = self.data[1:]
        else:
            self._data_nums = len(self.data)

    def iterator(self):
        while True:
            shard = self.data_shard_client.fetch_shard()
            if not shard:
                break
            logger.info("shard is {}".format(shard))
            for i in range(shard.start, shard.end):
                d = self.data[i]
                d = d.strip()
                dd = d.split(",")
                assert len(dd) == 40
                yield d

    def __del__(self):
        if self._file_handler is not None:
            self._file_handler.close()


reader_registery.register_reader("file", FileReader)
