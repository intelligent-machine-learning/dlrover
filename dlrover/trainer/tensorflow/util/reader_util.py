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

from dlrover.trainer.tensorflow.util.common_util import singleton
from dlrover.trainer.util.log_util import default_logger as logger


@singleton
class ReaderRegistry:
    """ """

    def __init__(self):
        self.reader_registry = {}

    def register_reader(self, reader_name, reader_class):
        if reader_name in self.reader_registry.keys():
            logger.info(
                "reader {} is registered in reader_registry before,\
                it would be replaced by {}".format(
                    reader_name, reader_class
                )
            )
        self.reader_registry[reader_name] = reader_class
        logger.info(
            "reader {} is registered in reader_registry,\
            reader class is {}".format(
                reader_name, reader_class
            )
        )

    def get_reader(self, reader_name):
        if reader_name not in self.reader_registry.keys():
            logger.warning(
                "reader is not registered in reader_registry before"
            )
        reader_class = self.reader_registry.get(reader_name, None)
        logger.info(
            "reader {} is registered in reader_registry,\
             reader class is {}".format(
                reader_name, reader_class
            )
        )
        return reader_class

    def unregister_reader(self, reader_name):
        if reader_name in self.reader_registry.keys():
            self.reader_registry.pop(reader_name)
        logger.info("reader {} is unregistered from reader_registry")


reader_registery = ReaderRegistry()
