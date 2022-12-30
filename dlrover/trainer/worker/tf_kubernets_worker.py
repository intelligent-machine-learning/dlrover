# Copyright 2022 The DLRover Authors. All rights reserved.
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

from dlrover.trainer.util.conf_util import get_conf


class TFKubernetesWorker:
    """KubemakerWorker"""

    def __init__(self, args):
        """
        Argument:
            args: result of parsed command line arguments
        """
        self._args = args
        task_conf = get_conf(py_conf=args.conf)
        self.init_executor(task_conf)

    def init_executor(self, task_conf):
        pass

    def start_failover_monitor(self):
        pass

    def run(self):
        pass
