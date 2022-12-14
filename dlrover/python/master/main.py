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

import os

from dlrover.python.common.constants import NodeType
from dlrover.python.common.global_context import Context
from dlrover.python.master.args import parse_master_args
from dlrover.python.master.master import Master
from dlrover.python.scheduler.factory import new_job_params
from dlrover.python.scheduler.job import JobParams

_dlrover_context = Context.instance()


def update_context(job_params: JobParams):
    for node_type, node_params in job_params.node_params.items():
        if node_type == NodeType.WORKER:
            _dlrover_context.easydl_worker_enabled = node_params.auto_scale
        elif node_type == NodeType.PS:
            _dlrover_context.easydl_ps_enabled = node_params.auto_scale
    _dlrover_context.print_config()


def run(args):
    job_params = new_job_params(args.platform, args.job_name, args.namespace)
    job_params.initilize()
    job_params.print()
    update_context(job_params)
    master = Master(args.port, job_params)
    master.prepare()
    return master.run()


def main():
    args = parse_master_args()
    exit_code = run(args)
    return exit_code


if __name__ == "__main__":
    os._exit(main())
