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

from dlrover.python.master.args import parse_master_args
from dlrover.python.master.master import Master
from dlrover.python.scheduler.factory import new_job_args


def run(args):
    job_args = new_job_args(args.platform, args.job_name, args.namespace)
    job_args.initilize()
    job_args.print()
    master = Master(args.port, job_args)
    master.prepare()
    return master.run()


def main():
    args = parse_master_args()
    exit_code = run(args)
    return exit_code


if __name__ == "__main__":
    os._exit(main())
