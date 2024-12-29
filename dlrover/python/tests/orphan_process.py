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

import os
import subprocess
import sys
import time


def myfork():
    pid = os.fork()

    if pid == 0:
        env_vars = os.environ.copy()
        for key, value in env_vars.items():
            print("environment variable %s=%s" % (key, value))

        # os.setsid()
        pid = os.getpid()
        print(f"child process: {pid}")
        time.sleep(300)
    else:
        pid = os.getpid()
        print(f"parent process: {pid}")
        exit(0)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "torch":
        os.environ["TORCHELASTIC_RUN_ID"] = "dlrover-test-job"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        subprocess.Popen(
            [
                "/usr/local/bin/python",
                "dlrover/python/tests/orphan_process.py",
            ],
            env=os.environ,
        )
        exit(0)
    else:
        myfork()
