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

"""
Users can use the script to check where the GPU is available before
running the training script. For example, we can the command
like
    python -m dlrover.trainer.check.gpu_check && dlrover-run ...

Note:
    If the return_code of gpu_inspector is 202, there may
    be the Pod residue on the machine. However, it may be a misjudgement.
"""

import os
import subprocess
import sys

if __name__ == "__main__":
    file_path = os.path.abspath(sys.argv[0])
    file_dir = os.path.dirname(file_path)
    check_path = os.path.join(file_dir, "gpu_inspector.sh")
    p = subprocess.run(["sh", check_path])
    os._exit(p.returncode)
