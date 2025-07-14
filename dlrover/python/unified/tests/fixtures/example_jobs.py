# Copyright 2025 The DLRover Authors. All rights reserved.
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

from dlrover.python.unified.common.workload_config import (
    ElasticWorkloadDesc,
    ResourceDesc,
)
from dlrover.python.unified.controller.config import (
    ACCELERATOR_TYPE,
    DLConfig,
    JobConfig,
)


def elastic_training_job():
    """Example job configuration for elastic training."""

    dl_config = DLConfig(
        workloads={
            "training": ElasticWorkloadDesc(
                num=2,
                entry_point="dlrover.trainer.torch.node_check.nvidia_gpu::run",
                resource=ResourceDesc(accelerator=1),
            )
        },
        accelerator_type=ACCELERATOR_TYPE.CPU,
    )
    return JobConfig(
        job_name="test_elastic_training",
        dl_config=dl_config,
    )
