#  Copyright 2025 The DLRover Authors. All rights reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dlrover.python.master.node.job_context import JobContext
from dlrover.python.unified.common.job_context import (
    get_job_context as get_unified_job_context,
)


class ElasticJobContext(JobContext):
    """
    To adapt dlrover.python.master.node.job_context::JobContext from
    dlrover.python.unified.common.job_context::JobContext
    """

    def __init__(self):
        super(ElasticJobContext, self).__init__()
        self._unified_job_ctx = get_unified_job_context()

    @property
    def graph(self):
        return self._unified_job_ctx.execution_graph


def get_elastic_job_context() -> ElasticJobContext:
    job_context = ElasticJobContext.singleton_instance()
    return job_context
