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

from dlrover.python.diagnosis.inferencechain.inferenceoperator.observer.check_resource_collection_operator import (  # noqa: E501
    CheckResourceCollectionOperator,
)
from dlrover.python.diagnosis.inferencechain.inferenceoperator.observer.metrics_collection_operator import (  # noqa: E501
    MetricsCollectionOperator,
)
from dlrover.python.diagnosis.inferencechain.inferenceoperator.observer.resource_collection_operator import (  # noqa: E501
    ResourceCollectionOperator,
)
from dlrover.python.diagnosis.inferencechain.inferenceoperator.resolver.resolve_gpu_errors_operator import (  # noqa: E501
    ResolveGPUErrorsOperator,
)


def get_training_failure_operators():
    return []


def get_worker_observe_operators():
    return [
        MetricsCollectionOperator(),
        CheckResourceCollectionOperator(),
        ResourceCollectionOperator(),
    ]


def get_worker_resolve_operators():
    return [
        ResolveGPUErrorsOperator(),
    ]
