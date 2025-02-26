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

import unittest
from unittest.mock import patch

from dlrover.python.common.constants import EventReportConstants
from dlrover.python.diagnosis.common.constants import InferenceConfigKey
from dlrover.python.diagnosis.common.inference_chain import (
    InferenceAttribute,
    InferenceDescription,
    InferenceName,
)
from dlrover.python.diagnosis.inferencechain.inference_chain import (
    Inference,
    InferenceChain,
)
from dlrover.python.diagnosis.inferencechain.inferenceoperator.observer.resource_collection_operator import (  # noqa: E501
    ResourceCollectionOperator,
)
from dlrover.python.diagnosis.inferencechain.inferenceoperator.resolver.resolve_gpu_errors_operator import (  # noqa: E501
    ResolveGPUErrorsOperator,
)


class InferenceChainTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    @patch(
        "dlrover.python.elastic_agent.monitor.resource"
        ".ResourceMonitor.report_resource"
    )
    def test_gpu_resource_error(self, mock_resource_monitor):
        error_logs = "Test the GPU is lost inference chain"
        mock_resource_monitor.side_effect = Exception(error_logs)
        operators = [
            ResolveGPUErrorsOperator(),
            ResourceCollectionOperator(),
        ]

        inf = Inference(
            name=InferenceName.WORKER,
            attribution=InferenceAttribute.COLLECT,
            description=InferenceDescription.RESOURCE,
        )

        ic = InferenceChain([inf], operators)
        results = ic.infer()
        self.assertEqual(len(results), 1)

        self.assertEqual(results[0].name, InferenceName.ACTION)
        self.assertEqual(
            results[0].configs[InferenceConfigKey.EVENT_TYPE],
            EventReportConstants.TYPE_WARN,
        )
        self.assertEqual(
            results[0].configs[InferenceConfigKey.EVENT_ACTION], "GPU is lost"
        )


if __name__ == "__main__":
    unittest.main()
