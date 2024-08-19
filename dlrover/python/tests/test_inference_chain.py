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
import unittest

from dlrover.python.diagnosis.common.constants import InferenceConfigKey
from dlrover.python.diagnosis.common.inference_chain import (
    Inference,
    InferenceAttribute,
    InferenceDescription,
    InferenceName,
    is_same_inference,
)
from dlrover.python.diagnosis.inferencechain.inference_chain import (
    InferenceChain,
)
from dlrover.python.diagnosis.inferencechain.inferenceoperator.check_failure_node_operator import (  # noqa: E501
    CheckFailureNodeOperator,
)
from dlrover.python.diagnosis.inferencechain.inferenceoperator.check_training_hang_operator import (  # noqa: E501
    CheckTrainingHangOperator,
)


class InferenceChainTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_CheckTrainingHangOperator(self):
        operator = CheckTrainingHangOperator(None)
        inf = Inference(
            name=InferenceName.TRAINING,
            attribution=InferenceAttribute.ISORNOT,
            description=InferenceDescription.HANG,
        )
        self.assertTrue(operator.is_compatible(inf))

        results = operator.infer([inf])
        self.assertEqual(results[0].name, InferenceName.END)

    def test_CheckFailureNodeOperator(self):
        file = "data/training.log"
        path = os.path.dirname(__file__)
        file_path = os.path.join(path, file)

        operator = CheckFailureNodeOperator()
        inf = Inference(
            name=InferenceName.NODE,
            attribution=InferenceAttribute.ISORNOT,
            description=InferenceDescription.FAILURE,
            configs={
                InferenceConfigKey.LOG_FILE: file_path,
                InferenceConfigKey.ERRORS: "error code is 507035",
            },
        )
        self.assertTrue(operator.is_compatible(inf))

        results = operator.infer([inf])
        failure_inf = Inference(
            name=InferenceName.NODE,
            attribution=InferenceAttribute.IS,
            description=InferenceDescription.FAILURE,
        )
        self.assertTrue(is_same_inference(results[0], failure_inf))

    def test_InferenceChain(self):
        file = "data/training.log"
        path = os.path.dirname(__file__)
        file_path = os.path.join(path, file)
        inf = Inference(
            name=InferenceName.NODE,
            attribution=InferenceAttribute.ISORNOT,
            description=InferenceDescription.FAILURE,
            configs={
                InferenceConfigKey.LOG_FILE: file_path,
                InferenceConfigKey.ERRORS: "error code is 507035",
            },
        )

        operators = [CheckFailureNodeOperator()]
        ic = InferenceChain([inf], operators)
        results = ic.infer()
        failure_inf = Inference(
            name=InferenceName.NODE,
            attribution=InferenceAttribute.IS,
            description=InferenceDescription.FAILURE,
        )
        self.assertTrue(is_same_inference(results[0], failure_inf))


if __name__ == "__main__":
    unittest.main()
