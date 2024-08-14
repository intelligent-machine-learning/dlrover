import unittest
from dlrover.python.diagnose.inferencechain.inferenceoperator.check_training_hang_operator import (  # noqa: E501
    CheckTrainingHangOperator,
)
from dlrover.python.diagnose.inferencechain.inferenceoperator.check_failure_node_operator import (  # noqa: E501
    CheckFailureNodeOperator,
)
from dlrover.python.diagnose.common.inference_chain import (
    Inference,
    InferenceName,
    InferenceAttribute,
    InferenceDescription,
    same_inference,
)
from dlrover.python.diagnose.common.constants import (
    InferenceConfigKey,
)
import os
from dlrover.python.diagnose.inferencechain.inference_chain import InferenceChain


class InferenceChainTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_CheckTrainingHangOperator(self):
        operator = CheckTrainingHangOperator()
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
        self.assertTrue(same_inference(results[0], failure_inf))

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

        ic = InferenceChain([inf])
        results = ic.infer()
        failure_inf = Inference(
            name=InferenceName.NODE,
            attribution=InferenceAttribute.IS,
            description=InferenceDescription.FAILURE,
        )
        self.assertTrue(same_inference(results[0], failure_inf))


if __name__ == '__main__':
    unittest.main()
