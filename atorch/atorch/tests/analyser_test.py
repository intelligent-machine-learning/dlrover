import unittest
from collections import namedtuple

import numpy
import torch
import torch.nn as nn
import transformers
from transformers import BertConfig, BertModel
from transformers.models.bert.modeling_bert import (
    BertAttention,
    BertEmbeddings,
    BertIntermediate,
    BertLayer,
    BertOutput,
    BertPooler,
    BertSelfAttention,
    BertSelfOutput,
)

from atorch.auto.analyser.analyser import get_analyser
from atorch.auto.model_context import ModelContext
from atorch.common.constants import AnalyserConstants
from atorch.tests.toy_module import ToyDataset, ToyModel, prepare_input

_BERT_SUBMODEL_TYPES = (
    torch.nn.modules.sparse.Embedding,
    transformers.models.bert.modeling_bert.BertSelfOutput,
    torch.nn.modules.container.ModuleList,
    torch.nn.modules.normalization.LayerNorm,
    transformers.models.bert.modeling_bert.BertPooler,
    transformers.models.bert.modeling_bert.BertEncoder,
    transformers.models.bert.modeling_bert.BertIntermediate,
    torch.nn.modules.linear.Linear,
    transformers.models.bert.modeling_bert.BertEmbeddings,
    torch.nn.modules.activation.Tanh,
    transformers.models.bert.modeling_bert.BertLayer,
    transformers.models.bert.modeling_bert.BertAttention,
    transformers.models.bert.modeling_bert.BertOutput,
    transformers.activations.GELUActivation,
    transformers.models.bert.modeling_bert.BertSelfAttention,
    torch.nn.modules.dropout.Dropout,
)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 10)
        )

    def forward(self, data):
        x = self.flatten(data[0])
        logits = self.linear_relu_stack(x)
        return logits


class AnalyserTest(unittest.TestCase):
    def setUp(self):
        model = BertModel(BertConfig())
        # construct sequence data
        input_ids, attention_mask = torch.tensor(
            [[3, 16, 87, 3, 9], [72, 12, 4, 90, 54]], dtype=torch.long
        ), torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(input_ids, attention_mask)
        self.model_context = ModelContext(
            model=model, optim_func=torch.optim.Adam, optim_args={"lr": 1e-3}, dataset=dataset
        )
        self.analyser = get_analyser()

    def test_analyse(self):
        analyse_methods = [AnalyserConstants.ANALYSE_BASIC, AnalyserConstants.ANALYSE_TRANSFORMER]

        res = self.analyser.analyse(self.model_context, analyse_methods)
        self.assertCountEqual(
            res[AnalyserConstants.ANALYSE_BASIC][AnalyserConstants.OPT_CONFIG_SUBMODULE_NAMES], ("BertLayer",)
        )
        self.assertEqual(
            res[AnalyserConstants.ANALYSE_BASIC][AnalyserConstants.MODEL_PARAMS_NUM],
            109482240,
        )
        self.assertAlmostEqual(
            res[AnalyserConstants.ANALYSE_BASIC][AnalyserConstants.MODEL_PARAMS_MB],
            417.6416015625,
        )
        self.assertTrue(res[AnalyserConstants.ANALYSE_BASIC][AnalyserConstants.HAS_MODULE_FOR_REPLACE])
        self.assertEqual(
            res[AnalyserConstants.ANALYSE_BASIC][AnalyserConstants.OPTIMIZER_TYPE],
            "Adam",
        )
        self.assertEqual(
            res[AnalyserConstants.ANALYSE_TRANSFORMER],
            {
                BertEmbeddings.__name__: 1,
                BertSelfAttention.__name__: 12,
                BertSelfOutput.__name__: 12,
                BertAttention.__name__: 12,
                BertIntermediate.__name__: 12,
                BertOutput.__name__: 12,
                BertLayer.__name__: 12,
                BertPooler.__name__: 1,
            },
        )
        self.assertCountEqual(
            res[AnalyserConstants.ANALYSE_BASIC][AnalyserConstants.SUBMODULE_TYPES],
            _BERT_SUBMODEL_TYPES,
        )

    @unittest.skipIf(torch.cuda.is_available(), "Failed on gpu")
    def test_dynamic_analyze(self):
        training_data = ToyDataset(100)
        model = ToyModel()

        model_context = ModelContext(
            model=model,
            optim_func=torch.optim.Adam,
            optim_args={"lr": 1e-3},
            dataset=training_data,
            loss_func=lambda data, pred: nn.CrossEntropyLoss()(pred, data[1]),
            prepare_input=prepare_input,
        )
        analyse_methods = [AnalyserConstants.ANALYSE_DYNAMIC]
        res = self.analyser.analyse(model_context, analyse_methods)[AnalyserConstants.ANALYSE_DYNAMIC]
        self.assertEqual(res[AnalyserConstants.DATA_SIZE], 16 + 4)
        self.assertGreater(res[AnalyserConstants.MODEL_FLOPS_AND_DYNAMIC_MEMORY_MB]["dynamic_memory_mb"], 0)
        self.assertGreater(res[AnalyserConstants.MODEL_FLOPS_AND_DYNAMIC_MEMORY_MB]["model_flops"], 0)
        self.assertGreater(res[AnalyserConstants.OPTIMIZER_STATE_NUM_AND_MEMORY_MB]["optimizer_state_num"], 0)
        self.assertGreater(res[AnalyserConstants.OPTIMIZER_STATE_NUM_AND_MEMORY_MB]["optimizer_state_memory_mb"], 0)

        self.assertEqual(self.analyser._get_data_size_recursively(6), 1)  # int
        self.assertEqual(self.analyser._get_data_size_recursively(1.2), 1)  # float
        self.assertEqual(self.analyser._get_data_size_recursively("abcd"), 4)  # str
        self.assertEqual(self.analyser._get_data_size_recursively(torch.as_tensor([[1, 1], [2, 2]])), 4)  # Tensor
        self.assertEqual(
            self.analyser._get_data_size_recursively(
                namedtuple("Data", ["x", "y"])(torch.as_tensor(1), torch.as_tensor(2))  # namedtuple
            ),
            2,
        )
        self.assertEqual(self.analyser._get_data_size_recursively([torch.as_tensor(1), torch.as_tensor(2)]), 2)  # list
        self.assertEqual(self.analyser._get_data_size_recursively((torch.as_tensor(1), torch.as_tensor(2))), 2)  # tuple
        self.assertEqual(
            self.analyser._get_data_size_recursively({"1": torch.as_tensor(1), "2": torch.as_tensor(2)}), 2  # dict
        )
        self.assertEqual(self.analyser._get_data_size_recursively(numpy.array([[1, 1], [2, 2]])), 4)  # ndarray


if __name__ == "__main__":
    unittest.main()
