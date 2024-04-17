import os
import unittest

import torch
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from atorch.rl.data.data_utils import GLMPromptDataset, PromptDataset


@unittest.skipIf(True, "skip this test and move the test to integration test when ready")
class PromptDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        if not os.path.exists("./glm-large-chinese"):
            os.system(
                "wget http://armor-test.oss-cn-hangzhou-zmf.aliyuncs.com/apps/elasticdl/atorch_ci/glm-large-chinese.tar.gz"  # noqa: E501
            )
            os.system("tar -zxvf glm-large-chinese.tar.gz")
        model_name = "/w/glm-large-chinese/"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer = tokenizer

    def test_prompt_dataset(self):
        prompts = ["今天天气真好 [MASK]", "请推荐一部纪录片 [MASK]"]
        max_prompt_length = 1024
        prompt_dataset = PromptDataset(prompts, max_prompt_length=max_prompt_length, tokenizer=self.tokenizer)
        self.assertTrue(isinstance(prompt_dataset[0], dict))
        self.assertTrue(isinstance(prompt_dataset[0]["input_ids"], list))
        data_loader = prompt_dataset.create_loader(2)
        prompt_data = [i for i in data_loader][0]
        self.assertTrue(isinstance(prompt_data["input_ids"], torch.Tensor))
        self.assertTrue((prompt_data, BatchEncoding))
        res = self.tokenizer.decode(prompt_data["input_ids"].tolist()[0])
        self.assertEqual(res, "今天天气真好 [MASK] <|endoftext|>")
        glm_generation_inputs = self.tokenizer.build_inputs_for_generation(prompt_data, max_gen_length=1024)
        res = self.tokenizer.decode(glm_generation_inputs.input_ids[0])
        self.assertEqual(res, "今天天气真好 [MASK] <|endoftext|> <|startofpiece|>")

    def test_glm_prompt_dataset(self):
        prompts = ["今天天气真好 [sMASK]", "请推荐一部纪录片 [sMASK]"]
        max_prompt_length = 1024
        prompt_dataset = GLMPromptDataset(prompts, max_prompt_length=max_prompt_length, tokenizer=self.tokenizer)
        self.assertTrue(isinstance(prompt_dataset[0], dict))
        self.assertTrue(isinstance(prompt_dataset[0]["input_ids"], list))
        data_loader = prompt_dataset.create_loader(2)
        prompt_data = [i for i in data_loader][0]
        self.assertTrue(isinstance(prompt_data["input_ids"], torch.Tensor))
        self.assertTrue((prompt_data, BatchEncoding))
        res = self.tokenizer.decode(prompt_data["input_ids"].tolist()[0])
        self.assertEqual(res, "[CLS] 今天天气真好 [sMASK] <|endoftext|> <|endoftext|>")
        glm_generation_inputs = self.tokenizer.build_inputs_for_generation(prompt_data, max_gen_length=1024)
        res = self.tokenizer.decode(glm_generation_inputs.input_ids[0])
        self.assertEqual(res, "[CLS] 今天天气真好 [sMASK] <|endoftext|> <|endoftext|> <|startofpiece|>")


if __name__ == "__main__":
    unittest.main()
