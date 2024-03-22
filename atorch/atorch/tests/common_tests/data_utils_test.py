import unittest
from collections import OrderedDict, abc

import torch

from atorch.data import expand_batch_dim


@unittest.skipIf(torch.cuda.is_available(), "Skip on gpu as cpu test covers it.")
class DataUtilTest(unittest.TestCase):
    def test_expand_torch_tensor(self):
        bs, c, h, w = 16, 3, 4, 4
        input_tensor = torch.ones([c, h, w], dtype=torch.float32)
        output_tensor = expand_batch_dim(input_tensor, batch_size=bs)
        self.assertTrue(torch.equal(output_tensor, torch.ones([bs, c, h, w], dtype=torch.float32)))

    def test_expand_list_of_tensors(self):
        bs = 4
        input_tensors = [torch.ones([i], dtype=torch.float32) for i in range(1, 4)]
        output_tensors = expand_batch_dim(input_tensors, batch_size=bs)
        for i in range(0, 3):
            self.assertTrue(torch.equal(output_tensors[i], torch.ones([bs, i + 1], dtype=torch.float32)))

    def test_expand_nested_list_of_tensors(self):
        dtype = torch.float32
        bs = 4
        input_tensors = tuple([torch.ones([4], dtype=dtype), [torch.ones([i], dtype=dtype) for i in range(1, 4)]])
        output_tensors = expand_batch_dim(input_tensors, batch_size=bs)
        expected_output = tuple(
            [torch.ones([bs, 4], dtype=dtype), [torch.ones([bs, i], dtype=dtype) for i in range(1, 4)]]
        )
        self.assertIsInstance(output_tensors, tuple)
        for output, expect in zip(output_tensors, expected_output):
            if isinstance(output, torch.Tensor):
                self.assertTrue(torch.equal(output, expect))
            elif isinstance(output, abc.Sequence):
                for o, e in zip(output, expect):
                    self.assertTrue(torch.equal(o, e))

    def test_expand_map_of_tensors(self):
        bs, c, h, w = 16, 3, 4, 4
        dtype = torch.float32
        input_dict = {
            "data": torch.ones([c, h, w], dtype=dtype),
            "label": torch.tensor(0, dtype=dtype),
        }
        output_dict = expand_batch_dim(input_dict, batch_size=bs)
        self.assertIsInstance(output_dict, dict)
        self.assertTrue(torch.equal(output_dict["data"], torch.ones([bs, c, h, w], dtype=dtype)))
        self.assertTrue(torch.equal(output_dict["label"], torch.zeros(bs, dtype=dtype)))

        input_ordered_dict = OrderedDict(
            {
                "data": torch.ones([c, h, w], dtype=dtype),
                "label": torch.tensor(0, dtype=dtype),
            }
        )
        output_ordered_dict = expand_batch_dim(input_ordered_dict, batch_size=bs)
        self.assertIsInstance(output_ordered_dict, OrderedDict)
        self.assertTrue(torch.equal(output_ordered_dict["data"], torch.ones([bs, c, h, w], dtype=dtype)))
        self.assertTrue(torch.equal(output_ordered_dict["label"], torch.zeros(bs, dtype=dtype)))


if __name__ == "__main__":
    unittest.main()
