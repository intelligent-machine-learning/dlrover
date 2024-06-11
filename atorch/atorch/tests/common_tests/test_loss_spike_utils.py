import os
import unittest

from atorch.utils.loss_spike_utils import TokenLossSpike


class LossSpikeTest(unittest.TestCase):
    def init(self):
        self.min_iter = 100
        self.min_loss = 4.0

        sample_data_paths = [("wikipedia", "corpus/base"), ("wikipedia", "corpus/base"), ("wikipedia", "corpus/base")]
        each_sample_len = 2

        if not os.path.exists("loss_spike"):
            os.mkdir("loss_spike")

        self.loss_ins = TokenLossSpike(
            "loss_spike",
            sample_data_paths,
            each_sample_len,
            self.min_iter,
            self.min_loss,
        )

    def setUp(self):
        self.init()

    def test_save(self):
        self.loss_ins.save_loss(
            "test_loss.txt",
            4.05,
            103,
            losses_str="2.44,2.33,4.05",
            sample_infos_str="20-1-1385697-14158189-2,20-1-1385697-14158189-2,20-1-1385697-14158189-2",
        )

    def test_decode(self):
        self.loss_ins.decode_loss_spike("res.txt", None)

    def test_parse(self):
        self.loss_ins.parse_sample_content(
            losses_str="2.44,2.33,4.05",
            sample_infos_str="20-1-1385697-14158189-2,20-1-1385697-14158189-2,20-1-1385697-14158189-2",
            tokenizer=None,
        )

    def test_fetch(self):
        self.loss_ins.fetch("20-1-1385697-14158189-2")
