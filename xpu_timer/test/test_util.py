import unittest
from collections import OrderedDict

import numpy as np
from py_xpu_timer.util import GetRankHelper


class UtilTest(unittest.TestCase):
    def test_rank(self):
        group_str = "tp2-dp2-pp2"
        groups_dict = OrderedDict((pair[:2], int(pair[2:])) for pair in group_str.split("-"))
        rank_helper = GetRankHelper(groups_dict)
        group_ranks = {group: rank_helper.get_ranks(group) for group in groups_dict}
        gt = {
            "tp": np.array([[0, 1], [2, 3], [4, 5], [6, 7]]),
            "dp": np.array([[0, 2], [1, 3], [4, 6], [5, 7]]),
            "pp": np.array([[0, 4], [1, 5], [2, 6], [3, 7]]),
        }
        for k in gt:
            self.assertTrue((group_ranks[k] == gt[k]).all())


if __name__ == "__main__":
    unittest.main()
