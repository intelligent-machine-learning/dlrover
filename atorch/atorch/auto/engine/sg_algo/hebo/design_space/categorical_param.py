import numpy as np

from .param import Parameter


class CategoricalPara(Parameter):
    def __init__(self, param):
        super().__init__(param)
        self.cates = list(param["categories"])
        try:
            self._categories_dict = {k: v for v, k in enumerate(self.cates)}
        except TypeError:  # there are unhashable types
            self._categories_dict = None
        self.lb = 0
        self.ub = len(self.cates) - 1

    def sample(self, num=1):
        assert num > 0
        return np.random.choice(self.cates, num, replace=True)

    def transform(self, x: np.ndarray):
        if self._categories_dict:
            ret = np.array(list(map(lambda a: self._categories_dict[a], x)))
        else:
            # otherwise, we fall back to searching in an array
            ret_li = list(map(lambda a: np.where(self.cates == a)[0][0], x))
            ret = np.array(ret_li)
        return ret.astype(float)

    def inverse_transform(self, x):
        return np.array([self.cates[x_] for x_ in x.round().astype(int)])

    @property
    def is_numeric(self):
        return False

    @property
    def is_discrete(self):
        return True

    @property
    def is_discrete_after_transform(self):
        return True

    @property
    def opt_lb(self):
        return self.lb

    @property
    def opt_ub(self):
        return self.ub

    @property
    def num_uniqs(self):
        return len(self.cates)
