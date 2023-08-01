import numpy as np
from sklearn.preprocessing import OneHotEncoder


class OneHotTransform(object):
    def __init__(self, num_uniqs):
        self.num_uniqs = num_uniqs

    @property
    def num_out_list(self):
        return self.num_uniqs

    @property
    def num_out(self) -> int:
        return sum(self.num_uniqs)

    def __call__(self, xe):
        return np.concatenate(
            [
                OneHotEncoder(categories=[list(np.arange(self.num_uniqs[i]))])
                .fit_transform(xe[:, i].reshape([-1, 1]))
                .toarray()
                for i in range(xe.shape[1])
            ],
            axis=1,
        ).astype("float")
