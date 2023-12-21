import numpy as np
from sklearn.ensemble import RandomForestRegressor

from atorch.auto.engine.sg_algo.hebo.models.base_model import BaseModel
from atorch.auto.engine.sg_algo.hebo.models.layers import OneHotTransform
from atorch.auto.engine.sg_algo.hebo.models.util import filter_nan


class RF(BaseModel):
    def __init__(self, num_cont, num_enum, num_out, **conf):
        super().__init__(num_cont, num_enum, num_out, **conf)
        self.n_estimators = self.conf.get("n_estimators", 100)
        self.rf = RandomForestRegressor(n_estimators=self.n_estimators)
        self.est_noise = np.zeros(self.num_out)
        if self.num_enum > 0:
            self.one_hot = OneHotTransform(self.conf["num_uniqs"])

    def xtrans(self, Xc: np.ndarray, Xe: np.ndarray) -> np.ndarray:
        if self.num_enum == 0:
            return Xc
        else:
            Xe_one_hot = self.one_hot(Xe)
            if Xc is None:
                Xc = np.zeros((Xe.shape[0], 0))
            return np.concatenate([Xc, Xe_one_hot], axis=1)

    def fit(self, Xc: np.ndarray, Xe: np.ndarray, y: np.ndarray):
        Xc, Xe, y = filter_nan(Xc, Xe, y, "all")
        Xtr = self.xtrans(Xc, Xe)
        ytr = y.reshape(-1)
        self.rf.fit(Xtr, ytr)
        var = (self.rf.predict(Xtr).reshape(-1) - ytr) ** 2
        self.est_noise = np.mean(var).reshape(self.num_out)

    @property
    def noise(self):
        return self.est_noise

    def predict(self, Xc: np.ndarray, Xe: np.ndarray):
        X = self.xtrans(Xc, Xe)
        mean = self.rf.predict(X).reshape(-1, 1)
        preds = []
        for estimator in self.rf.estimators_:
            preds.append(estimator.predict(X).reshape([-1, 1]))
        var = np.var(np.concatenate(preds, axis=1), axis=1)
        return mean.reshape([-1, 1]), var.reshape([-1, 1]) + self.noise
