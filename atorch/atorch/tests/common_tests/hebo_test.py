import unittest
import warnings

import numpy as np
import pandas as pd
import torch

from atorch.auto.engine.sg_algo.hebo.acquisitions.acq import MACE, Acquisition, Mean, Sigma
from atorch.auto.engine.sg_algo.hebo.design_space.design_space import DesignSpace


class ToyExample(Acquisition):
    def __init__(self, constr_v=1.0):
        super().__init__(None)
        self.constr_v = constr_v

    @property
    def num_obj(self):
        return 1

    @property
    def num_constr(self):
        return 1

    def eval(self, x, xe):
        # minimize L2norm(x) s.t. L2norm(x) > constr_v
        out = (xe**2).sum(axis=1).reshape(-1, 1)
        constr = self.constr_v - out
        return np.concatenate([out, constr], axis=1)


def obj(x: pd.DataFrame) -> np.ndarray:
    return x["x0"].values.astype(float).reshape(-1, 1) ** 2


class HEBOTest(unittest.TestCase):
    def setUp(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

    @unittest.skipIf(torch.cuda.is_available(), "run on cpu only")
    def test_design_sapce(self):
        space = DesignSpace().parse([{"name": "x3", "type": "cat", "categories": ["a", "b", "c"]}])
        self.assertEqual(space.numeric_names, [])
        self.assertEqual(space.enum_names, ["x3"])
        self.assertEqual(space.para_names, space.numeric_names + space.enum_names)
        self.assertEqual(space.num_paras, 1)
        self.assertEqual(space.num_numeric, 0)
        self.assertEqual(space.num_categorical, 1)
        samp = space.sample(10)
        x, xe = space.transform(samp)
        _, xe_ = space.transform(space.inverse_transform(x, xe))
        self.assertEqual((xe_ == xe).all(), True)
        self.assertEqual(space.paras["x3"].is_discrete, True)
        self.assertEqual(space.paras["x3"].is_discrete_after_transform, True)
        for _, para in space.paras.items():
            self.assertEqual(
                para.is_discrete == (type(para.sample(1).tolist()[0]) != float),
                True,
            )
            if para.is_discrete_after_transform:
                samp = para.sample(5)
                x = para.transform(samp)
                self.assertEqual((x == x.round()).all(), True)

    @unittest.skipIf(torch.cuda.is_available(), "run on cpu only")
    def test_models(self):
        from atorch.auto.engine.sg_algo.hebo.models import model_factory

        Xc = None
        Xe = np.random.randint(low=0, high=2, size=(50, 1))
        y = Xe.astype(float)
        model = model_factory.get_model("gpy", 0, 1, 1, num_uniqs=[2], num_epochs=1)
        model.fit(Xc, Xe, y + 1e-2 * np.random.randn(y.shape[0], y.shape[1]))
        py, ps2 = model.predict(Xc, Xe)
        y = y.reshape(-1)
        py = py.reshape(-1)
        self.assertEqual(y.shape == py.shape, True)
        self.assertEqual(np.isfinite(py).all(), True)
        self.assertEqual((ps2 > 0).all(), True)
        y = y.reshape([-1, 1])
        model = model_factory.get_model("rf", 0, 1, 1, num_uniqs=[2], num_epochs=1)
        model.fit(Xc, Xe, y + 1e-2 * np.random.randn(y.shape[0], y.shape[1]))
        py, ps2 = model.predict(Xc, Xe)
        y = y.reshape(-1)
        py = py.reshape(-1)
        self.assertEqual(y.shape == py.shape, True)
        self.assertEqual(np.isfinite(py).all(), True)
        self.assertEqual((ps2 > 0).all(), True)

    @unittest.skipIf(torch.cuda.is_available(), "run on cpu only")
    def test_acq(self):
        from atorch.auto.engine.sg_algo.hebo.models import model_factory

        X = np.random.randn(10, 1)
        y = X
        model = model_factory.get_model("rf", 1, 0, 1, num_epochs=10)
        model.fit(X, None, y)
        acq = Mean(model, best_y=0.0)
        acq_v = acq(X, None)
        assert acq.num_obj == 1
        assert acq.num_constr == 0
        acq = Sigma(model, best_y=0.0)
        acq_v = acq(X, None)
        assert acq.num_obj == 1
        assert acq.num_constr == 0
        acq = MACE(model, best_y=0.0)
        acq_v = acq(X, None)
        assert np.isfinite(acq_v).all()
        assert acq.num_obj == 3
        assert acq.num_constr == 0

    @unittest.skipIf(torch.cuda.is_available(), "run on cpu only")
    def test_acq_optimizer(self):
        from atorch.auto.engine.sg_algo.hebo.acq_optimizers import evolution_optimizer

        constr_v = 1.0
        space = DesignSpace().parse([{"name": "x1", "type": "cat", "categories": ["a", "b"]}])
        acq = ToyExample(constr_v=constr_v)
        opt = evolution_optimizer.EvolutionOpt(space, acq, pop=10, sobol_init="sobol")
        rec = opt.optimize(initial_suggest=space.sample(3))
        x, xe = space.transform(rec)
        self.assertAlmostEqual(acq(x, xe)[:, 0], 1.0, 0.01)

    @unittest.skipIf(torch.cuda.is_available(), "run on cpu only")
    def test_hebo(self):
        from atorch.auto.engine.sg_algo.hebo.optimizers.hebo import HEBO

        space = DesignSpace().parse(
            [
                {
                    "name": "x0",
                    "type": "cat",
                    "categories": [i for i in range(10)],
                }
            ]
        )
        opt = HEBO(space)
        num_suggest = 0
        for i in range(2):
            num_suggest = 2 if opt.support_parallel_opt else 1
            rec = opt.suggest(n_suggestions=num_suggest)
            y = obj(rec)
            opt.observe(rec, y)


if __name__ == "__main__":
    unittest.main()
