from atorch.auto.engine.sg_algo.hebo.models.base_model import BaseModel
from atorch.auto.engine.sg_algo.hebo.models.gauss_process.gpy_wgp import GPyGP
from atorch.auto.engine.sg_algo.hebo.models.random_forest.rf import RF

model_dict = {"gpy": GPyGP, "rf": RF}

model_names = [k for k in model_dict.keys()]


def get_model_class(model_name: str):

    assert model_name in model_dict
    model_class = model_dict[model_name]
    return model_class


def get_model(model_name: str, *params, **conf) -> BaseModel:
    model_class = get_model_class(model_name)
    return model_class(*params, **conf)
