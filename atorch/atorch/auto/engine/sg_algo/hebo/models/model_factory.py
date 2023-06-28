# Copyright 2022 The ElasticDL Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from atorch.auto.engine.sg_algo.hebo.models.base_model import BaseModel
from atorch.auto.engine.sg_algo.hebo.models.gp.gpy_wgp import GPyGP
from atorch.auto.engine.sg_algo.hebo.models.rf.rf import RF

model_dict = {"gpy": GPyGP, "rf": RF}

model_names = [k for k in model_dict.keys()]


def get_model_class(model_name: str):

    assert model_name in model_dict
    model_class = model_dict[model_name]
    return model_class


def get_model(model_name: str, *params, **conf) -> BaseModel:
    model_class = get_model_class(model_name)
    return model_class(*params, **conf)
