#  Copyright 2025 The DLRover Authors. All rights reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse

from omegaconf import DictConfig, ListConfig, OmegaConf


def args_2_omega_conf(args: argparse.Namespace) -> DictConfig:
    """Transfer args from argparse to omega config."""
    return OmegaConf.create(vars(args))


def omega_conf_2_args(conf: DictConfig) -> argparse.Namespace:
    """Transfer omega config to args."""
    return argparse.Namespace(**conf)


def convert_str_values(
    config: DictConfig, convert_none=True, convert_digit=True
):
    """
    Optimize omega config for str values.
    """

    if isinstance(config, DictConfig):
        for key, value in config.items():
            if isinstance(value, str):
                if convert_none and value == "None":
                    config[key] = None
                elif convert_digit and value.isdigit():
                    config[key] = int(value)
                else:
                    try:
                        if convert_digit:
                            config[key] = float(value)
                    except ValueError:
                        pass
            elif isinstance(value, (DictConfig, ListConfig)):
                convert_str_values(value)
    elif isinstance(config, ListConfig):
        for idx, value in enumerate(config):
            if isinstance(value, str):
                if convert_none and value == "None":
                    config[idx] = None
                elif convert_digit and value.isdigit():
                    config[idx] = int(value)
                else:
                    try:
                        if convert_digit:
                            config[idx] = float(value)
                    except ValueError:
                        pass
            elif isinstance(value, (DictConfig, ListConfig)):
                convert_str_values(value)


def read_dict_from_envs(prefix: str) -> dict[str, str]:
    """Read dict from environment variables with the given prefix."""
    import os

    return {
        k[len(prefix) :].lower(): v
        for k, v in os.environ.items()
        if k.startswith(prefix)
    }
