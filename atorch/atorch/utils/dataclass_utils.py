import copy
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Callable

from atorch.common.log_utils import default_logger as logger


@contextmanager
def clear_environment():
    """
    provide a blank environment with empty os envs,
    the previous envs will be resumed when exist the context
    """
    _old_os_env_backups = os.environ.copy()
    os.environ.clear()

    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(_old_os_env_backups)


class DataclassMixin:
    """
    provide a default to_dict and to_kwargs helper functions for dataclass
    """

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(self.to_dict(), indent=2)

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values and `Callable` by dictionaries (for JSON
        serialization support). It obfuscates the token values by removing their value.
        """

        def serialize_dict(d: dict):
            for k, v in d.items():
                if isinstance(v, Callable):  # type: ignore[arg-type]
                    d[k] = v.__name__ if hasattr(v, "__name__") else str(v)
                elif isinstance(v, (list, tuple)) and len(v) > 0:
                    if isinstance(v[0], Enum):
                        d[k] = [x.value for x in v]
                        if isinstance(v, tuple):
                            d[k] = tuple(d[k])
                    elif isinstance(v[0], Callable):  # type: ignore[arg-type]
                        d[k] = [x.__name__ if hasattr(x, "__name__") else str(x) for x in v]
                        if isinstance(v, tuple):
                            d[k] = tuple(d[k])
                elif isinstance(v, dict):
                    d[k] = serialize_dict(v)
            return d

        d = {field.name: getattr(self, field.name) for field in fields(self) if field.init}

        d = copy.deepcopy(d)

        return serialize_dict(d)

    def to_kwargs(self):
        """
        Returns a dictionary containing the attributes with values different from the default of this class.
        """
        with clear_environment():
            default_dict = self.__class__().to_dict()
        this_dict = self.to_dict()
        return {k: v for k, v in this_dict.items() if default_dict[k] != v}


@dataclass
class AutoMapperExtraConfigs:
    extra_configs: dict = field(default_factory=dict)

    def do_auto_mapping(self, name_map: dict, overwrite_extra_configs=False):
        """

        Args:
            name_map:
            overwrite_extra_configs: whether to overwrite `extra_configs` attribute. Be very careful to set to True,
            since once you overwrite it, the auto_mapping will not be able to change the value inside the extra_configs
            from the outside args wrapper

        Returns:

        """
        if overwrite_extra_configs:
            target_dict = self.extra_configs
        else:
            target_dict = copy.deepcopy(self.extra_configs)

        # Compat megatron args and atorch args
        # If there is an argument named A in Megatron and an argument named B in atorch,
        # with same meaning but different name, A is given higher priority.
        for atorch_args_name, megatron_args_name in name_map.items():
            # For example, logging interval is named "log_interval" in megatron but "logging_steps" in atorch,
            # you can set "logging_steps" in AtorchTrainingArgs or "log_interval" in AtorchTrainingArgs.extra_configs .
            # Just select one of them. If they are both set, give priority to "log_interval".
            if megatron_args_name not in target_dict:
                target_dict[megatron_args_name] = getattr(self, atorch_args_name)
            else:
                if target_dict[megatron_args_name] is None and getattr(self, atorch_args_name) is not None:
                    target_dict[megatron_args_name] = getattr(self, atorch_args_name)
                elif target_dict[megatron_args_name] != getattr(self, atorch_args_name):
                    logger.warning(
                        f"{atorch_args_name}:{getattr(self, atorch_args_name)} in AtorchTrainingArgs will "
                        f"be overridden by {megatron_args_name}:{target_dict[megatron_args_name]} in Megatron args."
                    )
                    setattr(self, atorch_args_name, target_dict[megatron_args_name])
        return target_dict


@dataclass
class DynamicDataClass(DataclassMixin):
    def __init__(self, **kwargs):
        field_names = {field.name for field in fields(self)}
        dirs = dir(self)

        for field_name in field_names:
            if field_name in kwargs:
                setattr(self, field_name, kwargs[field_name])
            elif field_name not in dirs:
                raise ValueError(f"{field_name} is not found in __init__ params and has no default value")
            else:
                attr_value = getattr(self, field_name)
                setattr(self, field_name, attr_value)

        # unknown args
        unknown_keys = set(kwargs.keys()) - field_names
        _hidden_extra_args = {}

        for key in unknown_keys:
            _hidden_extra_args[key] = copy.deepcopy(kwargs[key])

        self._additional_data = _hidden_extra_args

    def to_dict(self):
        field_dict = super().to_dict()
        field_dict.update(self._additional_data)
        return field_dict

    def __getattr__(self, item):
        if item in self._additional_data:
            return self._additional_data[item]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")
