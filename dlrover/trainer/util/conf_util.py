# Copyright 2022 The DLRover Authors. All rights reserved.
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

import inspect
import os

from pyhocon import ConfigFactory, ConfigMissingException, ConfigTree

from dlrover.trainer.util import reflect_util
from dlrover.trainer.util.log_util import default_logger as logger

config_tree_get_ori = ConfigTree.get


def new_func(*args, **kwds):
    default_in_kw = kwds.get("default", None)
    default_in_args = args[2] if len(args) > 2 else None
    default = default_in_kw or default_in_args
    try:
        return config_tree_get_ori(*args, **kwds)
    except ConfigMissingException:
        logger.info(f"key {args[1]}, def {default}")
        return default


ConfigTree.get = new_func


class Singleton:
    _instance = None

    def __new__(cls, *args, **kargs):
        if not isinstance(cls._instance, cls):
            cls._instance = super().__new__(cls)
        return cls._instance


class Configuration(Singleton):
    """wrapper get and set for new conf"""

    def __init__(self, inner_dict=None):
        object.__setattr__(
            self, "inner_dict", dict() if inner_dict is None else inner_dict
        )

    def __contains__(self, key):
        return key in self.inner_dict

    def get(self, key, default=None):
        # check whether the key exists before
        val_before = self.inner_dict.get(key, None)
        val = self.inner_dict.get(key, default)

        logger.info(
            f"get value from conf: the key is {key} and value is {val}"
        )
        logger.info(f"get value from conf: value type is {type(val)}")
        if type(val) is dict:
            # to avoid ConfigTree is a sub-class of dict
            val = ConfigFactory.from_dict(val)
            if val_before is not None:
                self.inner_dict[key] = val
        return val

    def put(self, key, value):
        self.inner_dict[key] = value

    def to_dict(self):
        return self.inner_dict

    def from_dict(self, dt):
        if not isinstance(dt, dict):
            logger.warning(f"you should pass dict not {type(dt)}")
            return
        for k, v in dt.items():
            self.set(k, v)

    def clear(self):
        self.inner_dict.clear()

    def __getitem__(self, key):
        return self.inner_dict.get(key, None)

    def __setitem__(self, key, value):
        self.inner_dict[key] = value

    def __str__(self):
        return str(self.inner_dict)

    def __getattr__(self, key):
        if key.startswith("__"):
            return object.__getattribute__(self, key)
        return self.inner_dict.get(key, None)

    def __setattr__(self, key, value):
        self.inner_dict.setdefault(key, value)

    def __reduce__(self):
        return (self.__class__, (self.inner_dict,))


class ConfigurationManagerMeta(type):
    _all_conf_by_name = {}  # type:dict
    _final_attrs = {}  # type:dict
    _override_order = []  # type:list
    _default_priority = 1 << 30  # type:int

    def __new__(cls, name, bases, attrs):
        clz = super().__new__(cls, name, bases, attrs)
        if name != "ConfigurationManagerInterface":
            cls.register(clz)
        return clz

    def __init__(self, name, bases, attrs):
        if "build" in attrs:
            self.build()

    @classmethod
    def register(cls, clz, priority=None):
        if priority is not None:
            setattr(clz, "priority", priority)
        name = clz.__name__
        if name in cls._all_conf_by_name:
            logger.warning(
                f"class {name} has been created, we will override it"
            )
        if not hasattr(clz, "priority"):
            setattr(clz, "priority", cls._default_priority)
            cls._default_priority -= 1

        if hasattr(clz, "build"):
            clz.build()
        cls._all_conf_by_name[name] = clz

    @classmethod
    def merge_configs(cls):
        def _get_attr(type_):
            return {
                k: v
                for k, v in inspect.getmembers(
                    type_, lambda x: not (inspect.isroutine(x))
                )
                if not k.startswith("__")
            }

        def _merge(a, b):
            for kb in b:
                if kb in a:
                    va = a[kb]
                    vb = b[kb]
                    if isinstance(va, dict) and isinstance(vb, dict):
                        _merge(va, vb)
                    else:
                        a[kb] = b[kb]
                else:
                    a[kb] = b[kb]

        names = []
        for k, v in sorted(
            cls._all_conf_by_name.items(), key=lambda x: -x[1].priority
        ):
            if hasattr(v, "to_dict"):
                _merge(cls._final_attrs, v.to_dict())
            else:
                _merge(cls._final_attrs, _get_attr(v))
            names.append(k)
        logger.info(f"override sequence is {' < '.join(names)}")
        cls._override_order = names
        conf = Configuration()
        for k, v in cls._final_attrs.items():
            conf.put(k, v)
        logger.info(f"conf after merge is {conf}")
        return conf


class ConfigurationManagerInterface(metaclass=ConfigurationManagerMeta):
    """only for configuration class to inherit"""

    pass


def get_conf(py_conf=None):
    """Get `ConfigurationManager` from args"""
    logger.info(f"Entering get_conf, original py_conf is {py_conf}")
    logger.info("current working director is %s" % os.getcwd())
    attribute_class = py_conf
    if py_conf:
        if isinstance(py_conf, str):
            attribute_class = reflect_util.get_class(py_conf)
    properties = dict()
    for i in dir(attribute_class):
        if not i.startswith("__"):
            properties[i] = getattr(attribute_class, i)

    attribute_class = type(
        "py_conf",
        (ConfigurationManagerInterface,),
        properties,
    )
    all_conf = ConfigurationManagerMeta.merge_configs()
    return all_conf
