# Copyright 2023 The DLRover Authors. All rights reserved.
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

from collections.abc import Iterable


def should_failover(failover_level):
    return failover_level > 0


def singleton(constructor):
    """
    Description:
        Always return a singleton instance which is created by the constructor.
    Args:
        constructor: An callable object
    Return:
        Singleton instance
    """
    env = [None]

    def wrap(*args, **kwargs):
        if env[0] is None:
            env[0] = constructor(*args, **kwargs)
        return env[0]

    return wrap


@singleton
class GlobalDict(dict):
    """A dictionary can be accessed from everywhere"""


class DatasetArgs(object):
    """DatasetArgs"""

    def __init__(self, args=None, kwargs=None):
        if isinstance(args, str):
            args = (args,)
        if args and not isinstance(args, Iterable):
            args = (args,)
        if kwargs and not isinstance(kwargs, dict):
            raise ValueError("kwargs for DatasetArgs must be a dict")
        self._args = args or ()
        self._kwargs = kwargs or {}

    @property
    def args(self):
        return self._args

    @property
    def kwargs(self):
        return self._kwargs


class Property(object):
    """Property descriptor"""

    def __init__(self, name, property_name, property_doc, default=None):
        self._name = name
        self._names = property_name
        self._default = default
        self._doc = property_doc

    def __get__(self, obj, objtype):
        return obj.__dict__.get(self._name, self._default)

    def __set__(self, obj, value):
        if obj is None:
            return self
        for name in self._names:
            obj.__dict__[name] = value

    @property
    def __doc__(self):
        return self._doc


def add_prop(*field_doc_pairs, **defaults):
    """Add a property to the class

    Args:
        *field_doc_pairs: list of tuple(name, doc)
        **defaults: possible default value of key, default is None
    """

    def decorator(clz):
        """Decorator function"""

        def create(**kwargs):
            instance = clz()
            for key, value in kwargs.items():
                if key not in instance.added_prop:
                    raise ValueError(
                        "Unknown property:%s, valid properties: %s"
                        % (key, instance.added_prop)
                    )
                setattr(instance, key, value)
            return instance

        added_prop = []
        for f in field_doc_pairs:
            names = f[0].split(" ")
            if len(f) > 2:
                default = f[2]
            else:
                default = None
            for name in names:
                if default is None:
                    default = defaults.get(name, None)
                setattr(
                    clz,
                    name,
                    Property(name, names, f[1], default),
                )
            added_prop.extend(names)
        setattr(clz, "added_prop", added_prop)
        setattr(clz, "create", staticmethod(create))
        return clz

    return decorator
