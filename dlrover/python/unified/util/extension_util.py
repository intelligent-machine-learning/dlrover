# Copyright 2025 The DLRover Authors. All rights reserved.
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

from collections import defaultdict
from typing import ClassVar, Dict, List

from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.util.test_hooks import after_test_cleanup


class Extensible:
    __extensions: ClassVar[List[type]]

    @classmethod
    def extensions(cls) -> List[type]:
        assert super(cls) is Extensible, (
            "Extensible class must be directly derived from Extensible"
        )
        if not hasattr(cls, "__extensions"):
            cls.__extensions = []
            after_test_cleanup(cls.__extensions.clear)
        return cls.__extensions

    @classmethod
    def register_extension(cls, ext: type):
        cls.extensions().append(ext)

    @classmethod
    def build_mixed_class(cls) -> type:
        extensions = cls.extensions()
        if len(extensions) == 0:
            return cls
        assert all(issubclass(ext, cls) for ext in extensions), (
            "All extensions must be subclasses of the base class."
        )

        try:
            ret = type(f"Mixed_{cls.__name__}", tuple(extensions) + (cls,), {})
            logger.info(f"{ret.__name__} built with {ret.__mro__}")
            return ret
        except Exception:
            logger.exception(f"Failed to create mixed class for {cls}")
            detect_mixin_conflicts(extensions, cls)
            raise RuntimeError(
                f"Cannot construct '{cls.__name__}' due to method conflicts"
            )


def get_overridden_methods(mixin: type, base: type) -> set:
    overridden = set()
    for name in dir(mixin):
        if name.startswith("__"):
            continue
        mixin_attr = getattr(mixin, name)
        base_attr = getattr(base, name, None)
        if mixin_attr != base_attr or callable(mixin_attr):
            overridden.add(name)
    return overridden


def detect_mixin_conflicts(mixins: List[type], base: type):
    method_map: Dict[str, List[type]] = defaultdict(list)
    for mixin in mixins:
        for method in get_overridden_methods(mixin, base):
            method_map[method].append(mixin)
    for method, impl in method_map.items():
        if len(impl) > 1:
            logger.warning(
                f"Method {method} is overridden by multiple mixins: {impl}"
            )

    return {k: v for k, v in method_map.items() if len(v) > 1}
