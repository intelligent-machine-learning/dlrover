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

import importlib
import importlib.util
import inspect
import pkgutil
from typing import Dict, Type, List, Set, Optional

from dlrover.python.common.log import default_logger as logger


class AutoExtensionRegistry:
    """
    Auto-discovery extension registry. It automatically discovers abstract
    classes decorated with @extensible and implementation classes decorated
    with @extension.
    """

    _scanned_modules: Set[str] = set()

    _original_impl: Dict[str, Type] = {}
    _extension_impl: Dict[str, Type] = {}

    @classmethod
    def auto_discover(cls, extension_package: str):
        """
        Do the auto-discovery for the extension package.

        Args:
            extension_package: extension package name
        """

        try:
            # scan basics and do registering
            basics = cls._scan_package_for_extensions("dlrover.python")
            for interface, impl_classes in basics.items():
                if impl_classes:
                    best_impl = max(
                        impl_classes,
                        key=lambda x: getattr(x, "_extension_priority", 0),
                    )
                    cls._original_impl[interface] = best_impl

            # scan extensions and do registering
            extensions = cls._scan_package_for_extensions(extension_package)
            for interface, impl_classes in extensions.items():
                if impl_classes:
                    best_impl = max(
                        impl_classes,
                        key=lambda x: getattr(x, "_extension_priority", 0),
                    )
                    cls._extension_impl[interface] = best_impl

        except Exception as e:
            logger.error(f"Extension discovery failed: {e}")
            raise e

    @classmethod
    def _scan_package_for_extensions(
        cls, package_name: str
    ) -> Dict[str, List[Type]]:
        extensions: Dict[str, List[Type]] = {}

        try:
            package = importlib.import_module(package_name)
            package_path = getattr(package, "__path__", [])

            for importer, module_name, ispkg in pkgutil.walk_packages(
                package_path, package_name + "."
            ):
                if module_name in cls._scanned_modules:
                    continue

                try:
                    module = importlib.import_module(module_name)
                    cls._scanned_modules.add(module_name)

                    # scan all the classed
                    for name, obj in inspect.getmembers(
                        module, inspect.isclass
                    ):
                        if cls._is_extension_class(obj):
                            # single inherited supported only
                            base_interface = obj.__bases__[0]
                            interface_key = f"{base_interface.__module__}.{base_interface.__name__}"

                            if interface_key not in extensions:
                                extensions[interface_key] = []
                            extensions[interface_key].append(obj)
                except (ImportError, AttributeError):
                    continue
        except (ImportError, AttributeError):
            pass
        except Exception:
            logger.exception(f"Package {package_name} scanning failed")
            raise

        return extensions

    @classmethod
    def _is_extension_class(cls, obj: Type) -> bool:
        if not inspect.isclass(obj):
            return False

        if not obj.__bases__:
            return False

        base = obj.__bases__[0]
        if not hasattr(base, "_is_extensible"):
            return False
        if not (
            inspect.isclass(base) and hasattr(base, "__abstractmethods__")
        ):
            return False

        return hasattr(base, "_is_extensible") or hasattr(
            obj, "_extension_priority"
        )

    @classmethod
    def get_original_class_by_interface(cls, interface: str) -> Optional[Type]:
        if interface in cls._original_impl:
            return cls._original_impl[interface]
        return None

    @classmethod
    def get_extension_class_by_interface(
        cls, interface: str
    ) -> Optional[Type]:
        if interface in cls._extension_impl:
            return cls._extension_impl[interface]
        return None


def extension(priority: int = 0):
    """Decorator for extension registry, for implementations."""

    def decorator(cls):
        cls._extension_priority = priority
        return cls

    return decorator


def extensible():
    """Decorator for extension registry, for interface."""

    def decorator(cls):
        cls._is_extensible = True
        return cls

    return decorator
