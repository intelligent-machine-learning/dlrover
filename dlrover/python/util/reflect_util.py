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

from __future__ import absolute_import, division, print_function

import importlib

from dlrover.python.common.log import default_logger as logger


class Importer(object):
    """Importer"""

    @staticmethod
    def import_module(module_name, msg=None, raise_if_error=True):
        """import module

        Args:
            module_name: "ab.c"
            msg: Error message
            raise_if_error: raise exception if error happens

        Returns:
            imported module: None for any exception
            status: True for a successfully import
        """
        try:
            module = importlib.import_module(module_name)
            return module, True
        except Exception as e:
            logger.debug(e)
            if msg:
                logger.warning(msg)
            if raise_if_error:
                raise
            return None, False

    @staticmethod
    def import_module_content(
        content_with_module, msg=None, raise_if_error=True
    ):
        """import a class or a function from a module

        Args:
            content_with_module: "ab.c.Clas"
            msg: Error message
            raise_if_error: raise exception if error happens

        Returns:
            imported content: None for any exception
            status: True for a successfully import
        """
        try:
            module, content = content_with_module.rsplit(".", 1)
            imported_module, _ = Importer.import_module(module, msg)
            imported_content = getattr(imported_module, content)
            return imported_content, True
        except Exception as e:
            logger.debug(e)
            if raise_if_error:
                raise
            return None, False

    @staticmethod
    def import_first_available_module_content(
        import_module_content_names, msg=None, raise_if_error=True
    ):
        """import multi class via `import_module_content`
           return first available class

        Args:
          import_module_content_names: tuple or list of name of module
          msg: same as `import_module_content`
          raise_if_error: same as `import_module_content`

        Returns:
            imported content: None for any exception
            status: True for a successfully import
        """
        ret_clazz, ok = None, False
        for name in import_module_content_names:
            ret_clazz, ok = Importer.import_module_content(
                name, msg=msg, raise_if_error=raise_if_error
            )
            if ok:
                break
        else:  # if end for loop normally fall into else
            logger.warning(
                f"can not find any module in {import_module_content_names}"
            )
        return ret_clazz, ok


def new_instance(module_class, *args, **kwargs):
    """Create a new instance using import_module
    Args:
        module_class: module class name, full path a.b.class_name
        args: passed to contructor of class
        kwargs: passed to contructor of class
    """
    ImportedClass, _ = Importer.import_module_content(module_class)
    return ImportedClass(*args, **kwargs)


def get_class(module_class):
    """Get the class"""
    return Importer.import_module_content(module_class)[0]
