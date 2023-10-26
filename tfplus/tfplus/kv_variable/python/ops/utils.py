# Copyright 2023 The TFPlus Authors. All rights reserved.
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
"""utils to use KvVariable"""

from __future__ import absolute_import, division, print_function

import linecache


def get_kv_variable_op_types():
  return ("KvVariable", "KvVariableV3", "KvVariableV4")


def is_kv_variable_op_type(op_type):
  return op_type in get_kv_variable_op_types()


def convert_stack(stack, include_func_start_lineno=False):
  """Converts a stack extracted using extract_stack() to a traceback stack.
    Copy from tf1.15 in tensorflow/python/util/tf_stack.py

    Args:
      stack: A list of n 5-tuples,
        (filename, lineno, name, frame_globals, func_start_lineno).
      include_func_start_lineno: True if function start line number should be
        included as the 5th entry in return tuples.

    Returns:
      A tuple of n 4-tuples or 5-tuples
      (filename, lineno, name, code, [optional: func_start_lineno]), where the
      code tuple element is calculated from the corresponding elements of the
      input tuple.
    """

  def _tuple_generator():  # pylint: disable=missing-docstring
    for frame in stack:
      filename = frame.filename
      lineno = frame.lineno
      linecache.checkcache(filename)
      line = linecache.getline(filename, lineno, frame.globals)
      if line:
        line = line.strip()
      else:
        line = None
      if include_func_start_lineno:
        yield (
            filename,
            lineno,
            frame.name,
            line,
            frame.func_start_lineno,
        )
      else:
        yield (filename, lineno, frame.name, line)

  return tuple(_tuple_generator())
