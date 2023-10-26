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
"""tfplus common"""
from __future__ import absolute_import, division, print_function

import ctypes as ct
import inspect
import os
import pkgutil
import sys

import tensorflow as tf
from packaging import version
from tensorflow import errors
from tensorflow.python.platform import tf_logging as logging


def _load_library(filename, lib="op", load_fn=None):
  """_load_library"""
  f = inspect.getfile(sys._getframe(1))  # pylint: disable=protected-access

  # Construct filename
  f = os.path.join(os.path.dirname(f), filename)
  suffix = get_suffix()
  if os.path.exists(f + suffix):
    f = f + suffix
  filenames = [f]

  # Add datapath to load if en var is set, used for running tests where shared
  # libraries are built in a different path
  datapath = os.environ.get("TFPLUS_DATAPATH")
  if datapath is not None:
    # Build filename from `datapath` + `package_name` + `relpath_to_library`
    f = os.path.join(datapath, os.path.relpath(f, os.path.dirname(filename)))
    suffix = get_suffix()
    if os.path.exists(f + suffix):
      f = f + suffix
    filenames.append(f)

  # Function to load the library, return True if file system library is loaded
  load_fn = load_fn or (
      tf.load_op_library if lib == "op" else
      lambda f: tf.compat.v1.load_file_system_library(f) is None)

  # Try to load all paths for file, fail if none succeed
  errs = []
  for f in filenames:
    try:
      l = load_fn(f)
      if l is not None:
        return l
      # if load_fn is tf.load_library:
      #   return
    except errors.NotFoundError as e:
      errs.append(str(e))
  raise NotImplementedError(
      "unable to open file: " +
      "{}, from paths: {}\ncaused by: {}".format(filename, filenames, errs))


def is_tf_1_13_or_higher():
  if version.parse(tf.__version__) >= version.parse("1.13.0"):
    return True
  return False


def is_pai_tf():
  return "pai" in tf.__version__.lower()


def get_suffix():
  """helper to get suffix of shared object"""
  suffix = os.getenv("SO_SUFFIX", None)
  if suffix:
    return suffix
  if "pai" in tf.__version__.lower():
    if tf.test.is_built_with_cuda():
      return ".eflops"
    if pkgutil.find_loader("xdl") is not None:
      return ".xdl"
    return ".pai"
  return ""


def ctypes_load_library(path):
  ct.cdll.LoadLibrary(path)
  logging.info("`%s` is loaded.", path)
