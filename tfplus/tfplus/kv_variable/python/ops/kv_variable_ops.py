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
"""Ops to use KvVariable. This file refers to original
tensorflow ResourceVariable class implementation in
https://github.com/tensorflow/tensorflow/blob/v1.13.1/tensorflow/python/ops/resource_variable_ops.py.
We inherit ResourceVariable to implement KvVariable.
"""

# pylint: disable=cyclic-import
from __future__ import absolute_import, division, print_function

import inspect
import itertools
import re
from collections import OrderedDict, defaultdict
from contextlib import contextmanager

import tensorflow as tf
from tensorflow.core.framework import attr_value_pb2, variable_pb2
from tensorflow.python.eager import context, tape
from tensorflow.python.framework import (
    dtypes,
    ops,
    registry,
    tensor_conversion_registry,
    tensor_shape,
)
from tensorflow.python.ops import variables
from tensorflow.python.ops.variables import PartitionedVariable

try:
  from tensorflow.python.keras.backend import get_graph
except ImportError:
  get_graph = ops.get_default_graph
from tensorflow.python.ops import (
    array_ops,
    control_flow_ops,
    io_ops,
    resource_variable_ops,
    state_ops,
)
from tensorflow.python.ops import variable_scope as tf_variable_scope
from tensorflow.python.ops.resource_variable_ops import _dense_var_to_tensor
from tensorflow.python.ops.variables import (
    _try_guard_against_uninitialized_dependencies,
)
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import device_setter
from tensorflow.python.training.saver import BaseSaverBuilder

try:
  from tensorflow.python.training.saving import saveable_object_util
except ImportError:
  from tensorflow.python.training import saver as saveable_object_util

from tensorflow.python.training.tracking import base as checkpointable
from tensorflow.python.util import compat

from tfplus.common import _load_library, is_tf_1_13_or_higher
from tfplus.kv_variable.kernels.hybrid_embedding.storage_config_pb2 import *  # pylint: disable=unused-wildcard-import
from tfplus.kv_variable.python.ops import utils, variable_scope
from tfplus.kv_variable.python.ops.kv_variable_options import parse_from_string

gen_kv_variable_ops = _load_library("_kv_variable_ops.so")

# add KvVariableV2 op to PS ops tuple collection
device_setter.STANDARD_PS_OPS = (device_setter.STANDARD_PS_OPS +
                                 utils.get_kv_variable_op_types())  # pylint: disable=bad-whitespace

_TF_PLUS_SAVER_MODE = 1

_QUERY_VALID_IDS = {}

_ENABLE_DELTA_EXPORT = False
_ENABLE_SAVE_V3 = False
_ENABLE_FULL_OR_DELTA_IMPORT_V2 = False
_ENABLE_AUTO_PARTITION = True
_ENABLE_AUTO_HYBRID_EMBEDDING = []
_CKPT_DIR_OR_FILE = None
_NEW_NUM_SHARDS_DICT = None
TFPLUS_SAVER_BUILDER = [BaseSaverBuilder]
ORI_SAVE_V2 = None
ORI_ADDRESTOREOPS = None
IN_RESTORE_STATE = False
IS_TRAINING = True


@contextmanager
def set_restore_state():
  global IN_RESTORE_STATE
  IN_RESTORE_STATE = True
  yield
  IN_RESTORE_STATE = False


def _BaseSaverBuilder_AddRestoreOps(  # pylint: disable=invalid-name
    self,
    filename_tensor,
    saveables,
    restore_sequentially,
    reshape,
    preferred_shard=-1,
    name="restore_all",
):
  """Add operations to restore saveables.

  Args:
    filename_tensor: Tensor for the path of the file to load.
    saveables: A list of SaveableObject objects.
    restore_sequentially: True if we want to restore variables sequentially
      within a shard.
    reshape: True if we want to reshape loaded tensors to the shape of
      the corresponding variable.
    preferred_shard: Shard to open first when loading a sharded file.
    name: Name for the returned op.

  Returns:
    An Operation that restores the variables.
  """
  first_kv_saveable = None
  one_kv_variable = None
  for saveable in saveables:
    if isinstance(saveable, KvVariableSaveable):
      first_kv_saveable = saveable
      one_kv_variable = first_kv_saveable.var
      break
  if (first_kv_saveable is not None
      and first_kv_saveable.var.is_ignore_eflops_device_fn):
    # Due eflops_device_fn is only support KvVariable,
    # We colocate RestoreV2 op with Kv variables.
    from tfplus.kv_variable.python.ops.eflops_embedding_ops import (
        eflops_force_colocate_with,
    )

    with eflops_force_colocate_with(first_kv_saveable.var.device):  # pylint: disable=protected-access
      all_tensors = self.bulk_restore(
          filename_tensor,
          saveables,
          preferred_shard,
          restore_sequentially,
      )
  else:
    all_tensors = self.bulk_restore(filename_tensor, saveables,
                                    preferred_shard, restore_sequentially)
  assign_ops = []
  idx = 0
  # Load and optionally reshape on the CPU, as string tensors are not
  # available on the GPU.
  # TODO(touts): Re-enable restore on GPU when we can support annotating
  # string tensors as "HostMemory" inputs.
  for saveable in saveables:
    shapes = None
    if reshape:
      # Compute the shapes, let the restore op decide if and how to do
      # the reshape.
      shapes = []
      for spec in saveable.specs:
        v = spec.tensor
        shape = v.get_shape()
        if not shape.is_fully_defined():
          shape = array_ops.shape(v)
        shapes.append(shape)
    saveable_tensors = all_tensors[idx:idx + len(saveable.specs)]
    idx += len(saveable.specs)
    if isinstance(saveable, KvVariableSaveable):
      assign_ops.append(
          saveable.restore(saveable_tensors, shapes, filename_tensor))
    else:
      assign_ops.append(saveable.restore(saveable_tensors, shapes))
  if (one_kv_variable is not None
      and one_kv_variable.is_ignore_eflops_device_fn):
    # colocate restore_shard with saveable
    from tfplus.kv_variable.python.ops.eflops_embedding_ops import (
        eflops_force_colocate_with,
    )

    with eflops_force_colocate_with(one_kv_variable.device):  # pylint: disable=protected-access
      with ops.colocate_with(one_kv_variable.device):
        return control_flow_ops.group(*assign_ops, name=name)
  return control_flow_ops.group(*assign_ops, name=name)


BaseSaverBuilder._AddRestoreOps = ( # pylint: disable=protected-access
    _BaseSaverBuilder_AddRestoreOps
)


def enable_delta_export():
  global _ENABLE_DELTA_EXPORT
  _ENABLE_DELTA_EXPORT = True


def delta_export_enabled():
  return _ENABLE_DELTA_EXPORT


def disable_delta_export():
  global _ENABLE_DELTA_EXPORT
  _ENABLE_DELTA_EXPORT = False


def enable_full_or_delta_import_v2():
  global _ENABLE_FULL_OR_DELTA_IMPORT_V2
  _ENABLE_FULL_OR_DELTA_IMPORT_V2 = True


def full_or_delta_import_v2_enabled():
  return _ENABLE_FULL_OR_DELTA_IMPORT_V2


def disable_full_or_delta_import_v2():
  global _ENABLE_FULL_OR_DELTA_IMPORT_V2
  _ENABLE_FULL_OR_DELTA_IMPORT_V2 = False


def enable_save_v3():
  """Enable SaveV3 OP"""
  global _ENABLE_SAVE_V3
  global ORI_SAVE_V2
  global ORI_ADDRESTOREOPS
  if _ENABLE_SAVE_V3:
    return
  _ENABLE_SAVE_V3 = True
  ORI_SAVE_V2 = io_ops.save_v2

  def save_v2_hook(filename_tensor,
                   tensor_names,
                   tensor_slices,
                   tensors,
                   has_ev=None):  # pylint: disable=unused-argument
    first_n_input = get_or_create_first_n()
    do_full_export = get_or_create_do_full_export()
    return gen_kv_variable_ops.save_v3(
        filename_tensor,
        tensor_names,
        tensor_slices,
        first_n_input,
        do_full_export,
        tensors,
        freq_use_uint32=delta_export_enabled(),
    )

  ORI_ADDRESTOREOPS = []

  def hook_add_restore_ops(origin_fn):

    def wrapper(*args, **kwargs):
      with set_restore_state():
        return origin_fn(*args, **kwargs)

    return wrapper

  for saver_builder in TFPLUS_SAVER_BUILDER:
    origin_add_restore_ops = (saver_builder._AddRestoreOps)  # pylint: disable=protected-access
    ORI_ADDRESTOREOPS.append(origin_add_restore_ops)
    saver_builder._AddRestoreOps = hook_add_restore_ops(  # pylint: disable=protected-access
        origin_add_restore_ops)

  io_ops.save_v2 = save_v2_hook
  logging.info("enable save_v3, disable save_v2.")


def save_v3_enabled():
  """Check SaveV3 OP is enabled"""
  return _ENABLE_SAVE_V3


def disable_save_v3():
  """Disable SaveV3 OP"""
  global _ENABLE_SAVE_V3
  global ORI_SAVE_V2
  global ORI_ADDRESTOREOPS
  if not _ENABLE_SAVE_V3:
    return
  _ENABLE_SAVE_V3 = False
  io_ops.save_v2 = ORI_SAVE_V2
  ORI_SAVE_V2 = None
  for saver_builder, origin_add_restore_ops in zip(TFPLUS_SAVER_BUILDER,
                                                   ORI_ADDRESTOREOPS):
    saver_builder._AddRestoreOps = (  # pylint: disable=protected-access
        origin_add_restore_ops)
  ORI_ADDRESTOREOPS = None
  logging.info("disable save_v3, enable save_v2.")


def _query_kv_feature_size(var):
  if not isinstance(var, KvVariable):
    return  # do nothing
  var_name = var.handle.op.name
  graph = get_graph()
  if graph not in _QUERY_VALID_IDS:
    _QUERY_VALID_IDS[graph] = {}
  if var_name not in _QUERY_VALID_IDS[graph]:
    _QUERY_VALID_IDS[graph][var_name] = var.total_count


def get_ckpt_dir_or_file():
  return _CKPT_DIR_OR_FILE


def set_new_num_shards(new_num_shards):
  """Set new num shards

    Args:
        new_num_shards (dict), optional): a dict mapping for
          variable name to num shards, variable name should remove part
          info and ':0-xxx', it can be generated by KvVariable.get_generic_name.
          For example, a variable name 'embedding/GroupAdaHessian' with
          new num shards 4, the name of the corresponding tensor may be
          'embedding/part_1/GroupAdaHessian:0-keys'. Defaults to None.
          For example, a key-value pair can be
          {"embedding/GroupAdaHessian": 4}
    """
  global _NEW_NUM_SHARDS_DICT
  if _ENABLE_AUTO_PARTITION:
    _NEW_NUM_SHARDS_DICT = new_num_shards


def get_new_num_shards():
  return _NEW_NUM_SHARDS_DICT


def get_kv_feature_size():
  graph = get_graph()
  if graph not in _QUERY_VALID_IDS:
    return {}
  return _QUERY_VALID_IDS[graph]


def reset_session():
  _QUERY_VALID_IDS.clear()
  ops.reset_default_graph()


def tfplus_saver_mode():
  global _TF_PLUS_SAVER_MODE  # pylint: disable=global-variable-not-assigned
  if context.executing_eagerly():
    return 1  # Fallback to train mode as default
  return _TF_PLUS_SAVER_MODE


def set_tfplus_saver_mode(value):
  """Sets the learning phase to a fixed value.

  Arguments:
    value: Learning phase value, either 0 or 1 (integers).
        When set 1, we will export all blacklist/frequencylist
        for restore to retrain.
      When set 0, we will only export valid keys and values, these are enough
        for predict, and will reduce model size.

  Raises:
      ValueError: if `value` is neither `0` nor `1`.
  """
  global _TF_PLUS_SAVER_MODE  # pylint: disable=global-variable-not-assigned
  if value not in {0, 1}:
    raise ValueError("Expected tfplus saver mode to be 0 or 1.")
  with ops.init_scope():
    if not context.executing_eagerly():
      _TF_PLUS_SAVER_MODE = value
  if value == 0:
    logging.info("tfplus saver mode in Prediction mode.")
  else:
    logging.info("tfplus saver mode in Training mode.")


def get_or_create_do_full_export():
  """
    get or create a placeholder tensor for do_full_export with
    default value true and name equal to do_full_export
    """
  try:
    do_full_export = ops.get_default_graph().get_tensor_by_name(
        "do_full_export:0")
  except Exception:  # pylint: disable=broad-except
    with ops.name_scope(""):
      do_full_export = array_ops.placeholder_with_default(
          True, shape=(), name='do_full_export')
  return do_full_export


def get_or_create_is_loading_finished():
  """
    get or create a placeholder tensor for is_loading_finished with
    default value 0 and name equal to is_loading_finished
    """
  try:
    is_loading_finished = ops.get_default_graph().get_tensor_by_name(
        "is_loading_finished:0")
  except Exception:  # pylint: disable=broad-except
    with ops.name_scope(""):
      is_loading_finished = array_ops.placeholder_with_default(
          True, shape=(), name="is_loading_finished")
  return is_loading_finished


def get_or_create_first_n():
  """
    get or create a placeholder tensor for first_n with
    default value 3 and name equal to first_n
    """
  try:
    first_n = ops.get_default_graph().get_tensor_by_name("first_n:0")
  except Exception:  # pylint: disable=broad-except
    with ops.name_scope(""):
      first_n = array_ops.placeholder_with_default(8,
                                                   shape=(),
                                                   name="first_n")
  return first_n


def _eager_safe_variable_handle(
    shape,
    key_dtype,
    value_dtype,
    shared_name,
    name,
    graph_mode,
    enter_threshold=0,
    kv_options=variable_scope.default_kv_option(),
):
  """Creates a kvvariable handle with information to do shape inference."""
  # TODO(tongsuo.ts): We should add tests for eager mode later
  container = (ops.get_default_graph()._container or "")  # pylint: disable=protected-access
  shape = tensor_shape.as_shape(shape.as_list()[1])
  handle = gen_kv_variable_ops.kv_variable(
      value_shape=shape,
      key_dtype=key_dtype,
      value_dtype=value_dtype,
      shared_name=shared_name,
      name=name,
      container=container,
      enter_threshold=enter_threshold,
  )
  if graph_mode:
    # pylint: disable=protected-access
    handle._handle_data = resource_variable_ops.get_resource_handle_data(
        handle)
    return handle

  exists = gen_kv_variable_ops.kv_variable_is_initialized_v2(handle)
  if exists:
    raise ValueError("variable object with name '%s' already created. Use "
                     "get_kv_variable() if reuse is desired." % shared_name)
  with context.graph_mode(), ops.Graph().as_default() as graph:
    if kv_options.has_path():
      h = gen_kv_variable_ops.kv_variable_v4(
          value_shape=shape,
          key_dtype=key_dtype,
          value_dtype=value_dtype,
          shared_name=shared_name,
          name=name,
          container=container,
          storage_option=kv_options.serialize_string(),
      )
    else:
      h = gen_kv_variable_ops.kv_variable(
          value_shape=shape,
          key_dtype=key_dtype,
          value_dtype=value_dtype,
          shared_name=shared_name,
          name=name,
          container=container,
      )
    # Tensor._handle_data contains information for the shape-inference code to
    # know the shape and dtype of the variable pointed to by a handle. Since
    # shape inference doesn't run in eager mode we copy this data here for when
    # the handle is captured by an eager mode function.
    # pylint: disable=protected-access
    handle._handle_data = resource_variable_ops.get_resource_handle_data(h)
    # pylint: enable=protected-access
  # Clean up op->graph->op reference cycles.
  ops.dismantle_graph(graph)
  return handle


class EagerResourceDeleter(resource_variable_ops.EagerResourceDeleter):
  """An object which cleans up a resource handle.

  An alternative to defining a __del__ method on an object. The intended use is
  that ResourceVariables or other objects with resource handles will maintain a
  single reference to this object. When the parent object is collected, this
  object will be too. Even if the parent object is part of a reference cycle,
  the cycle will be collectable.
  """

  def __del__(self):
    # Resources follow object-identity when executing eagerly, so it is safe to
    # delete the resource we have a handle to.
    try:
      # This resource was created in eager mode. However, this destructor may be
      # running in graph mode (especially during unit tests). To clean up
      # successfully, we switch back into eager mode temporarily.
      with context.eager_mode():
        with ops.device(self._handle_device):
          gen_kv_variable_ops.destroy_kv_variable_op_v2(
              self._handle, ignore_lookup_error=True)
    except TypeError:
      # Suppress some exceptions, mainly for the case when we're running on
      # module deletion. Things that can go wrong include the context module
      # already being unloaded, self._handle._handle_data no longer being
      # valid, and so on. Printing warnings in these cases is silly
      # (exceptions raised from __del__ are printed as warnings to stderr).
      pass  # 'NoneType' object is not callable when the handle has been
      # partially unloaded.
    except AttributeError:
      pass  # 'NoneType' object has no attribute 'eager_mode' when context has
      # been unloaded. Will catch other module unloads as well.


def add_inner_table_for_multi_hash(params, names):
  """
    add inner table for multi hash
    caller should add all subtable once
    """
  scope_name = tf_variable_scope.get_variable_scope().name
  if scope_name != "":
    scope_name += "/"
  if isinstance(params, PartitionedVariable):
    var = list(params)
    for part, v in enumerate(var):
      v.add_multi_level_hash(
          [f"{scope_name}{name}/part_{part}:0" for name in names], names)
  else:
    params.add_multi_level_hash([f"{scope_name}{name}:0" for name in names],
                                names)


class KvVariable(resource_variable_ops.ResourceVariable): # pylint: disable=abstract-method
  """KvVariable extends ResourceVariable"""

  # Create a unique counter for this class
  unique_counter = itertools.count()

  # pylint: disable=super-init-not-called
  def __init__(
      self,
      initial_value=None,
      trainable=True,
      collections=None,
      name=None,
      key_dtype=None,
      value_dtype=None,
      variable_def=None,
      import_scope=None,
      constraint=None,
      enter_threshold=0,
      kv_options=variable_scope.default_kv_option(),
      **kwargs,
  ):
    """Creates a variable.

    Args:
      initial_value: A `Tensor`, or Python object convertible to a `Tensor`,
        which is the initial value for the Variable. The initial value must have
        a shape specified unless `validate_shape` is set to False. Can also be a
        callable with no argument that returns the initial value when called.
        (Note that initializer functions from init_ops.py must first be bound
         to a shape before being used here.)
      trainable: If `True`, the default, also adds the variable to the graph
        collection `GraphKeys.TRAINABLE_VARIABLES`. This collection is used as
        the default list of variables to use by the `Optimizer` classes.
      collections: List of graph collections keys. The new variable is added to
        these collections. Defaults to `[GraphKeys.GLOBAL_VARIABLES]`.
      name: Optional name for the variable. Defaults to `'Variable'` and gets
        uniquified automatically.
      key_dtype: Key dtype.
      value_dtype: Value dtype. If set, initial_value will be converted to the
        given type. If None, either the datatype will be kept (if initial_value
        is a Tensor) or float32 will be used (if it is a Python object
        convertible to a Tensor).
      variable_def: `VariableDef` protocol buffer. If not None, recreates the
        `ResourceVariable` object with its contents. `variable_def` and other
        arguments (except for import_scope) are mutually exclusive.
      import_scope: Optional `string`. Name scope to add to the
        ResourceVariable. Only used when `variable_def` is provided.
      constraint: An optional projection function to be applied to the variable
        after being updated by an `Optimizer` (e.g. used to implement norm
        constraints or value constraints for layer weights). The function must
        take as input the unprojected Tensor representing the value of the
        variable and return the Tensor for the projected value
        (which must have the same shape). Constraints are not safe to
        use when doing asynchronous distributed training.
      enter_threshold: An optional scalar value to be used on frequency filter.
        If we set it larger than 0, a key can be add to table and update by
        optimizer only when it appears more than enter_threshold times during
        training phase.

    Raises:
      ValueError: If the initial value is not specified, or does not have a
        shape and `validate_shape` is `True`.

    @compatibility(eager)
    When Eager Execution is enabled, the default for the `collections` argument
    is `None`, which signifies that this `Variable` will not be added to any
    collections.
    @end_compatibility
    """
    if variable_def:
      if initial_value is not None:
        raise ValueError("variable_def and initial_value are mutually "
                         "exclusive.")
      if context.executing_eagerly():
        raise ValueError("Creating KvVariable from variable_def is "
                         "not supported when eager execution is enabled.")

      self._init_from_proto(variable_def, import_scope=import_scope)
    else:
      self._init_from_arguments(
          initial_value=initial_value,
          trainable=trainable,
          collections=collections,
          name=name,
          key_dtype=key_dtype,
          value_dtype=value_dtype,
          constraint=constraint,
          enter_threshold=enter_threshold,
          kv_options=kv_options,
      )
    self._enther_threshold = enter_threshold
    self._is_multi_level = kwargs.get("is_multi_level", False)
    self._is_ignore_eflops_device_fn = kwargs.get(
        "is_ignore_eflops_device_fn", False)
    self._multi_level_names = []
    self._multi_level_names_origin = []
    self._multi_level_names_index = {}
    self._kv_options = kv_options
    self._num_concat_opt_vars = 1
    self._use_ps_os_eflops = kwargs.get("use_ps_on_eflops", False)
    if not ops.get_collection(variable_scope.GRAPHSHARDS_KEY):
      ops.add_to_collection(variable_scope.GRAPHSHARDS_KEY, defaultdict(int))
    graph_embedding_num_shards = ops.get_collection(
        variable_scope.GRAPHSHARDS_KEY)[0]
    graph_embedding_num_shards[self.get_generic_name(self.handle.name)] += 1

  def add_multi_level_hash(self, names, origin_names):
    """
        add multi hash, names is for savespec, concat with scope and partition
        """
    if self._is_multi_level:
      d = defaultdict(int)
      for n in origin_names:
        d[n] += 1
      msg = [f'input "{k}" has count {v}' for k, v in d.items() if v > 1]
      if msg:
        msg = "\n".join(msg)
        raise ValueError("Found duplicate names when add sub hash "
                         f"your input is \n {msg}")

      duplicate_names = set(origin_names) & set(
          self._multi_level_names_origin)
      if duplicate_names:
        raise ValueError("Found duplicate names when add sub hash "
                         f"duplicate names are {duplicate_names}")
      names_in_tensor = ops.convert_to_tensor(names, dtype=dtypes.string)
      for name, origin_name in zip(names, origin_names):
        self._multi_level_names_index[name] = len(self._multi_level_names)
        self._multi_level_names.append(name)
        self._multi_level_names_origin.append(origin_name)
        logging.info(f"set {name} for multi hash table {self._handle.name}")
      insert = gen_kv_variable_ops.append_kv_variable_for_multi_hash(
          self._handle, names_in_tensor)
      ops.add_to_collections(ops.GraphKeys.TABLE_INITIALIZERS, insert)
    else:
      logging.warning("Kvvariable is not set to multi level, pass")

  def copy_multi_level_hash_config(self, other):
    """
        copy multi config from other kv variable
        this function is called when optimizer create
        slot variable which type is KvVariable
        """
    if isinstance(other, KvVariable):
      add_inner_table_for_multi_hash(other, self._multi_level_names_origin)
    else:
      logging.warning("Variable is not Kvvariable, "
                      "skip set attr for multi level hash")

  def _init_from_arguments(
      self,
      initial_value=None,
      trainable=True,
      collections=None,
      name=None,
      key_dtype=None,
      value_dtype=None,
      constraint=None,
      enter_threshold=0,
      kv_options=variable_scope.default_kv_option(),
  ):
    """Creates a variable.

    Args:
      initial_value: A `Tensor`, or Python object convertible to a `Tensor`,
        which is the initial value for the Variable. The initial value must have
        a shape specified unless `validate_shape` is set to False. Can also be a
        callable with no argument that returns the initial value when called.
        (Note that initializer functions from init_ops.py must first be bound
         to a shape before being used here.)
      trainable: If `True`, the default, also adds the variable to the graph
        collection `GraphKeys.TRAINABLE_VARIABLES`. This collection is used as
        the default list of variables to use by the `Optimizer` classes.
      collections: List of graph collections keys. The new variable is added to
        these collections. Defaults to `[GraphKeys.GLOBAL_VARIABLES]`.
      name: Optional name for the variable. Defaults to `'Variable'` and gets
        uniquified automatically.
      key_dtype: Key dtype.
      value_dtype: If set, initial_value will be converted to the given type.
        If None, either the datatype will be kept (if initial_value is
       a Tensor) or float32 will be used (if it is a Python object convertible
       to a Tensor).
      constraint: An optional projection function to be applied to the variable
        after being updated by an `Optimizer` (e.g. used to implement norm
        constraints or value constraints for layer weights). The function must
        take as input the unprojected Tensor representing the value of the
        variable and return the Tensor for the projected value
        (which must have the same shape). Constraints are not safe to
        use when doing asynchronous distributed training.
      enter_threshold: An optional scalar value to be used on frequency filter.
        If we set it larger than 0, a key can be add to table and update by
        optimizer only when it appears more than enter_threshold times during
        training phase.
      kv_options: KvVariable storage option.

    Raises:
      ValueError: If the initial value is not specified, or does not have a
        shape and `validate_shape` is `True`.

    @compatibility(eager)
    When Eager Execution is enabled, variables are never added to collections.
    It is not implicitly added to the `GLOBAL_VARIABLES` or
    `TRAINABLE_VARIABLES` collections, and the `collections` argument is
    ignored.
    @end_compatibility
    """
    if initial_value is None:
      raise ValueError("initial_value must be specified.")
    init_from_fn = callable(initial_value)

    if (isinstance(initial_value, ops.Tensor)
        and hasattr(initial_value, "graph")
        and initial_value.graph.building_function):
      raise ValueError("Tensor-typed variable initializers must either be "
                       "wrapped in an init_scope or callable "
                       "(e.g., `tfplus.KvVariable(lambda : "
                       "tf.truncated_normal([10, 40]))`) when building "
                       "functions. Please file a feature request if this "
                       "restriction inconveniences you.")

    if collections is None:
      collections = [ops.GraphKeys.GLOBAL_VARIABLES]
    if not isinstance(collections, (list, tuple, set)):
      raise ValueError(
          "collections argument to KvVariable constructor must be a list,"
          " tuple, or set. Got %s of type %s" %
          (collections, type(collections)))
    if constraint is not None and not callable(constraint):
      raise ValueError("The `constraint` argument must be a callable.")

    if isinstance(initial_value, checkpointable.CheckpointInitialValue):
      self._maybe_initialize_checkpointable()
      self._update_uid = initial_value.checkpoint_position.restore_uid
      initial_value = initial_value.wrapped_value

    self._trainable = trainable
    if trainable and ops.GraphKeys.TRAINABLE_VARIABLES not in collections:
      collections = list(collections) + [ops.GraphKeys.TRAINABLE_VARIABLES]
    self._save_slice_info = None
    # Store the graph key so optimizers know how to only retrieve variables from
    # this graph.
    self._graph_key = (ops.get_default_graph()._graph_key)  # pylint: disable=protected-access
    self._key_dtype = key_dtype
    self._kv_options = kv_options

    with ops.init_scope():
      self._in_graph_mode = not context.executing_eagerly()
      with ops.name_scope(
          name, "KvVariable",
          [] if init_from_fn else [initial_value]) as var_scope_name:
        handle_name = name[:-1] if name[-1] == "/" else name

        if self._in_graph_mode:
          shared_name = handle_name
        else:
          shared_name = "%s_%d" % (handle_name, ops.uid())

        attr = attr_value_pb2.AttrValue(
            list=attr_value_pb2.AttrValue.ListValue(
                s=[compat.as_bytes("loc:@%s" % handle_name)]))
        with ops.get_default_graph()._attr_scope({"_class": attr}):  # pylint: disable=protected-access
          with ops.name_scope("Initializer"), ops.device(None):
            initial_value = ops.convert_to_tensor(
                initial_value() if init_from_fn else initial_value,
                name="initial_value",
                dtype=value_dtype,
            )
            self._shape = initial_value.shape

          # Under the scope of attribute
          self._handle = _eager_safe_variable_handle(
              shape=self._shape,
              key_dtype=key_dtype,
              value_dtype=initial_value.dtype.base_dtype,
              shared_name=shared_name,
              name=var_scope_name,
              graph_mode=self._in_graph_mode,
              enter_threshold=enter_threshold,
              kv_options=kv_options,
          )

        # pylint: disable=protected-access
        if (self._in_graph_mode and initial_value is not None
            and initial_value.op._get_control_flow_context() is not None):
          raise ValueError(
              "Initializer for KvVariable %s is from inside a control-flow "
              "construct, such as a loop or conditional. When creating a "
              "variable inside a loop or conditional, use a lambda as the "
              "initializer." % var_scope_name)
        # pylint: enable=protected-access
        self._value_dtype = initial_value.dtype.base_dtype
        self._unique_id = shared_name
        self._initial_value = (initial_value if self._in_graph_mode else None)
        self._handle_name = handle_name + ":0"
        self._constraint = constraint

        if self._in_graph_mode:
          with ops.name_scope("IsInitialized"):
            self._is_initialized_op = (
                gen_kv_variable_ops.kv_variable_is_initialized_v2(
                    self._handle))
          if initial_value is not None:
            with ops.name_scope("Init") as n, ops.colocate_with(self._handle):
              self._initializer_op = gen_kv_variable_ops.init_kv_variable_v2(
                  self._handle,
                  _try_guard_against_uninitialized_dependencies(
                      n, initial_value),
                  name=n,
              )
          with ops.name_scope("Read"), ops.colocate_with(self._handle):
            # Manually assign reads to the handle's device to avoid log
            # messages.
            with ops.device(self._handle.device):
              value = self._read_variable_op()[1]
            self._graph_element = value
        else:
          gen_kv_variable_ops.init_kv_variable_v2(self._handle, initial_value)
          self._is_initialized_op = None
          self._initializer_op = None
          self._graph_element = None
        if not context.executing_eagerly():
          ops.add_to_collections(collections, self)
        elif ops.GraphKeys.GLOBAL_STEP in collections:
          ops.add_to_collections(ops.GraphKeys.GLOBAL_STEP, self)

    if not self._in_graph_mode:
      self._handle_deleter = EagerResourceDeleter(
          handle=self._handle, handle_device=self._handle.device)
    self._cached_shape_as_list = None

  def _init_from_proto(self, variable_def, import_scope=None):
    """Initializes from `VariableDef` proto."""
    # Note that init_from_proto is currently not supported in Eager mode.
    assert not context.executing_eagerly()
    self._in_graph_mode = True
    assert isinstance(variable_def, variable_pb2.VariableDef)
    if not variable_def.is_resource:
      raise ValueError("Trying to restore Variable as ResourceVariable.")

    # Create from variable_def.
    g = ops.get_default_graph()
    self._handle = g.as_graph_element(
        ops.prepend_name_scope(variable_def.variable_name,
                               import_scope=import_scope))
    self._shape = tensor_shape.as_shape([10000]).concatenate(
        self._handle.op.get_attr("value_shape"))
    self._handle_name = self._handle.name
    self._unique_id = self._handle_name
    self._initializer_op = g.as_graph_element(
        ops.prepend_name_scope(variable_def.initializer_name,
                               import_scope=import_scope))
    if (hasattr(variable_def, "initial_value_name")
        and variable_def.initial_value_name):
      self._initial_value = g.as_graph_element(
          ops.prepend_name_scope(variable_def.initial_value_name,
                                 import_scope=import_scope))
    else:
      self._initial_value = None

    if variable_def.HasField("save_slice_info_def"):
      self._save_slice_info = variables.Variable.SaveSliceInfo(
          save_slice_info_def=variable_def.save_slice_info_def,
          import_scope=import_scope,
      )
    else:
      self._save_slice_info = None
    self._caching_device = None
    self._key_dtype = dtypes.as_dtype(self._handle.op.get_attr("key_dtype"))
    self._value_dtype = dtypes.as_dtype(
        self._handle.op.get_attr("value_dtype"))
    self._graph_element = g.get_tensor_by_name(
        self._handle.op.name + "/Read/ReadKvVariableOpV2:0")[1]
    if variable_def.HasField("save_slice_info_def"):
      self._save_slice_info = variables.Variable.SaveSliceInfo(
          save_slice_info_def=variable_def.save_slice_info_def,
          import_scope=import_scope,
      )
    self._constraint = None
    self._cached_shape_as_list = None
    self._trainable = getattr(variable_def, "trainable", True)
    try:
      storage_option_string = self._handle.op.get_attr("storage_option")
      self._kv_options = parse_from_string(storage_option_string)
    except ValueError as e:
      logging.warning(
          "can't find kv_options of variable: %s, %s",
          self._handle_name,
          str(e),
      )
      self._kv_options = variable_scope.default_kv_option()

  @property
  def is_multi_level(self):
    return self._is_multi_level

  @property
  def is_ignore_eflops_device_fn(self):
    return self._is_ignore_eflops_device_fn

  @property
  def use_ps_on_eflops(self):
    return self._use_ps_os_eflops

  @property
  def enter_threshold(self):
    return self._enther_threshold

  @property
  def kv_options(self):
    return self._kv_options

  @property
  def storage_combination(self):
    return self._kv_options.storage_combination

  def has_storage(self, storage_type):
    return self._kv_options.has_storage(storage_type)

  def get_storage_config(self, storage_type):
    return self._kv_options.get_storage_config(storage_type)

  def has_path(self):
    return self._kv_options.has_path()

  def has_remote_storage(self):
    return self._kv_options.has_remote_storage()

  @property
  def storage_size_count(self):
    return gen_kv_variable_ops.kv_variable_size_v3(self._handle,
                                                   name=self._handle.op.name +
                                                   "/StorageSize")

  @property
  def num_concat_opt_vars(self):
    return self._num_concat_opt_vars

  @num_concat_opt_vars.setter
  def num_concat_opt_vars(self, value):
    self._num_concat_opt_vars = value

  @property
  def key_dtype(self):
    """The key dtype of this variable."""
    return self._key_dtype

  @property
  def dtype(self):
    """Override ResourceVariable.dtype. The value dtype of this variable."""
    return self._value_dtype

  @property
  def total_count(self):
    """The count of keys in kvvariable"""
    return gen_kv_variable_ops.kv_variable_size_v2(self._handle,
                                                   name=self._handle.op.name +
                                                   "/Size")

  @property
  def total_freq(self):
    """Calculate the freq of kvvariable"""
    return gen_kv_variable_ops.kv_variable_frequency(
        self._handle, name=self._handle.op.name + "/Freq")

  def _read_variable_op(self):
    if hasattr(self, "_trainable") and self._trainable:
      tape.variable_accessed(self)
    result = gen_kv_variable_ops.read_kv_variable_op_v2(
        self._handle, self._key_dtype, self._value_dtype)
    return result

  def value(self):
    """Override ResourceVariable.value. A cached operation which
        reads the value of this variable."""
    with ops.colocate_with(None, ignore_existing=True):
      with ops.device(self._handle.device):
        return self._read_variable_op()[1]

  def read_value(self):
    """Constructs an op which reads the value of this kv variable.

        Should be used when there are multiple reads, or when it is desirable to
        read the value only after some condition is true.

        Returns:
         the read operation.
        """
    with ops.name_scope("Read"):
      with ops.device(self._handle.device):
        value = self._read_variable_op()
    return array_ops.identity(value[1])

  # pylint: disable=missing-docstring
  def _smart_cond(self, pred, true_fn, false_fn):
    from tfplus.common import util

    if (self._trainable is False and self.has_path()
        and not util.is_tfplus_opt_slot_var(self.handle.name)):
      logging.info(
          "kv variable %s is not trainable, use predict_fn",
          self.handle.name,
      )
      return false_fn()

    if isinstance(pred, ops.Tensor):
      # TODO: former code add a identity op after tf.cond, don't know why, ignore for now
      return tf.cond(pred, true_fn, false_fn)

    if pred in {0, 1}:
      pred_value = bool(pred)
    elif isinstance(pred, bool):
      pred_value = pred
    else:
      raise TypeError("`pred` must be a Tensor, or a Python bool, or 1 or 0. "
                      "Found instead: %s" % pred)
    return true_fn() if pred_value else false_fn()

  def sparse_read(self, indices, name=None):
    """Reads the value of this variable sparsely, using `tf.gather`."""

    with ops.name_scope(name or "Gather") as gather_name:
      if self._trainable:
        tape.variable_accessed(self)

      def train_fn():
        return gen_kv_variable_ops.kv_variable_gather_or_insert_v2(
            self._handle,
            indices,
            dtype=self._value_dtype,
            name=gather_name + "train",
        )

      def predict_fn():
        return gen_kv_variable_ops.kv_variable_gather_or_zeros_v2(
            self._handle,
            indices,
            dtype=self._value_dtype,
            name=gather_name + "predict",
        )

      return train_fn() if IS_TRAINING else predict_fn()

  def sparse_read_with_counts(self, indices, counts=None, name=None):
    """Read the value of this variable sparsely, using `tf.gather`."""

    with ops.name_scope(name or "Gather") as gather_name:
      if self._trainable:
        tape.variable_accessed(self)

      def train_fn():
        if counts is not None:
          return gen_kv_variable_ops.kv_variable_gather_or_insert_with_counts(
              self._handle,
              indices,
              counts,
              dtype=self._value_dtype,
              name=gather_name + "train_with_counts",
          )
        return gen_kv_variable_ops.kv_variable_gather_or_insert_v2(
            self._handle,
            indices,
            dtype=self._value_dtype,
            name=gather_name + "train",
        )

      def predict_fn():
        return gen_kv_variable_ops.kv_variable_gather_or_zeros_v2(
            self._handle,
            indices,
            dtype=self._value_dtype,
            name=gather_name + "predict",
        )

      return train_fn() if IS_TRAINING else predict_fn()

  def increase_counting(self, indices, counts, name=None):
    """Increase counting for indices"""

    def train_fn():
      return gen_kv_variable_ops.kv_variable_increase_count_v2(self._handle,
                                                               indices,
                                                               counts,
                                                               name=name)

    def predict_fn():
      return tf.no_op()

    return train_fn() if IS_TRAINING else predict_fn()

  def get_counting(self, indices, name=None):
    return gen_kv_variable_ops.kv_variable_get_count_v2(self._handle,
                                                        indices,
                                                        name=name)

  def apply_snapshot(self,
                     filename_tensor,
                     need_full_import,
                     ckpt_name=None,
                     name=None):
    if ckpt_name is None:
      return gen_kv_variable_ops.kv_variable_apply_snapshot(self._handle,
                                                            filename_tensor,
                                                            need_full_import,
                                                            name=name)
    return gen_kv_variable_ops.kv_variable_apply_snapshot_v2(
        self._handle,
        filename_tensor,
        need_full_import,
        ckpt_name,
        name=name,
    )

  def to_proto(self, export_scope=None):
    """Converts a `KvVariable` to a `VariableDef` protocol buffer.
    Args:
      export_scope: Optional `string`. Name scope to remove.

    Raises:
      RuntimeError: If run in EAGER mode.

    Returns:
      A `VariableDef` protocol buffer, or `None` if the `Variable` is not
      in the specified name scope.
    """
    if context.executing_eagerly():
      raise RuntimeError("to_proto not supported in EAGER mode.")
    if export_scope is None or self.handle.name.startswith(export_scope):
      var_def = variable_pb2.VariableDef()
      var_def.variable_name = ops.strip_name_scope(self.handle.name,
                                                   export_scope)
      if self._initial_value is not None:
        # This is inside an if-statement for backwards compatibility, since
        # self._initial_value might be None for variables constructed from old
        # protos.
        var_def.initial_value_name = ops.strip_name_scope(
            self._initial_value.name, export_scope)
      var_def.initializer_name = ops.strip_name_scope(self.initializer.name,
                                                      export_scope)
      # We set snapshot_name to kv_variable, and use it to identify
      # whether variable_def is KvVariable
      var_def.snapshot_name = ops.strip_name_scope("kv_variable",
                                                   export_scope)
      var_def.is_resource = True
      var_def.trainable = self._trainable
      # TODO(tongsuo.ts): variable.proto is used for save and load for variable.
      # But now the proto can not distinguish KvVariable and ResourceVariable type,
      # We need to resolve it later.
      if self._save_slice_info:
        var_def.save_slice_info_def.MergeFrom(
            self._save_slice_info.to_proto(export_scope=export_scope))
      return var_def
    return None

  @staticmethod
  def from_proto(variable_def, import_scope=None):
    if context.executing_eagerly():
      raise RuntimeError("from_proto not supported in EAGER mode.")
    return KvVariable(variable_def=variable_def, import_scope=import_scope)

  def count_up_to(self, limit):  # pylint: disable=unused-argument
    """Unsupported."""
    raise RuntimeError("KvVariable does not implement count_up_to")

  def _ref(self):
    """Unsupported."""
    raise RuntimeError("KvVariable does not implement _ref")

  def set_shape(self, shape):  # pylint: disable=unused-argument
    """Unsupported."""
    raise RuntimeError("KvVariable does not implement set_shape")

  @property
  def shape(self):
    return self._shape

  def is_initialized(self, name=None):
    return gen_kv_variable_ops.kv_variable_is_initialized_v2(
        self.handle, name)

  # pylint: disable=unused-argument
  def assign(self, value, use_locking=None, name=None, read_value=True):
    # TODO(tongsuo.ts): add implement later
    if not isinstance(value, KvVariable):
      raise ValueError("KvVariable does not implement assign() for type %s" %
                       type(value))
    return gen_kv_variable_ops.kv_variable_import(
        self._handle,
        *gen_kv_variable_ops.kv_variable_export(
            value._handle,  # pylint: disable=protected-access
            value._key_dtype,  # pylint: disable=protected-access
            value._value_dtype,  # pylint: disable=protected-access
            enable_cutoff=True,
            cutoff_value=1.0e-20,
            first_n=4,
        ),
    )

  # pylint: disable=unused-argument
  def assign_sub(self, delta, use_locking=None, name=None, read_value=True):
    """Unsupported."""
    raise RuntimeError("KvVariable does not implement assign_sub")

  # pylint: disable=unused-argument
  def assign_add(self, delta, use_locking=None, name=None, read_value=True):
    """Unsupported."""
    raise RuntimeError("KvVariable does not implement assign_add")

  def _lazy_read(self, op):
    if hasattr(self, "_trainable") and self._trainable:
      tape.variable_accessed(self)

    return _UnreadVariable(
        self._handle,
        self._key_dtype,
        self._value_dtype,
        self._shape,
        self._in_graph_mode,
        self._handle_deleter if not self._in_graph_mode else None,
        op,
        self._unique_id,
    )

  # pylint: disable=unused-argument
  def scatter_sub(self, sparse_delta, use_locking=False, name=None):
    if not isinstance(sparse_delta, ops.IndexedSlices):
      raise ValueError("sparse_delta is not IndexedSlices: %s" % sparse_delta)
    return gen_kv_variable_ops.kv_variable_scatter_sub_v2(
        self.handle,
        sparse_delta.indices,
        ops.convert_to_tensor(sparse_delta.values, self.dtype),
        name=name,
    )

  # pylint: disable=unused-argument
  def scatter_add(self, sparse_delta, use_locking=False, name=None):
    if not isinstance(sparse_delta, ops.IndexedSlices):
      raise ValueError("sparse_delta is not IndexedSlices: %s" % sparse_delta)
    return gen_kv_variable_ops.kv_variable_scatter_add_v2(
        self.handle,
        sparse_delta.indices,
        ops.convert_to_tensor(sparse_delta.values, self.dtype),
        name=name,
    )

  # pylint: disable=unused-argument
  def scatter_mul(self, sparse_delta, use_locking=False, name=None):
    if not isinstance(sparse_delta, ops.IndexedSlices):
      raise ValueError("sparse_delta is not IndexedSlices: %s" % sparse_delta)
    return gen_kv_variable_ops.kv_variable_scatter_mul_v2(
        self.handle,
        sparse_delta.indices,
        ops.convert_to_tensor(sparse_delta.values, self.dtype),
        name=name,
    )

  # pylint: disable=unused-argument
  def scatter_div(self, sparse_delta, use_locking=False, name=None):
    if not isinstance(sparse_delta, ops.IndexedSlices):
      raise ValueError("sparse_delta is not IndexedSlices: %s" % sparse_delta)
    return gen_kv_variable_ops.kv_variable_scatter_div_v2(
        self.handle,
        sparse_delta.indices,
        ops.convert_to_tensor(sparse_delta.values, self.dtype),
        name=name,
    )

  # pylint: disable=unused-argument
  def scatter_update(self, sparse_delta, use_locking=False, name=None):
    if not isinstance(sparse_delta, ops.IndexedSlices):
      raise ValueError("sparse_delta is not IndexedSlices: %s" % sparse_delta)
    return gen_kv_variable_ops.kv_variable_scatter_update_v2(
        self.handle,
        sparse_delta.indices,
        ops.convert_to_tensor(sparse_delta.values, self.dtype),
        name=name,
    )

  # pylint: disable=unused-argument
  def scatter_max(self, sparse_delta, use_locking=False, name=None):
    if not isinstance(sparse_delta, ops.IndexedSlices):
      raise ValueError("sparse_delta is not IndexedSlices: %s" % sparse_delta)
    return gen_kv_variable_ops.kv_variable_scatter_max_v2(
        self.handle,
        sparse_delta.indices,
        ops.convert_to_tensor(sparse_delta.values, self.dtype),
        name=name,
    )

  # pylint: disable=unused-argument
  def scatter_min(self, sparse_delta, use_locking=False, name=None):
    if not isinstance(sparse_delta, ops.IndexedSlices):
      raise ValueError("sparse_delta is not IndexedSlices: %s" % sparse_delta)
    return gen_kv_variable_ops.kv_variable_scatter_min_v2(
        self.handle,
        sparse_delta.indices,
        ops.convert_to_tensor(sparse_delta.values, self.dtype),
        name=name,
    )

  # pylint: disable=unused-argument
  def scatter_nd_sub(self, indices, updates, name=None):
    """Unsupported."""
    raise RuntimeError("KvVariable does not implement scatter_nd_sub")

  # pylint: disable=unused-argument
  def scatter_nd_add(self, indices, updates, name=None):
    """Unsupported."""
    raise RuntimeError("KvVariable does not implement scatter_nd_add")

  # pylint: disable=unused-argument
  def scatter_nd_update(self, indices, updates, name=None):
    """Unsupported."""
    raise RuntimeError("KvVariable does not implement scatter_nd_update")

  # pylint: disable=unused-argument
  def _strided_slice_assign(
      self,
      begin,
      end,
      strides,
      value,
      name,
      begin_mask,
      end_mask,
      ellipsis_mask,
      new_axis_mask,
      shrink_axis_mask,
  ):
    """Unsupported."""
    raise RuntimeError("KvVariable does not implement _strided_slice_assign")

  def __int__(self):
    """Unsupported."""
    raise RuntimeError("KvVariable int(value) not supported")

  def get_generic_name(self, var_name=None):
    """[Get repartition generic name]

        Returns:
            str: remove `part_*` and `:0`
        """
    prefix, suffix, _ = self.get_name_info(var_name)
    return prefix + suffix[:len(suffix) - 2]

  def get_name_info(self, var_name=None):
    """Get variable's name and part info"""
    if var_name:
      name = var_name
    else:
      name = self.handle.name
    match = re.search(r"/part_\d+", name, 0)
    if match is None:
      return name[:-2], ":0", 0

    span = match.span()
    prefix = name[:span[0]]
    suffix = name[span[1]:]
    index = int(name[span[0] + 6:span[1]])
    return prefix, suffix, index

  def dynamic_restore(
      self,
      ckpt_path_tensor,
      restore_tensors,
      restore_mode,
      num_shards,
      ckpt_num_shards,
  ):
    """Dynamic restoring tensors from checkpoint files"""
    prefix, suffix, index = self.get_name_info()
    load_op = gen_kv_variable_ops.kv_variable_dynamic_restore(
        self.handle,
        ckpt_path_tensor,
        restore_tensors,
        restore_mode,
        ckpt_num_shards,
        num_shards,
        index,
    )
    with ops.control_dependencies([load_op]):
      init_op = gen_kv_variable_ops.init_kv_variable_v2(
          self._handle,
          _try_guard_against_uninitialized_dependencies(
              "{}_{}_{}".format(prefix, suffix, index),
              self._initial_value,
          ),
      )
      return init_op

  def load_remote_storage(self, table_name):
    return gen_kv_variable_ops.kv_variable_load_remote_table(
        self.handle, table_name)

  def export(self, name=None):
    """Export the content of KvVariable"""
    if _ENABLE_DELTA_EXPORT:
      do_full_export = get_or_create_do_full_export()
      with ops.colocate_with(self._handle):
        if tfplus_saver_mode() == 0:
          # Inference mode, only export keys, value, init-tables
          # and other three tensors will be empty
          first_n = 3
        else:
          first_n = 8
        tensors = gen_kv_variable_ops.kv_variable_full_or_delta_export(
            self._handle,
            do_full_export,
            self._key_dtype,
            self._value_dtype,
            enable_cutoff=True,
            cutoff_value=1.0e-20,
            first_n=first_n,
        )
        # Check the nubmer of tensors exported.
        if len(tensors) != 8:
          raise RuntimeError(
              "KvVariable: Number of exported tensors is not equal to 8" %
              name)
        return OrderedDict(
            (name + "-" + k, v) for k, v in tensors._asdict().items())
    with ops.colocate_with(self._handle):
      if tfplus_saver_mode() == 0:
        # Inference mode, only export keys, value, init-tables
        # and other three tensors will be empty
        first_n = 3
      else:
        first_n = 6
      # multi level hash will export list of dict
      if self._is_multi_level:
        tensors = []
        for sub_hash_name in self._multi_level_names:
          tensors.append(
              gen_kv_variable_ops.kv_variable_export_for_multi_hash(
                  self._handle,
                  self._key_dtype,
                  self._value_dtype,
                  enable_cutoff=True,
                  cutoff_value=1.0e-20,
                  first_n=first_n,
                  variable_name=sub_hash_name,
              ))
        return [
            OrderedDict((name + "-" + k, v)
                        for k, v in tensor._asdict().items())
            for name, tensor in zip(self._multi_level_names, tensors)
        ]
      # normal full save v3
      tensors = gen_kv_variable_ops.kv_variable_export(
          self._handle,
          self._key_dtype,
          self._value_dtype,
          enable_cutoff=True,
          cutoff_value=1.0e-20,
          first_n=first_n,
      )

      # Return an ordered dict that keeps the ordering of output tensors
      return OrderedDict(
          (name + "-" + k, v) for k, v in tensors._asdict().items())

  def delete(self, indices, name=None):
    return gen_kv_variable_ops.kv_variable_delete(self._handle,
                                                  indices,
                                                  name=name)

  def get_timestamp(self, indices, name=None):
    return gen_kv_variable_ops.kv_variable_get_time_stamp(self._handle,
                                                          indices,
                                                          name=name)

  def delete_with_timestamp(self, threshold, name=None):
    """Delete the tensors whose last updated time exceed the threshold.
        Args:
          threshold: largest time that the tensors haven't updated,
            otherwise should be deleted. The unit is day.
        """
    return gen_kv_variable_ops.kv_variable_delete_with_timestamp(
        self._handle, self._key_dtype, threshold, name=name)


class KvVariableSaveable(BaseSaverBuilder.SaveableObject):
  """SaveableObject implementation that handles KvVariable"""

  def __init__(self, var, name):
    # Export the content of KvVariable.
    # Note tensors_dict is an ordered dict that keeps the ordering
    # of operation output tensors
    self._var = var
    tensors_dict = var.export(name=name)
    self._key_dtype = var.key_dtype
    self._value_dtype = var.dtype
    self._embedding_dim = var.shape.as_list()[1]
    self._is_loading_finished = get_or_create_is_loading_finished()
    # Build a save spec.
    specs = [
        BaseSaverBuilder.SaveSpec(tensor, "", tensor_name)
        for tensor_name, tensor in tensors_dict.items()
    ]
    orig_ordering = [k.split("-")[-1] for k in tensors_dict.keys()]
    if tfplus_saver_mode() == 0:
      specs = specs[:3] + specs[6:]
      self._ordering = orig_ordering[:3] + orig_ordering[6:]
      self._empty_ordering = orig_ordering[3:6]
    else:
      self._ordering = orig_ordering
      self._empty_ordering = None

    super(KvVariableSaveable, self).__init__(var, specs, name)

  def get_generic_name(self, var_name=None):
    """[Get repartition generic name]

        Returns:
            [str]: remove `part_*` and `:0`
        """
    return self.op.get_generic_name(var_name)

  def dynamic_restore(
      self,
      ckpt_path_tensor,
      restore_tensors,
      restore_mode,
      num_shards,
      ckpt_num_shards,
  ):
    """Create load op on variable's device"""
    if self.device:
      from tensorflow.python.framework import device as pydev

      device = pydev.DeviceSpec.from_string(self.device)
      device.device_type = "CPU"
      device.device_index = 0
      device = device.to_string()
    else:
      device = None

    with ops.device(device):
      return self.op.dynamic_restore(
          ckpt_path_tensor,
          restore_tensors,
          restore_mode,
          num_shards,
          ckpt_num_shards,
      )

  # pylint: disable=unused-argument, missing-docstring
  def restore(
      self,  # pylint: disable=arguments-differ
      restored_tensors,
      restored_shapes,
      filename_tensor=None,
      ckpt_name=None,
  ):

    def _restore():
      kwargs = {k: restored_tensors[i] for i, k in enumerate(self._ordering)}
      if self._empty_ordering is not None:
        # blacklist / freq_keys / freq_values
        empty_keys = tf.zeros([0], self._key_dtype)
        if _ENABLE_DELTA_EXPORT:
          empty_values = tf.zeros([0, self._embedding_dim], dtypes.uint32)
        else:
          empty_values = tf.zeros([0, self._embedding_dim], dtypes.uint16)
        kwargs.update({
            self._empty_ordering[0]: empty_keys,
            self._empty_ordering[1]: empty_keys,
            self._empty_ordering[2]: empty_values,
        })

      if _ENABLE_DELTA_EXPORT:
        with ops.colocate_with(self.op.handle):
          # Restore KvVariable with the tensors.
          if tfplus_saver_mode() == 0:
            first_n = 3
          else:
            first_n = 8
          if full_or_delta_import_v2_enabled():
            kwargs.update({"is_loading_finished": self._is_loading_finished})
            return gen_kv_variable_ops.kv_variable_full_or_delta_import_v2(
                self.op.handle, first_n=first_n, **kwargs)
          return (gen_kv_variable_ops.kv_variable_full_or_delta_import(
              self.op.handle, first_n=first_n, **kwargs))
      else:
        with ops.colocate_with(self.op.handle):
          if tfplus_saver_mode() == 0:
            first_n = 3
          else:
            first_n = 6
          # Restore KvVariable with the tensors.
          return gen_kv_variable_ops.kv_variable_import(self.op.handle,
                                                        first_n=first_n,
                                                        **kwargs)

    restore_op = _restore()
    if self._var.is_ignore_eflops_device_fn:
      return restore_op
    with ops.control_dependencies([restore_op]):
      # pylint: disable=protected-access
      init_op = gen_kv_variable_ops.init_kv_variable_v2(
          self._var._handle,
          variables._try_guard_against_uninitialized_dependencies(
              self.var.name, self._var._initial_value),
      )
    return init_op

  @property
  def var(self):
    return self._var


class KvVariableSaveableV3(KvVariableSaveable):
  """SaveableObject implementation that handles KvVariable"""

  def __init__(self, var, name):
    # Export the content of KvVariable.
    # Note tensors_dict is an ordered dict that keeps the ordering
    # of operation output tensors
    tensors_dict = var.export(name=name)

    self._var = var
    self._key_dtype = var.key_dtype
    self._value_dtype = var.dtype
    self._embedding_dim = var.shape.as_list()[1]
    # Build a save spec.
    if var.is_multi_level or var.is_ignore_eflops_device_fn:
      # due to eflops_embedding_ops.eflops_device_fn
      # kvvariabe colocate will be broken, so we use
      # force colocate with
      from tfplus.kv_variable.python.ops.eflops_embedding_ops import (
          eflops_force_colocate_with,
      )

      with eflops_force_colocate_with(var._handle.device):  # pylint: disable=protected-access
        with ops.device(var._handle.device):  # pylint: disable=protected-access
          handles = array_ops.identity([var._handle])  # pylint: disable=protected-access
    else:
      with ops.device(var._handle.device):  # pylint: disable=protected-access
        handles = array_ops.identity([var._handle])  # pylint: disable=protected-access
    specs = [BaseSaverBuilder.SaveSpec(handles, "", var._handle_name)]  # pylint: disable=protected-access
    self._save_specs = specs

    # Build a restore spec.
    if var.is_multi_level:
      self._restore_specs = []
      orig_ordering = []
      for tensor_dict in tensors_dict:
        for tensor_name, tensor in tensor_dict.items():
          self._restore_specs.append(
              BaseSaverBuilder.SaveSpec(tensor, "", tensor_name))
          orig_ordering.append(tensor_name.split("-")[-1])
      # TODO(zhangji.zhang) support serving?
      self._ordering = orig_ordering
      self._empty_ordering = None
      self.restore = self._restore_multi_level_hash
    else:
      self._restore_specs = [
          BaseSaverBuilder.SaveSpec(tensor, "", tensor_name)
          for tensor_name, tensor in tensors_dict.items()
      ]
      orig_ordering = [k.split("-")[-1] for k in tensors_dict.keys()]

      if tfplus_saver_mode() == 0:
        self._restore_specs = (self._restore_specs[:3] +
                               self._restore_specs[6:])
        self._ordering = orig_ordering[:3] + orig_ordering[6:]
        self._empty_ordering = orig_ordering[3:6]
      else:
        self._ordering = orig_ordering
        self._empty_ordering = None
    self._is_loading_finished = get_or_create_is_loading_finished()
    super(KvVariableSaveable, self).__init__(var, specs, name)  # pylint: disable=bad-super-call

  def _restore_multi_level_hash(
      self,
      restored_tensors,
      restored_shapes,
      filename_tensor,  # pylint: disable=unused-argument
  ):  # pylint: disable=unused-argument
    """
        for hook restore in save v3.
        only works when is_multi_level=True in KvVariable
        """
    all_kwargs = []
    inner_kwargs = {}
    for i, k in enumerate(self._ordering):
      if k in inner_kwargs:
        all_kwargs.append(inner_kwargs)
        inner_kwargs = {k: restored_tensors[i]}
      else:
        inner_kwargs[k] = restored_tensors[i]
    all_kwargs.append(inner_kwargs)
    # TODO(zhangji.zhang) support _ENABLE_DELTA_EXPORT
    # TODO(zhangji.zhang) support serving model?
    with ops.colocate_with(self.op.handle):
      with ops.control_dependencies(
          ops.get_collection(ops.GraphKeys.TABLE_INITIALIZERS)):
        return control_flow_ops.group([
            gen_kv_variable_ops.kv_variable_import(self.op.handle,
                                                   first_n=6,
                                                   **kw) for kw in all_kwargs
        ])

  @property
  def specs(self):
    if IN_RESTORE_STATE:
      return self._restore_specs
    return self._save_specs

  @specs.setter
  def specs(self, value):
    self._specs = value


# Register a conversion function which reads the value of the variable,
# allowing instances of the class to be used as tensors.
class _UnreadVariable(KvVariable):
  """Represents a future for a read of a variable.

    Pretends to be the tensor if anyone looks.
    """

  # pylint: disable=super-init-not-called
  def __init__(
      self,
      handle,
      key_dtype,
      value_dtype,
      shape,
      in_graph_mode,
      deleter,
      parent_op,
      unique_id,
  ):
    self._trainable = False
    self._save_slice_info = None
    self._graph_key = (ops.get_default_graph()._graph_key)  # pylint: disable=protected-access
    self._in_graph_mode = in_graph_mode
    self._handle = handle
    self._shape = shape
    self._initial_value = None
    if isinstance(self._handle, ops.EagerTensor):
      self._handle_name = ""
    else:
      self._handle_name = self._handle.name
    self._unique_id = unique_id
    self._key_dtype = key_dtype
    self._value_dtype = value_dtype
    self._constraint = None
    self._cached_value = None
    self._is_initialized_op = None
    self._initializer_op = None
    self._parent_op = parent_op
    if context.executing_eagerly():
      self._graph_element = None
    else:
      self._graph_element = self.read_value()
    self._handle_deleter = deleter

  def value(self):
    return self._read_variable_op()[1]

  def read_value(self):
    return self._read_variable_op()[1]

  def _read_variable_op(self):
    with ops.control_dependencies([self._parent_op]):
      return gen_kv_variable_ops.read_kv_variable_op_v2(
          self._handle, self._key_dtype, self._value_dtype)

  def set_shape(self, shape):
    self._shape = shape
    self._cached_shape_as_list = None

  @property
  def op(self):
    """The op for this variable."""
    return self._parent_op


tensor_conversion_registry.register_tensor_conversion_function(
    _UnreadVariable, _dense_var_to_tensor)

@ops.RegisterGradient("KvVariableGatherOrInsertWithCounts")
def _GatherWithCountGrad(op, grad):  # pylint: disable=invalid-name
  result = _GatherGrad(op, grad)
  result.append(None)
  return result


@ops.RegisterGradient("KvVariableGatherOrInsertV2")
def _GatherGrad(op, grad):  # pylint: disable=invalid-name
  """Gradient for gather op."""
  # Build appropriately shaped IndexedSlices
  handle = op.inputs[0]
  indices = op.inputs[1]

  if utils.is_kv_variable_op_type(handle.op.type):
    # No tf.cond in forward pass, just KvVariableV2 -> KvVariableGatherOrInsertV2
    kv_op = handle.op
  else:
    # Here we use tf.cond for sparse_read, its structure is like
    # KvVariableV2 -> Switch -> KvVariableGatherOrInsertV2, so we
    # get handle.op.inputs[0].op to get KvVariableV2 op
    kv_op = handle.op.inputs[0].op

  params_shape = ops.convert_to_tensor(
      tensor_shape.TensorShape(kv_op.get_attr("value_shape")))
  size = array_ops.expand_dims(array_ops.size(indices), 0)
  values_shape = array_ops.concat([size, params_shape[0:]], 0)
  values = array_ops.reshape(grad, values_shape)
  indices = array_ops.reshape(indices, size)

  # return [ops.IndexedSlices(values, indices, params_shape), None]
  return [
      tf.IndexedSlices(values, indices, params_shape),
      None,
  ]  # tf2.13 change


ops.NotDifferentiable("KvVariableGatherOrZerosV2")
ops.NotDifferentiable("ReadKvVariableOpV2")
ops.NotDifferentiable("KvVariableIsInitializedV2")
ops.NotDifferentiable("KvVariableShape")

# Make state_ops.scatter_update/scatter_add/scatter_sub support KvVariable.
# Some optimizers use scatter functions in state_ops by _apply_sparse
# There are alos some optimizers directly use scatter functions in resource_variable_ops
# by _resource_apply_sparse, but its first argument is handle, we can not distinguish
# between KvVariable and ResourceVariable, for those optimizer we need to inherit and
# override _resource_apply_sparse function.

orignal_scatter_update = state_ops.scatter_update
orignal_scatter_add = state_ops.scatter_add
orignal_scatter_sub = state_ops.scatter_sub
orignal_is_variable_initialized = state_ops.is_variable_initialized


def scatter_update(ref, indices, updates, use_locking=True, name=None):
  if utils.is_kv_variable_op_type(ref.op.type):
    # pylint: disable=protected-access
    return gen_kv_variable_ops.kv_variable_scatter_update_v2(
        ref.handle,
        indices,
        ops.convert_to_tensor(updates, ref.dtype),
        name=name,
    )
  return orignal_scatter_update(ref,
                                indices,
                                updates,
                                use_locking=use_locking,
                                name=name)


def scatter_add(ref, indices, updates, use_locking=True, name=None):
  if utils.is_kv_variable_op_type(ref.op.type):
    # pylint: disable=protected-access
    return gen_kv_variable_ops.kv_variable_scatter_add_v2(
        ref.handle,
        indices,
        ops.convert_to_tensor(updates, ref.dtype),
        name=name,
    )
  return orignal_scatter_add(ref,
                             indices,
                             updates,
                             use_locking=use_locking,
                             name=name)


def scatter_sub(ref, indices, updates, use_locking=True, name=None):
  if utils.is_kv_variable_op_type(ref.op.type):
    # pylint: disable=protected-access
    return gen_kv_variable_ops.kv_variable_scatter_sub_v2(
        ref.handle,
        indices,
        ops.convert_to_tensor(updates, ref.dtype),
        name=name,
    )
  return orignal_scatter_sub(ref,
                             indices,
                             updates,
                             use_locking=use_locking,
                             name=name)


def is_variable_initialized(ref, name=None):
  if utils.is_kv_variable_op_type(ref.op.type):
    return ref.is_initialized(name=name)
  return orignal_is_variable_initialized(ref, name=name)


state_ops.scatter_update = scatter_update
state_ops.scatter_add = scatter_add
state_ops.scatter_sub = scatter_sub
state_ops.is_variable_initialized = is_variable_initialized

if is_tf_1_13_or_higher():
  original_op_list_to_dict = saveable_object_util.op_list_to_dict
  original_saveable_objects_for_op = (
      saveable_object_util.saveable_objects_for_op)
else:
  original_op_list_to_dict = (
      saveable_object_util.BaseSaverBuilder.OpListToDict)
  original_saveable_objects_for_op = (
      saveable_object_util.BaseSaverBuilder.SaveableObjectsForOp)


def op_list_to_dict(op_list, convert_variable_to_tensor=True):
  """
    op_list_to_dict hook, to support KvVariable
    """
  if not isinstance(op_list, (list, tuple, set)):
    raise TypeError("Variables to save should be passed in a dict or a "
                    "list: %s" % op_list)
  op_list = sorted(op_list, key=lambda x: x.name)
  names_to_saveables = {}
  orignal_tf_op_list = []
  for var in op_list:
    # Revised to enable TFPlus KvVariable
    if isinstance(var, KvVariable):
      if save_v3_enabled():
        saved = KvVariableSaveableV3(var, var.name)
      else:
        saved = KvVariableSaveable(var, var.name)
      names_to_saveables[var.name] = saved
    else:
      orignal_tf_op_list.append(var)
  names_to_saveables.update(
      original_op_list_to_dict(orignal_tf_op_list,
                               convert_variable_to_tensor))
  return names_to_saveables


def saveable_objects_for_op(op, name):
  """
    saveable_objects_for_op hook, to support KvVariable
    """
  if isinstance(op, KvVariable):
    if save_v3_enabled():
      yield KvVariableSaveableV3(op, name)
    else:
      yield KvVariableSaveable(op, name)
  else:
    yield from original_saveable_objects_for_op(op, name)


if is_tf_1_13_or_higher():
  saveable_object_util.op_list_to_dict = op_list_to_dict
  saveable_object_util.saveable_objects_for_op = saveable_objects_for_op
else:
  saveable_object_util.BaseSaverBuilder.OpListToDict = staticmethod(
      op_list_to_dict)
  saveable_object_util.BaseSaverBuilder.SaveableObjectsForOp = staticmethod(
      saveable_objects_for_op)


def register(self, candidate, name=None):
  """Registers a Python object "candidate" for the given "name".

  Args:
    candidate: The candidate object to add to the registry.
    name: An optional string specifying the registry key for the candidate.
          If None, candidate.__name__ will be used.
  Raises:
    KeyError: If same name is used twice.
  """
  if not name:
    name = candidate.__name__
  # pylint: disable=protected-access
  if name in self._registry:
    (filename, line_number, function_name,
     _) = self._registry[name][registry._LOCATION_TAG]
    logging.vlog(
        1,
        "Registering two %s with name '%s'! "
        "(Previous registration was in %s %s:%d)" %
        (self._name, name, function_name, filename, line_number),
    )

  logging.vlog(1, "Registering %s (%s) in %s.", name, candidate, self._name)
  # stack trace is [this_function, Register(), user_function,...]
  # so the user function is #2.g
  # tf2.13 change
  stack = inspect.stack()
  user_function = stack[2]
  location_tag = "{}:{}".format(user_function.filename, user_function.lineno)

  self._registry[name] = {
      registry._TYPE_TAG: candidate,
      registry._LOCATION_TAG: location_tag,
  }


registry.Registry.register = register


def _from_proto_fn(v, import_scope=None):
  """Creates Variable or ResourceVariable or KvVariable from VariableDef as needed."""
  if hasattr(v, "snapshot_name") and v.snapshot_name == "kv_variable":
    return KvVariable.from_proto(v, import_scope=import_scope)
  # pylint: disable=protected-access
  return resource_variable_ops._from_proto_fn(v, import_scope=import_scope)


# pylint: disable=protected-access
ops.register_proto_function(
    ops.GraphKeys.GLOBAL_VARIABLES,
    proto_type=variable_pb2.VariableDef,
    to_proto=tf_variable_scope._to_proto_fn,  # tf2.13 change
    from_proto=_from_proto_fn,
)
ops.register_proto_function(
    ops.GraphKeys.TRAINABLE_VARIABLES,
    proto_type=variable_pb2.VariableDef,
    to_proto=tf_variable_scope._to_proto_fn,
    from_proto=_from_proto_fn,
)

# tf2.13 change disable this hook
# original_zeros_link_outside_loop = control_flow_ops.ZerosLikeOutsideLoop

# def ZerosLikeOutsideLoop(op, index):
#   """Hook create zeros_like for the specified output of an op for kvvariable."""
#   if control_flow_util.IsSwitch(op):
#     op_ctxt = op._get_control_flow_context()
#     if op_ctxt and utils.is_kv_variable_op_type(op.inputs[0].op.type):
#       pred = op_ctxt.pred
#       branch = op_ctxt.branch
#       switch_val = control_flow_ops.switch(op.inputs[0], pred)[1 - branch]
#       pivot = array_ops.identity(switch_val)
#       with ops.control_dependencies([pivot]):
#         # For cond_grad need construct zeros, when we support
#         # string type, we should also create relative IndexedSlices
#         # with same dtype, else tf will convert a Tensor to IndexedSlices
#         # with dtype int64, this will cause type mismatch error when
#         # tf put string original indices and converted int64 indices
#         # in Merge op.
#         values_shape = tensor_shape.as_shape([10000]).concatenate(
#             op.inputs[0].op.get_attr("value_shape"))
#         values = array_ops.zeros(values_shape,
#                                  dtype=op.inputs[0].op.get_attr("value_dtype"))
#         indices = array_ops.zeros([10000],
#                                   dtype=op.inputs[0].op.get_attr("key_dtype"))
#         params_shape = ops.convert_to_tensor(values_shape)
#         return ops.IndexedSlices(values, indices, params_shape)
#   return original_zeros_link_outside_loop(op, index)

# control_flow_ops.ZerosLikeOutsideLoop = ZerosLikeOutsideLoop
