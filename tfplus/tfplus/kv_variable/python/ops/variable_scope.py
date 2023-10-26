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
"""A class to store named kv variables and a scope operator to manage sharing.
This file is based on
https://github.com/tensorflow/tensorflow/blob/v1.13.1/tensorflow/python/ops/variable_scope.py
and add some subclasses for get_kv_variable. get_kv_variable is similar
to tf.get_variable which support variable scope and PartitionedVariable
for sharding.
We have checked the custom_getter do not accept other use define args,
so we have to use many codes and comments from tensorflow repo,
otherwise our implementation would be much more simple.
We should simplify our implementation by using custom_getter which
tensorflow accept pass user define args.
Thanks to the original authors.
"""

# pylint: disable=cyclic-import
from __future__ import absolute_import, division, print_function

import collections as collections_lib
import copy
import traceback
from contextlib import contextmanager

import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes, ops, tensor_shape
from tensorflow.python.ops import array_ops, resource_variable_ops
from tensorflow.python.ops import variable_scope as tf_variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import optimizer, slot_creator
from tensorflow.python.util import tf_inspect

from tfplus.common import is_tf_1_13_or_higher
from tfplus.kv_variable.kernels.hybrid_embedding.storage_config_pb2 import *  # pylint: disable=unused-wildcard-import
from tfplus.kv_variable.python.ops import kv_variable_ops
from tfplus.kv_variable.python.ops.kv_variable_options import KvOptions

# add get_kv_variable signature to tf_variable_scope.__all__
tf_variable_scope.__all__.append("get_kv_variable")
_NUM_SHARDS_DICT_BASE_ON_CKPT = None

# for KV_HASH_TYPE=4
# multi level hash needs flag to mark behavior
# flag for marking this kv_variable is multi_level_hash
_IS_MULTI_LEVEL = False

# for save v3, due to there is an identity ops in save_v3,
# colocate will be broken in eflops device fn, so we add flag
# to mark disable eflops_device_fn in scope
_IS_IGNORE_EFLOPS_DEVICE_FN = False

_DEFAULT_KV_OPTION = KvOptions()


@contextmanager
def change_global_flag_for_multi_level_hash(is_multi_level,
                                            is_ignore_eflops_device_fn):
  """
    global switch for multi_level_hash_creator
    """
  global _IS_IGNORE_EFLOPS_DEVICE_FN
  global _IS_MULTI_LEVEL

  before_ignore_device_fn = _IS_IGNORE_EFLOPS_DEVICE_FN
  before_level = _IS_MULTI_LEVEL

  _IS_IGNORE_EFLOPS_DEVICE_FN = is_ignore_eflops_device_fn
  _IS_MULTI_LEVEL = is_multi_level
  yield
  _IS_IGNORE_EFLOPS_DEVICE_FN = before_ignore_device_fn
  _IS_MULTI_LEVEL = before_level


def multi_level_hash_creator(
    is_multi_level=True,
    is_ignore_eflops_device_fn=False,
    use_ps_on_eflops=False,
):
  """
    variable creator for eflops
    """

  def inner(next_creator, **kwargs):
    if "is_kvvar" in kwargs:
      kwargs["is_multi_level"] = _IS_MULTI_LEVEL or is_multi_level
      kwargs["is_ignore_eflops_device_fn"] = (_IS_IGNORE_EFLOPS_DEVICE_FN
                                              or is_ignore_eflops_device_fn)
      kwargs["use_ps_on_eflops"] = use_ps_on_eflops
    return next_creator(**kwargs)

  return inner


def set_num_shards_dict_base_on_ckpt(num_shards_dict_base_on_ckpt, ):  # pylint: disable=unused-argument
  pass


def set_ckpt_num_shards_dict(num_shards_dict_base_on_ckpt):
  global _NUM_SHARDS_DICT_BASE_ON_CKPT
  _NUM_SHARDS_DICT_BASE_ON_CKPT = num_shards_dict_base_on_ckpt


def get_num_shards_dict_base_on_ckpt():
  return _NUM_SHARDS_DICT_BASE_ON_CKPT


def default_kv_option():
  return _DEFAULT_KV_OPTION


class _KvVariableStore(tf_variable_scope._VariableStore):  # pylint: disable=protected-access
  """KvVariable store extends _VariableStore in
    https://github.com/tensorflow/tensorflow/blob/v1.13.1/tensorflow/python/ops/variable_scope.py#L246
    Add new get_kv_variable function.
    """

  def get_kv_variable(
      self,
      name,
      shape=None,
      key_dtype=dtypes.int64,
      dtype=dtypes.float32,
      initializer=None,
      regularizer=None,
      reuse=None,
      trainable=None,
      collections=None,
      partitioner=None,
      constraint=None,
      enter_threshold=0,
      kv_options=default_kv_option(),
  ):
    """
        Similar to get_variable in
        https://github.com/tensorflow/tensorflow/blob/v1.13.1/tensorflow/python/ops/variable_scope.py#L263
        and use KvVariable instead. The args are also same to get_variable
        except we add key_dtype and enter_threshold which are used by KvVariable
        """
    # Note that it's fine to reuse eager variables whose initialization was
    # lifted from a function-building graph into the eager context (that's why
    # the following clause is not wrapped in an `init_scope`); lifted variables
    # are tracked by the graph's `VariableStore`.
    if context.executing_eagerly():
      if not self._store_eager_variables and reuse:
        raise RuntimeError(
            "When eager execution is enabled variable reuse is only supported"
            " when an EagerVariableStore is active. See the documentation on"
            " EagerVariableStore for example usage.")
      if self._store_eager_variables:
        reuse = AUTO_REUSE

    # If a *_ref type is passed in an error would be triggered further down the
    # stack. We prevent this using base_dtype to get a non-ref version of the
    # type, before doing anything else. When _ref types are removed in favor of
    # resources, this line can be removed.
    try:
      dtype = dtype.base_dtype
    except AttributeError:
      # .base_dtype not existing means that we will try and use the raw dtype
      # which was passed in - this might be a NumPy type which is valid.
      pass

    # Set trainable value based on synchronization value.
    # tf2 change
    trainable = tf.VariableSynchronization.AUTO
    is_scalar = (shape is not None
                 and isinstance(shape, collections_lib.abc.Sequence)
                 and not shape)
    # Partitioned variable case
    if partitioner is not None and not is_scalar:
      if not callable(partitioner):
        raise ValueError("Partitioner must be callable, but received: %s" %
                         partitioner)
      with ops.name_scope(None):
        return self._get_partitioned_kv_variable(
            name=name,
            shape=shape,
            key_dtype=key_dtype,
            dtype=dtype,
            initializer=initializer,
            regularizer=regularizer,
            reuse=reuse,
            trainable=trainable,
            collections=collections,
            partitioner=partitioner,
            constraint=constraint,
            enter_threshold=enter_threshold,
            kv_options=kv_options,
        )

    # Special case for partitioned variable to allow reuse without having to
    # specify partitioner.
    if (reuse is True and partitioner is None
        and name in self._partitioned_vars):
      return self._get_partitioned_kv_variable(
          name=name,
          shape=shape,
          key_dtype=key_dtype,
          dtype=dtype,
          initializer=initializer,
          regularizer=regularizer,
          reuse=reuse,
          trainable=trainable,
          collections=collections,
          partitioner=None,
          constraint=constraint,
          enter_threshold=enter_threshold,
          kv_options=kv_options,
      )

    # Single variable case
    if "%s/part_0" % name in self._vars:
      raise ValueError(
          "No partitioner was provided, but a partitioned version of the "
          "variable was found: %s/part_0. Perhaps a variable of the same "
          "name was already created with partitioning?" % name)
    if shape is not None:
      shape = tensor_shape.as_shape([10000]).concatenate(
          tensor_shape.as_shape(shape))
    return self._get_single_kv_variable(
        name=name,
        shape=shape,
        key_dtype=key_dtype,
        dtype=dtype,
        initializer=initializer,
        regularizer=regularizer,
        reuse=reuse,
        trainable=trainable,
        collections=collections,
        constraint=constraint,
        enter_threshold=enter_threshold,
        kv_options=kv_options,
    )

  def _get_partitioned_kv_variable(
      self,
      name,
      partitioner,
      shape=None,
      key_dtype=dtypes.int64,
      dtype=dtypes.float32,
      initializer=None,
      regularizer=None,
      reuse=None,
      trainable=None,
      collections=None,
      constraint=None,
      enter_threshold=0,
      kv_options=default_kv_option(),
  ):
    """
        Similar to get_variable in https://github.com/tensorflow/tensorflow/blob/v1.13.1/tensorflow/python/ops/variable_scope.py#L549
        and use KvVariable instead. The args are also same to get_variable
        except we add key_dtype and enter_threshold which are used by KvVariable
        """
    initializing_from_value = initializer is not None and isinstance(
        initializer, ops.Tensor)
    if name in self._vars:
      raise ValueError(
          "A partitioner was provided, but an unpartitioned version of the "
          "variable was found: %s.  Perhaps a variable of the same name was "
          "already created without partitioning?" % name)
    # (tfplus): embedding_dim is 1 rank, we insert 100000 to dim0
    # and use this shape for initializer when partitioner is None
    embedding_dim = tensor_shape.as_shape(shape)
    if initializing_from_value:
      shape = initializer.get_shape()
      embedding_dim = tensor_shape.as_shape([shape.as_list()[1]])
    else:
      shape = tensor_shape.as_shape([100000]).concatenate(embedding_dim)

    partitions = None
    if not reuse or partitioner:
      # pylint: disable=protected-access
      if is_tf_1_13_or_higher():
        partitions = tf_variable_scope._call_partitioner(
            partitioner, shape, dtype)
      else:
        partitions = partitioner(shape=shape, dtype=dtype)
      # (tfplus): When use partitioner, we use allocate
      # [10000, embedding_dim] init table for each shard
      shards = partitions[0]
      shape = tensor_shape.as_shape([10000 * shards
                                     ]).concatenate(embedding_dim)

    if name in self._partitioned_vars:
      if reuse is False:
        raise ValueError(
            "Partitioned variable with name %s already exists. Did you mean to "
            "set reuse=True or reuse=tf.AUTO_REUSE in VarScope?" % name)

      existing_var = self._partitioned_vars[name]
      if not shape.is_compatible_with(existing_var.get_shape()):
        raise ValueError(
            "Trying to reuse partitioned variable %s, but specified shape %s "
            "and found shape %s." % (name, shape, existing_var.get_shape()))
      if not dtype.is_compatible_with(existing_var.dtype):
        raise ValueError(
            "Trying to reuse partitioned variable %s, but specified dtype %s "
            "and found dtype %s." %
            (name, dtype.name, existing_var.dtype.name))
      if not key_dtype.is_compatible_with(
          existing_var._variable_list[0].key_dtype):  # pylint: disable=protected-access
        raise ValueError(
            "Trying to reuse partitioned variable %s, but specified "
            "key dtype %s and found key dtype %s." % (
                name,
                key_dtype.name,
                existing_var._variable_list[0].key_dtype.name,  # pylint: disable=protected-access
            ))

      if (partitions is not None
          and existing_var._get_partitions() != partitions):  # pylint: disable=protected-access
        raise ValueError(
            "Trying to reuse partitioned variable %s, but specified partitions "
            "%s and found partitions %s." %
            (name, partitions, existing_var._get_partitions()))  # pylint: disable=protected-access
      # pylint: enable=protected-access
      return existing_var

    if reuse is True:
      raise ValueError("PartitionedVariable %s does not exist, or was not "
                       "created with tf.get_variable(). Did you mean to set "
                       "reuse=False or reuse=tf.AUTO_REUSE in VarScope?" %
                       name)
    if is_tf_1_13_or_higher():
      # pylint: disable=protected-access
      (
          slice_dim,
          num_slices,
      ) = tf_variable_scope._get_slice_dim_and_num_slices(partitions)
    else:
      (
          slice_dim,
          _,
      ) = tf_variable_scope._compute_slice_dim_and_shape(  # pylint: disable=protected-access
          shape.as_list(), partitions)
      num_slices = partitions[slice_dim]

    if "%s/part_0" % name in self._vars:
      if "%s/part_%d" % (name, num_slices - 1) not in self._vars:
        raise ValueError(
            "Partitioner returned a different partitioning than what was "
            "already found.  Partitioner returned %d shards, and shard "
            "%s/part_0 was found, but %s/part_%d was not." %
            (num_slices, name, name, num_slices - 1))
      if "%s/part_%d" % (name, num_slices) in self._vars:
        raise ValueError(
            "Partitioner returned a different partitioning than what was "
            "already found.  Partitioner returned %d shards, and shard "
            "%s/part_0 was found, but so was the extra shard %s/part_%d." %
            (num_slices, name, name, num_slices))

    vs = []
    if is_tf_1_13_or_higher():
      iter_slices = (tf_variable_scope._iter_slices)  # pylint: disable=protected-access
    else:

      def _iter_slices(full_shape, num_slices, slice_dim):
        """Slices a given a shape along the specified dimension."""
        num_slices_with_excess = full_shape[slice_dim] % num_slices
        offset = [0] * len(full_shape)
        min_slice_len = full_shape[slice_dim] // num_slices
        for i in range(num_slices):
          shape = full_shape[:]
          shape[slice_dim] = min_slice_len + bool(i < num_slices_with_excess)
          yield offset[:], shape
          offset[slice_dim] += shape[slice_dim]

      iter_slices = _iter_slices
    for i, (var_offset, var_shape) in enumerate(
        iter_slices(shape.as_list(), num_slices, slice_dim)):
      partition_info = tf_variable_scope._PartitionInfo(  # pylint: disable=protected-access
          full_shape=shape.as_list(),
          var_offset=var_offset)
      var_full_name = "%s/part_%d" % (name, i)
      with ops.name_scope(var_full_name + "/PartitionedInitializer"):
        # Create the tensor to initialize the variable with default value.
        if initializer is None:
          (
              init,
              initializing_from_value,
          ) = self._get_default_initializer(name=name,
                                            shape=shape,
                                            dtype=dtype)
          if initializing_from_value:
            init_shape = None
          else:
            init_shape = var_shape
        elif callable(initializer):
          init = initializer
          init_shape = var_shape
        elif isinstance(initializer, ops.Tensor):
          init = array_ops.slice(initializer, var_offset, var_shape)
          # Use the dtype of the given tensor.
          dtype = init.dtype.base_dtype
          init_shape = None
        else:
          init = ops.convert_to_tensor(initializer, dtype=dtype)
          init = array_ops.slice(init, var_offset, var_shape)
          init_shape = None

      with ops.name_scope(None):
        var = self._get_single_kv_variable(
            name=var_full_name,
            shape=init_shape,
            key_dtype=key_dtype,
            dtype=dtype,
            initializer=init,
            partition_info=partition_info,
            regularizer=regularizer,
            reuse=reuse,
            trainable=trainable,
            collections=collections,
            constraint=constraint,
            enter_threshold=enter_threshold,
            kv_options=kv_options,
        )

      # pylint: disable=protected-access
      var._set_save_slice_info(
          variables.Variable.SaveSliceInfo(name, shape.as_list(), var_offset,
                                           var_shape))
      vs.append(var)
      # pylint: enable=protected-access
    partitioned_var = variables.PartitionedVariable(
        name=name,
        shape=shape,
        dtype=dtype,
        variable_list=vs,
        partitions=partitions,
    )
    if not context.executing_eagerly() or self._store_eager_variables:
      self._partitioned_vars[name] = partitioned_var
    return partitioned_var

  def _default_variable_creator(self, _, **kwargs):
    # pop unused kwargs
    kwargs.pop("is_kvvar", None)
    kwargs.pop("use_resource", None)
    return kv_variable_ops.KvVariable(**kwargs)

  def _get_single_kv_variable(
      self,
      name,
      shape=None,
      key_dtype=dtypes.int64,
      dtype=dtypes.float32,
      initializer=None,
      regularizer=None,
      partition_info=None,
      reuse=None,
      trainable=None,
      collections=None,
      constraint=None,
      enter_threshold=0,
      kv_options=default_kv_option(),
  ):
    """
        Similar to get_variable in https://github.com/tensorflow/tensorflow/blob/v1.13.1/tensorflow/python/ops/variable_scope.py#L780
        and use KvVariable instead. The args are also same to get_variable
        except we add key_dtype and enter_threshold which are used by KvVariable
        """
    # Set to true if initializer is a constant.
    initializing_from_value = False
    if initializer is not None and not callable(initializer):
      initializing_from_value = True
    if shape is not None and initializing_from_value:
      raise ValueError("If initializer is a constant, do not specify shape.")

    dtype = dtypes.as_dtype(dtype)
    shape = tensor_shape.as_shape(shape)
    if name in self._vars:
      # Here we handle the case when returning an existing variable.
      if reuse is False:
        var = self._vars[name]
        err_msg = ("Variable %s already exists, disallowed."
                   " Did you mean to set reuse=True or "
                   "reuse=tf.AUTO_REUSE in VarScope?" % name)
        # ResourceVariables don't have an op associated with so no traceback
        if isinstance(var, resource_variable_ops.ResourceVariable):
          raise ValueError(err_msg)
        tb = var.op.traceback[::-1]
        # Throw away internal tf entries and only take a few lines.
        tb = [x for x in tb if "tensorflow/python" not in x[0]][:3]
        raise ValueError("%s Originally defined at:\n\n%s" %
                         (err_msg, "".join(traceback.format_list(tb))))
      found_var = self._vars[name]
      if not shape.is_compatible_with(found_var.get_shape()):
        raise ValueError("Trying to share variable %s, but specified shape %s"
                         " and found shape %s." %
                         (name, shape, found_var.get_shape()))
      if not dtype.is_compatible_with(found_var.dtype):
        dtype_str = dtype.name
        found_type_str = found_var.dtype.name
        raise ValueError("Trying to share variable %s, but specified dtype %s"
                         " and found dtype %s." %
                         (name, dtype_str, found_type_str))
      if not key_dtype.is_compatible_with(found_var.key_dtype):
        dtype_str = key_dtype.name
        found_type_str = found_var.key_dtype.name
        raise ValueError(
            "Trying to share variable %s, but specified key dtype %s"
            " and found key dtype %s." % (name, dtype_str, found_type_str))
      return found_var

    # The code below handles only the case of creating a new variable.
    if reuse is True:
      raise ValueError("Variable %s does not exist, or was not created with "
                       "tf.get_variable(). Did you mean to set "
                       "reuse=tf.AUTO_REUSE in VarScope?" % name)

    # Create the tensor to initialize the variable with default value.
    if initializer is None:
      (
          initializer,
          initializing_from_value,
      ) = self._get_default_initializer(name=name, shape=shape, dtype=dtype)
    # Enter an init scope when creating the initializer.
    with ops.init_scope():
      if initializing_from_value:
        init_val = initializer
        variable_dtype = None
      else:
        # Instantiate initializer if provided initializer is a type object.
        if tf_inspect.isclass(initializer):
          initializer = initializer(dtype=dtype)
        if shape is not None and shape.is_fully_defined():
          init_val = (
              lambda: initializer(  # pylint: disable=g-long-lambda
                  shape.as_list(),
                  dtype=dtype,
                  partition_info=partition_info,
              ))
          variable_dtype = dtype.base_dtype
        elif len(tf_inspect.getargspec(initializer).args) == len(
            tf_inspect.getargspec(initializer).defaults or []):
          init_val = initializer
          variable_dtype = None
        else:
          raise ValueError("The initializer passed is not valid. It should "
                           "be a callable with no arguments and the "
                           "shape should not be provided or an instance of "
                           "`tf.keras.initializers.*' and `shape` should be "
                           "fully defined.")
    # (tfplus): Here is our truely custom code that
    # we call KvVariable instead of Variable
    kwargs_ = {
        "initial_value": init_val,
        "name": name,
        "trainable": trainable,
        "collections": collections,
        "key_dtype": key_dtype,
        "value_dtype": variable_dtype,
        "constraint": constraint,
        "enter_threshold": enter_threshold,
        "kv_options": kv_options,
        "is_kvvar": True,
        "use_ps_on_eflops": False,
    }

    previous_getter = lambda **kwargs: self._default_variable_creator(  # pylint: disable=protected-access
        None, **kwargs)
    for getter in tf.compat.v1.get_default_graph()._variable_creator_stack: # pylint: disable=protected-access
      previous_getter = variables._make_getter(getter[1], previous_getter) # pylint: disable=protected-access
    v = previous_getter(**kwargs_)

    if context.executing_eagerly() and self._store_eager_variables:
      if collections:
        ops.add_to_collections(collections, v)
      else:
        ops.add_to_collection(ops.GraphKeys.GLOBAL_VARIABLES, v)
      if trainable:
        ops.add_to_collection(ops.GraphKeys.TRAINABLE_VARIABLES, v)

    if not context.executing_eagerly() or self._store_eager_variables:
      # In eager mode we do not want to keep default references to Variable
      # objects as this will prevent their memory from being released.
      self._vars[name] = v
    logging.vlog(
        1,
        "Created variable %s with shape %s and init %s",
        v.name,
        format(shape),
        initializer,
    )

    # Run the regularizer if requested and save the resulting loss.
    if regularizer:
      with ops.colocate_with(v):
        with ops.name_scope(name + "/Regularizer/"):
          with ops.init_scope():
            loss = regularizer(v)
        if loss is not None:
          if context.executing_eagerly():
            v_name = "v_%s" % type(v)
            loss_name = "loss_%s" % type(loss)
          else:
            v_name = v.name
            loss_name = loss.name
          logging.vlog(
              1,
              "Applied regularizer to %s and added the result %s "
              "to REGULARIZATION_LOSSES.",
              v_name,
              loss_name,
          )
          ops.add_to_collection(ops.GraphKeys.REGULARIZATION_LOSSES, loss)
    return v


class KvVariableScope(tf_variable_scope.VariableScope):
  """
    KvVariableScope extends VariableScope in
    https://github.com/tensorflow/tensorflow/blob/v1.13.1/tensorflow/python/ops/variable_scope.py#L988

    Add new get_kv_variable function
    """

  def __init__(
      self,
      reuse,
      name="",
      initializer=None,
      regularizer=None,
      caching_device=None,
      partitioner=None,
      custom_getter=None,
      name_scope="",
      key_dtype=dtypes.int64,
      dtype=dtypes.float32,
      use_resource=None,
      constraint=None,
      enter_threshold=0,
      kv_options=default_kv_option(),
  ):
    """Create KvvariableScope"""
    super(KvVariableScope, self).__init__(
        reuse,
        name=name,
        initializer=initializer,
        regularizer=regularizer,
        caching_device=caching_device,
        partitioner=partitioner,
        custom_getter=custom_getter,
        name_scope=name_scope,
        dtype=dtype,
        use_resource=use_resource,
        constraint=constraint,
    )
    self._key_dtype = key_dtype
    self._enter_threshold = enter_threshold
    self._kv_options = kv_options

  def get_kv_variable(
      self,
      var_store,
      name,
      shape=None,
      key_dtype=None,
      dtype=None,
      initializer=None,
      regularizer=None,
      reuse=None,
      trainable=None,
      collections=None,
      partitioner=None,
      constraint=None,
      enter_threshold=None,
      kv_options=default_kv_option(),
  ):
    """Similar to get_variable in
        https://github.com/tensorflow/tensorflow/blob/v1.13.1/tensorflow/python/ops/variable_scope.py#L1150
        and use KvVariable instead. The args are also same to get_variable
        except we add key_dtype and enter_threshold which are used by KvVariable
        """
    if regularizer is None:
      regularizer = self._regularizer
    if partitioner is None:
      partitioner = self._partitioner
    if context.executing_eagerly():
      reuse = False
    else:
      if reuse is None:
        reuse = self._reuse

    full_name = self.name + "/" + name if self.name else name

    # Variable names only depend on variable_scope (full_name here),
    # not name_scope, so we reset it below for the time of variable creation.
    with ops.name_scope(None):
      # Check that `initializer` dtype and `dtype` are consistent before
      # replacing them with defaults.
      if (dtype is not None and initializer is not None
          and not callable(initializer)):
        init_dtype = ops.convert_to_tensor(initializer).dtype.base_dtype
        if init_dtype != dtype:
          raise ValueError("Initializer type '%s' and explicit dtype '%s' "
                           "don't match." % (init_dtype, dtype))
      if initializer is None:
        initializer = self._initializer
      if constraint is None:
        constraint = self._constraint
      if dtype is None:
        dtype = self._dtype
      if key_dtype is None:
        key_dtype = self._key_dtype
      if enter_threshold is None:
        enter_threshold = self._enter_threshold
      if kv_options is None:
        kv_options = self._kv_options

      return var_store.get_kv_variable(
          full_name,
          shape=shape,
          key_dtype=key_dtype,
          dtype=dtype,
          initializer=initializer,
          regularizer=regularizer,
          reuse=reuse,
          trainable=trainable,
          collections=collections,
          partitioner=partitioner,
          constraint=constraint,
          enter_threshold=enter_threshold,
          kv_options=kv_options,
      )


# The argument list for get_kv_variable must match arguments to get_local_kv_variable.
# So, if you are updating the arguments, also update arguments to
# get_local_kv_variable below.
def get_kv_variable(
    name,
    embedding_dim=None,
    key_dtype=dtypes.int64,
    value_dtype=dtypes.float32,
    initializer=None,
    regularizer=None,
    trainable=None,
    collections=None,
    partitioner=None,
    constraint=None,
    enter_threshold=0,
    kv_options=default_kv_option(),
):
  """The same usage with tf.get_variable, you can
    also use tf.variable_scope as same as the doc in
    https://github.com/tensorflow/tensorflow/blob/v1.13.1/tensorflow/python/ops/variable_scope.py#L1448
    """
  return get_kv_variable_scope().get_kv_variable(
      _get_default_kv_variable_store(),
      name,
      shape=embedding_dim,
      key_dtype=key_dtype,
      dtype=value_dtype,
      initializer=initializer,
      regularizer=regularizer,
      trainable=trainable,
      collections=collections,
      partitioner=partitioner,
      constraint=constraint,
      enter_threshold=enter_threshold,
      kv_options=kv_options,
  )


_VARSTORE_KEY = ("__kv_variable_store", )
_VARSCOPESTORE_KEY = ("__kv_varscope", )
GRAPHSHARDS_KEY = ("__kv_variable_num_shards", )


class _KvVariableScopeStore(tf_variable_scope._VariableScopeStore):  # pylint: disable=protected-access
  """VariableScopeStore for kvvariable"""

  def __init__(self):
    super(_KvVariableScopeStore, self).__init__()
    self.current_scope = KvVariableScope(False)


def _get_default_kv_variable_store():
  store = ops.get_collection(_VARSTORE_KEY)
  if store:
    return store[0]
  store = _KvVariableStore()
  ops.add_to_collection(_VARSTORE_KEY, store)
  return store


def get_kv_variable_scope_store():
  """Returns the kv variable scope store for current thread."""
  scope_store = ops.get_collection(_VARSCOPESTORE_KEY)

  if not scope_store:
    scope_store = _KvVariableScopeStore()
    ops.add_to_collection(_VARSCOPESTORE_KEY, scope_store)
  else:
    scope_store = scope_store[0]

  return scope_store


def get_kv_variable_scope():
  return get_kv_variable_scope_store().current_scope


def default_kv_variable_creator(next_creator=None, **kwargs):
  """Default variable creator."""
  assert next_creator is None
  initial_value = kwargs.get("initial_value", None)
  trainable = kwargs.get("trainable", None)
  name = kwargs.get("name", None)
  variable_def = kwargs.get("variable_def", None)
  dtype = kwargs.get("dtype", None)
  import_scope = kwargs.get("import_scope", None)
  constraint = kwargs.get("constraint", None)
  key_dtype = kwargs.get("key_dtype", None)
  enter_threshold = kwargs.get("enter_threshold", 0)
  kv_options = kwargs.get("kv_options", default_kv_option())

  # Set trainable value based on synchronization value.
  # tf2 change
  synchronization = kwargs.get("synchronization",
                               tf.VariableSynchronization.AUTO)
  if synchronization == tf.VariableSynchronization.ON_READ:
    trainable = False
  else:
    trainable = kwargs.get("trainable", True)

  return kv_variable_ops.KvVariable(
      initial_value=initial_value,
      trainable=trainable,
      name=name,
      key_dtype=key_dtype,
      value_dtype=dtype,
      constraint=constraint,
      variable_def=variable_def,
      import_scope=import_scope,
      enter_threshold=enter_threshold,
      kv_options=kv_options,
  )


def variable_creator_wrapper(func):

  def wrapper(next_creator=None, **kwargs):
    if "key_dtype" in kwargs:
      return default_kv_variable_creator(next_creator=next_creator, **kwargs)
    return func(next_creator=next_creator, **kwargs)

  return wrapper


tf_variable_scope.default_variable_creator = variable_creator_wrapper(
    tf_variable_scope.default_variable_creator)
tf_variable_scope.variables.default_variable_creator_v2 = (
    variable_creator_wrapper(  # pylint: disable=line-too-long
        resource_variable_ops.default_variable_creator_v2))  # tf2.13 change
tf_variable_scope.get_variable_scope = get_kv_variable_scope
tf_variable_scope._get_default_variable_store = (  # pylint: disable=protected-access
    _get_default_kv_variable_store
)
tf_variable_scope.get_variable_scope_store = get_kv_variable_scope_store


class _pure_kv_variable_scope:  # pylint: disable=protected-access, invalid-name
  """
    This function is base on
    https://github.com/tensorflow/tensorflow/blob/v1.13.1/tensorflow/python/ops/variable_scope.py#L1761
    We rewrite its logic to support KvVariableScope
    """

  def __init__(
      self,
      name_or_scope,
      reuse=None,
      initializer=None,
      regularizer=None,
      caching_device=None,
      partitioner=None,
      custom_getter=None,
      old_name_scope=None,
      dtype=dtypes.float32,
      use_resource=None,
      constraint=None,
  ):
    self._name_or_scope = name_or_scope
    self._reuse = reuse
    self._initializer = initializer
    self._regularizer = regularizer
    self._caching_device = caching_device
    self._partitioner = partitioner
    self._custom_getter = custom_getter
    self._old_name_scope = old_name_scope
    self._dtype = dtype
    self._use_resource = use_resource
    self._constraint = constraint

    # (tfplus): Use relative kv functions
    self._var_store = _get_default_kv_variable_store()
    self._var_scope_store = get_kv_variable_scope_store()
    if isinstance(self._name_or_scope, tf_variable_scope.VariableScope):
      self._new_name = self._name_or_scope.name
      name_scope = (self._name_or_scope._name_scope)  # pylint: disable=protected-access
      variable_scope_object = KvVariableScope(
          self._name_or_scope.reuse if not self._reuse else self._reuse,
          name=self._new_name,
          initializer=self._name_or_scope.initializer,
          regularizer=self._name_or_scope.regularizer,
          caching_device=self._name_or_scope.caching_device,
          partitioner=self._name_or_scope.partitioner,
          dtype=self._name_or_scope.dtype,
          custom_getter=self._name_or_scope.custom_getter,
          name_scope=name_scope,
          use_resource=self._name_or_scope.use_resource,
          constraint=self._constraint,
      )
      if self._initializer is not None:
        variable_scope_object.set_initializer(self._initializer)
      if self._regularizer is not None:
        variable_scope_object.set_regularizer(self._regularizer)
      if self._caching_device is not None:
        variable_scope_object.set_caching_device(self._caching_device)
      if self._partitioner is not None:
        variable_scope_object.set_partitioner(self._partitioner)
      if self._custom_getter is not None:
        variable_scope_object.set_custom_getter(
            tf_variable_scope._maybe_wrap_custom_getter(
                self._custom_getter, self._name_or_scope.custom_getter))
      if self._dtype is not None:
        variable_scope_object.set_dtype(self._dtype)
      if self._use_resource is not None:
        variable_scope_object.set_use_resource(self._use_resource)
      self._cached_variable_scope_object = variable_scope_object

  def __enter__(self):
    """Begins the scope block.

    Returns:
      A VariableScope.
    Raises:
      ValueError: when trying to reuse within a create scope, or create within
        a reuse scope, or if reuse is not `None` or `True`.
      TypeError: when the types of some arguments are not appropriate.
    """
    self._old = self._var_scope_store.current_scope
    if isinstance(self._name_or_scope, tf_variable_scope.VariableScope):
      self._var_scope_store.open_variable_scope(self._new_name)
      self._old_subscopes = copy.copy(
          self._var_scope_store.variable_scopes_count)
      variable_scope_object = self._cached_variable_scope_object
    else:
      # Handler for the case when we just prolong current variable scope.
      #   VariableScope with name extended by the provided one, and inherited
      #   reuse and initializer (except if the user provided values to set).
      self._new_name = (self._old.name + "/" + self._name_or_scope
                        if self._old.name else self._name_or_scope)
      self._reuse = (self._reuse or self._old.reuse
                     )  # Re-using is inherited by sub-scopes.
      if self._old_name_scope is None:
        name_scope = self._name_or_scope
      else:
        name_scope = self._old_name_scope
      variable_scope_object = KvVariableScope(
          self._reuse,
          name=self._new_name,
          initializer=self._old.initializer,
          regularizer=self._old.regularizer,
          caching_device=self._old.caching_device,
          partitioner=self._old.partitioner,
          dtype=self._old.dtype,
          use_resource=self._old.use_resource,
          custom_getter=self._old.custom_getter,
          name_scope=name_scope,
          constraint=self._constraint,
      )
      if self._initializer is not None:
        variable_scope_object.set_initializer(self._initializer)
      if self._regularizer is not None:
        variable_scope_object.set_regularizer(self._regularizer)
      if self._caching_device is not None:
        variable_scope_object.set_caching_device(self._caching_device)
      if self._partitioner is not None:
        variable_scope_object.set_partitioner(self._partitioner)
      if self._custom_getter is not None:
        variable_scope_object.set_custom_getter(
            tf_variable_scope._maybe_wrap_custom_getter(
                self._custom_getter, self._old.custom_getter))
      if self._dtype is not None:
        variable_scope_object.set_dtype(self._dtype)
      if self._use_resource is not None:
        variable_scope_object.set_use_resource(self._use_resource)
      self._var_scope_store.open_variable_scope(self._new_name)
    self._var_scope_store.current_scope = variable_scope_object
    return variable_scope_object

  def __exit__(self, type_arg, value_arg, traceback_arg):
    if isinstance(self._name_or_scope, tf_variable_scope.VariableScope):
      self._var_scope_store.variable_scopes_count = self._old_subscopes
    else:
      self._var_scope_store.close_variable_subscopes(self._new_name)
    self._var_scope_store.current_scope = self._old


# (tfplus): We hook tf _pure_variable_scope to _pure_kv_variable_scope
# here which support user can still use tf.variable_scope both for
# tf.get_variable and our get_kv_variable.
# This is also why we must use _pure_variable_scope's implementation
# and rewrite it. If we can have any better way, pls contact us, thanks.
tf_variable_scope._pure_variable_scope = (  # pylint:disable=protected-access
    _pure_kv_variable_scope
)


def create_slot_wrapper(func):
  """
    Wrapper function for slot_creator._create_slot_var to
    call get_kv_variable if its primary is an instance of
    KvVariable.
    """

  # pylint: disable=missing-docstring
  def wrapper(primary, val, scope, validate_shape, shape, dtype,
              copy_xla_sharding):
    shape = shape if callable(val) else None
    if isinstance(primary, kv_variable_ops.KvVariable):
      if shape is not None:
        shape = tensor_shape.as_shape(
            [shape.as_list()[1] * int(primary.num_concat_opt_vars)])
      current_partitioner = (
          get_kv_variable_scope_store().current_scope.partitioner)
      get_kv_variable_scope_store().current_scope.set_partitioner(None)
      if primary.is_multi_level:
        # if kv_variable and multi level, all state variable
        # of optimizer should be multi level hash
        with tf_variable_scope.variable_creator_scope(
            multi_level_hash_creator()):
          slot = get_kv_variable(
              scope,
              embedding_dim=shape,
              initializer=val,
              key_dtype=primary.key_dtype,
              value_dtype=primary.dtype,
              trainable=False,
              kv_options=primary.kv_options,
          )
          primary.copy_multi_level_hash_config(slot)
      else:
        with tf_variable_scope.variable_creator_scope(
            multi_level_hash_creator(
                is_multi_level=False,
                is_ignore_eflops_device_fn=primary.is_ignore_eflops_device_fn,
            )):
          slot = get_kv_variable(
              scope,
              embedding_dim=shape,
              initializer=val,
              key_dtype=primary.key_dtype,
              value_dtype=primary.dtype,
              trainable=False,
              kv_options=primary.kv_options,
          )
      get_kv_variable_scope_store().current_scope.set_partitioner(
          current_partitioner)
      return slot
    return func(
        primary,
        val,
        scope,
        validate_shape,
        shape,
        dtype,
        copy_xla_sharding=copy_xla_sharding,
    )

  return wrapper


# pylint:disable=protected-access
slot_creator._create_slot_var = create_slot_wrapper(
    slot_creator._create_slot_var)


def _get_processor_wrapper(func):

  def wrapper(v):
    if isinstance(v, kv_variable_ops.KvVariable):
      return optimizer._DenseResourceVariableProcessor(v)  # pylint:disable=protected-access
    return func(v)

  return wrapper


optimizer._get_processor = _get_processor_wrapper(optimizer._get_processor)  # pylint:disable=protected-access


def create_local_var(primary, val, scope, validate_shape, shape, dtype):
  """Create a slot initialized to the given value.
    The type of the slot is determined by the given value.
    Args:
        primary: The primary `Variable` or `Tensor`.
        val: A `Tensor` specifying the initial value of the slot.
    Returns:
        A `Variable` object.
    """
  shape = shape if callable(val) else None
  # Handle kv_variable first
  if isinstance(primary, kv_variable_ops.KvVariable):
    if shape is not None:
      shape = tensor_shape.as_shape([shape.as_list()[1]])
    current_partitioner = (
        get_kv_variable_scope_store().current_scope.partitioner)

    get_kv_variable_scope_store().current_scope.set_partitioner(None)
    local_var = get_kv_variable(
        scope,
        embedding_dim=shape,
        initializer=val,
        key_dtype=primary.key_dtype,
        value_dtype=primary.dtype,
        collections=[ops.GraphKeys.LOCAL_VARIABLES],
        trainable=False,
    )

    get_kv_variable_scope_store().current_scope.set_partitioner(
        current_partitioner)

    return local_var

  # Handle tf variable
  current_partitioner = tf_variable_scope.get_variable_scope().partitioner
  tf_variable_scope.get_variable_scope().set_partitioner(None)
  local_var = tf_variable_scope.get_variable(
      scope,
      initializer=val,
      trainable=False,
      use_resource=resource_variable_ops.is_resource_variable(primary),
      shape=shape,
      dtype=dtype,
      collections=[ops.GraphKeys.LOCAL_VARIABLES],
      validate_shape=validate_shape,
  )

  tf_variable_scope.get_variable_scope().set_partitioner(current_partitioner)
  # pylint: disable=protected-access
  if isinstance(primary, variables.Variable) and primary._save_slice_info:
    # Primary is a partitioned variable, so we need to also indicate that
    # the local_var is a partitioned variable.  local_var have the same partitioning
    # as their primaries.
    # For examples when using AdamOptimizer in linear model, local_var.name
    # here can be "linear//weights/Adam:0", while primary.op.name is
    # "linear//weight". We want to get 'Adam' as real_var_name, so we
    # remove "'linear//weight' + '/'" and ':0'.
    real_var_name = local_var.name[len(primary.op.name + "/"):-2]
    slice_info = primary._save_slice_info
    local_var._set_save_slice_info(
        variables.Variable.SaveSliceInfo(
            slice_info.full_name + "/" + real_var_name,
            slice_info.full_shape[:],
            slice_info.var_offset[:],
            slice_info.var_shape[:],
        ))

  # pylint: enable=protected-access
  return local_var
