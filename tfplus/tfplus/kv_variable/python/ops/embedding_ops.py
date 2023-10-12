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
"""Operations for embeddings both support tf Variable and our KvVariable.
This file is based on
https://github.com/tensorflow/tensorflow/blob/v1.13.1/tensorflow/python/ops/embedding_ops.py
Thanks to the original authors.
"""
from __future__ import absolute_import, division, print_function

import re

import tensorflow as tf
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.python.framework import (
    constant_op,
    dtypes,
    ops,
    sparse_tensor,
    tensor_shape,
)
# Imports gradient definitions.
from tensorflow.python.ops import (
    array_ops,
    data_flow_ops,
    embedding_ops,
    math_ops,
    resource_variable_ops,
    sparse_ops,
    string_ops,
    variables,
)
from tensorflow.python.platform import tf_logging as logging

from tfplus.common import ranking_utils
from tfplus.kv_variable.python.ops import kv_variable_ops


def _embedding_lookup_and_transform(
    params,
    ids,
    partition_strategy="mod",
    name=None,
    max_norm=None,
    transform_fn=None,
    counts=None,
):
  """
    Based on https://github.com/tensorflow/tensorflow/blob/v1.13.1/tensorflow/python/ops/embedding_ops.py#L84.
    We add counts param for frequency statistics, and add
    mod partition_strategy for KvVariable.
    """
  if params is None or params in ((), []):
    raise ValueError("Need at least one param")
  if isinstance(params, variables.PartitionedVariable):
    params = list(params)  # Iterate to get the underlying Variables.
  if not isinstance(params, list):
    params = [params]
  is_kv_variable = isinstance(params[0], kv_variable_ops.KvVariable)
  if is_kv_variable and not all(
      isinstance(p, kv_variable_ops.KvVariable) for p in params):
    raise ValueError("All params should be KvVariable")

  with ops.name_scope(name, "embedding_lookup", params + [ids]) as name:  # pylint:disable=redefined-argument-from-local
    np = len(params)  # Number of partitions
    # Preserve the resource variable status to avoid accidental dense reads.
    if not any(
        isinstance(p, resource_variable_ops.ResourceVariable) for p in params):
      params = ops.convert_n_to_tensor_or_indexed_slices(params, name="params")
    ids = ops.convert_to_tensor(ids, name="ids")
    if np == 1 and (not transform_fn or ids.get_shape().ndims == 1):  # pylint:disable=no-else-return
      if ids.get_shape().ndims > 1:
        flattern_ids = array_ops.reshape(ids, [-1])
      else:
        flattern_ids = ids

      with ops.colocate_with(params[0]):
        if is_kv_variable:
          kv_variable_ops._query_kv_feature_size(params[0])  # pylint: disable=protected-access
          gather_op = params[0].sparse_read_with_counts(flattern_ids,
                                                        counts,
                                                        name=name)
        else:
          gather_op = array_ops.gather(params[0], flattern_ids, name=name)
        # pylint: disable=protected-access
        result = embedding_ops._clip(gather_op, flattern_ids, max_norm)
        if transform_fn:
          result = transform_fn(result)
      # Make sure the final result does not have colocation contraints on the
      # params. Similar to the case np > 1 where parallel_dynamic_stitch is
      # outside the scioe of all with ops.colocate_with(params[p]).
      if ids.get_shape().ndims > 1:
        result = array_ops.reshape(
            result,
            array_ops.concat([array_ops.shape(ids), params[0].get_shape()[1:]],
                             0),
        )
      result.set_shape(ids.get_shape().concatenate(params[0].get_shape()[1:]))
      return result
    else:
      # Flatten the ids. There are two cases where we need to do this.
      # - There is more than one params tensor.
      # - There is a transform_fn and ids is not statically known to be 1-D.
      #   We must flatten in this case because transform_fn expects a flat
      #   tensor of embeddings.
      flat_ids = array_ops.reshape(ids, [-1])
      original_indices = math_ops.range(array_ops.size(flat_ids))
      if counts is not None:
        counts = array_ops.reshape(counts, [-1])

      # Create p_assignments and set new_ids depending on the strategy.
      if is_kv_variable:
        # We use mod strategy for kv_variable.
        new_ids = flat_ids
        if flat_ids.dtype == dtypes.string:
          p_assignments = string_ops.string_to_hash_bucket_fast(flat_ids, np)
        else:
          p_assignments = flat_ids % np
      elif partition_strategy == "mod":
        p_assignments = flat_ids % np
        new_ids = flat_ids // np
      elif partition_strategy == "div":
        # Compute num_total_ids as the sum of dim-0 of params, then assign to
        # partitions based on a constant number of ids per partition. Optimize
        # if we already know the full shape statically.
        dim_0_size = tensor_shape.Dimension(
            tensor_shape.dimension_value(params[0].get_shape()[0]))
        for p in xrange(1, np):
          dim_0_size += tensor_shape.Dimension(
              tensor_shape.dimension_value(params[p].get_shape()[0]))
        if dim_0_size.value:
          num_total_ids = constant_op.constant(dim_0_size.value, flat_ids.dtype)
        else:
          dim_0_sizes = []
          for p in xrange(np):
            param_p_dim = tensor_shape.dimension_value(
                params[p].get_shape()[0])
            if param_p_dim is not None:
              dim_0_sizes.append(param_p_dim)
            else:
              with ops.colocate_with(params[p]):
                dim_0_sizes.append(array_ops.shape(params[p])[0])
          num_total_ids = math_ops.reduce_sum(
              math_ops.cast(tf.stack(dim_0_sizes), flat_ids.dtype))
        ids_per_partition = num_total_ids // np
        extras = num_total_ids % np

        p_assignments = math_ops.maximum(
            flat_ids // (ids_per_partition + 1),
            (flat_ids - extras) // ids_per_partition,
        )

        # Emulate a conditional using a boolean indicator tensor
        new_ids = array_ops.where(
            p_assignments < extras,
            flat_ids % (ids_per_partition + 1),
            (flat_ids - extras) % ids_per_partition,
        )
      else:
        raise ValueError("Unrecognized partition strategy: " +
                         partition_strategy)

      # Cast partition assignments to int32 for use in dynamic_partition.
      # There really should not be more than 2^32 partitions.
      p_assignments = math_ops.cast(p_assignments, dtypes.int32)
      # Partition list of ids based on assignments into np separate lists
      gather_ids = data_flow_ops.dynamic_partition(new_ids, p_assignments, np)
      if counts is not None and is_kv_variable:
        counts_ids = data_flow_ops.dynamic_partition(counts, p_assignments, np)
      else:
        counts_ids = [None] * np
      # Similarly, partition the original indices.
      pindices = data_flow_ops.dynamic_partition(original_indices,
                                                 p_assignments, np)
      # Do np separate lookups, finding embeddings for plist[p] in params[p]
      partitioned_result = []
      for p in xrange(np):
        pids = gather_ids[p]
        counts_id = counts_ids[p]
        with ops.colocate_with(params[p]):
          if is_kv_variable:
            kv_variable_ops._query_kv_feature_size(params[p])  # pylint: disable=protected-access
            result = params[p].sparse_read_with_counts(pids, counts_id)
          else:
            result = array_ops.gather(params[p], pids)
          if transform_fn:
            # If transform_fn is provided, the clip_by_norm precedes
            # the transform and hence must be co-located. See below
            # for the counterpart if transform_fn is not proveded.
            result = transform_fn(embedding_ops._clip(result, pids, max_norm))  # pylint: disable=protected-access
        partitioned_result.append(result)
      # Stitch these back together
      ret = data_flow_ops.parallel_dynamic_stitch(pindices,
                                                  partitioned_result,
                                                  name=name)

      # Determine the static element shape.
      if transform_fn is None:
        element_shape_s = params[0].get_shape()[1:]
        for p in params[1:]:
          element_shape_s = element_shape_s.merge_with(p.get_shape()[1:])
      else:
        element_shape_s = ret.get_shape()[1:]

      # Compute the dynamic element shape.
      if element_shape_s.is_fully_defined():
        element_shape_d = element_shape_s
      elif transform_fn is None:
        # It's important that we compute params[0].shape on the right device
        # to avoid data motion.
        with ops.colocate_with(params[0]):
          params_shape = array_ops.shape(params[0])
        element_shape_d = params_shape[1:]
      else:
        element_shape_d = array_ops.shape(ret)[1:]

      # Reshape to reverse the flattening of ids.
      ret = array_ops.reshape(
          ret,
          array_ops.concat([array_ops.shape(ids), element_shape_d], 0),
      )

      # Normally the reshape is sufficient, but setting shape explicitly
      # teaches shape inference that params[1:].get_shape() matters
      # (in the case that transform_fn is None).
      ret.set_shape(ids.get_shape().concatenate(element_shape_s))
      if not transform_fn:
        # If transform_fn was provided, the clip_by_norm was done above.
        ret = embedding_ops._clip(ret, ids, max_norm)  # pylint: disable=protected-access
      return ret


def embedding_lookup(
    params,
    ids,
    partition_strategy="mod",
    name=None,
    validate_indices=True,  # pylint: disable=unused-argument
    max_norm=None,
):
  """
    Based on https://github.com/tensorflow/tensorflow/blob/v1.13.1/tensorflow/python/ops/embedding_ops.py#L251
    """
  # compatibility with list/PartitionedVariable/Variable types
  if isinstance(params, ranking_utils.SUPPORTED_TYPES):
    unique_name = "%s_lookup" % params.name.split(":")[0]
    bucket_size = params.shape.as_list()[0]
  result = _embedding_lookup_and_transform(
      params=params,
      ids=ids,
      partition_strategy=partition_strategy,
      name=name,
      max_norm=max_norm,
      transform_fn=None,
  )
  if isinstance(params, ranking_utils.SUPPORTED_TYPES):
    ranking_utils.update_embedding_for_ranking(
        unique_name,
        params,
        None,
        None,
        bucket_size=bucket_size,
        partition_strategy=partition_strategy,
    )
    ranking_utils.append_embedding_result_for_ranking(unique_name, result)
    ranking_utils.append_embedding_input_for_ranking(unique_name, ids)
  return result


def embedding_lookup_sparse(
    params,
    sp_ids,
    sp_weights,
    partition_strategy="mod",
    name=None,
    combiner=None,
    max_norm=None,
):
  """
    Based on https://github.com/tensorflow/tensorflow/blob/v1.13.1/tensorflow/python/ops/embedding_ops.py#L379.
    We use array_ops.unique_with_counts instead of array_ops.unique so we can
    increase the frequency counting for KvVariable.
    """
  # compatibility with list/PartitionedVariable/Variable types
  if isinstance(params, ranking_utils.SUPPORTED_TYPES):
    unique_name = "%s_lookup" % params.name.split(":")[0]
    bucket_size = params.shape.as_list()[0]
  result = _embedding_lookup_sparse(
      params=params,
      sp_ids=sp_ids,
      sp_weights=sp_weights,
      partition_strategy=partition_strategy,
      name=name,
      combiner=combiner,
      max_norm=max_norm,
  )
  if isinstance(params, ranking_utils.SUPPORTED_TYPES):
    ranking_utils.update_embedding_for_ranking(
        unique_name,
        params,
        combiner,
        None,
        bucket_size=bucket_size,
        partition_strategy=partition_strategy,
    )
    ranking_utils.append_embedding_result_for_ranking(unique_name, result)
    ranking_utils.append_embedding_input_for_ranking(unique_name, sp_ids)
  return result


def _embedding_lookup_sparse(
    params,
    sp_ids,
    sp_weights,
    partition_strategy="mod",
    name=None,
    combiner=None,
    max_norm=None,
):
  """
    Based on https://github.com/tensorflow/tensorflow/blob/v1.13.1/tensorflow/python/ops/embedding_ops.py#L379.
    We use array_ops.unique_with_counts instead of array_ops.unique so we can
    increase the frequency counting for KvVariable.
    """
  if combiner is None:
    logging.warn('The default value of combiner will change from "mean" '
                 'to "sqrtn" after 2016/11/01.')
    combiner = "mean"
  if combiner not in ("mean", "sqrtn", "sum"):
    raise ValueError("combiner must be one of 'mean', 'sqrtn' or 'sum'")
  if isinstance(params, variables.PartitionedVariable):
    params = list(params)  # Iterate to get the underlying Variables.
  if not isinstance(params, list):
    params = [params]
  if not isinstance(sp_ids, sparse_tensor.SparseTensor):
    raise TypeError("sp_ids must be SparseTensor")
  ignore_weights = sp_weights is None
  if not ignore_weights:
    if not isinstance(sp_weights, sparse_tensor.SparseTensor):
      raise TypeError("sp_weights must be either None or SparseTensor")
    sp_ids.values.get_shape().assert_is_compatible_with(
        sp_weights.values.get_shape())
    sp_ids.indices.get_shape().assert_is_compatible_with(
        sp_weights.indices.get_shape())
    sp_ids.dense_shape.get_shape().assert_is_compatible_with(
        sp_weights.dense_shape.get_shape())
    # TODO(yleon): Add enhanced node assertions to verify that sp_ids and
    # sp_weights have equal indices and shapes.
  with ops.name_scope(name, "embedding_lookup_sparse",
                      params + [sp_ids]) as name:  # pylint:disable=redefined-argument-from-local
    segment_ids = sp_ids.indices[:, 0]
    if segment_ids.dtype != dtypes.int32:
      segment_ids = math_ops.cast(segment_ids, dtypes.int32)

    ids = sp_ids.values
    need_counts = (isinstance(params[0], kv_variable_ops.KvVariable)
                   and params[0].enter_threshold > 0)
    # use unique_with_counts to get each ids' counts
    if need_counts:
      ids, idx, counts = array_ops.unique_with_counts(ids)
    else:
      ids, idx = array_ops.unique(ids)
      counts = None

    embeddings = _embedding_lookup_and_transform(
        params,
        ids,
        counts=counts,
        partition_strategy=partition_strategy,
        max_norm=max_norm,
    )
    if embeddings.dtype in (dtypes.float16, dtypes.bfloat16):
      embeddings = math_ops.to_float(embeddings)
    if not ignore_weights:
      weights = sp_weights.values
      if weights.dtype != embeddings.dtype:
        weights = math_ops.cast(weights, embeddings.dtype)

      embeddings = array_ops.gather(embeddings, idx)

      # Reshape weights to allow broadcast
      ones = array_ops.fill(
          array_ops.expand_dims(array_ops.rank(embeddings) - 1, 0), 1)
      bcast_weights_shape = array_ops.concat([array_ops.shape(weights), ones],
                                             0)

      orig_weights_shape = weights.get_shape()
      weights = array_ops.reshape(weights, bcast_weights_shape)

      # Set the weight shape, since after reshaping to bcast_weights_shape,
      # the shape becomes None.
      if embeddings.get_shape().ndims is not None:
        weights.set_shape(
            orig_weights_shape.concatenate(
                [1 for _ in range(embeddings.get_shape().ndims - 1)]))

      embeddings *= weights

      if combiner == "sum":
        embeddings = math_ops.segment_sum(embeddings, segment_ids, name=name)
      elif combiner == "mean":
        embeddings = math_ops.segment_sum(embeddings, segment_ids)
        weight_sum = math_ops.segment_sum(weights, segment_ids)
        embeddings = math_ops.div(embeddings, weight_sum, name=name)
      elif combiner == "sqrtn":
        embeddings = math_ops.segment_sum(embeddings, segment_ids)
        weights_squared = math_ops.pow(weights, 2)
        weight_sum = math_ops.segment_sum(weights_squared, segment_ids)
        weight_sum_sqrt = math_ops.sqrt(weight_sum)
        embeddings = math_ops.div(embeddings, weight_sum_sqrt, name=name)
      else:
        assert False, "Unrecognized combiner"
    else:
      assert idx is not None
      if combiner == "sum":
        embeddings = math_ops.sparse_segment_sum(embeddings,
                                                 idx,
                                                 segment_ids,
                                                 name=name)
      elif combiner == "mean":
        embeddings = math_ops.sparse_segment_mean(embeddings,
                                                  idx,
                                                  segment_ids,
                                                  name=name)
      elif combiner == "sqrtn":
        embeddings = math_ops.sparse_segment_sqrt_n(embeddings,
                                                    idx,
                                                    segment_ids,
                                                    name=name)
      else:
        assert False, "Unrecognized combiner"
    return embeddings


def safe_embedding_lookup_sparse(
    embedding_weights,
    sparse_ids,
    sparse_weights=None,
    combiner="mean",
    default_id=None,
    name=None,
    partition_strategy="div",
    max_norm=None,
):
  """
  Based on https://github.com/tensorflow/tensorflow/blob/v1.13.1/tensorflow/python/ops/embedding_ops.py#L632
  Lookup embedding results, accounting for empty features.
  We do accept ID < 0 as valid id.

  The partitioned embedding in `embedding_weights` must all be the same shape
  except for the first dimension. The first dimension is allowed to vary as the
  vocabulary size is not necessarily a multiple of `P`.  `embedding_weights`
  may be a `PartitionedVariable`.

  For an entry with no features, the embedding vector
  for `default_id` is returned, or the 0-vector if `default_id` is not supplied.

  The ids and weights may be multi-dimensional. Embeddings are always aggregated
  along the last dimension.

  Args:
    embedding_weights:  A list of `P` float `Tensor`s or values representing
        partitioned embedding `Tensor`s.  Alternatively, a `PartitionedVariable`
        created by partitioning along dimension 0.  The total unpartitioned
        shape should be `[e_0, e_1, ..., e_m]`, where `e_0` represents the
        vocab size and `e_1, ..., e_m` are the embedding dimensions.
    sparse_ids: `SparseTensor` of shape `[d_0, d_1, ..., d_n]` containing the
        ids. `d_0` is typically batch size.
    sparse_weights: `SparseTensor` of same shape as `sparse_ids`, containing
        float weights corresponding to `sparse_ids`, or `None` if all weights
        are be assumed to be 1.0.
    combiner: A string specifying how to combine embedding results for each
        entry. Currently "mean", "sqrtn" and "sum" are supported, with "mean"
        the default.
    default_id: The id to use for an entry with no features.
    name: A name for this operation (optional).
    partition_strategy: A string specifying the partitioning strategy.
        Currently `"div"` and `"mod"` are supported. Default is `"div"`.
    max_norm: If not `None`, all embeddings are l2-normalized to max_norm before
        combining.


  Returns:
    Dense `Tensor` of shape `[d_0, d_1, ..., d_{n-1}, e_1, ..., e_m]`.

  Raises:
    ValueError: if `embedding_weights` is empty.
  """
  if embedding_weights is None:
    raise ValueError("Missing embedding_weights %s." % embedding_weights)
  # compatibility with list/PartitionedVariable/Variable types
  if isinstance(embedding_weights, ranking_utils.SUPPORTED_TYPES):
    bucket_size = embedding_weights.shape.as_list()[0]
  origin_weights = embedding_weights
  origin_sparse_ids = sparse_ids
  if isinstance(embedding_weights, variables.PartitionedVariable):
    embedding_weights = list(embedding_weights)  # get underlying Variables.
  if not isinstance(embedding_weights, list):
    embedding_weights = [embedding_weights]
  if not embedding_weights:
    raise ValueError("Missing embedding_weights %s." % embedding_weights)

  dtype = sparse_weights.dtype if sparse_weights is not None else None
  embedding_weights = [
      w if
      (isinstance(w, resource_variable_ops.ResourceVariable)
       and dtype in (None, w.dtype)) else ops.convert_to_tensor(w, dtype=dtype)
      for w in embedding_weights
  ]

  is_kv_variable = isinstance(embedding_weights[0], kv_variable_ops.KvVariable)
  if is_kv_variable and not all(
      isinstance(p, kv_variable_ops.KvVariable) for p in embedding_weights):
    raise ValueError("All embedding_weights should be KvVariable")

  with ops.name_scope(
      name,
      "embedding_lookup",
      embedding_weights + [sparse_ids, sparse_weights],
  ) as scope:
    # Reshape higher-rank sparse ids and weights to linear segment ids.
    original_shape = sparse_ids.dense_shape
    original_rank_dim = tensor_shape.dimension_value(
        sparse_ids.dense_shape.get_shape()[0])
    original_rank = (array_ops.size(original_shape)
                     if original_rank_dim is None else original_rank_dim)
    if original_rank_dim is not None and original_rank_dim > 2:
      sparse_ids = sparse_ops.sparse_reshape(
          sparse_ids,
          [
              math_ops.reduce_prod(
                  array_ops.slice(original_shape, [0], [original_rank - 1])),
              array_ops.gather(original_shape, original_rank - 1),
          ],
      )
    if sparse_weights is not None:
      sparse_weights = sparse_tensor.SparseTensor(
          sparse_ids.indices,
          sparse_weights.values,
          sparse_ids.dense_shape,
      )

    if not is_kv_variable:
      # Prune invalid ids only for non kv_variable.
      # pylint: disable = protected-access
      sparse_ids, sparse_weights = embedding_ops._prune_invalid_ids(
          sparse_ids, sparse_weights)
    if combiner != "sum":
      # pylint: disable = protected-access
      sparse_ids, sparse_weights = embedding_ops._prune_invalid_weights(
          sparse_ids, sparse_weights)

    # Fill in dummy values for empty features, if necessary.
    sparse_ids, is_row_empty = sparse_ops.sparse_fill_empty_rows(
        sparse_ids, default_id or 0)
    if sparse_weights is not None:
      sparse_weights, _ = sparse_ops.sparse_fill_empty_rows(sparse_weights, 1.0)

    result = _embedding_lookup_sparse(
        embedding_weights,
        sparse_ids,
        sparse_weights,
        combiner=combiner,
        partition_strategy=partition_strategy,
        name=None if default_id is None else scope,
        max_norm=max_norm,
    )

    if default_id is None:
      # Broadcast is_row_empty to the same shape as embedding_lookup_result,
      # for use in Select.
      is_row_empty = array_ops.tile(
          array_ops.reshape(is_row_empty, [-1, 1]),
          tf.stack([1, array_ops.shape(result)[1]]),
      )

      result = array_ops.where(is_row_empty,
                               array_ops.zeros_like(result),
                               result,
                               name=scope)

    if original_rank_dim is not None and original_rank_dim > 2:
      # Reshape back from linear ids back into higher-dimensional dense result.
      final_result = array_ops.reshape(
          result,
          array_ops.concat(
              [
                  array_ops.slice(
                      math_ops.cast(original_shape, dtypes.int32),
                      [0],
                      [original_rank - 1],
                  ),
                  array_ops.slice(array_ops.shape(result), [1], [-1]),
              ],
              0,
          ),
      )
      final_result.set_shape(
          tensor_shape.unknown_shape(
              (tensor_shape.Dimension(original_rank_dim) -
               1).value).concatenate(result.get_shape()[1:]))
    else:
      final_result = result
    # compatibility with list/PartitionedVariable/Variable types
    if isinstance(origin_weights, ranking_utils.SUPPORTED_TYPES):
      unique_name = "%s_safe_sparse" % origin_weights.name.split(":")[0]
      ranking_utils.update_embedding_for_ranking(
          unique_name,
          origin_weights,
          combiner,
          None,
          bucket_size=bucket_size,
          partition_strategy=partition_strategy,
      )
      ranking_utils.append_embedding_result_for_ranking(unique_name,
                                                        final_result)
      ranking_utils.append_embedding_input_for_ranking(unique_name,
                                                       origin_sparse_ids)
    return final_result


original_embedding_lookup = embedding_ops.embedding_lookup
original_embedding_lookup_sparse = embedding_ops.embedding_lookup_sparse
original_safe_embedding_lookup_sparse = (
    embedding_ops.safe_embedding_lookup_sparse)
original_safe_embedding_lookup_sparse_v2 = (
    embedding_ops.safe_embedding_lookup_sparse_v2)

safe_embedding_lookup_sparse_v2 = None

# tf2.13 change
# original_embedding_lookup_unique = contrib_embedding_ops.embedding_lookup_unique


def embedding_lookup_unique(params, ids, partition_strategy="mod", name=None):
  """Version of embedding_lookup that avoids duplicate lookups.

  This can save communication in the case of repeated ids.
  Same interface as embedding_lookup. Except it supports multi-dimensional `ids`
  which allows to not reshape input/output to fit gather.

  Args:
    params: A list of tensors with the same shape and type, or a
      `PartitionedVariable`. Shape `[index, d1, d2, ...]`.
    ids: A one-dimensional `Tensor` with type `int32` or `int64` containing
      the ids to be looked up in `params`. Shape `[ids1, ids2, ...]`.
    partition_strategy: A string specifying the partitioning strategy, relevant
      if `len(params) > 1`. Currently `"div"` and `"mod"` are supported. Default
      is `"mod"`.
    name: A name for this operation (optional).

  Returns:
    A `Tensor` with the same type as the tensors in `params` and dimension of
    `[ids1, ids2, d1, d2, ...]`.

  Raises:
    ValueError: If `params` is empty.
  """
  # compatibility with list/PartitionedVariable/Variable types
  if isinstance(params, ranking_utils.SUPPORTED_TYPES):
    unique_name = "%s_lookup_unique" % params.name.split(":")[0]
    bucket_size = params.shape.as_list()[0]
  with ops.name_scope(name, "EmbeddingLookupUnique", [params, ids]):
    ids = ops.convert_to_tensor(ids)
    shape = array_ops.shape(ids)
    ids_flat = array_ops.reshape(ids, math_ops.reduce_prod(shape,
                                                           keepdims=True))
    unique_ids, idx = array_ops.unique(ids_flat)
    unique_embeddings = _embedding_lookup_and_transform(params, unique_ids,
                                                        partition_strategy)
    embeds_flat = array_ops.gather(unique_embeddings, idx)
    embed_shape = array_ops.concat(
        [shape, array_ops.shape(unique_embeddings)[1:]], 0)
    embeds = array_ops.reshape(embeds_flat, embed_shape)
    embeds.set_shape(ids.get_shape().concatenate(
        unique_embeddings.get_shape()[1:]))
  if isinstance(params, ranking_utils.SUPPORTED_TYPES):
    ranking_utils.update_embedding_for_ranking(
        unique_name,
        params,
        None,
        None,
        bucket_size=bucket_size,
        partition_strategy=partition_strategy,
    )
    ranking_utils.append_embedding_result_for_ranking(unique_name, embeds)
    ranking_utils.append_embedding_input_for_ranking(unique_name, ids)
  return embeds


# tf2.13 change
# contrib_embedding_ops.embedding_lookup_unique = embedding_lookup_unique


def insert_kv_embedding(params, ids, values, name=None):
  """Insert id-value pair(s) into existing kv embedding"""
  if params is None or params in ((), []):
    raise ValueError("Need at least one param")
  if isinstance(params, variables.PartitionedVariable):
    params = list(params)  # Iterate to get the underlying Variables.
  if not isinstance(params, list):
    params = [params]
  if not all(isinstance(p, kv_variable_ops.KvVariable) for p in params):
    raise ValueError("All params should be KvVariable")

  pattern = r"(.+)/part_(\d+):0$"
  match = re.match(pattern, params[0].name)
  if not match:
    raise ValueError("Unknown KvVariable %s" % params[0].name)
  prefix = match.group(1)
  for param in params:
    if not param.name.startswith(prefix):
      raise ValueError("All KvVariable should be the same, "
                       "found %s" % param.name)

  assert (ids.dtype == params[0].key_dtype
          ), "ids' dtype: {} not matched with embedding's {}".format(
              ids.dtype, params[0].key_dtype)
  assert (values.dtype == params[0].dtype
          ), "values' dtype: {} not matched with embedding's {}".format(
              values.dtype, params[0].dtype)
  assert (values.shape.as_list()[-1] == params[0].shape.as_list(
  )[-1]), "values' shape: {} not matched with " "embedding dimension {}".format(
      values.shape, params[0].shape)

  with ops.name_scope(name, "load_kv_embedding", params + [ids, values]):
    num_partition = len(params)
    if ids.dtype == dtypes.string:
      p_assignments = string_ops.string_to_hash_bucket_fast(ids, num_partition)
    else:
      p_assignments = ids % num_partition

    # Cast partition assignments to int32 for use in dynamic_partition.
    # There really should not be more than 2^32 partitions.
    p_assignments = math_ops.cast(p_assignments, dtypes.int32)
    # Partition list of ids based on assignments into np separate lists
    gather_keys = data_flow_ops.dynamic_partition(ids, p_assignments,
                                                  num_partition)
    gather_values = data_flow_ops.dynamic_partition(values, p_assignments,
                                                    num_partition)

    scatter_update_ops = [
        kv_variable_ops.scatter_update(params[i], gather_keys[i],
                                       gather_values[i])
        for i in range(num_partition)
    ]
    return scatter_update_ops


def use_tfplus_embedding_ops():
  """use_tfplus_embedding_ops"""
  embedding_ops.embedding_lookup = embedding_lookup
  embedding_ops.embedding_lookup_sparse = embedding_lookup_sparse
  embedding_ops.safe_embedding_lookup_sparse = safe_embedding_lookup_sparse
  tf.nn.embedding_lookup = embedding_lookup
  tf.nn.embedding_lookup_sparse = embedding_lookup_sparse
  tf.nn.safe_embedding_lookup_sparse = safe_embedding_lookup_sparse
  global safe_embedding_lookup_sparse_v2
  safe_embedding_lookup_sparse_v2 = (
      embedding_ops.safe_embedding_lookup_sparse_v2)

def use_tf_embedding_ops():
  """use_tf_embedding_ops"""
  embedding_ops.embedding_lookup = original_embedding_lookup
  embedding_ops.embedding_lookup_sparse = original_embedding_lookup_sparse
  embedding_ops.safe_embedding_lookup_sparse = (
      original_safe_embedding_lookup_sparse)
  tf.nn.embedding_lookup = original_embedding_lookup
  tf.nn.embedding_lookup_sparse = original_embedding_lookup_sparse
  tf.nn.safe_embedding_lookup_sparse = original_safe_embedding_lookup_sparse

  embedding_ops.safe_embedding_lookup_sparse_v2 = (
      original_safe_embedding_lookup_sparse_v2)
  global safe_embedding_lookup_sparse_v2
  safe_embedding_lookup_sparse_v2 = original_safe_embedding_lookup_sparse_v2


def _change_embedding_to_eflops_gen():
  """
    switch all2all/normal embedding ops
    """
  from functools import wraps

  embedding_ops_name = {
      "embedding_lookup",
      "embedding_lookup_sparse",
      "safe_embedding_lookup_sparse",
      "embedding_lookup_unique",
  }

  origin_functions_ = dict(
      embedding_lookup=embedding_lookup,
      embedding_lookup_sparse=embedding_lookup_sparse,
      safe_embedding_lookup_sparse=safe_embedding_lookup_sparse,
      embedding_lookup_unique=embedding_lookup_unique,
  )

  def _is_eflops_mode():
    if not tf.test.is_built_with_cuda():
      return False
    try:
      from tfplus.kv_variable.python.ops import eflops_embedding_ops
    except Exception:  # pylint: disable=W0703,bare-except
      logging.warn("eflops mode is not supported")
      return False
    return all(globals()[name] is getattr(eflops_embedding_ops, name)
               for name in embedding_ops_name)

  def _wrap(fn):

    @wraps(fn)
    def wrap(*args, **kwargs):
      # fall back is true IFF lookup non kv in eflops' lookup op
      fall_back = kwargs.pop("fall_back", False)
      if fn.__name__ not in embedding_ops_name:
        raise ValueError("not wrap embedding ops")
      if _is_eflops_mode() and not fall_back:
        from tfplus.kv_variable.python.ops import eflops_embedding_ops

        logging.info("call eflops embedding ops")
        return getattr(eflops_embedding_ops, fn.__name__)(*args, **kwargs)
      logging.info("call normal embedding ops")
      kwargs.pop("allow_fast_lookup", False)  # tf2.13 change
      return origin_functions_[fn.__name__](*args, **kwargs)

    return wrap

  def _exchange_embedding_ops(use_eflops=True):
    if use_eflops:
      logging.info("use embedding ops of eflops")
      from tfplus.kv_variable.python.ops import eflops_embedding_ops

      for name in embedding_ops_name:
        globals()[name] = getattr(eflops_embedding_ops, name)
    else:
      logging.info("use embedding ops of origin tfplus")
      for origin_fn_name, origin_fn in origin_functions_.items():
        globals()[origin_fn_name] = origin_fn
    use_tfplus_embedding_ops()

  wrap_origin_functions_ = dict(
      embedding_lookup=_wrap(embedding_lookup),
      embedding_lookup_sparse=_wrap(embedding_lookup_sparse),
      safe_embedding_lookup_sparse=_wrap(safe_embedding_lookup_sparse),
      embedding_lookup_unique=_wrap(embedding_lookup_unique),
  )

  def _get_origin_tfplus_embedding_ops(name):
    if name not in wrap_origin_functions_:
      raise ValueError(f"{name} is not in embedding ops")
    return wrap_origin_functions_[name]

  return (
      _exchange_embedding_ops,
      _is_eflops_mode,
      _get_origin_tfplus_embedding_ops,
      _wrap,
  )


(
    change_embedding_to_eflops,
    is_eflops_mode,
    get_origin_tfplus_embedding_ops,
    wrap_embedding_ops,
) = _change_embedding_to_eflops_gen()

embedding_lookup = wrap_embedding_ops(embedding_lookup)
embedding_lookup_sparse = wrap_embedding_ops(embedding_lookup_sparse)
safe_embedding_lookup_sparse = wrap_embedding_ops(safe_embedding_lookup_sparse)
embedding_lookup_unique = wrap_embedding_ops(embedding_lookup_unique)

use_tfplus_embedding_ops()
