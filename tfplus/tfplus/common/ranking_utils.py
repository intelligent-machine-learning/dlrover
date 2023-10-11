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
"""ranking embedding util functions used by feature column.
"""
from __future__ import absolute_import, division, print_function

import json

from tensorflow.python.framework import ops, sparse_tensor
from tensorflow.python.ops import resource_variable_ops, variables
from tensorflow.python.saved_model import signature_def_utils, utils

from tfplus.kv_variable.python.ops import kv_variable_ops

RANKING_SERVICE_EMBEDDING = "__rank_service_embedding"

__all__ = [
    "add_tensor_to_collection",
    "append_tensor_to_collection",
    "update_embedding_for_ranking",
    "append_embedding_result_for_ranking",
    "append_embedding_input_for_ranking",
]

SUPPORTED_TYPES = (
    resource_variable_ops.ResourceVariable,
    variables.PartitionedVariable,
    variables.Variable,
)


def _tensor_to_map(tensor):
  return {
      "node_path": tensor.name,
      "shape": tensor.shape.as_list() if tensor.shape else None,
      "dtype": tensor.dtype.name,
  }


# pylint: disable=missing-docstring
def _tensor_to_tensorinfo(tensor):
  tensor_info = {}
  if isinstance(tensor, sparse_tensor.SparseTensor):
    tensor_info["is_dense"] = False
    tensor_info["values"] = _tensor_to_map(tensor.values)
    tensor_info["indices"] = _tensor_to_map(tensor.indices)
    tensor_info["dense_shape"] = _tensor_to_map(tensor.dense_shape)
  else:
    tensor_info["is_dense"] = True
    tensor_info.update(_tensor_to_map(tensor))
  return tensor_info


def add_tensor_to_collection(collection_name, name, tensor):
  tensor_info = _tensor_to_tensorinfo(tensor)
  tensor_info["name"] = name
  update_attr_to_collection(collection_name, tensor_info)


def append_tensor_to_collection(collection_name, name, key, tensor):
  tensor_info = _tensor_to_tensorinfo(tensor)
  append_attr_to_collection(collection_name, name, key, tensor_info)


# pylint: disable=missing-docstring
def _process_item(collection_name, name, func):
  col = ops.get_collection_ref(collection_name)
  item_found = {}
  idx_found = -1
  for idx, c in enumerate(col):
    item = json.loads(c)
    if item["name"] == name:
      item_found = item
      idx_found = idx
      break
  func(item_found)
  if idx_found == -1:
    col.append(json.dumps(item_found))
  else:
    col[idx_found] = json.dumps(item_found)


def append_attr_to_collection(collection_name, name, key, value):

  def append(item_found):
    if key not in item_found:
      item_found[key] = []
    item_found[key].append(value)

  _process_item(collection_name, name, append)


def update_attr_to_collection(collection_name, attrs):

  def update(item_found):
    item_found.update(attrs)

  _process_item(collection_name, attrs["name"], update)


# pylint: disable=missing-docstring
def update_embedding_for_ranking(
    column_name,
    variable,
    combiner,
    max_norm,
    bucket_size=1000,
    partition_strategy="mod",
):
  attrs = {
      "bucket_size": bucket_size,
      "combiner": combiner,
      "max_norm": max_norm,
  }
  if partition_strategy:
    attrs["partition_strategy"] = partition_strategy
  attrs["name"] = column_name
  attrs["weights_op_path"] = variable.name
  attrs["is_embedding_var"] = False
  if isinstance(variable, kv_variable_ops.KvVariable):
    attrs["is_embedding_var"] = True
    attrs["embedding_var_keys"] = [variable._shared_name + ":0-keys"]  # pylint: disable=protected-access
    attrs["embedding_var_values"] = [variable._shared_name + ":0-values"]  # pylint: disable=protected-access
    attrs["partition_strategy"] = "mod"
  elif isinstance(variable, variables.PartitionedVariable):
    variables_list = list(variable)
    if isinstance(variables_list[0], kv_variable_ops.KvVariable):
      attrs["is_embedding_var"] = True
      attrs["embedding_var_keys"] = [
          v._shared_name + ":0-keys" for v in variables_list  # pylint: disable=protected-access
      ]
      attrs["embedding_var_values"] = [
          v._shared_name + ":0-values" for v in variables_list  # pylint: disable=protected-access
      ]
      attrs["partition_strategy"] = "mod"
  update_attr_to_collection(RANKING_SERVICE_EMBEDDING, attrs)


def append_embedding_result_for_ranking(column_name, embedding_result):
  append_tensor_to_collection(RANKING_SERVICE_EMBEDDING, column_name, "tensor",
                              embedding_result)


def append_embedding_input_for_ranking(column_name, input_tensors):
  append_tensor_to_collection(RANKING_SERVICE_EMBEDDING, column_name, "input",
                              input_tensors)


def generate_signature(features, predictions):
  if not isinstance(features, dict):
    raise ValueError(
        "generate_signature excepted features to be dict, but got %s" %
        features)
  inputs = dict(
      zip(
          features,
          map(lambda x: utils.build_tensor_info(x), features.values()),  # pylint: disable=unnecessary-lambda
      ))

  if not isinstance(predictions, dict):
    predictions = {"prediction": predictions}
  outputs = dict(
      zip(
          predictions,
          map(lambda x: utils.build_tensor_info(x), predictions.values()),  # pylint: disable=unnecessary-lambda
      ))

  signature = signature_def_utils.build_signature_def(inputs, outputs)
  ops.get_collection_ref("FEATURE_INPUTS").append(signature.SerializeToString())
  return signature
