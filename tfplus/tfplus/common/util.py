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
"""common utils"""
import collections
import json
import os
import re

from tensorflow.python.framework import ops
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training import training as train

from tfplus.common import constants, is_tf_1_13_or_higher
from tfplus.common.constants import Constants
from tfplus.kv_variable.python.ops import kv_variable_ops, variable_scope


def skip(*tests):
  """
    used for skipping unittests
    which are in-compatiable with tf 1.12
    """

  def wrap(clz):

    def wrap_v(func):

      def inner(*args, **kwargs):
        if not is_tf_1_13_or_higher():  # change your condition here
          return
        func(*args, **kwargs)

      return inner

    v = vars(clz)
    for vv in v:
      if vv in tests:
        setattr(clz, vv, wrap_v(getattr(clz, vv)))
    return clz

  return wrap


def is_tfplus_opt_slot_var(var_name):
  """Judge whether it is an slot variable used by optimizer.

    For Adam optimizer, we match the '/Adam:0' or '/Adam_'+any numbers
    +':0'(such as '/Adam_2:0') in var_name.
    """
  for opt_name in constants.OPT_NAMES:
    pattern = r"(.+)/%s(_\d+)?:0$" % opt_name
    if re.match(pattern, var_name):
      return True
  return False


def group_var_by_embedding(var_names):
  """Group vars by corresponding embedding.

    Args:
        var_names: a list of kv variable name, such as
          ['embedding/part_0:0', 'embedding/part_0/GroupAdam_0:0',
           'embedding/part_0/GroupAdam_1:0',
           'embedding/part_1:0', 'embedding/part_1/GroupAdam_0:1',
           'embedding/part_1/GroupAdam_1:0']
    Returns:
        A dict of {embedding_name: slot_var_list}, such as
        {
          'embedding/part_0:0': ['embedding/part_0/GroupAdam_0:0',
                                 'embedding/part_0/GroupAdam_1:0'],
          'embedding/part_1:0': ['embedding/part_1/GroupAdam_0:1',
                                 'embedding/part_1/GroupAdam_1:0']
        }
    """
  embedding_group = {}
  slot_names = [
      var_name for var_name in var_names if is_tfplus_opt_slot_var(var_name)
  ]
  for var_name in var_names:
    if not is_tfplus_opt_slot_var(var_name):
      embedding_name = re.findall("(.*):0", var_name)[0]
      embedding_group[var_name] = []
      for slot_name in slot_names:
        slot_pattern = "((.*)/)?%s/(.*):0" % embedding_name
        slot_result = re.match(slot_pattern, slot_name)
        if slot_result:
          embedding_group[var_name].append(slot_name)
  for slots in embedding_group.values():
    slots.sort()
    if len(slots) == 4 and "GroupAdam" in slots[0]:
      slots.pop(1)
  return embedding_group


def get_generic_name(var_name):
  """Get repartition generic name.

    Args:
        var_name: kv variable real name in runtime,
          such as embedding_table/part_10/GroupAdam_3:0;
    Returns:
        A str var_name remove `part_*` and `:0`
    """
  var_name = var_name[:var_name.rfind(":")]
  match = re.search(r"/part_\d+", var_name, 0)
  if match is not None:
    span = match.span()
    return var_name[:span[0]] + var_name[span[1]:]
  return var_name


def get_name_info(var_name):
  """Get variable's name and part info"""
  match = re.search(r"/part_\d+", var_name, 0)
  if match is None:
    return var_name[:-2], ":0", 0
  span = match.span()
  prefix = var_name[:span[0]]
  suffix = var_name[span[1]:]
  index = int(var_name[span[0] + 6:span[1]])
  return prefix, suffix, index


def check_num_shards_in_checkpoint(var_list, checkpoint, var_mapping=None):
  """Check checkpoint and graph num shards."""
  if not ops.get_collection(variable_scope.GRAPHSHARDS_KEY) or not var_list:
    return

  if not is_tf_1_13_or_higher():
    return
  ckpt_parser = CheckpointParser(checkpoint)
  ckpt_num_shards = ckpt_parser.embedding_num_shards
  graph_num_shards = ops.get_collection(variable_scope.GRAPHSHARDS_KEY)[0]
  for var in var_list:
    if isinstance(var, kv_variable_ops.KvVariable):
      var_name = var.name
      if not is_tfplus_opt_slot_var(var_name):
        generic_name = get_generic_name(var_name)
        if var_mapping and var_name in var_mapping:
          ckpt_var_name = var_mapping[var_name]
          ckpt_generic_name = get_generic_name(ckpt_var_name)
        else:
          ckpt_generic_name = generic_name
        if (graph_num_shards[generic_name]
            != ckpt_num_shards[ckpt_generic_name]):
          raise ValueError("Num shards for %s in checkpoint %s "
                           "is different from num shards "
                           "in graph (%s != %s).\n"
                           "checkpoint num shards: %s.\n"
                           "graph num shards: %s.\n"
                           "var mapping: %s" % (
                               generic_name,
                               checkpoint,
                               ckpt_num_shards[ckpt_generic_name],
                               graph_num_shards[generic_name],
                               ckpt_num_shards,
                               graph_num_shards,
                               var_mapping,
                           ))


def get_graph_num_shards():
  if not ops.get_collection(variable_scope.GRAPHSHARDS_KEY):
    return {}
  return ops.get_collection(variable_scope.GRAPHSHARDS_KEY)[0]


class CheckpointParser:
  """
    CheckpointParser is a class for analyzing checkpoints.
    """

  def __init__(self, ckpt_path):
    """
        Args:
            ckpt_path: checkpoint path for analyzing
        """
    self._ckpt_path = None
    self._ckpt_file = None
    self._ckpt_reader = None
    self._var_dtype_map = None
    self._var_shape_map = None
    self._embedding_num_shards = None
    self._merge_var_dtype_map = None
    self._merge_var_shape_map = None
    self._ckpt_num_shards = None

    if not ckpt_path:
      logging.info("Ckpt path is None, invalid parser or no ckpt before")
    else:
      self._ckpt_file = self._get_checkpoint_filename(ckpt_path)
      if not self._ckpt_file:
        logging.info("Ckpt file not exist, invalid parser or no ckpt before")
        return
      self._ckpt_path = ckpt_path
      self._ckpt_reader = train.NewCheckpointReader(self._ckpt_file)

      self._var_dtype_map = self._ckpt_reader.get_variable_to_dtype_map()
      self._var_shape_map = self._ckpt_reader.get_variable_to_shape_map()
      kv_vars = [
          var_name[:-5] for var_name in self._var_dtype_map
          if var_name.endswith(":0-keys")
      ]
      self._embedding_group = group_var_by_embedding(kv_vars)
      self._embedding_num_shards = (self._load_embedding_num_shards_from_ckpt())
      self._ckpt_num_shards = self._generate_ckpt_partition_info()

  @property
  def ckpt_path(self):
    return self._ckpt_path

  @property
  def embedding_num_shards(self):
    return self._embedding_num_shards

  @property
  def ckpt_num_shards(self):
    return self._ckpt_num_shards

  @property
  def var_shape_map(self):
    return self._var_shape_map

  def _get_checkpoint_filename(self, ckpt_dir_or_file):
    """
        Returns checkpoint filename given directory or
        specific checkpoint file.
        """
    if gfile.IsDirectory(ckpt_dir_or_file):
      return checkpoint_management.latest_checkpoint(ckpt_dir_or_file)
    if gfile.Exists(ckpt_dir_or_file + ".index"):
      return ckpt_dir_or_file
    return None

  def _load_embedding_num_shards_from_ckpt(self):
    """Get embedding num shards dict from ckpt dir."""
    if not self.ckpt_path:
      return {}

    ckpt_num_shards = collections.defaultdict(int)
    for embedding_name in self._embedding_group:
      generic_name = get_generic_name(embedding_name)
      ckpt_num_shards[generic_name] += 1
    return ckpt_num_shards

  def load_auto_partition_plan(self):
    """get num shards dict from json file in ckpt dir"""
    if not self._ckpt_path:
      return {}

    if gfile.IsDirectory(self._ckpt_path):
      partition_path = self._ckpt_path
    else:
      match = re.search(r"/[0-9a-zA-Z-_]+.ckpt-\d+", self._ckpt_path, 0)
      if match is None:
        return {}
      span = match.span()
      partition_path = self._ckpt_path[:span[0]]
    partition_file = os.path.join(partition_path,
                                  "kv_var_repartition_plan.json")

    new_num_shards = {}
    if gfile.Exists(partition_file):
      with gfile.GFile(partition_file, "r") as f:
        new_num_shards = json.loads(f.read())
    return new_num_shards

  def genertate_partition_info_of_new_graph(self, new_num_shards):
    """get num shards dict for variable requiring repartition"""
    if not self._ckpt_reader:
      return {}

    repartition_num_shards = self._embedding_num_shards.copy()
    for var_name, num_shards in new_num_shards.items():
      repartition_num_shards[var_name] = num_shards

    for embedding, slots in self._embedding_group.items():
      for slot in slots:
        repartition_num_shards[get_generic_name(slot)] = repartition_num_shards[
            get_generic_name(embedding)]
    return repartition_num_shards

  def _generate_ckpt_partition_info(self):
    """get num shards dict for ckpt"""
    if not self._ckpt_reader:
      return {}
    ckpt_num_shards = self._embedding_num_shards.copy()
    for embedding, slots in self._embedding_group.items():
      for slot in slots:
        ckpt_num_shards[get_generic_name(slot)] = ckpt_num_shards[
            get_generic_name(embedding)]
    return ckpt_num_shards

  def generate_dynamic_restore_group(self, var_list, assignment_map=None):
    """get dynamic restore group"""
    restore_group = collections.defaultdict(lambda: [])
    restore_mode = {}
    restore_var_names = [var.handle.name for var in var_list]
    graph_num_shards = get_graph_num_shards()
    if not self._ckpt_reader:
      raise InvalidArgumentError("Checkpoint path {} is not valid".format(
          self._ckpt_path))
      # for var_name in restore_var_names:
      #   restore_mode[var_name] = Constants.NORMAL_RESTORE
      #   restore_group[var_name] = var_name
      # return restore_group, restore_mode, graph_num_shards, graph_num_shards

    graph_embedding_group = group_var_by_embedding(restore_var_names)

    ckpt_num_shards = self._generate_ckpt_partition_info()
    ckpt_embedding_group = self._embedding_group
    print("--- self._ckpt_path {}".format(self._ckpt_path))
    print("--- graph_embedding_group {}".format(graph_embedding_group))
    print("--- ckpt_embedding_group {}".format(ckpt_embedding_group))
    print("--- ckpt_num_shards {}".format(ckpt_num_shards))
    print("--- graph_num_shards {}".format(graph_num_shards))
    print("--- assignment_map {}".format(assignment_map))
    for var_name in restore_var_names:
      if var_name not in graph_embedding_group:
        continue
      if assignment_map:
        ckpt_name = assignment_map[var_name]
      else:
        ckpt_name = var_name
      generic_name = get_generic_name(var_name)
      ckpt_generic_name = get_generic_name(ckpt_name)
      graph_shard = graph_num_shards[generic_name]
      ckpt_shard = ckpt_num_shards[ckpt_generic_name]
      if ckpt_generic_name + ":0" in ckpt_embedding_group:
        ckpt_shard = 1
        ckpt_name = ckpt_generic_name + ":0"
        has_part_info = False
      else:
        has_part_info = True

      if ckpt_shard == graph_shard:  # no need to repartition
        restore_group[var_name] = [ckpt_name]
        restore_mode[var_name] = Constants.NORMAL_RESTORE
        graph_slots = graph_embedding_group[var_name]
        ckpt_slots = ckpt_embedding_group[ckpt_name]
        if not graph_slots:
          continue
        if len(graph_slots) == len(ckpt_slots):  # normal restore
          for graph_slot, ckpt_slot in zip(graph_slots, ckpt_slots):
            restore_group[graph_slot] = [ckpt_slot]
            restore_mode[graph_slot] = Constants.NORMAL_RESTORE
        elif (len(graph_slots) == 1 and len(ckpt_slots) > 1):  # need to merge
          slot = graph_slots[0]
          restore_group[slot] = ckpt_slots
          restore_mode[slot] = Constants.MERGE_RESTORE
        else:
          raise ValueError(
              "Unsupport restore mode from variable %s: [%s] to %s: [%s]" % (
                  ckpt_name,
                  ckpt_embedding_group[ckpt_name],
                  var_name,
                  graph_embedding_group[var_name],
              ))
      else:  # need to repartition
        pre, suf, _ = get_name_info(ckpt_name)
        restore_mode[var_name] = Constants.REPARTITION_RESTORE
        for i in range(ckpt_shard):
          if has_part_info:
            part_info = "/part_" + str(i)
          else:
            part_info = ""
          restore_name = pre + part_info + suf
          restore_group[var_name].append(restore_name)
          graph_slots = graph_embedding_group[var_name]
          ckpt_slots = ckpt_embedding_group[restore_name]
          if not graph_slots:
            continue
          if len(graph_slots) == len(ckpt_slots):
            for graph_slot, ckpt_slot in zip(graph_slots, ckpt_slots):
              restore_group[graph_slot].append(ckpt_slot)
              restore_mode[graph_slot] = Constants.REPARTITION_RESTORE
          elif len(graph_slots) == 1 and len(ckpt_slots) > 1:
            graph_slot = graph_slots[0]
            restore_group[graph_slot].append(ckpt_slots)
            restore_mode[graph_slot] = Constants.REPARTITION_MERGE_RESTORE
          else:
            raise ValueError(
                "Unsupport restore mode from variable %s: [%s] to %s: [%s]" % (
                    restore_name,
                    ckpt_embedding_group[restore_name],
                    var_name,
                    graph_embedding_group[var_name],
                ))
    return restore_group, restore_mode, graph_num_shards, ckpt_num_shards

  def get_ckpt_kv_size(self):
    """Get kv num shards dict and kv sum size size dict from ckpt."""
    if not self._ckpt_path:
      return {}

    ckpt_kv_size = collections.defaultdict(int)
    for embedding, slots in self._embedding_group.items():
      generic_name = get_generic_name(embedding)
      for var_name in [embedding] + slots:
        var_val_name = var_name + "-values"
        var_key_name = var_name + "-keys"
        key_type = self._var_dtype_map[var_key_name]
        val_type = self._var_dtype_map[var_val_name]

        shape = self._var_shape_map[var_val_name]
        var_part_size = 2.0 * (shape[0] * key_type.size + shape[0] * shape[1] *
                               val_type.size + shape[0] * 4)
        ckpt_kv_size[generic_name] += var_part_size
    return ckpt_kv_size
