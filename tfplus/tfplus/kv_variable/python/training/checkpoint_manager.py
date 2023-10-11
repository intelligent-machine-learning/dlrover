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

# pylint: disable=invalid-name
"""Save and restore variables."""
from __future__ import absolute_import, division, print_function

import copy
import os
import time

from google.protobuf import text_format
from tensorflow.python.framework import errors
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import \
    checkpoint_management as tf_checkpoint_management

from tfplus.kv_variable.python.training.checkpoint_state_extend_pb2 import (
    CheckpointStateExt,
)


class CheckpointStateManager(object):  # pylint: disable=useless-object-inheritance
  """
    Manage metadata for full and delta versions checkpoint.
    It keeps a CheckpointStateExt and serializes value
      to checkpoint_dir/checkpoint_ext file.
    """

  def __init__(self, checkpoint_dir, latest_filename=None):
    self._checkpoint_dir = checkpoint_dir
    self._ckpt_state_ext = self.load_checkpoint_state_ext(
        latest_filename=latest_filename)

  def delete_specified_history_version_in_ckpt_state(self, ckpt):
    if ckpt in self._ckpt_state_ext.history_versions:
      del self._ckpt_state_ext.history_versions[ckpt]
      self.update_checkpiont_state_ext()

  # pylint: disable=missing-docstring
  def update_latest_full_checkpoint(self, full_checkpoint_path):
    prev_current_full_path = None
    prev_all_delta_paths = None
    if self._ckpt_state_ext is None:
      self._ckpt_state_ext = CheckpointStateExt()
    else:
      prev_current_full_path = (
          self._ckpt_state_ext.current_full_checkpoint_path)
      prev_all_delta_paths = (
          self._ckpt_state_ext.all_valid_delta_checkpoint_paths)
    self._ckpt_state_ext.current_full_checkpoint_path = (full_checkpoint_path)
    self._ckpt_state_ext.current_full_checkpoint_timestamp = time.time()

    if prev_current_full_path is not None:
      self._ckpt_state_ext.history_versions[
          prev_current_full_path].deltas.extend(prev_all_delta_paths)
    del self._ckpt_state_ext.all_valid_delta_checkpoint_paths[:]
    self.update_checkpiont_state_ext()

  # pylint: disable=missing-docstring
  def add_delta_checkpoint(self, delta_checkpoint_path):
    if self._ckpt_state_ext is None:
      raise ValueError(
          "current ckpt_ext_state is None, can not add delta checkpoint %s" %
          delta_checkpoint_path)

    if self._ckpt_state_ext.current_full_checkpoint_path is None:
      raise ValueError("current_full_checkpoint_path is None,\
          can not add delta checkpoint %s" % delta_checkpoint_path)
    self._ckpt_state_ext.all_valid_delta_checkpoint_paths.append(
        delta_checkpoint_path)
    self.update_checkpiont_state_ext()

  def get_checkpoint_ext_filename(self, latest_filename=None):
    """Returns a filename for storing the CheckpointState.

        Args:
          save_dir: The directory for saving and restoring checkpoints.
          latest_filename: Name of the file in 'save_dir' that is used
            to store the CheckpointState.

        Returns:
          The path of the file that contains the CheckpointState proto.
        """
    if latest_filename is None:
      latest_filename = "checkpoint_ext"
    return os.path.join(self._checkpoint_dir, latest_filename)

  def load_checkpoint_state_ext(self, latest_filename=None):
    """
        Deserialized CheckpointStateExt from checkpoint_dir/checkpoint_ext file.
        """
    ckpt = None
    coord_checkpoint_filename = self.get_checkpoint_ext_filename(
        latest_filename)
    f = None
    checkpoint_dir = self._checkpoint_dir
    try:
      if file_io.file_exists(coord_checkpoint_filename):
        file_content = file_io.read_file_to_string(coord_checkpoint_filename)
        ckpt = CheckpointStateExt()
        text_format.Merge(file_content, ckpt)
        if not ckpt.current_full_checkpoint_path:
          raise ValueError("Invalid ext checkpoint state loaded from " +
                           checkpoint_dir)

        def fill_if_non_abs(checkpoint_dir, p):
          return os.path.join(checkpoint_dir, p)

        process_non_abs_path(ckpt, fill_if_non_abs, checkpoint_dir)
    except (errors.OpError, text_format.ParseError) as e:
      # It's ok if the file cannot be read
      logging.warning("%s: %s", type(e).__name__, e)
      logging.warning("%s: Checkpoint ignored", coord_checkpoint_filename)
    finally:
      if f:
        f.close()
    return ckpt

  def get_all_relative_delta_checkpoints(self, full_checkpoint_path):
    """
        Get all delta versions for full_checkpoint_path.

        Args:
          full_checkpoint_path: path for full checkpoint

        Returns:
          All delta versions for full_checkpoint_path
        """
    if self._ckpt_state_ext is None:
      return None
    if (full_checkpoint_path ==
        self._ckpt_state_ext.current_full_checkpoint_path):
      return self._ckpt_state_ext.all_valid_delta_checkpoint_paths

    delta_list = self._ckpt_state_ext.history_versions.get(
        full_checkpoint_path, None)
    if delta_list is None:
      return None
    return delta_list.deltas

  def get_current_full_checkpoint(self):
    if self._ckpt_state_ext is None:
      return None
    return self._ckpt_state_ext.current_full_checkpoint_path

  def get_history_full_checkpoints(self):
    if self._ckpt_state_ext is None:
      return []
    return list(self._ckpt_state_ext.history_versions)

  @property
  def checkpoint_dir(self):
    return self._checkpoint_dir

  def update_checkpiont_state_ext(self):
    ckpt_ext_file = self.get_checkpoint_ext_filename()

    def basename_if_non_abs(checkpoint_dir, p):  # pylint: disable=unused-argument
      return os.path.basename(p)

    ckpt_state_ext = copy.deepcopy(self._ckpt_state_ext)
    process_non_abs_path(ckpt_state_ext, basename_if_non_abs, "")
    file_io.atomic_write_string_to_file(
        ckpt_ext_file, text_format.MessageToString(ckpt_state_ext))

  @property
  def latest_checkpoint(self):
    full_ckpt_path = self.get_current_full_checkpoint()
    delta_list = self.get_all_relative_delta_checkpoints(full_ckpt_path)
    if delta_list:
      return delta_list[-1]
    if full_ckpt_path:
      return full_ckpt_path
    model_ckpt = os.path.join(self._checkpoint_dir, "model.ckpt")
    logging.error("Couldn't match files for checkpoint %s", model_ckpt)
    return None

  @property
  def checkpoint_state_ext(self):
    return self._ckpt_state_ext

  def remove_old_checkpoint(self,
                            removed_full_ckpt_path,
                            meta_graph_suffix="meta"):
    """
        Remove specific full checkpoint and all relative deltas files.

        Args:
          removed_full_ckpt_path: path for full checkpoint to remove
          meta_graph_suffix: Suffix for `MetaGraphDef` file. Defaults to 'meta'.
        """
    if self._ckpt_state_ext is None:
      return

    if (removed_full_ckpt_path ==
        self._ckpt_state_ext.current_full_checkpoint_path):
      raise ValueError("Can not remove %s which is the only verion checkpoint" %
                       removed_full_ckpt_path)
    to_be_removed = [removed_full_ckpt_path]
    delta_list = self._ckpt_state_ext.history_versions.get(
        removed_full_ckpt_path, None)
    if delta_list and delta_list.deltas:
      to_be_removed.extend(delta_list.deltas)
    del self._ckpt_state_ext.history_versions[removed_full_ckpt_path]
    for ckpt in to_be_removed:
      tf_checkpoint_management.remove_checkpoint(
          ckpt, meta_graph_suffix=meta_graph_suffix)
      self.remove_snapshot_directory(ckpt)
    self.update_checkpiont_state_ext()
    logging.info("Delete old full ckpt %s and all relative delta versions." %
                 removed_full_ckpt_path)

  def remove_snapshot_directory(self, ckpt):
    filespec = ckpt + ".snapshot"
    for pathname in file_io.get_matching_files(filespec):
      file_io.delete_recursively(pathname)
      logging.warning("Delete snapshot directory: %s" % pathname)


def process_non_abs_path(ckpt, func, checkpoint_dir):
  """
    Process all non abs path in ckpt by applying func(checkpoint_dir, path).
    When save_ext pass a save_path is not an abs path, such as ./checkpoint_dir,
      we should remove the prefix path, only record basename of variable file
      to checkpoint_ext file.
    When we load checkpoint_ext file, we join all the paths with checkpoint_dir
      if the path stored in the checkpoint_ext file is not abs path.
    The rule is follow the tf.train.Saver which records meta in checkpoint file.

    Example:
      If we call SaverExt.save_ext('/tmp/tmpwhm295lz'), all paths in
      checkpoint_ext are abs path.
    ```text
      current_full_checkpoint_path: "/tmp/tmpwhm295lz/my-model-ext.ckpt-17"
      current_full_checkpoint_timestamp: 1582718843.3135512
      all_valid_delta_checkpoint_paths: "/tmp/tmpwhm295lz/my-model-ext.ckpt-18"
      all_valid_delta_checkpoint_paths: "/tmp/tmpwhm295lz/my-model-ext.ckpt-19"
      all_valid_delta_checkpoint_paths: "/tmp/tmpwhm295lz/my-model-ext.ckpt-20"
      all_valid_delta_checkpoint_paths: "/tmp/tmpwhm295lz/my-model-ext.ckpt-21"
      all_valid_delta_checkpoint_paths: "/tmp/tmpwhm295lz/my-model-ext.ckpt-22"
      history_versions {
        key: "/tmp/tmpwhm295lz/my-model-ext.ckpt-11"
        value {
          deltas: "/tmp/tmpwhm295lz/my-model-ext.ckpt-12"
          deltas: "/tmp/tmpwhm295lz/my-model-ext.ckpt-13"
          deltas: "/tmp/tmpwhm295lz/my-model-ext.ckpt-14"
          deltas: "/tmp/tmpwhm295lz/my-model-ext.ckpt-15"
          deltas: "/tmp/tmpwhm295lz/my-model-ext.ckpt-16"
        }
      }
    ```
     If we call SaverExt.save_ext('./my_ckpt'), all paths in
      checkpoint_ext are basename path.
    ```text
      current_full_checkpoint_path: "my-model-ext.ckpt-17"
      current_full_checkpoint_timestamp: 1582719093.7972286
      all_valid_delta_checkpoint_paths: "my-model-ext.ckpt-18"
      all_valid_delta_checkpoint_paths: "my-model-ext.ckpt-19"
      all_valid_delta_checkpoint_paths: "my-model-ext.ckpt-20"
      all_valid_delta_checkpoint_paths: "my-model-ext.ckpt-21"
      all_valid_delta_checkpoint_paths: "my-model-ext.ckpt-22"
      history_versions {
        key: "my-model-ext.ckpt-11"
        value {
          deltas: "my-model-ext.ckpt-12"
          deltas: "my-model-ext.ckpt-13"
          deltas: "my-model-ext.ckpt-14"
          deltas: "my-model-ext.ckpt-15"
          deltas: "my-model-ext.ckpt-16"
        }
      }
    ```
    """
  if ckpt is None:
    return

  if not os.path.isabs(ckpt.current_full_checkpoint_path):
    ckpt.current_full_checkpoint_path = func(checkpoint_dir,
                                             ckpt.current_full_checkpoint_path)
  for i, _ in enumerate(ckpt.all_valid_delta_checkpoint_paths):
    p = ckpt.all_valid_delta_checkpoint_paths[i]
    if not os.path.isabs(p):
      ckpt.all_valid_delta_checkpoint_paths[i] = func(checkpoint_dir, p)
  history_with_full_path = {}
  replace_full = []
  for full in ckpt.history_versions:
    deltas_list = ckpt.history_versions[full]
    for i, _ in enumerate(deltas_list.deltas):
      p = deltas_list.deltas[i]
      if not os.path.isabs(p):
        deltas_list.deltas[i] = func(checkpoint_dir, p)
    if not os.path.isabs(full):
      new_path = func(checkpoint_dir, full)
      history_with_full_path[new_path] = [  # pylint: disable=unnecessary-comprehension
          delta for delta in deltas_list.deltas
      ]
      replace_full.append(full)
  for full in replace_full:
    del ckpt.history_versions[full]

  for full, deltas_list in history_with_full_path.items():
    ckpt.history_versions[full].deltas.extend(deltas_list)


def latest_checkpoint(checkpoint_dir, latest_filename=None):
  """Finds the filename of latest saved checkpoint file.

    Args:
      checkpoint_dir: Directory where the variables were saved.
      latest_filename: Optional name for the protocol buffer file that
        contains the list of most recent checkpoint filenames.
        See the corresponding argument to `Saver.save()`.

    Returns:
      The full path to the latest checkpoint or `None` if
      no checkpoint was found.
    """
  # Pick the latest checkpoint based on checkpoint state.
  ckpt_manager = CheckpointStateManager(checkpoint_dir,
                                        latest_filename=latest_filename)
  return ckpt_manager.latest_checkpoint
