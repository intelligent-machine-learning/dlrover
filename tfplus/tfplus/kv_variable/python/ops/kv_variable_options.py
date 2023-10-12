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
"""A class to define KvVariable options"""

from __future__ import absolute_import, division, print_function

from tfplus.kv_variable.kernels.hybrid_embedding import storage_config_pb2
from tfplus.kv_variable.kernels.hybrid_embedding.storage_config_pb2 import *  # pylint: disable=unused-wildcard-import

COMBINATION_MAPPING = {
    StorageCombination.MEM: [StorageType.MEM_STORAGE],
}


def parse_from_string(storage_option_string):
  kv_options = KvOptions()
  storage_opt = storage_config_pb2.StorageOption()
  storage_opt.ParseFromString(storage_option_string)
  kv_options.storage_option = storage_opt
  kv_options.storage_option_string = storage_option_string
  return kv_options


def get_remote_option(storage_path):
  return KvOptions(
      combination=StorageCombination.REMOTE,
      configs={
          StorageType.REMOTE_STORAGE:
          KvStorageConfig(storage_path=storage_path),
      },
  )


class KvOptions:
  """KvVariable options"""

  def __init__(self, combination=StorageCombination.MEM, configs=None):
    """
        Args:
        combination: pb enum, combination of storage.
          supported combinations are MEM, REMOTE, MEM_SSD, MEM_SSD_REMOTE.
        configs: a dict mapping storage type to a KvStorageConfig instance.
        """
    if configs is None:
      configs = {StorageType.MEM_STORAGE: KvStorageConfig()}

    storage_opt = storage_config_pb2.StorageOption()

    for storage_type in COMBINATION_MAPPING[combination]:
      if storage_type not in configs:
        raise ValueError("Storage type %s should in combination %s" %
                         (storage_type, combination))
      config = configs[storage_type]
      if not isinstance(config, KvStorageConfig):
        raise ValueError("Storage config should be instance of StorageConfig")
      config.trans_to_pb(storage_opt.configs[storage_type])

    storage_opt.combination = combination
    self.storage_option = storage_opt
    self.storage_option_string = storage_opt.SerializeToString()

  def has_storage(self, storage_type):
    return (storage_type
            in COMBINATION_MAPPING[self.storage_option.combination])

  def get_storage_config(self, storage_type):
    if storage_type in self.storage_option.configs:
      return self.storage_option.configs[storage_type]
    return None

  def has_path(self):
    return False

  def has_remote_storage(self):
    return False

  def __hash__(self):
    return hash(id(self))

  def __str__(self):
    return str(self.storage_option)

  def serialize_string(self):
    return self.storage_option_string


class KvStorageConfig:
  """KvVariable storage configuration"""

  def __init__(
      self,
      storage_path="",
      training_storage_size=-1,
      inference_storage_size=-1,
  ):
    """
    Args:
    storage_path: storage path
    training_storage_size: num of features in this storage during training,
      if not provided, then the auto size is used.
    inference_storage_size: num of features in this storage during inferencing,
      if not provided, then the auto size is used.
    """
    self.training_storage_size_ = training_storage_size
    self.inference_storage_size_ = inference_storage_size
    self.storage_path_ = storage_path

  @property
  def storage_path(self):
    return self.storage_path_

  @property
  def training_storage_size(self):
    return self.training_storage_size_

  @property
  def inference_storage_size(self):
    return self.inference_storage_size_

  def trans_to_pb(self, storage_option):
    storage_option.training_storage_size = self.training_storage_size_
    storage_option.inference_storage_size = self.inference_storage_size_
    storage_option.storage_path = str(self.storage_path_)
