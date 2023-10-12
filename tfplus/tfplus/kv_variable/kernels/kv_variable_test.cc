// Copyright 2023 The TFPlus Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tfplus/kv_variable/kernels/kv_variable.h"

#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <set>

#include "gtest/gtest.h"

namespace {
using namespace tfplus;      // NOLINT(build/namespaces)
using namespace tensorflow;  // NOLINT(build/namespaces)

// check the error status of the statement
#define TFPLUS_EXPECT_OK(statement) \
  EXPECT_EQ(::tensorflow::OkStatus(), (statement))

Status GenerateRandomRealTensor(float start, float end, Tensor* tensor) {
  std::default_random_engine generator;
  std::uniform_real_distribution<float> dist(start, end);

  auto flat_tensor = tensor->flat<float>();
  int64_t size = flat_tensor.size();
  for (int64_t i = 0; i < size; i++) {
    flat_tensor(i) = dist(generator);
  }

  return tensorflow::OkStatus();
}

template <typename V>
Status GenerateIndexTensor(int64_t start, Tensor* tensor) {
  auto flat_tensor = tensor->template flat<V>();
  int64_t size = flat_tensor.size();
  for (int64_t i = 0; i < size; i++) {
    flat_tensor(i) = start + i;
  }

  return tensorflow::OkStatus();
}

StorageOption GetStorageOption(StorageCombination sc) {
  StorageOption storage_options;
  storage_options.set_combination(sc);
  auto configs = storage_options.mutable_configs();
  StorageConfig mem_storage_config;
  mem_storage_config.set_storage_path("");
  mem_storage_config.set_training_storage_size(500);
  mem_storage_config.set_training_storage_size(1000);
  (*configs)[StorageType::MEM_STORAGE] = mem_storage_config;
  return storage_options;
}

TEST(KvVariableTest, InitRandomValues) {
  VLOG(0) << "?????";
  // create a KvVariable object
  const int64_t embedding_dim = 64;
  // StorageOption storage_options = GetStorageOption(StorageCombination::MEM);
  auto table =
      std::unique_ptr<KvVariableInterface>(new KvVariable<int64, float>(
          std::string("test_kv_variable1"), TensorShape({embedding_dim}), 0));
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  VLOG(0) << "?????";
  // create a relatively small initialization table
  int64_t num_rows = 1024;
  TensorShape shape({num_rows, embedding_dim});
  Tensor random_init(DataTypeToEnum<float>::v(), shape);

  // using uniform distribution function to initialize the table
  std::default_random_engine generator;
  std::uniform_real_distribution<float> dist(0.0, 1.0);
  auto flat_tensor = random_init.flat<float>();
  int64_t size = flat_tensor.size();
  for (int64_t i = 0; i < size; i++) {
    flat_tensor(i) = dist(generator);
  }

  // fill the initialization table to KvVariable
  TFPLUS_EXPECT_OK(table->InitRandomValues(random_init));

  // get the initialization table
  Tensor other;
  TFPLUS_EXPECT_OK(table->GetInitTable(&other));
}

TEST(KvVariableTest, Find) {
  const int64_t embedding_dim = 64;
  StorageOption storage_options = GetStorageOption(StorageCombination::MEM);
  auto table =
      std::unique_ptr<KvVariableInterface>(new KvVariable<int64, float>(
          std::string("test_kv_variable2"), TensorShape({embedding_dim}), 0,
          storage_options));
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  // create a relatively small initialization table
  int64_t num_rows = 1024;
  TensorShape shape({num_rows, embedding_dim});
  Tensor random_init(DataTypeToEnum<float>::v(), shape);
  TFPLUS_EXPECT_OK(GenerateRandomRealTensor(0.1, 1.0, &random_init));
  TFPLUS_EXPECT_OK(table->InitRandomValues(random_init));

  TensorShape keys_shape({1});
  Tensor keys(DataTypeToEnum<int64>::v(), keys_shape);
  auto keys_flat = keys.flat<int64>();
  keys_flat(0) = static_cast<int64>(1000);
  Tensor values(DataTypeToEnum<float>::v(),
                TensorShape({1, embedding_dim}));  // [1, embedding_dim]
  TFPLUS_EXPECT_OK(table->FindOrZeros(nullptr, keys, &values));
  auto flat_values = values.flat<float>();
  for (int64_t i = 0; i < flat_values.size(); i++) {
    if (flat_values(i) != 0) {
      TFPLUS_EXPECT_OK(errors::InvalidArgument("FindOrZeros failed: %f != %f",
                                               flat_values(i), 0));
      break;
    }
  }

  // Now FindOrInsert, each value should in range [0.1, 1.0]
  TFPLUS_EXPECT_OK(table->FindOrInsert(nullptr, keys, &values));
  flat_values = values.flat<float>();
  for (int64_t i = 0; i < flat_values.size(); i++) {
    if (flat_values(i) <= 0 || flat_values(i) > 1.0) {
      TFPLUS_EXPECT_OK(errors::InvalidArgument(
          "FindOrInsert failed: %f not in range [0.1, 1.0]", flat_values(i)));
      break;
    }
  }

  // TODO(tongsuo):  need add test for filter_out
}

TEST(KvVariableTest, InsertOrUpdate) {
  const int embedding_dim = 64;
  // create and initialize the KvVariable
  StorageOption storage_options = GetStorageOption(StorageCombination::MEM);
  auto table =
      std::unique_ptr<KvVariableInterface>(new KvVariable<int64, float>(
          std::string("test_kv_variable3"), TensorShape({embedding_dim}), 0,
          storage_options));
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  // create a relatively small initialization table
  int64_t num_rows = 1024;
  TensorShape shape({num_rows, embedding_dim});
  Tensor random_init(DataTypeToEnum<float>::v(), shape);

  // using uniform distribution function to initialize the table
  std::default_random_engine generator;
  std::uniform_real_distribution<float> dist(0.0, 1.0);
  auto flat_tensor = random_init.flat<float>();
  for (int64_t i = 0; i < flat_tensor.size(); ++i) {
    flat_tensor(i) = dist(generator);
  }

  // fill the initialization table to KvVariable
  TFPLUS_EXPECT_OK(table->InitRandomValues(random_init));

  // regular cases
  int num_keys = 10;
  Tensor keys(DataTypeToEnum<int64>::v(), TensorShape({num_keys}));
  Tensor values(DataTypeToEnum<float>::v(),
                TensorShape({num_keys, embedding_dim}));
  auto keys_flat = keys.flat<int64>();
  auto values_flat = values.flat_outer_dims<float>();
  for (int i = 0; i < num_keys; ++i) {
    keys_flat(i) = i;
    for (int j = 0; j < embedding_dim; ++j) {
      values_flat(i, j) = dist(generator);
    }
  }
  TFPLUS_EXPECT_OK(table->InsertOrUpdate(nullptr, keys, values));

  // check the number of key-value pairs in KvVariable
  int64_t table_size = static_cast<int64>(table->size());
  int64_t expected_size = keys_flat.size();
  if (table_size != expected_size) {
    TFPLUS_EXPECT_OK(
        errors::InvalidArgument("After inserting, the number of elements in "
                                "KvVariable does not match: ",
                                table_size, " != ", expected_size));
  }

  // apply filter out
  Tensor filterout(DataTypeToEnum<bool>::v(), TensorShape({keys_flat.size()}));
  auto filterout_flat = filterout.flat<bool>();
  for (int64_t i = 0; i < keys_flat.size(); ++i) {
    filterout_flat(i) = i % 2 ? true : false;
  }
  TFPLUS_EXPECT_OK(table->InsertOrUpdate(nullptr, keys, values, &filterout));
}

TEST(KvVariableTest, IsInitialized) {
  StorageOption storage_options = GetStorageOption(StorageCombination::MEM);
  auto table =
      std::unique_ptr<KvVariableInterface>(new KvVariable<int64, float>(
          std::string("test_kv_variable4"), TensorShape({1000, 64}), 0,
          storage_options));
  std::this_thread::sleep_for(std::chrono::milliseconds(5000));
  EXPECT_FALSE(table->IsInitialized());

  // TODO(tongsuo): after call InitRandomValues, and then EXPECT_TRUE
}

TEST(KvVariableTest, ImportValues) {
  // TODO(chengyi): tested only in Python because of the lack of TF context in
  // C++
}

TEST(KvVariableTest, ExportValues) {
  // TODO(chengyi): tested only in Python because of the lack of TF context in
  // C++
}

TEST(KvVariableTest, ScatterUpdate) {
  const int embedding_dim = 64;
  StorageOption storage_options = GetStorageOption(StorageCombination::MEM);
  // Create and initialize the KvVariable.
  auto table =
      std::unique_ptr<KvVariableInterface>(new KvVariable<int64, float>(
          std::string("test_kv_variable5"), TensorShape({embedding_dim}), 0,
          storage_options));
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  // Create a relatively small initialization table.
  int64_t num_rows = 1024;
  TensorShape shape({num_rows, embedding_dim});
  Tensor random_init(DataTypeToEnum<float>::v(), shape);

  // Using uniform distribution function to initialize the table.
  std::default_random_engine generator;
  std::uniform_real_distribution<float> dist(0.0, 1.0);
  auto flat_tensor = random_init.flat<float>();
  for (int64_t i = 0; i < flat_tensor.size(); ++i) {
    flat_tensor(i) = dist(generator);
  }

  // Fill the initialization table to KvVariable.
  TFPLUS_EXPECT_OK(table->InitRandomValues(random_init));

  // Generate data for scatter update.
  int num_keys = 10;
  Tensor keys(DataTypeToEnum<int64>::v(), TensorShape({num_keys}));
  Tensor values(DataTypeToEnum<float>::v(),
                TensorShape({num_keys, embedding_dim}));
  Tensor updates(DataTypeToEnum<float>::v(),
                 TensorShape({num_keys, embedding_dim}));
  Tensor updates_2(DataTypeToEnum<float>::v(),
                   TensorShape({num_keys, embedding_dim}));
  auto keys_flat = keys.flat<int64>();
  auto updates_flat = updates.flat_outer_dims<float>();
  auto updates_2_flat = updates_2.flat_outer_dims<float>();
  for (int i = 0; i < num_keys; ++i) {
    keys_flat(i) = i;
    for (int j = 0; j < embedding_dim; ++j) {
      updates_flat(i, j) = 1.0;
      updates_2_flat(i, j) = 2.0;
    }
  }

  // Perform scatter updates.
  OpKernelContext* ctx = nullptr;
  TFPLUS_EXPECT_OK(
      table->ScatterUpdate(ctx, keys, updates, SCATTER_UPDATE_ASSIGN));
  TFPLUS_EXPECT_OK(table->FindOrZeros(ctx, keys, &values));
  auto values_flat = values.flat_outer_dims<float>();
  for (int i = 0; i < num_keys; i++) {
    for (int j = 0; j < embedding_dim; ++j) {
      if (values_flat(i, j) != 1.0) {
        TFPLUS_EXPECT_OK(errors::InvalidArgument(
            "Error in SCATTER_UPDATE_ASSIGN: ", values_flat(i, j)));
      }
    }
  }

  TFPLUS_EXPECT_OK(
      table->ScatterUpdate(ctx, keys, updates, SCATTER_UPDATE_ADD));
  TFPLUS_EXPECT_OK(table->FindOrZeros(ctx, keys, &values));
  for (int i = 0; i < num_keys; i++) {
    for (int j = 0; j < embedding_dim; ++j) {
      if (values_flat(i, j) != 2.0) {
        TFPLUS_EXPECT_OK(errors::InvalidArgument("Error in SCATTER_UPDATE_ADD:",
                                                 values_flat(i, j)));
      }
    }
  }

  TFPLUS_EXPECT_OK(
      table->ScatterUpdate(ctx, keys, updates, SCATTER_UPDATE_SUB));
  TFPLUS_EXPECT_OK(table->FindOrZeros(ctx, keys, &values));
  for (int i = 0; i < num_keys; i++) {
    for (int j = 0; j < embedding_dim; ++j) {
      if (values_flat(i, j) != 1.0) {
        TFPLUS_EXPECT_OK(errors::InvalidArgument("Error in SCATTER_UPDATE_SUB:",
                                                 values_flat(i, j)));
      }
    }
  }

  TFPLUS_EXPECT_OK(
      table->ScatterUpdate(ctx, keys, updates_2, SCATTER_UPDATE_MUL));
  TFPLUS_EXPECT_OK(table->FindOrZeros(ctx, keys, &values));
  for (int i = 0; i < num_keys; i++) {
    for (int j = 0; j < embedding_dim; ++j) {
      if (values_flat(i, j) != 2.0) {
        TFPLUS_EXPECT_OK(errors::InvalidArgument("Error in SCATTER_UPDATE_MUL:",
                                                 values_flat(i, j)));
      }
    }
  }

  TFPLUS_EXPECT_OK(
      table->ScatterUpdate(ctx, keys, updates_2, SCATTER_UPDATE_DIV));
  TFPLUS_EXPECT_OK(table->FindOrZeros(ctx, keys, &values));
  for (int i = 0; i < num_keys; i++) {
    for (int j = 0; j < embedding_dim; ++j) {
      if (values_flat(i, j) != 1.0) {
        TFPLUS_EXPECT_OK(errors::InvalidArgument("Error in SCATTER_UPDATE_DIV:",
                                                 values_flat(i, j)));
      }
    }
  }

  TFPLUS_EXPECT_OK(
      table->ScatterUpdate(ctx, keys, updates_2, SCATTER_UPDATE_MIN));
  TFPLUS_EXPECT_OK(table->FindOrZeros(ctx, keys, &values));
  for (int i = 0; i < num_keys; i++) {
    for (int j = 0; j < embedding_dim; ++j) {
      if (values_flat(i, j) != 1.0) {
        TFPLUS_EXPECT_OK(errors::InvalidArgument("Error in SCATTER_UPDATE_MIN:",
                                                 values_flat(i, j)));
      }
    }
  }

  TFPLUS_EXPECT_OK(
      table->ScatterUpdate(ctx, keys, updates_2, SCATTER_UPDATE_MAX));
  TFPLUS_EXPECT_OK(table->FindOrZeros(ctx, keys, &values));
  for (int i = 0; i < num_keys; i++) {
    for (int j = 0; j < embedding_dim; ++j) {
      if (values_flat(i, j) != 2.0) {
        TFPLUS_EXPECT_OK(errors::InvalidArgument("Error in SCATTER_UPDATE_MAX:",
                                                 values_flat(i, j)));
      }
    }
  }
}

TEST(KvVariableTest, TestKvStat) {
  struct kv_stat {
    // (low bits)total update frequency for current key.
    uint16 frequency;
    // (high bits)
    // save unix time by days unit, instead of ms.
    // then uint16 is enough. i.e 1581427427089 ms to 18303 day.
    // 65536 as limit is far enough right now.
    uint16 last_update_time_in_days;
  };

  uint32_t source = 0;
  auto ptr = reinterpret_cast<kv_stat*>(&source);
  auto ptr16 = reinterpret_cast<uint16_t*>(&source);
  ptr->frequency = 65535;
  ptr->last_update_time_in_days = 65534;
  EXPECT_EQ(GetUint16FromUint32(source, true), 65535);
  EXPECT_EQ(GetUint16FromUint32(source, false), 65534);
  EXPECT_EQ(ptr16[0], 65535);
  EXPECT_EQ(ptr16[1], 65534);

  uint32_t target = MakeUint32FromUint16(65534, 65535);
  EXPECT_EQ(source, target);
}

TEST(KvVariableTest, Delete) {
  const int embedding_dim = 64;
  StorageOption storage_options = GetStorageOption(StorageCombination::MEM);

  // Create and initialize the KvVariable.
  auto table =
    std::unique_ptr<KvVariableInterface>(new KvVariable<int64, float>(
      std::string("test_kv_variable6"), TensorShape({embedding_dim}), 0,
          storage_options));
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  // Create a relatively small initialization table.
  int64_t num_rows = 1024;
  TensorShape shape({num_rows, embedding_dim});
  Tensor random_init(DataTypeToEnum<float>::v(), shape);

  // Using uniform distribution function to initialize the table.
  std::default_random_engine generator;
  std::uniform_real_distribution<float> dist(0.0, 1.0);
  auto flat_tensor = random_init.flat<float>();
  for (int64_t i = 0; i < flat_tensor.size(); ++i) {
    flat_tensor(i) = dist(generator);
  }

  // Fill the initialization table to KvVariable.
  TFPLUS_EXPECT_OK(table->InitRandomValues(random_init));

  // Generate data for scatter update and delete.
  int num_keys = 10;
  int num_delete_keys = num_keys / 2;
  Tensor keys(DataTypeToEnum<int64>::v(), TensorShape({num_keys}));
  Tensor delete_keys(DataTypeToEnum<int64>::v(),
                     TensorShape({num_delete_keys}));
  Tensor values(DataTypeToEnum<float>::v(),
                TensorShape({num_keys, embedding_dim}));
  Tensor updates(DataTypeToEnum<float>::v(),
                 TensorShape({num_keys, embedding_dim}));
  auto keys_flat = keys.flat<int64>();
  auto delete_keys_flat = delete_keys.flat<int64>();
  auto updates_flat = updates.flat_outer_dims<float>();
  for (int i = 0; i < num_keys; ++i) {
    keys_flat(i) = i;
    if (i % 2 == 0) {
      delete_keys_flat(i / 2) = i;
    }
    for (int j = 0; j < embedding_dim; ++j) {
      updates_flat(i, j) = 1.0;
    }
  }

  // Perform scatter updates.
  OpKernelContext* ctx = nullptr;
  TFPLUS_EXPECT_OK(
    table->ScatterUpdate(ctx, keys, updates, SCATTER_UPDATE_ASSIGN));

  // Perform delete operation.
  TFPLUS_EXPECT_OK(table->Delete(delete_keys));

  // Check the size of the table after deleting.
  int table_size = static_cast<int>(table->size());
  int expected_size = num_keys - num_delete_keys;
  if (table_size != expected_size) {
    TFPLUS_EXPECT_OK(
      errors::InvalidArgument("After deleting, the number of elements in "
                              "KvVariable does not match: ",
                              table_size, " != ", expected_size));
  }
}


}  // namespace

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
