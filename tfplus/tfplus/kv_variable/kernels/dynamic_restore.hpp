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

#ifndef TFPLUS_KV_VARIABLE_KERNELS_DYNAMIC_RESTORE_HPP_
#define TFPLUS_KV_VARIABLE_KERNELS_DYNAMIC_RESTORE_HPP_

#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <regex>
namespace tfplus {
using ::tensorflow::Status;

template <typename K, typename V>
Status KvVariable<K, V>::DeltaImport(OpKernelContext* ctx, int first_n,
                                     const Tensor& keys, const Tensor& values,
                                     const std::vector<Tensor>& others) {
  // Check key and value data type.
  TF_RETURN_IF_ERROR(CheckKvVariableDataTypes(keys.dtype(), values.dtype()));
  TF_RETURN_IF_ERROR(CheckKvVariableValueShape(values));

  // Check the size of vector others.
  if (others.size() < 6) {
    return ::tensorflow::errors::InvalidArgument(
        "KvVariable ", variable_name_.c_str(),
        ": number of other tensors for DeltaImport should >= 6 as "
        "expected");
  }

  // Check if the 1st dimension matches between keys and values.
  TensorShape value_shape = values.shape();
  if (keys.dim_size(0) != value_shape.dim_size(0)) {
    return ::tensorflow::errors::InvalidArgument(
        "KvVariable ", variable_name_.c_str(), ": number of keys (",
        keys.dim_size(0), ") and number of values (", value_shape.dim_size(0),
        ") do not match");
  }

  mutex_write_lock l(*mu());
  // DeltaImport must been initialized
  // If FullImport finished, all variable must been initialized
  // TF_RETURN_IF_ERROR(CheckInitializedInternal());

  // Stage 1: insert keys and values.
  const auto& keys_flat = keys.template flat<K>();
  const auto& values_flat = values.flat_outer_dims<V>();
  if (HasMemTable()) {
    for (int64 i = 0; i < keys_flat.size(); ++i) {
      auto& key = keys_flat(i);
      auto update_embedding = [this, &values_flat, &key, &i,
                               &values](EVContext<V>* context) {
        auto embedding_val =
            reinterpret_cast<const V*>(values.tensor_data().data()) +
            i * embedding_dim_;
        // insert: allocate and memcpy, update: memcpy
        context->UpdateValue(embedding_val, false, value_bytes_);
        context->Meta()->RemoveBlacklist();
        UpdateUnderThreshold(context);
      };
      // TODO(jinwang) Should avoid repeated inserts here.
      auto succeed = table_->UpdateWithFn(key, update_embedding);
      if (!succeed) {
        table_->InsertWithFn(key, update_embedding);
      }
    }
  } else {
    for (int64 i = 0; i < keys_flat.size(); ++i) {
      auto& key = keys_flat(i);
      auto update_embedding = [this, &values_flat, &key, &i,
                               &values](EVContext<V>* context) {
        auto val = reinterpret_cast<const V*>(values.tensor_data().data()) +
                   i * embedding_dim_;
        context->UpdateValue(val, false, value_bytes_);
      };
      table_->InsertWithFn(key, update_embedding);
    }
  }

  // Stage 2: blacklist.
  const auto& blacklist = others[1];
  if (blacklist.NumElements() > 0) {
    // Maybe an empty tensor.
    const auto& blacklist_flat = blacklist.template flat<K>();
    if (first_n > 3) {
      for (int64 i = 0; i < blacklist_flat.size(); ++i) {
        auto& key = blacklist_flat(i);
        auto l = GetScopedKeyLock(key, LockType::WRITE_LOCK);
        MarkBlacklistUnsafe(key, nullptr);
      }
    } else {
      // Inference load mode, just remove all keys in blacklist
      for (int64 i = 0; i < blacklist_flat.size(); ++i) {
        auto& blacklist_key = blacklist_flat(i);
        table_->DeleteKey(blacklist_key);
      }
    }
  }

  // Stage 3/4: Insert to frequency table.
  const auto& freq_keys = others[2];
  const auto& freq_values = others[3];
  if (freq_keys.NumElements() > 0) {
    // Maybe an empty tensor.
    const auto& freq_keys_flat = freq_keys.template flat<K>();
    const auto& freq_values_flat = freq_values.template flat<uint32_t>();

    CHECK(freq_keys_flat.size() == freq_values_flat.size());
    for (int64 i = 0; i < freq_keys_flat.size(); ++i) {
      auto& input_key = freq_keys_flat(i);
      auto& input_value = freq_values_flat(i);
      if (HasMemTable()) {
        auto find_func = [this, &freq_values, &i,
                          &input_value](EVContext<V>* context) {
          context->Meta()->UpdateFrequency(input_value);
        };
        table_->UpdateWithFn(input_key, find_func);
      }
    }
  }

  // Stage 5: Delete unused keys
  const auto& delete_keys = others[5];
  if (delete_keys.NumElements() > 0) {
    const auto& delete_keys_flat = delete_keys.template flat<K>();
    for (int64 i = 0; i < delete_keys_flat.size(); ++i) {
      auto& delete_key = delete_keys_flat(i);
      table_->DeleteKey(delete_key);
    }
  }

  if (others.size() <= 6 || others[6].template flat<bool>()(0)) {
    if (others.size() > 6) {
      LOG(INFO) << "There is no delta ckpt left to load, "
                << "restore is complete";
    }
    random_init_table_set_ = true;
  }
  return ::tensorflow::OkStatus();
}

template <typename K, typename V>
Status KvVariable<K, V>::ImportValues(OpKernelContext* ctx, const Tensor& keys,
                                      const Tensor& values,
                                      const std::vector<Tensor>& others) {
  // Check key and value data type.
  TF_RETURN_IF_ERROR(CheckKvVariableDataTypes(keys.dtype(), values.dtype()));
  TF_RETURN_IF_ERROR(CheckKvVariableValueShape(values));

  // Check the size of vector others.
  if (others.size() < IMPORT_OP_OTHERS_SIZE) {
    // return ::tensorflow::errors::InvalidArgument(
    //     "KvVariable ", variable_name_.c_str(),
    //     ": number of other tensors should be", IMPORT_OP_OTHERS_SIZE, " & ",
    //     FULL_OR_DELTA_IMPORT_OP_OTHERS_SIZE, " as expected, but we got ",
    //     others.size());
  }

  // Check if the 1st dimension matches between keys and values.
  TensorShape value_shape = values.shape();
  mutex_write_lock l(*mu());

  // Stage 1: import keys and values.
  table_->clear();
  const auto& keys_flat = keys.template flat<K>();

  // Cache values for hash table reference
  p_values_ = values;
  // bool values_found = (keys.dim_size(0) == value_shape.dim_size(0));
  // if (values_found) {
  for (int64 i = 0; i < keys_flat.size(); ++i) {
    // TODO(jianmu.scj): multi thread optimize
    auto insert_func = [this, i, &values](EVContext<V>* context) {
      V* value_ptr = const_cast<V*>(
          reinterpret_cast<const V*>(values.tensor_data().data()) +
          i * embedding_dim_);
      context->InitValue(value_ptr,
                         false);  // no need to allocate or copy here
    };
    table_->InsertWithFn(keys_flat(i), insert_func);
  }
  // }

  // Stage 2: initialization table (may be empty).
  const auto& init_table = others[0];
  if (init_table.NumElements() > 0) {
    random_init_table_ = DeepCopy(init_table);
  }

  // Stage 3: blacklist.
  const auto& blacklist = others[1];
  if (blacklist.NumElements() > 0) {
    // Maybe an empty tensor.
    const auto& blacklist_flat = blacklist.template flat<K>();
    for (int64 i = 0; i < blacklist_flat.size(); ++i) {
      auto& key = blacklist_flat(i);
      auto l = GetScopedKeyLock(key, LockType::WRITE_LOCK);
      MarkBlacklistUnsafe(key, nullptr);
    }
  }

  // Stage 4: frequency table.
  // Clear the old frequency table and insert the new one.
  const auto& freq_keys = others[2];
  const auto& freq_values = others[3];
  if (freq_keys.NumElements() > 0) {
    int start, stop;
    TF_RETURN_IF_ERROR(
        ctx->op_kernel().InputRange("freq_values", &start, &stop));
    if (stop != start + 1) {
      // return ::tensorflow::errors::InvalidArgument(
      //     "OpKernel used list-valued input name "
      //     "freq_values when single-valued input was "
      //     "expected");
    }
    bool freq_use_uint32 = ctx->input_dtype(start) == ::tensorflow::DT_UINT32;
    // Maybe an empty tensor.
    // size_t mem_data_count = 0;
    const auto& freq_keys_flat = freq_keys.template flat<K>();
    for (int64 i = 0; i < freq_keys_flat.size(); ++i) {
      auto& input_key = freq_keys_flat(i);
      auto find_func = [this, freq_use_uint32, &freq_values,
                        i](EVContext<V>* context) {
        uint32_t input_value = 0;
        if (freq_use_uint32) {
          input_value = freq_values.template flat<uint32_t>()(i);
        } else {
          input_value =
              static_cast<uint32_t>(freq_values.template flat<uint16>()(i));
        }
        context->Meta()->UpdateFrequency(input_value);
      };
      table_->UpdateWithFn(input_key, find_func);
    }
  }

  // If remain ckpt num is 0, set KvVariable as intialized.
  if (others.size() <= 6 || others[6].template flat<bool>()(0)) {
    if (others.size() > 6) {
      LOG(INFO) << "There is no delta ckpt left to load, "
                << "restore is complete";
    }
    random_init_table_set_ = true;
  }
  train_deltalist_.clear();  // not thread safe here
  prediction_deltalist_.clear();

  return ::tensorflow::OkStatus();
}

}  // namespace tfplus

#endif  // TFPLUS_KV_VARIABLE_KERNELS_DYNAMIC_RESTORE_HPP_
