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

#ifndef TFPLUS_KV_VARIABLE_KERNELS_DYNAMIC_SAVE_HPP_
#define TFPLUS_KV_VARIABLE_KERNELS_DYNAMIC_SAVE_HPP_

#include <map>
#include <unordered_set>
#include <vector>
#include <memory>
#include <string>

namespace tfplus {

template <typename K, typename V>
Status KvVariable<K, V>::FullExport(OpKernelContext* ctx, int first_n,
                                    bool no_copy, const string& tensor_key,
                                    BundleWriter* writer, bool enable_cutoff,
                                    float cutoff_value, bool freq_use_uint32) {
  CHECK(ctx != nullptr);
  TF_RETURN_IF_ERROR(ExportValues(ctx, first_n, enable_cutoff, cutoff_value));

  // Set need_full_import true
  Tensor* need_full_import;
  TF_RETURN_IF_ERROR(ctx->allocate_output(
      "need_full_import", TensorShape({1}), &need_full_import));
  auto&& need_full_import_flat = need_full_import->template flat<bool>();
  need_full_import_flat(0) = true;

  // Empty delete_keys for full export.
  Tensor* delete_keys;
  TF_RETURN_IF_ERROR(
      ctx->allocate_output("delete_keys", TensorShape({0}), &delete_keys));
  return ::tensorflow::OkStatus();
}

template <typename K, typename V>
Status KvVariable<K, V>::ExportValues(OpKernelContext* ctx, int first_n,
                                      bool enable_cutoff, float cutoff_value,
                                      void* table_handler) {
  CHECK(ctx != nullptr);

  // The order of locks: mu_ -> train_deltalist_mu_ -> table_.locks
  mutex_write_lock l(*mu());
  TableManager<K, V>* handler =
      table_handler ? reinterpret_cast<TableManager<K, V>*>(table_handler)
                    : table_;
  TF_RETURN_IF_ERROR(CheckInitializedInternal());
  // Calcuate the number of valid key-value pairs, blacklist, frequency.
  int64_t num_rows = 0;
  int64_t blacklist_nums = 0;
  int64_t freq_nums = first_n > FIRST_N_EXPORT_FREQUENCY ? handler->size() : 0;
  int64_t key_row = 0;
  int64_t blacklist_row = 0;
  int64_t freq_row = 0;
  size_t value_rows = 0;
  {
    typename TableManager<K, V>::ScopedLock table_scoped_lock(handler);
    RefreshAllUnderThresholds(enable_cutoff, cutoff_value, handler);
  }
  auto counting_iter = [this, &blacklist_nums, &num_rows, &enable_cutoff,
                        &cutoff_value,
                        &first_n](const K& key, const EVContext<V>* context) {
    auto v = context->Meta();
    if (v->InBlacklist()) {
      blacklist_nums++;
    } else if ((first_n <= FIRST_N_EXPORT_BLACK_LIST ||
                !HasLowFrequency(v->GetFrequency())) &&
               !v->IsUnderThreshold()) {
      num_rows++;
    }
  };
  handler->ForEach(counting_iter);
  // Allocate output tensors for key-value pairs.
  Tensor* keys;
  Tensor* values;
  TensorShape value_tensor_shape = value_shape_;
  value_tensor_shape.InsertDim(0, num_rows);
  TF_RETURN_IF_ERROR(
      ctx->allocate_output("keys", TensorShape({num_rows}), &keys));
  TF_RETURN_IF_ERROR(
      ctx->allocate_output("values", value_tensor_shape, &values));
  auto&& keys_flat = keys->template flat<K>();
  auto&& values_flat = values->flat_outer_dims<V>();

  // Stage 2: output initialization table (e.g.for inference only).
  // Attribute first_n controls how much information will be exported.
  Tensor* init_table = nullptr;
  Tensor* blacklist = nullptr;
  Tensor* freq_keys = nullptr;
  Tensor* freq_values = nullptr;
  bool freq_use_uint32 = true;

  if (first_n > FIRST_N_EXPORT_KEY_AND_VALUES) {
    if (first_n > FIRST_N_EXPORT_BLACK_LIST) {
      TF_RETURN_IF_ERROR(ctx->allocate_output(
          "init_table", random_init_table_.shape(), &init_table));
      ExportInitTable(init_table);
    } else {
      // Export an empty initialization table.
      TensorShape init_table_shape = value_shape_;
      init_table_shape.InsertDim(0, 0);
      TF_RETURN_IF_ERROR(
          ctx->allocate_output("init_table", init_table_shape, &init_table));
    }

    if (first_n <= FIRST_N_EXPORT_BLACK_LIST) {
      blacklist_nums = 0;
    }
    TF_RETURN_IF_ERROR(ctx->allocate_output(
        "blacklist", TensorShape({blacklist_nums}), &blacklist));

    if (first_n <= FIRST_N_EXPORT_FREQUENCY) {
      freq_nums = 0;
    }
    TF_RETURN_IF_ERROR(ctx->allocate_output(
        "freq_keys", TensorShape({freq_nums}), &freq_keys));
    TF_RETURN_IF_ERROR(ctx->allocate_output(
        "freq_values", TensorShape({freq_nums}), &freq_values));
    int start, stop;
    TF_RETURN_IF_ERROR(
        ctx->op_kernel().OutputRange("freq_values", &start, &stop));
    if (stop != start + 1) {
      return ::tensorflow::errors::InvalidArgument(
          "OpKernel used list-valued output name "
          "freq_values when single-valued output was "
          "expected");
    }
    freq_use_uint32 =
        ctx->expected_output_dtype(start) == ::tensorflow::DT_UINT32;
  }
  auto do_export = [this, &key_row, &num_rows, &keys_flat, &values_flat,
                    &blacklist, &blacklist_nums, &blacklist_row,
                    &freq_use_uint32, &freq_nums, &freq_row, &freq_keys,
                    &freq_values, &enable_cutoff, &first_n,
                    &cutoff_value](const K& key, const EVContext<V>* context) {
    auto v = context->Meta();
    if (v->InBlacklist() && blacklist_nums > 0 &&
        blacklist_row < blacklist_nums && blacklist != nullptr) {
      // Output blacklist if needed
      blacklist->template flat<K>()(blacklist_row++) = key;
    } else if ((first_n <= FIRST_N_EXPORT_BLACK_LIST ||
                !HasLowFrequency(v->GetFrequency())) &&
               !v->IsUnderThreshold() && key_row < num_rows) {
      // Output keys and values
      keys_flat(key_row) = key;
      context->OutputEmbeddingData(values_flat.template chip<0>(key_row),
                                   embedding_dim_);
      key_row++;
    }

    if (freq_nums > 0 && freq_row < freq_nums && freq_keys != nullptr &&
        freq_values != nullptr) {
      // Output frequency if needed
      freq_keys->template flat<K>()(freq_row) = key;
      if (freq_use_uint32) {
        freq_values->template flat<uint32_t>()(freq_row) = v->GetFrequency();
      } else {
        freq_values->template flat<uint16_t>()(freq_row) =
            GetUint16FromUint32(v->GetFrequency(), true);
      }
      freq_row++;
    }
  };
  handler->ForEach(do_export);

  VLOG(0) << "Export " << variable_name_ << " with num of ids=" << num_rows
          << " blacklists=" << blacklist_nums << " freqs=" << freq_nums
          << " first_n " << first_n;
  if (first_n > FIRST_N_EXPORT_KEY_AND_VALUES) {
    if (first_n <= FIRST_N_EXPORT_BLACK_LIST) {
      // For prediction full export, just clear prediction_deltalist_
      prediction_deltalist_.clear();
    } else {
      if (support_prediction_delta_) {
        // Copy train_deltalist_ to prediction_deltalist_
        prediction_deltalist_.insert(train_deltalist_.begin(),
                                     train_deltalist_.end());
      }
      // Erase all elements in train_deltalist_
      train_deltalist_.clear();
    }
  }
  return ::tensorflow::OkStatus();
}

template <typename K, typename V>
Status KvVariable<K, V>::DeltaExport(OpKernelContext* ctx, int first_n,
                                     bool no_copy, const string& tensor_key,
                                     BundleWriter* writer, bool enable_cutoff,
                                     float cutoff_value) {
  CHECK(ctx != nullptr);
  VLOG(0) << "Start DeltaExport " << variable_name_;

  mutex_write_lock lock(*mu());
  typename TableManager<K, V>::ScopedDisableRecordRequest
      table_disable_record_request(table_);
  TF_RETURN_IF_ERROR(CheckInitializedInternal());
  // Stage 1: ouput key-value pairs.
  // Calcuate the number of valid key-value pairs.
  // training mode: train_deltalist_
  std::unordered_set<K> all_delta(train_deltalist_.begin(),
                                  train_deltalist_.end());
  std::vector<K> update_list;
  // black_list is only used by group lasso, its logical is more
  // complex that a key appears again will insert 0 as its vector
  std::vector<K> black_list;
  // delete_list is used for key removed by under_threshold or
  // other feature reduction strategy in future
  std::vector<K> delete_list;

  if (first_n <= FIRST_N_EXPORT_BLACK_LIST) {
    // inference mode: train_deltalist_ + prediction_deltalist_
    for (auto it = prediction_deltalist_.begin();
         it != prediction_deltalist_.end(); ++it) {
      all_delta.insert(*it);
    }
  }

  for (auto iter = all_delta.begin(); iter != all_delta.end(); ++iter) {
    const auto v = table_->FindOrNull(*iter);

    if (v == nullptr) {
      delete_list.push_back(*iter);
      continue;
    }
    // Just ignore low frequency here
    if (HasLowFrequency(v->GetFrequency())) {
      continue;
    }

    if (v->InBlacklist()) {
      black_list.push_back(*iter);
      continue;
    }
    update_list.push_back(*iter);
  }

  // Allocate output tensors for key-value pairs.
  Tensor* keys;
  Tensor* values;
  TensorShape value_tensor_shape = value_shape_;
  Tensor temp_keys;
  Tensor temp_values;
  value_tensor_shape.InsertDim(0, static_cast<int64>(update_list.size()));
  if (!no_copy) {
    TF_RETURN_IF_ERROR(ctx->allocate_output(
        "keys", TensorShape({static_cast<int64>(update_list.size())}), &keys));
  } else {
    TF_RETURN_IF_ERROR(ctx->allocate_temp(
        this->key_dtype(),
        TensorShape({static_cast<int64>(update_list.size())}), &temp_keys));
    keys = &temp_keys;
  }

  if (!no_copy) {
    TF_RETURN_IF_ERROR(
        ctx->allocate_output("values", value_tensor_shape, &values));
  } else {
    TF_RETURN_IF_ERROR(ctx->allocate_temp(this->value_dtype(),
                                          value_tensor_shape, &temp_values));
    values = &temp_values;
  }

  // Copy out key-value pairs with multithreading
  int64_t num_rows = 0;
  auto&& keys_flat = keys->template flat<K>();
  auto&& values_flat = values->flat_outer_dims<V>();
  auto DoOutputKVPairs = [this, &keys_flat, &values_flat, &update_list](
                             int64_t start, int64_t end) {
    std::unique_ptr<V, void (*)(V*)> buf(
        static_cast<V*>(AllocateRaw(value_bytes_)), DeallocateRaw<V>);
    for (int64_t row = start; row < end; ++row) {
      EVContext<V> context(buf.get(), false);
      auto key = update_list[row];
      keys_flat(row) = key;
      auto out_keys_values = [this, &values_flat, row,
                              &key](EVContext<V>* context) {
        if (!HasLowFrequency(context->Meta()->GetFrequency())) {
          context->OutputEmbeddingData(values_flat.template chip<0>(row),
                                       embedding_dim_);
        }
      };
      table_->FindWithFn(key, out_keys_values, &context);
    }
  };
  if (ctx != nullptr) {
    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
    ::tensorflow::Shard(worker_threads.num_threads, worker_threads.workers,
                        update_list.size(), 5000, DoOutputKVPairs);
  } else {
    DoOutputKVPairs(0, update_list.size());
  }
  if (no_copy) {
    // write keys/values
    string tensor_name = tensor_key + "-keys";
    TF_RETURN_IF_ERROR(writer->Add(tensor_name, *keys));
    tensor_name = tensor_key + "-values";
    TF_RETURN_IF_ERROR(writer->Add(tensor_name, *values));
  }

  // Export an empty initialization table.
  if (no_copy) {
    // write init_table
    TensorShape init_table_shape = value_shape_;
    init_table_shape.InsertDim(0, 0);
    Tensor random_init_table(DataTypeToEnum<float>::v(), init_table_shape);
    TF_RETURN_IF_ERROR(
        writer->Add(tensor_key + "-init_table", random_init_table));
  } else {
    Tensor* init_table;
    TensorShape init_table_shape = value_shape_;
    init_table_shape.InsertDim(0, 0);
    TF_RETURN_IF_ERROR(
        ctx->allocate_output("init_table", init_table_shape, &init_table));
  }

  // Output blacklist.
  Tensor* blacklist;
  Tensor temp_blacklist;
  if (first_n <= FIRST_N_EXPORT_BLACK_LIST) {
    // In prediction mode, move black_list to delete_list
    delete_list.insert(delete_list.end(),
                       std::make_move_iterator(black_list.begin()),
                       std::make_move_iterator(black_list.end()));
    black_list.erase(black_list.begin(), black_list.end());
  }
  int64_t blacklist_size = static_cast<int64>(black_list.size());
  if (!no_copy) {
    TF_RETURN_IF_ERROR(ctx->allocate_output(
        "blacklist", TensorShape({blacklist_size}), &blacklist));
  } else {
    TF_RETURN_IF_ERROR(ctx->allocate_temp(
        this->key_dtype(), TensorShape({blacklist_size}), &temp_blacklist));
    blacklist = &temp_blacklist;
  }
  CopyListToTensor(blacklist, black_list);
  if (no_copy) {
    // write blacklist
    TF_RETURN_IF_ERROR(writer->Add(tensor_key + "-blacklist", temp_blacklist));
  }

  // Stage 3: optionally output frequency table.
  Tensor *freq_keys, *freq_values;
  if (first_n > FIRST_N_EXPORT_FREQUENCY) {
    int64_t delta_size = static_cast<int64>(all_delta.size());
    if (!no_copy) {
      TF_RETURN_IF_ERROR(ctx->allocate_output(
          "freq_keys", TensorShape({delta_size}), &freq_keys));
      TF_RETURN_IF_ERROR(ctx->allocate_output(
          "freq_values", TensorShape({delta_size}), &freq_values));
      ExportFrequencyDelta(freq_keys, freq_values, all_delta);
    } else {
      Tensor temp_freq_keys, temp_freq_values;
      TF_RETURN_IF_ERROR(ctx->allocate_temp(
          this->key_dtype(), TensorShape({delta_size}), &temp_freq_keys));
      freq_keys = &temp_freq_keys;
      TF_RETURN_IF_ERROR(ctx->allocate_temp(DataTypeToEnum<uint32_t>::v(),
                                            TensorShape({delta_size}),
                                            &temp_freq_values));
      freq_values = &temp_freq_values;
      ExportFrequencyDelta(freq_keys, freq_values, all_delta);
      // write freq_keys/freq_values
      TF_RETURN_IF_ERROR(
          writer->Add(tensor_key + "-freq_keys", temp_freq_keys));
      TF_RETURN_IF_ERROR(
          writer->Add(tensor_key + "-freq_values", temp_freq_values));
    }
  } else {
    if (!no_copy) {
      // Export empty frequency tensors.
      TF_RETURN_IF_ERROR(
          ctx->allocate_output("freq_keys", TensorShape({0}), &freq_keys));
      TF_RETURN_IF_ERROR(
          ctx->allocate_output("freq_values", TensorShape({0}), &freq_values));
    } else {
      Tensor temp_freq_keys(this->key_dtype(), TensorShape({0}));
      Tensor temp_freq_values(DataTypeToEnum<uint32_t>::v(), TensorShape({0}));
      // write freq_keys/freq_values
      TF_RETURN_IF_ERROR(
          writer->Add(tensor_key + "-freq_keys", temp_freq_keys));
      TF_RETURN_IF_ERROR(
          writer->Add(tensor_key + "-freq_values", temp_freq_values));
    }
  }

  // Set need_full_import false
  if (!no_copy) {
    Tensor* need_full_import;
    TF_RETURN_IF_ERROR(ctx->allocate_output(
        "need_full_import", TensorShape({1}), &need_full_import));
    auto&& need_full_import_flat = need_full_import->template flat<bool>();
    need_full_import_flat(0) = false;
  } else {
    Tensor temp_need_full_import(DataTypeToEnum<bool>::v(), TensorShape({1}));
    auto&& need_full_import_flat = temp_need_full_import.template flat<bool>();
    need_full_import_flat(0) = false;
    // write need_full_import
    TF_RETURN_IF_ERROR(
        writer->Add(tensor_key + "-need_full_import", temp_need_full_import));
  }

  // Set delete_keys
  Tensor* delete_keys;
  if (!no_copy) {
    TF_RETURN_IF_ERROR(ctx->allocate_output(
        "delete_keys", TensorShape({static_cast<int64>(delete_list.size())}),
        &delete_keys));
    CopyListToTensor(delete_keys, delete_list);
  } else {
    Tensor temp_delete_keys;
    TF_RETURN_IF_ERROR(ctx->allocate_temp(
        this->key_dtype(),
        TensorShape({static_cast<int64>(delete_list.size())}),
        &temp_delete_keys));
    CopyListToTensor(&temp_delete_keys, delete_list);
    // write delete_keys
    TF_RETURN_IF_ERROR(
        writer->Add(tensor_key + "-delete_keys", temp_delete_keys));
  }

  // Update deltalist
  if (first_n <= FIRST_N_EXPORT_BLACK_LIST) {
    // inference mode: clear prediction_deltalist_
    prediction_deltalist_.clear();
  } else {
    if (support_prediction_delta_) {
      // training mode: move train_deltalist_ to prediction_deltalist_
      prediction_deltalist_.insert(train_deltalist_.begin(),
                                   train_deltalist_.end());
    }
    train_deltalist_.clear();
  }
  VLOG(0) << "DeltaExport " << variable_name_
          << " with num of ids=" << update_list.size()
          << " blacklists=" << black_list.size()
          << " delete_keys=" << delete_list.size() << " first_n=" << first_n;
  return ::tensorflow::OkStatus();
}

}  // namespace tfplus

#endif  // TFPLUS_KV_VARIABLE_KERNELS_DYNAMIC_SAVE_HPP_
