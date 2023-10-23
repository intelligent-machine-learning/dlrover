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

#ifndef TFPLUS_KV_VARIABLE_KERNELS_KV_VARIABLE_H_
#define TFPLUS_KV_VARIABLE_KERNELS_KV_VARIABLE_H_

#include <algorithm>
#include <atomic>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "tbb/concurrent_unordered_set.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/gtl/map_util.h"
// #include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/work_sharder.h"
#include "tfplus/kv_variable/kernels/embedding_value.h"
#include "tfplus/kv_variable/kernels/hashmap.h"
#include "tfplus/kv_variable/kernels/hybrid_embedding/embedding_context.h"
#include "tfplus/kv_variable/kernels/hybrid_embedding/storage_config.pb.h"
#include "tfplus/kv_variable/kernels/hybrid_embedding/table_manager.h"
#include "tfplus/kv_variable/kernels/kv_variable_cwise_op.h"
#include "tfplus/kv_variable/kernels/kv_variable_interface.h"
#include "tfplus/kv_variable/kernels/mutex.h"
#include "tfplus/kv_variable/kernels/utility.h"
#include "tfplus/kv_variable/utils/progress_bar.h"
#include "tfplus/kv_variable/utils/utils.h"

namespace tfplus {
using ::tensorflow::DataType;
using ::tensorflow::DataTypeToEnum;
using ::tensorflow::int32;
using ::tensorflow::int64;
using ::tensorflow::OpKernelContext;
using ::tensorflow::Status;
using ::tensorflow::Tensor;
using ::tensorflow::TensorShape;
using ::tensorflow::uint16;
using ::tensorflow::BundleWriter;
using ::tensorflow::gtl::InlinedVector;
using ::tensorflow::tensor::DeepCopy;
using mutex_read_lock = tfplus_shared_lock;
using mutex_write_lock = tfplus_mutex_lock;

// Used to periodically clean up keys that are
// lazily removed from the blacklist and table
constexpr int KV_RECOVERY_COUNTS = 500000;

constexpr int IMPORT_OP_OTHERS_SIZE = 4;
constexpr int FULL_OR_DELTA_IMPORT_OP_OTHERS_SIZE = 6;

constexpr int FIRST_N_EXPORT_KEY_AND_VALUES = 2;
constexpr int FIRST_N_EXPORT_BLACK_LIST = 3;
constexpr int FIRST_N_EXPORT_FREQUENCY = 4;

template <typename K, typename V>
class KvVariable : public KvVariableInterface {
 public:
  using TensorChip = ::Eigen::Tensor<V, 1, ::Eigen::RowMajor>;
  KvVariable(const std::string& variable_name, const TensorShape& value_shape,
             int32 enter_threshold,
             StorageOption storage_option = StorageOption())
      : random_init_table_set_(false),
        variable_name_(variable_name),
        value_shape_(value_shape),
        embedding_dim_(value_shape.num_elements()),
        enter_threshold_(SaturateMaxFrequency(enter_threshold)),
        value_bytes_(embedding_dim_ * sizeof(V)) {
    support_prediction_delta_ =
        GetEnvVar<bool>("SUPPORT_PREDICTION_DELTA_EXPORT", false);
    support_delta_export_ = GetEnvVar<bool>("SUPPORT_DELTA_EXPORT", false);
    VLOG(1) << "new kv variable: " << variable_name_
            << " enter_threshold: " << enter_threshold_
            << " SUPPORT_DELTA_EXPORT: " << support_delta_export_
            << " SUPPORT_PREDICTION_DELTA_EXPORT: "
            << support_prediction_delta_;
    table_ = new TableManager<K, V>(
        storage_option, variable_name, embedding_dim_,
        support_delta_export_ ? &train_deltalist_ : nullptr);
    lockable_ = table_->NeedExplicitLock();
    has_mem_table_ = table_->HasMemTable();
  }

  virtual ~KvVariable() { delete table_; }

  string DebugString() const override {
    return variable_name_;
  }

  int64_t MemoryUsed() const override {
    int64_t ret = 0;
    mutex_read_lock l(*mu());

    // get the memory for the initializer table.
    if (random_init_table_set_) {
      ret += random_init_table_.AllocatedBytes();
    }

    return sizeof(KvVariable) + ret;
  }

  template <typename TL>
  inline int ModKey(const TL& key, const int& num_shards) {
    return tfplus::ModKeyImpl<TL>(key, num_shards);
  }

  size_t size() const override {
    mutex_read_lock lock(*mu());
    return size_unsafe();
  }

  size_t size_unsafe() const {
    int64_t num_rows = 0;
    auto count_size = [this, &num_rows](const K& key,
                                        const EVContext<V>* context) {
      auto embedding_val = context->Meta();
      if (!embedding_val->InBlacklist() &&
          !HasLowFrequency(embedding_val->GetFrequency())) {
        num_rows++;
      }
    };
    table_->ForEach(count_size);
    return num_rows;
  }

  size_t sum_freq() const override {
    mutex_read_lock lock(*mu());
    return sum_freq_unsafe();
  }

  size_t sum_freq_unsafe() const {
    int64_t freq_count = 0;
    auto count_freq = [this, &freq_count](const K& key,
                                          const EVContext<V>* context) {
      auto embedding_val = context->Meta();
      if (!embedding_val->InBlacklist() &&
          !HasLowFrequency(embedding_val->GetFrequency())) {
        freq_count += GetUint16FromUint32(embedding_val->GetFrequency(), true);
      }
    };
    table_->ForEach(count_freq);
    return freq_count;
  }

  TensorShape GetShape() const override {
    TensorShape shape = value_shape_;
    shape.InsertDim(0, table_->size());

    return shape;
  }

  Status InitRandomValues(const Tensor& random_init) override {
    mutex_write_lock lock(*mu());

    // Initialization table can only be set once.
    if (CheckInitializedInternal() == ::tensorflow::OkStatus() &&
        random_init_table_.NumElements() > 0) {
      VLOG(1) << "Random value table already initialized for: "
              << variable_name_ << " re-initialization ignored.";
      return ::tensorflow::OkStatus();
    }

    // Check consistency between value shape.
    TensorShape shape = random_init.shape();
    shape.RemoveDim(0);

    // Copy the initialization table.
    random_init_table_ = random_init;

    // Set that the initialization table is already filled.
    random_init_table_set_ = true;

    return ::tensorflow::OkStatus();
  }

  Status GetInitTableShape(TensorShape* shape) const override {
    CHECK(shape != nullptr);

    mutex_read_lock lock(*mu());

    // If initialization table is not set yet, report error.
    TF_RETURN_IF_ERROR(CheckInitializedInternal());

    // Get the shape
    *shape = random_init_table_.shape();

    return ::tensorflow::OkStatus();
  }

  Status GetInitTable(Tensor* tensor) const override {
    CHECK(tensor != nullptr);

    mutex_read_lock lock(*mu());

    // If initialization table is not set yet, report error.
    TF_RETURN_IF_ERROR(CheckInitializedInternal());

    // Deep copy the content.
    *tensor = DeepCopy(random_init_table_);

    return ::tensorflow::OkStatus();
  }

  // FindOrZeros retrieves the tensor with given keys. All zeros will be
  // returned if the given key doesn't exist in the table_. The function is
  // typically used in KvVariableGather OP for model prediction.
  Status FindOrZeros(OpKernelContext* ctx, const Tensor& keys,
                     Tensor* values) const override {
    CHECK(values != nullptr);
    TF_RETURN_IF_ERROR(CheckInitializedInternal());
    auto values_flat = values->flat_outer_dims<V>();
    auto find_fn = [this, &values_flat](const K& key, EVContext<V>* context,
                                        size_t row) {
      context->OutputEmbeddingData(values_flat.template chip<0>(row),
                                   embedding_dim_);
    };
    auto set_zero = [this, &values_flat](const K& key, EVContext<V>* context,
                                         size_t row) {
      values_flat.chip(row, 0).setZero();
    };
    return table_->BatchGetWithFn(ctx, keys, find_fn, set_zero);
  }

  //  FindOrInsert is typically used in KvVariableGather OP for model training.
  //  If the key already in the table_, we return its value, otherwise we will
  //  insert a random tensor from random_init_table_ as its initializer tensor.
  //  The 3rd param filter_out is used to filter out the low frequency. If the
  //  key is low frequency, we do not insert it, and do not update its gradient
  //  in backward computing. Please note that if a key exists in the blacklist,
  //  it means that its value is a zero tensor.
  Status FindOrInsert(OpKernelContext* ctx, const Tensor& keys, Tensor* values,
                      const Tensor* counts, Tensor* filter_out,
                      bool apply_filter = true) override {
    CHECK(values != nullptr);
    if (counts != nullptr) {
      if (keys.shape() != counts->shape()) {
        return ::tensorflow::errors::InvalidArgument(
            "KvVariable ", variable_name_.c_str(),
            ": increment count, indices shape ", keys.shape().DebugString(),
            " does not match with counts shape ",
            counts->shape().DebugString());
      }

      if (counts->dtype() != DataType::DT_INT32) {
        return ::tensorflow::errors::InvalidArgument(
            "KvVariable ", variable_name_.c_str(),
            ": increment count, counts dtype must be int32");
      }
    }
    mutex_read_lock l(*mu());
    return FindOrInsertLocally(ctx, keys, values, counts, filter_out,
                                apply_filter);
  }

  Status FindOrInsertLocally(
      OpKernelContext* ctx, const Tensor& keys, Tensor* values,
      const Tensor* counts, Tensor* filter_out, bool apply_filter = true,
      std::vector<std::pair<K, size_t>>* key_and_index = nullptr) {
    const auto& keys_flat = keys.flat<K>();
    auto values_flat = values->flat_outer_dims<V>();
    uint16_t last_update_time_in_days = GetCurrentUnixTimeByDivisor();

    TF_RETURN_IF_ERROR(CheckInitializedInternal());
    // thread can't be wait for mutex lock
    Status st = ::tensorflow::OkStatus();
    auto DoWork = [this, &keys_flat, &values_flat, &filter_out, &apply_filter,
                   &last_update_time_in_days, &counts, key_and_index,
                   &st](int64_t start, int64_t end) {
      std::unique_ptr<V, void (*)(V*)> buf(
          static_cast<V*>(AllocateRaw(value_bytes_)), DeallocateRaw<V>);
      for (int64_t i = start; i < end; ++i) {
        // Check if key exists in filter_out.
        size_t row = i;
        K key = keys_flat(row);
        if (key_and_index) {
          row = (*key_and_index)[i].second;
          key = (*key_and_index)[i].first;
        }
        if (filter_out != nullptr && apply_filter &&
            filter_out->flat<bool>()(row)) {
          continue;
        }

        if (NeedDeltaInfo()) {
          train_deltalist_.insert(key);
        }

        auto find_func = [this, &values_flat, &filter_out, &counts,
                          &apply_filter, &last_update_time_in_days, &row,
                          &key](EVContext<V>* context) {
          uint16_t frequency = 1;
          if (counts != nullptr) {
            frequency =
                SaturateMaxFrequency(counts->template flat<int32>()(row));
          }
          context->Meta()->AddFrequency(frequency, last_update_time_in_days);
          UpdateUnderThreshold(context);
          context->OutputEmbeddingData(values_flat.template chip<0>(row),
                                        embedding_dim_);
        };

        bool should_filter = false;
        uint32_t freq = 1;
        if (filter_out != nullptr && !apply_filter) {
          should_filter = HasLowFrequency(freq);
        }
        auto insert_func = [this, &values_flat, &freq, &apply_filter,
                            &should_filter, &filter_out,
                            &last_update_time_in_days, &counts, &key,
                            &row](EVContext<V>* context) {
          auto stat = reinterpret_cast<uint16_t*>(&freq);
          if (counts != nullptr) {
            stat[0] = SaturateMaxFrequency(counts->template flat<int32>()(row));
          } else {
            stat[0] = static_cast<uint16_t>(1);
          }
          stat[1] = last_update_time_in_days + 0;
          context->Meta()->UpdateFrequency(freq);
          if (filter_out != nullptr && !apply_filter) {
            filter_out->template flat<bool>()(row) = should_filter;
          }
          if (context->Meta()->GetStorageType() == StorageType::MEM_STORAGE) {
            // needs allocating new buffer for mem_storage
            context->UpdateValue(static_cast<V*>(AllocateRaw(value_bytes_)),
                                 true, value_bytes_);
          }
          GenerateRandomInitialValue(context->Value());
          UpdateUnderThreshold(context);
          context->OutputEmbeddingData(values_flat.template chip<0>(row),
                                       embedding_dim_);
        };
        EVContext<V> context(buf.get(), false);
        context.SetStatus(&st);
        table_->FindOrInsertWithDifferentFn(key, find_func, insert_func,
                                            &context);
      }
    };
    size_t key_size =
        key_and_index == nullptr ? keys_flat.size() : key_and_index->size();
    if (ctx != nullptr && !lockable_) {
      auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
      ::tensorflow::Shard(worker_threads.num_threads, worker_threads.workers,
                          key_size, 5000, DoWork);
    } else {
      DoWork(0, key_size);
    }
    return st;
  }

  void FindOrInsertUnsafe(const K& key, EVContext<V>* context,
                          bool* filter_out) {
    auto insert_func = [this, key](EVContext<V>* context) {
      if (context->Meta()->GetStorageType() == StorageType::MEM_STORAGE) {
        // needs allocating new buffer for mem_storage
        context->UpdateValue(static_cast<V*>(AllocateRaw(value_bytes_)), true,
                             value_bytes_);
      } else {
        // TODO(mochen.bmc): change training_ops.cc to removing the if
        // statement, add context_buf
        if (!context->Value()) {
          context->UpdateValue(static_cast<V*>(AllocateRaw(value_bytes_)), true,
                               value_bytes_);
        }
      }
      GenerateRandomInitialValue(context->Value());
      UpdateUnderThreshold(context);
    };
    bool succ = table_->FindOrInsertWithFnUnsafe(key, insert_func, context);
    if (succ) {
      if (filter_out != nullptr) {
        // forward variables
        auto should_filter = HasLowFrequency(context->Meta()->GetFrequency());
        *filter_out = should_filter;
        if (context->Meta()->InBlacklist() && !should_filter) {
          table_->RemoveBlacklistUnsafe(key, context);
        }
      } else {
        // backward variables, evict needs the frequency information
        uint16_t last_update_time_in_days = GetCurrentUnixTimeByDivisor();
        uint16_t frequency = 1;
        context->Meta()->AddFrequency(frequency, last_update_time_in_days);
      }
    }
  }

  void MarkBlacklistUnsafe(const K& key, EVContext<V>* context) {
    // mutex_write_lock lock(mu_, lockable_);
    table_->MarkBlacklistUnsafe(key, context);
  }

  Status InsertOrUpdate(OpKernelContext* ctx, const Tensor& keys,
                        const Tensor& values, const Tensor* filterout,
                        const Tensor* blacklist) override {
    // Check if data type matches.
    CHECK_EQ(keys.dtype(), DataTypeToEnum<K>::v());
    CHECK_EQ(values.dtype(), DataTypeToEnum<V>::v());
    if (filterout != nullptr) {
      CHECK_EQ(filterout->dtype(), DataTypeToEnum<bool>::v());
    }

    if (blacklist != nullptr) {
      CHECK_EQ(blacklist->dtype(), DataTypeToEnum<bool>::v());
    }

    const auto& keys_flat = keys.flat<K>();
    const auto& values_inner_flat = values.template flat_inner_dims<V, 2>();
    auto values_flat = values.flat_outer_dims<V>();

    mutex_read_lock l(*mu());
    TF_RETURN_IF_ERROR(CheckInitializedInternal());
    auto DoWork = [this, &keys_flat, &values_inner_flat, &values_flat,
                   &filterout, &blacklist](int64_t start, int64_t end) {
      // Apply filtering.
      for (int64_t i = start; i < end; ++i) {
        const auto& key = keys_flat(i);
        bool mark_blacklist =
            blacklist == nullptr ? false : blacklist->flat<bool>()(i);

        if (NeedDeltaInfo()) {
          train_deltalist_.insert(key);
        }
        if ((filterout != nullptr && filterout->flat<bool>()(i))) {
          continue;
        }
        std::function<void(EVContext<V> * context)> update_fn;
        if (!mark_blacklist) {
          auto insert_or_update_fn = [this, &mark_blacklist, &values_flat, &i,
                                      &key](EVContext<V>* context) {
            const V* value_ptr = &values_flat(i, 0);
            context->UpdateValue(value_ptr, false, value_bytes_);
            UpdateUnderThreshold(context);
          };
          // update: memcpy, insert: allocate and memcpy
          bool succeed = table_->UpdateWithFn(key, insert_or_update_fn);
          if (!succeed) {
            table_->InsertWithFn(key, insert_or_update_fn);
          }
        } else {
          auto l = GetScopedKeyLock(key, LockType::WRITE_LOCK);
          MarkBlacklistUnsafe(key, nullptr);
        }
      }
    };

    if (ctx != nullptr) {
      auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
      ::tensorflow::Shard(worker_threads.num_threads, worker_threads.workers,
                          keys_flat.size(), 5000, DoWork);
    } else {
      DoWork(0, keys_flat.size());
    }
    return ::tensorflow::OkStatus();
  }

  void CoverUpdate(const K& key, EVContext<V>* context) const {
    auto l = GetScopedKeyLock(key, LockType::WRITE_LOCK);
    CoverUpdateUnsafe(key, context);
  }

  void CoverUpdateUnsafe(const K& key, EVContext<V>* context) const {
    if (context->Meta()->GetStorageType() == StorageType::MEM_STORAGE) {
      UpdateUnderThreshold(context);
    } else {
      auto update_fn = [this, &key](EVContext<V>* context) {
        UpdateUnderThreshold(context);
      };
      table_->UpdateWithFnUnsafe(key, update_fn, context);
    }
  }

  Status GetCount(const Tensor& indices, Tensor* counts) {
    CHECK(counts != nullptr);

    // Check indices dtypes.
    TF_RETURN_IF_ERROR(CheckKvVariableKeyTypes(indices.dtype()));

    const auto& indices_values = indices.template flat<K>();
    auto&& counts_values = counts->template flat<int32>();
    mutex_read_lock lock(*mu());
    TF_RETURN_IF_ERROR(CheckInitializedInternal());

    for (int64_t i = 0; i < indices_values.size(); ++i) {
      const auto& key = indices_values(i);
      const auto v = table_->FindOrNull(key);
      if (v == nullptr) {
        counts_values(i) = 0;
      } else {
        counts_values(i) = GetUint16FromUint32(v->GetFrequency(), true);
      }
    }
    return ::tensorflow::OkStatus();
  }

  Status GetTimeStamp(const Tensor& indices, Tensor* timestamps) {
    CHECK(timestamps != nullptr);
    // The shape of indices should match the shape of timestamps.
    if (indices.shape() != timestamps->shape()) {
      return ::tensorflow::errors::InvalidArgument(
          "KvVariable ", variable_name_.c_str(),
          ": get timestamp, indices shape ", indices.shape().DebugString(),
          " not match with timestamps shape ",
          timestamps->shape().DebugString());
    }

    // Check indices dtypes.
    TF_RETURN_IF_ERROR(CheckKvVariableKeyTypes(indices.dtype()));
    // Timestamps dtype must be uint32.
    if (timestamps->dtype() != DataType::DT_UINT32) {
      return ::tensorflow::errors::InvalidArgument(
          "KvVariable ", variable_name_.c_str(),
          ": get timestamp, timestamps dtype must be uint32");
    }

    const auto& indices_values = indices.template flat<K>();
    auto&& timestamps_values = timestamps->template flat<uint32_t>();
    mutex_read_lock lock(*mu());
    TF_RETURN_IF_ERROR(CheckInitializedInternal());

    for (int64_t i = 0; i < indices_values.size(); ++i) {
      const auto& key = indices_values(i);
      const auto v = table_->FindOrNull(key);
      if (v == nullptr) {
        timestamps_values(i) = GetCurrentUnixTimeByDivisor();
      } else {
        timestamps_values(i) = GetUint16FromUint32(v->GetFrequency(), false);
      }
    }
    return ::tensorflow::OkStatus();
  }

  bool IsInitialized() const override { return random_init_table_set_; }

  Status ImportValues(OpKernelContext* ctx, const Tensor& keys,
                      const Tensor& values,
                      const std::vector<Tensor>& others) override;

  Status ExportValuesForMultiHash(OpKernelContext* ctx, int first_n,
                                  bool enable_cutoff, float cutoff_value,
                                  const std::string& variable_name) override {
    return ::tensorflow::OkStatus();
  }

  Status CountStorageSize(OpKernelContext* ctx) override {
    std::vector<size_t> storage_size = table_->CountStorageSize();
    Tensor* storage_size_tensor;
    TF_RETURN_IF_ERROR(ctx->allocate_output(
        "sizes", TensorShape({storage_size.size()}), &storage_size_tensor));
    auto&& storage_size_flat = storage_size_tensor->template flat<int64>();
    for (int i = 0; i < storage_size.size(); i++) {
      storage_size_flat(i) = storage_size[i];
    }
    return ::tensorflow::OkStatus();
  }

  Status ExportValues(OpKernelContext* ctx, int first_n,
                      bool enable_cutoff = false, float cutoff_value = 0.0,
                      void* table_handler = nullptr) override;

  Status FullOrDeltaImport(OpKernelContext* ctx, int first_n,
                           const Tensor& keys, const Tensor& values,
                           const std::vector<Tensor>& others) override {
    // Check the size of vector others.

    auto need_full_export = others[4].template flat<bool>()(0);
    if (need_full_export) {
      return ImportValues(ctx, keys, values, others);
    } else {
      return DeltaImport(ctx, first_n, keys, values, others);
    }
  }

  Status FullOrDeltaExport(OpKernelContext* ctx, int first_n,
                           bool enable_cutoff = false, float cutoff_value = 0.0,
                           bool need_full_export = true) override {
    if (need_full_export) {
      return FullExport(ctx, first_n, false, "", nullptr, enable_cutoff,
                        cutoff_value, false);
    } else {
      return DeltaExport(ctx, first_n, false, "", nullptr, enable_cutoff,
                         cutoff_value);
    }
  }

  Status ScatterUpdate(OpKernelContext* ctx, const Tensor& keys,
                       const Tensor& updates,
                       ScatterUpdateOps update_op) override {
    // Check if data types match.
    CHECK_EQ(keys.dtype(), DataTypeToEnum<K>::v());
    CHECK_EQ(updates.dtype(), DataTypeToEnum<V>::v());

    // Check if the first dimensions match.
    auto updates_shape = updates.shape();
    if (keys.dim_size(0) != updates_shape.dim_size(0)) {
      // return ::tensorflow::errors::InvalidArgument(
      //     "KvVariable ", variable_name_.c_str(), ": number of keys ",
      //     keys.dim_size(0), " and number of values updates ",
      //     updates_shape.dim_size(0), " do not match");
    }

    // Check if the value shape matches.
    updates_shape.RemoveDim(0);
    if (value_shape_ != updates_shape) {
      // return ::tensorflow::errors::InvalidArgument(
      //     "KvVariable ", variable_name_.c_str(),
      //     ": value shape does not match in ScatterUpdate");
    }
    TF_RETURN_IF_ERROR(CheckInitializedInternal());

    // Create an operator as per the update operation.
    CwiseOperationBase<TensorChip>* op_obj = nullptr;
    switch (update_op) {
      case SCATTER_UPDATE_ASSIGN:
        op_obj = new CwiseOperationAssign<TensorChip>();
        break;
      case SCATTER_UPDATE_ADD:
        op_obj = new CwiseOperationAdd<TensorChip>();
        break;
      case SCATTER_UPDATE_SUB:
        op_obj = new CwiseOperationSub<TensorChip>();
        break;
      case SCATTER_UPDATE_MUL:
        op_obj = new CwiseOperationMul<TensorChip>();
        break;
      case SCATTER_UPDATE_DIV:
        op_obj = new CwiseOperationDiv<TensorChip>();
        break;
      case SCATTER_UPDATE_MIN:
        op_obj = new CwiseOperationMin<TensorChip>();
        break;
      case SCATTER_UPDATE_MAX:
        op_obj = new CwiseOperationMax<TensorChip>();
        break;
      default:
        break;
        // return ::tensorflow::errors::InvalidArgument(
        //     "KvVariable ", variable_name_.c_str(),
        //     ": unsupported update operation ", update_op);
    }

    // Perform value updates.
    const auto num_elements = embedding_dim_;
    const auto& keys_flat = keys.flat<K>();
    const auto& updates_flat = updates.flat_outer_dims<V>();

    // Run in parallel, kernel function per thread.
    mutex_read_lock lock(*mu());
    auto DoWork = [this, &keys_flat, &updates_flat, op_obj, num_elements](
                      int64_t start_row, int64_t end_row) {
      // Iterate over the slice.
      for (int64_t row = start_row; row < end_row; ++row) {
        const auto& key = keys_flat(row);
        updates_flat.chip(row, 0);
        if (NeedDeltaInfo()) {
          train_deltalist_.insert(key);
        }
        auto scatter_update_fn = [this, &row, &updates_flat, &num_elements,
                                  &op_obj, &key](EVContext<V>* context) {
          if (!context->Meta()->InBlacklist()) {
            typename ::tensorflow::TTypes<V>::ConstTensor lhs(context->Value(),
                                                              num_elements);
            const auto& res = (*op_obj)(lhs, updates_flat.chip(row, 0));
            std::copy_n(res.data(), num_elements, context->Value());
            UpdateUnderThreshold(context);
          }
        };
        auto succeed = table_->FetchAndUpdateWithFn(key, scatter_update_fn);
        if (!succeed) {
          auto insert_func = [this, &num_elements, &op_obj, &updates_flat, &row,
                              &key](EVContext<V>* context) {
            V* init_vec =
                reinterpret_cast<V*>(::tensorflow::cpu_allocator()->AllocateRaw(
                    ::tensorflow::Allocator::kAllocatorAlignment,
                    value_bytes_));
            GenerateRandomInitialValue(init_vec);
            context->UpdateValue(init_vec, true, value_bytes_);
            typename ::tensorflow::TTypes<V>::ConstTensor lhs(context->Value(),
                                                              num_elements);
            const auto& res = (*op_obj)(lhs, updates_flat.chip(row, 0));
            std::copy_n(res.data(), num_elements, context->Value());
            UpdateUnderThreshold(context);
          };
          table_->InsertWithFn(key, insert_func);
        }
      }
    };

    // Invoke multiple threads.
    if (ctx != nullptr) {
      auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
      ::tensorflow::Shard(worker_threads.num_threads, worker_threads.workers,
                          keys_flat.size(), 5000, DoWork);
    } else {
      DoWork(0, keys_flat.size());
    }

    // Release operator object
    if (op_obj != nullptr) {
      delete op_obj;
    }

    return ::tensorflow::OkStatus();
  }

  // Delete keys.
  Status Delete(const Tensor& indices) {
    // Check indices dtypes.
    TF_RETURN_IF_ERROR(CheckKvVariableKeyTypes(indices.dtype()));

    const auto& indices_values = indices.template flat<K>();
    mutex_write_lock lock(*mu());
    TF_RETURN_IF_ERROR(CheckInitializedInternal());
    for (int64_t i = 0; i < indices_values.size(); ++i) {
      const auto& delete_key = indices_values(i);
      table_->DeleteKey(delete_key);
      if (NeedDeltaInfo()) {
        train_deltalist_.insert(delete_key);
      }
    }

    return ::tensorflow::OkStatus();
  }

  // Delete keys with timestamp.
  Status DeleteWithTimestamp(OpKernelContext* ctx, int threshold) {
    CHECK(ctx != nullptr);

    mutex_write_lock l(*mu());
    TF_RETURN_IF_ERROR(CheckInitializedInternal());

    std::vector<K> delete_list;
    auto current_time = GetCurrentUnixTimeByDivisor();

    auto delete_iter = [this, &threshold, &delete_list, &current_time](
                           const K& key, const EVContext<V>* context) {
      auto v = context->Meta();
      auto key_time = GetUint16FromUint32(v->GetFrequency(), false);
      if (key_time > 0 &&
          current_time - key_time >= static_cast<uint16_t>(threshold)) {
        delete_list.push_back(key);
        if (NeedDeltaInfo()) {
          train_deltalist_.insert(key);
        }
      }
    };
    table_->ForEach(delete_iter);
    for (auto iter = delete_list.begin(); iter != delete_list.end(); ++iter) {
      table_->DeleteKey(*iter);
    }

    Tensor* delete_keys;
    int64_t deletelist_size = static_cast<int64>(delete_list.size());
    TF_RETURN_IF_ERROR(ctx->allocate_output(
        "delete_keys", TensorShape({deletelist_size}), &delete_keys));
    CopyListToTensor(delete_keys, delete_list);

    return ::tensorflow::OkStatus();
  }

  Status MarkAsDeltaListElements(OpKernelContext* ctx, const Tensor& keys,
                                 const std::vector<int64>& indices) override {
    if (NeedDeltaInfo()) {
      // mutex_read_lock lock(*mu());
      auto keys_flat = keys.flat<K>();
      for (auto& index : indices) {
        train_deltalist_.insert(keys_flat(index));
      }
    }
    return ::tensorflow::OkStatus();
  }

  const bool is_support_delta_export() const override {
    return support_delta_export_;
  }

  const std::string& name() const override { return variable_name_; }

  DataType key_dtype() const override { return DataTypeToEnum<K>::v(); }
  DataType value_dtype() const override { return DataTypeToEnum<V>::v(); }
  TensorShape value_shape() const override { return value_shape_; }
  int embedding_dim() const override { return embedding_dim_; }
  tf_mutex* mu() const override { return table_->mu(); }
  IMap<K, EmbeddingValue<V>>* kv_map() const { return table_->GetKVMap(); }

  inline bool NeedDeltaInfo() const override {
    return support_delta_export_ || table_->MayNeedDeltaInfo();
  }

  Status ApplySnapshot(const std::vector<std::string>& src_files) override {
    typename TableManager<K, V>::ScopedLock table_scoped_lock(table_);
    TF_RETURN_IF_ERROR(
        reinterpret_cast<tfplus::TableManager<K, V>*>(table_)->PrepareSnapshots(
            src_files));
    return reinterpret_cast<tfplus::TableManager<K, V>*>(table_)
        ->ApplySnapshot();
  }

  Status LoadRemoteTable(const std::string& table_name) override {
    typename TableManager<K, V>::ScopedLock table_scoped_lock(table_);
    TF_RETURN_IF_ERROR(
        reinterpret_cast<tfplus::TableManager<K, V>*>(table_)->InitRemoteTable(
            table_name));
    return ::tensorflow::OkStatus();
  }

  bool UpdateUnderThreshold(const EVContext<V>* context,
                            bool enable_cutoff = DEFAULT_ENABLE_CUTOFF,
                            float cutoff_value = DEFAULT_CUTOFF_VALUE) const {
    if (context->Meta()->InBlacklist()) {
      context->Meta()->SetUnderThreshold(true);
      return true;
    }
    if (!context->Value()) {
      context->Meta()->SetUnderThreshold(true);
      return true;
    }
    // Check if none of values in vec exceeds the cutoff_value threshold.
    if (!enable_cutoff) {
      context->Meta()->SetUnderThreshold(false);
      return false;
    }
    for (int64_t i = 0; i < embedding_dim_; i++) {
      if (std::abs(static_cast<float>(context->Value()[i])) >= cutoff_value) {
        context->Meta()->SetUnderThreshold(false);
        return false;
      }
    }
    context->Meta()->SetUnderThreshold(true);
    return true;
  }

  ScopedSpinLock GetScopedKeyLock(const K& key, LockType lock_type) {
    return table_->GetScopedKeyLock(key, lock_type);
  }

 private:
  TableManager<K, V>* table_;
  bool has_mem_table_;
  tbb::concurrent_unordered_set<K> train_deltalist_;
  tbb::concurrent_unordered_set<K> prediction_deltalist_;
  bool support_prediction_delta_;
  bool support_delta_export_;
  Tensor random_init_table_;
  std::atomic<bool> random_init_table_set_;
  const std::string variable_name_;
  const TensorShape value_shape_;
  const uint16 enter_threshold_;
  const int embedding_dim_;
  std::string ray_actor_path_;
  size_t value_bytes_;
  std::string phstore_path_;
  std::string ph_table_name_;
  bool lockable_;
  Tensor p_values_;
  MapType map_type_;

  // Randomly generates initial value.
  inline void GenerateRandomInitialValue(V* value) const {
    const int64_t init_dim_size = random_init_table_.dim_size(0);
    const int r1 = std::rand() % init_dim_size;
    const int r2 = std::rand() % init_dim_size;
    const auto& t = random_init_table_.flat_outer_dims<V>();
    typename ::Eigen::Tensor<V, 1, ::Eigen::RowMajor> avg =
        (t.chip(r1, 0) + t.chip(r2, 0)) * V(0.5f);

    std::copy_n(avg.data(), embedding_dim_, value);
  }

  // Check if the variable is already initialized.
  Status CheckInitializedInternal() const {
    if (!random_init_table_set_) {
      // return ::tensorflow::errors::InvalidArgument(
      //     "KvVariable is not initialized for ", variable_name_);
    }
    return ::tensorflow::OkStatus();
  }

  // Check if the key is of low frequency.
  bool HasLowFrequency(uint32_t frequency) const {
    return GetUint16FromUint32(frequency, true) < enter_threshold_;
  }

  // Export initialization table (optinoal for checkpoint/saved model).
  void ExportInitTable(Tensor* init_table) const {
    CHECK(init_table != nullptr);
    CHECK(init_table->NumElements() == random_init_table_.NumElements());

    // Deep copy tensor content
    *init_table = DeepCopy(random_init_table_);
  }

  inline void CopyListToTensor(Tensor* tensor,
                               const std::vector<K>& list) const {
    CHECK(tensor != nullptr);
    CHECK(tensor->NumElements() == list.size());

    int64_t num_rows = 0;
    auto&& tensor_flat = tensor->template flat<K>();
    for (auto iter = list.begin(); iter != list.end(); ++iter) {
      tensor_flat(num_rows) = *iter;
      ++num_rows;
    }
  }

  // Export delta frequency table (optional for checkpoint/saved model).
  void ExportFrequencyDelta(Tensor* freq_keys, Tensor* freq_values,
                            const std::unordered_set<K>& delta_keys) const {
    CHECK(freq_keys != nullptr);
    CHECK(freq_keys->NumElements() == delta_keys.size());
    CHECK(freq_values->NumElements() == delta_keys.size());

    int64_t num_rows = 0;
    auto&& freq_keys_flat = freq_keys->template flat<K>();
    auto&& freq_values_flat = freq_values->template flat<uint32_t>();
    for (auto iter = delta_keys.begin(); iter != delta_keys.end(); ++iter) {
      const auto v = table_->FindOrNull(*iter);
      freq_keys_flat(num_rows) = *iter;
      freq_values_flat(num_rows) = (v == nullptr ? 0 : v->GetFrequency());
      ++num_rows;
    }
  }

  Status FullExport(OpKernelContext* ctx, int first_n, bool no_copy,
                    const string& tensor_key, BundleWriter* writer,
                    bool enable_cutoff = false, float cutoff_value = 0.0,
                    bool freq_use_uint32 = false);

  Status DeltaExport(OpKernelContext* ctx, int first_n, bool no_copy,
                     const string& tensor_key, BundleWriter* writer,
                     bool enable_cutoff = false, float cutoff_value = 0.0);

  Status DeltaImport(OpKernelContext* ctx, int first_n, const Tensor& keys,
                     const Tensor& values, const std::vector<Tensor>& others);

  Status CheckKvVariableDataTypes(DataType key_dtype, DataType value_dtype) {
    if (this->key_dtype() != key_dtype || this->value_dtype() != value_dtype) {
      // return ::tensorflow::errors::InvalidArgument(
      //     "Conflicting key/value dtypes ", key_dtype, "->", value_dtype,
      //     " with ", this->key_dtype(), "-", this->value_dtype());
    }
    return ::tensorflow::OkStatus();
  }

  Status CheckKvVariableKeyTypes(DataType key_dtype) {
    if (this->key_dtype() != key_dtype) {
      // return ::tensorflow::errors::InvalidArgument(
      //     "Conflicting key dtypes ", key_dtype, " with ", this->key_dtype());
    }
    return ::tensorflow::OkStatus();
  }

  Status CheckKvVariableValueShape(const Tensor& values) {
    TensorShape expected_value_shape = this->value_shape();
    TensorShape value_shape = values.shape();
    value_shape.RemoveDim(0);
    // if (value_shape != expected_value_shape) {
    //   return ::tensorflow::errors::InvalidArgument(
    //       "Expected shape ", expected_value_shape.DebugString(), " got ",
    //       values.shape().DebugString(), "for KvVariable " + name());
    // }
    return ::tensorflow::OkStatus();
  }

  void RefreshAllUnderThresholds(bool enable_cutoff, float cutoff_value,
                                 TableManager<K, V>* table_handler) {
    // Using default threshold, but users can change the values leads to cutoff
    // failed. We should check the values whether equals the export-op attribute
    // values.
    bool need_refresh_under_threshold_ =
        (enable_cutoff != DEFAULT_ENABLE_CUTOFF ||
         cutoff_value != DEFAULT_CUTOFF_VALUE);
    if (need_refresh_under_threshold_) {
      auto refresh_threshold = [this, &enable_cutoff, &cutoff_value](
                                   const K& key, const EVContext<V>* context) {
        if (context->Meta()->GetStorageType() == StorageType::MEM_STORAGE) {
          UpdateUnderThreshold(context, enable_cutoff, cutoff_value);
        }
      };
      table_handler->ForEachUnsafe(refresh_threshold);
    }
  }

  bool HasMemTable() const { return has_mem_table_; }
};

};  // namespace tfplus

#include "tfplus/kv_variable/kernels/dynamic_restore.hpp"
#include "tfplus/kv_variable/kernels/dynamic_save.hpp"
#endif  // TFPLUS_KV_VARIABLE_KERNELS_KV_VARIABLE_H_
