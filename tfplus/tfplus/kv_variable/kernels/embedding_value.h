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

#ifndef TFPLUS_KV_VARIABLE_KERNELS_EMBEDDING_VALUE_H_
#define TFPLUS_KV_VARIABLE_KERNELS_EMBEDDING_VALUE_H_

#include <atomic>
#include <cstring>
#include <iostream>
#include <utility>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/work_sharder.h"
#include "tfplus/kv_variable/kernels/hybrid_embedding/storage_config.pb.h"
#include "tfplus/kv_variable/kernels/kv_variable_cwise_op.h"
#include "tfplus/kv_variable/kernels/utility.h"
#include "tfplus/kv_variable/utils/utils.h"

namespace tfplus {
extern GlobalConfigs gConf;
using ::tensorflow::int64;
template <typename V>
class EmbeddingValue {
 public:
  using TensorChip = ::Eigen::Tensor<V, 1, ::Eigen::RowMajor>;
  using MatrixChip =
      ::Eigen::TensorChippingOp<0l,
                                typename ::tensorflow::TTypes<V, 2>::Matrix>;

  EmbeddingValue()
      : in_black_(false),
        buf_owner_(true),
        under_threshold_(false),
        freq_val_(1),
        embedding_val_(nullptr),
        storage_type_(0) {}
  explicit EmbeddingValue(bool buf_owner)
      : in_black_(false),
        buf_owner_(buf_owner),
        under_threshold_(false),
        freq_val_(1),
        embedding_val_(nullptr),
        storage_type_(0) {}
  explicit EmbeddingValue(V* val)
      : in_black_(false),
        buf_owner_(true),
        under_threshold_(false),
        freq_val_(1),
        embedding_val_(std::move(val)),
        storage_type_(0) {}
  EmbeddingValue(V* val, bool in_black)
      : in_black_(in_black),
        buf_owner_(true),
        under_threshold_(false),
        freq_val_(1),
        embedding_val_(std::move(val)),
        storage_type_(0) {}
  EmbeddingValue(V* val, bool in_black, uint32_t freq_val)
      : in_black_(in_black),
        buf_owner_(true),
        under_threshold_(false),
        freq_val_(freq_val),
        embedding_val_(std::move(val)),
        storage_type_(0) {}
  EmbeddingValue(V* val, bool in_black, uint32_t freq_val, bool buf_owner)
      : in_black_(in_black),
        buf_owner_(buf_owner),
        under_threshold_(false),
        freq_val_(freq_val),
        embedding_val_(std::move(val)),
        storage_type_(0) {}
  EmbeddingValue(V* val, bool in_black, uint32_t freq_val, bool buf_owner,
                 uint8_t storage_type)
      : in_black_(in_black),
        buf_owner_(buf_owner),
        under_threshold_(false),
        freq_val_(freq_val),
        embedding_val_(std::move(val)),
        storage_type_(storage_type) {}

  EmbeddingValue(const EmbeddingValue& v) = delete;
  EmbeddingValue& operator=(const EmbeddingValue&) = delete;

  EmbeddingValue& operator=(EmbeddingValue&& other) {
    if (this != &other || embedding_val_ != other.embedding_val_) {
      if (buf_owner_.load() && (embedding_val_ != nullptr)) {
        ::tensorflow::cpu_allocator()->DeallocateRaw(
            reinterpret_cast<void*>(embedding_val_));
      }
      embedding_val_ = other.embedding_val_;
      in_black_ = other.in_black_.load();
      buf_owner_ = other.buf_owner_.load();
      under_threshold_ = other.under_threshold_;
      storage_type_ = other.storage_type_;
      freq_val_ = other.freq_val_;
      other.embedding_val_ = nullptr;
    }
    return *this;
  }

  EmbeddingValue(EmbeddingValue&& other) : embedding_val_(nullptr) {
    *this = std::move(other);
  }

  ~EmbeddingValue() {
    if (CASBufferOwner(true, false) && (embedding_val_ != nullptr)) {
      ::tensorflow::cpu_allocator()->DeallocateRaw(
          reinterpret_cast<void*>(embedding_val_));
      embedding_val_ = nullptr;
    }
  }

  V* Value() { return embedding_val_; }
  V* Value() const { return embedding_val_; }


  void OutputEmbedding(MatrixChip out, int64_t num_elements) const {
    typename ::tensorflow::TTypes<V>::ConstTensor src(embedding_val_,
                                                      num_elements);
    out = src;
  }

  void OutputEmbedding(V* out, int64_t num_elements) const {
    std::memcpy(static_cast<void*>(out), static_cast<void*>(embedding_val_),
                num_elements * sizeof(V));
  }

  void UpdateEmbedding(const void* v_base) {
    V* val_ptr = (V*)(v_base);  // NOLINT
    UpdateEmbedding(val_ptr);
  }

  void UpdateEmbedding(V* val_ptr) {
    if (embedding_val_ && embedding_val_ != val_ptr && buf_owner_.load()) {
      ::tensorflow::cpu_allocator()->DeallocateRaw(
          reinterpret_cast<void*>(embedding_val_));
    }
    embedding_val_ = val_ptr;
  }

  void CwiseOpUpdateEmbedding(CwiseOperationBase<TensorChip>* func,
                              const TensorChip& rhs, int64_t num_elements) {
    typename ::tensorflow::TTypes<V>::ConstTensor lhs(embedding_val_,
                                                      num_elements);
    const auto& res = (*func)(lhs, rhs);
    std::copy_n(res.data(), num_elements, embedding_val_);
  }

  bool InBlacklist() { return in_black_; }

  bool InBlacklist() const { return in_black_; }

  bool BufOwner() const {return buf_owner_;}

  void MarkBlacklist() {
    if (in_black_) {
      return;
    }
    in_black_ = true;
  }

  void RemoveBlacklist() { in_black_ = false; }

  uint32_t GetFrequency() const { return freq_val_; }

  void UpdateFrequency(uint32_t freq) { freq_val_ = freq; }

  void AddFrequency(uint16_t freq, uint16_t last_update_time_in_days) {
    auto stat = reinterpret_cast<uint16_t*>(&freq_val_);
    stat[0] = SaturateAddFrequency(stat[0], freq);
    stat[1] = last_update_time_in_days;
  }

  bool IsUnderThreshold() const { return under_threshold_; }

  void SetUnderThreshold(bool value) { under_threshold_ = value; }

  void SetBufOwner(bool value) { buf_owner_ = value; }

  void SetValue(V* value) {
    if (CASBufferOwner(false, true)) {
      embedding_val_ = value;
    }
  }

  void DeleteValue() {
    if (CASBufferOwner(true, false) && embedding_val_) {
      ::tensorflow::cpu_allocator()->DeallocateRaw(
          reinterpret_cast<void*>(embedding_val_));
    }
    embedding_val_ = nullptr;
  }

  void SetStorage(StorageType storage_type) {
    storage_type_ = static_cast<uint8_t>(storage_type);
  }

  StorageType GetStorageType() { return StorageType(storage_type_); }

 private:
  bool CASBufferOwner(bool expected, bool desired) {
    return buf_owner_.compare_exchange_strong(expected, desired);
  }
  std::atomic<bool> in_black_;
  std::atomic<bool> buf_owner_;
  bool under_threshold_;
  uint8_t storage_type_;
  // (low bits) total update frequency for current uint16 frequency;
  // (high bits) save unix time by days unit, instead of ms.
  // then uint16 is enough. i.e 1581427427089 ms to 18303 day.
  // 65536 as limit is far enough right now.
  // uint16 last_update_time_in_days;
  uint32_t freq_val_;
  V* embedding_val_{nullptr};
};
}  // namespace tfplus
#endif  // TFPLUS_KV_VARIABLE_KERNELS_EMBEDDING_VALUE_H_
