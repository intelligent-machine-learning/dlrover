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

#ifndef TFPLUS_KV_VARIABLE_KERNELS_HYBRID_EMBEDDING_EMBEDDING_CONTEXT_H_
#define TFPLUS_KV_VARIABLE_KERNELS_HYBRID_EMBEDDING_EMBEDDING_CONTEXT_H_
#include <atomic>

#include "tfplus/kv_variable/kernels/embedding_value.h"

namespace tfplus {
template <typename V>
class EVContext {
 public:
  using MatrixChip =
      ::Eigen::TensorChippingOp<0l,
                                typename ::tensorflow::TTypes<V, 2>::Matrix>;
  EVContext()
      : meta_(nullptr),
        buf_owner_(false),
        val_(nullptr),
        status_(nullptr),
        is_valid_(false) {}

  EVContext(V* val, bool buf_owner)
      : meta_(nullptr),
        buf_owner_(buf_owner),
        val_(val),
        status_(nullptr),
        is_valid_(false) {}

  EVContext(EmbeddingValue<V>* ev, V* val)
      : meta_(ev),
        buf_owner_(false),
        val_(val),
        status_(nullptr),
        is_valid_(false) {}

  explicit EVContext(V* val)
      : meta_(nullptr),
        buf_owner_(true),
        val_(val),
        status_(nullptr),
        is_valid_(false) {}

  explicit EVContext(EmbeddingValue<V>* ev)
      : meta_(ev),
        buf_owner_(false),
        val_(ev->Value()),
        status_(nullptr),
        is_valid_(false) {}

  ~EVContext() {
    meta_ = nullptr;
    if (CASBufferOwner(true, false) && val_) {
      ::tensorflow::cpu_allocator()->DeallocateRaw(
          reinterpret_cast<void*>(val_));
      val_ = nullptr;
    }
  }

  EVContext(const EVContext& v) = delete;

  EVContext& operator=(const EVContext&) = delete;

  V* Value() const { return val_; }

  void MarkBlacklist() { meta_->MarkBlacklist(); }

  void InitValue(const V* new_val, bool buffer_owner) {
    if (CASBufferOwner(true, false) && val_) {
      ::tensorflow::cpu_allocator()->DeallocateRaw(
          reinterpret_cast<void*>(val_));
      val_ = nullptr;
    }
    val_ = const_cast<V*>(new_val);
    SetBufOwner(buffer_owner);
  }

  void UpdateValue(const V* new_val, bool input_buf_owner, size_t value_bytes) {
    if (new_val == val_) {
      return;
    }
    if (!buf_owner_.load()) {
      if (input_buf_owner) {
        // false, true -> true
        val_ = const_cast<V*>(new_val);
        buf_owner_ = input_buf_owner;
      } else {
        if (!val_) {
          // false(nullptr), false -> true
          val_ =
              reinterpret_cast<V*>(::tensorflow::cpu_allocator()->AllocateRaw(
                  ::tensorflow::Allocator::kAllocatorAlignment, value_bytes));
          buf_owner_ = true;
        }
        // false, false -> false / true
        memcpy(val_, new_val, value_bytes);
      }
    } else {
      if (input_buf_owner) {
        // true, true -> true
        if (val_) {
          ::tensorflow::cpu_allocator()->DeallocateRaw(
              reinterpret_cast<void*>(val_));
        }
        val_ = const_cast<V*>(new_val);
      } else {
        // true, false -> true
        memcpy(val_, new_val, value_bytes);
      }
    }
  }

  EmbeddingValue<V>* Meta() const { return meta_; }

  void OutputEmbeddingData(MatrixChip out, int64_t num_elements) const {
    if (val_) {
      typename ::tensorflow::TTypes<V>::ConstTensor src(val_, num_elements);
      out = src;
    } else if (meta_ && meta_->Value()) {
      typename ::tensorflow::TTypes<V>::ConstTensor src(meta_->Value(),
                                                        num_elements);
      out = src;
    }
  }

  void UpdateMeta(EmbeddingValue<V>* meta) { meta_ = meta; }

  uint16_t GetFrequency() {
    return GetUint16FromUint32(meta_->GetFrequency(), true);
  }

  void SetBufOwner(bool val) { buf_owner_ = val; }

  void DeleteValue() {
    if (CASBufferOwner(true, false) && val_) {
      ::tensorflow::cpu_allocator()->DeallocateRaw(
          reinterpret_cast<void*>(val_));
      val_ = nullptr;
    } else {
      meta_->DeleteValue();
    }
  }

  bool GetBufOwner() { return buf_owner_; }

  ::tensorflow::Status* Status() { return status_; }

  void SetStatus(::tensorflow::Status* status) { status_ = status; }

  inline void SetValid(bool valid) { is_valid_ = valid; }

  inline bool IsValid() { return is_valid_; }

 private:
  bool CASBufferOwner(bool expected, bool desired) {
    return buf_owner_.compare_exchange_strong(expected, desired);
  }
  std::atomic<bool> buf_owner_;
  bool is_valid_;
  V* val_{nullptr};
  EmbeddingValue<V>* meta_;
  ::tensorflow::Status* status_{nullptr};
};

}  // namespace tfplus
#endif  // TFPLUS_KV_VARIABLE_KERNELS_HYBRID_EMBEDDING_EMBEDDING_CONTEXT_H_"
