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

#ifndef TFPLUS_KV_VARIABLE_KERNELS_HYBRID_EMBEDDING_STORAGE_TABLE_H_
#define TFPLUS_KV_VARIABLE_KERNELS_HYBRID_EMBEDDING_STORAGE_TABLE_H_

#include <future>  // NOLINT(build/c++11)
#include <memory>
#include <string>
#include <utility>
#include <vector>

// #include "phstore/client/cpp/include/Client.h"
// #include "phstore/client/cpp/include/Request.h"
// #include "phstore/phdb/include/PhDB.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/mem.h"
#include "tfplus/kv_variable/kernels/embedding_value.h"
#include "tfplus/kv_variable/kernels/hashmap.h"
#include "tfplus/kv_variable/kernels/hybrid_embedding/embedding_context.h"
#include "tfplus/kv_variable/kernels/utility.h"

namespace tfplus {
extern GlobalConfigs gConf;


constexpr int REMOTE_BATCH_GET_SIZE = 100;

template <typename K, typename V>
class StorageTableInterface {
 public:
  using KvMap = IMap<K, EmbeddingValue<V>>;
  // virtual ~StorageTableInterface() = default;
  virtual void Get(const K& key, EVContext<V>* context) = 0;
  virtual void Put(const K& key, EVContext<V>* context,
                   bool is_insert = false) = 0;

  virtual Status ApplySnapshot() { return ::tensorflow::OkStatus(); }
  virtual Status PrepareSnapshots(const std::vector<std::string>& src_files) {
    return ::tensorflow::OkStatus();
  }
  virtual void AddStorageSize() {}
  // virtual void SetBufOwner(EVContext<V>* context) = 0;
  virtual StorageType GetStorageType() = 0;
  // // Caches kv pairs in slower storage
  // // TODO(jianmu.scj) The logic of using forced permutation
  // // can then be abstracted as a separate asynchronous permutation scheme
  // virtual void CacheSlowKV(const K& key, EVContext<V>* context) = 0;
  virtual void Evict(const K& key) {}
  virtual int64_t Capacity() = 0;
  virtual int64_t Size() = 0;
  virtual int64_t RequestCount() { return 0; }
  virtual void ResetRequestCount() {}
  virtual void StartRecordRequest() {}
  virtual void StopRecordRequest() {}
  virtual void SetCapacity(size_t size) {}
  virtual bool StoageReady() { return true; }
  void Clear() { return; }
  virtual bool AutoSize() { return false; }
};

template <typename K, typename V>
class MemStorageTable : public StorageTableInterface<K, V> {
 public:
  using KvMap = IMap<K, EmbeddingValue<V>>;
  MemStorageTable(int64_t storage_size, size_t embedding_dim, KvMap* ev_table,
                  const std::string& table_name)
      : embedding_dim_(embedding_dim),
        table_name_(table_name),
        ev_table_(ev_table) {
    storage_size_ = 0;
    if (storage_size == -1) {
      storage_capacity_ = 0;
      auto_size_ = true;
    } else {
      storage_capacity_ = storage_size;
      auto_size_ = false;
    }
  }

  ~MemStorageTable() {}

  void Clear() {
    ev_table_->clear();
    storage_size_ = 0;
  }

  StorageType GetStorageType() override { return StorageType::MEM_STORAGE; }

  int64_t RequestCount() override { return request_count_; }

  void ResetRequestCount() override { request_count_ = 0; }

  void Evict(const K& key) {
    auto ev = ev_table_->FindOrNullUnsafe(key);
    if (ev) {
      ev->DeleteValue();
    }
    storage_size_--;
  }
  /*
    1. get from ev_table_
  */
  void Get(const K& key, EVContext<V>* context) override {
    auto ev = context->Meta();
    if (!ev) {
      ev = ev_table_->FindOrNullUnsafe(key);
      context->UpdateMeta(ev);
    }
    context->InitValue(ev->Value(),
                       false);  // no need to allocate or copy here
    if (record_request_) {
      request_count_++;
    }
    // VLOG(0)<<"request_count_" <<request_count_;
  }

  // TODO(jianmu.scj): You need to integrate the logic to insert ev_table_ into
  // this function
  void Put(const K& key, EVContext<V>* context,
           bool is_insert = false) override {
    if (is_insert && context->Meta()->Value() == nullptr) {
      storage_size_++;
    }
    context->Meta()->UpdateEmbedding(context->Value());
    // 如果是buffer owner，则直接赋值给ev table
    if (context->GetBufOwner()) {
      context->Meta()->SetBufOwner(true);
      context->SetBufOwner(false);
    }
  }

  // void SetBufOwner(EVContext<V>* context) {
  //   if (with_cache_) {
  //     context->SetBufOwner();
  //   } else {
  //     context->Meta()->SetValue(context->Value());
  //   }
  // }

  void LockAll() { ev_table_->LockAll(); }

  void ReleaseAll() { ev_table_->ReleaseAll(); }

  void SetCapacity(size_t size) { storage_capacity_ = size; }

  int64_t Capacity() { return storage_capacity_; }

  int64_t Size() { return storage_size_; }

  bool AutoSize() override { return auto_size_; }

  void AddStorageSize() override { storage_size_++; }

  void StartRecordRequest() override { record_request_ = true; }

  void StopRecordRequest() override { record_request_ = false; }

  // Status ExportSnapshot(BundleWriter* writer, TensorShape& value_shape,
  //                       const string& tensor_key) {}
  // void StartEvict();
  // void Clear();
  // Status ExportSnapshot(BundleWriter* writer, TensorShape& value_shape,
  //                       const string& tensor_key) {
  //                         return ::tensorflow::OkStatus();
  //                       }
  // void StartEvict() {
  //   return;
  // }
  // void Clear() {
  //   return;
  // }

 private:
  int64_t storage_capacity_;
  std::atomic<size_t> storage_size_{0};

  std::string table_name_;
  KvMap* ev_table_;
  size_t embedding_dim_;
  size_t buffer_size_;
  bool auto_size_;
  bool record_request_{true};
  std::atomic<size_t> request_count_{0};
};

}  // namespace tfplus
#endif  // TFPLUS_KV_VARIABLE_KERNELS_HYBRID_EMBEDDING_STORAGE_TABLE_H_ NOLINT
