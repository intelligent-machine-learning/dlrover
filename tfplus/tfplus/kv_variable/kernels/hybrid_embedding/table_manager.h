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

#ifndef TFPLUS_KV_VARIABLE_KERNELS_HYBRID_EMBEDDING_TABLE_MANAGER_H_
#define TFPLUS_KV_VARIABLE_KERNELS_HYBRID_EMBEDDING_TABLE_MANAGER_H_

#include <algorithm>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "tbb/concurrent_queue.h"
#include "tbb/concurrent_unordered_set.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/env.h"
#include "tfplus/kv_variable/kernels/embedding_value.h"
#include "tfplus/kv_variable/kernels/hashmap.h"
#include "tfplus/kv_variable/kernels/hybrid_embedding/embedding_context.h"
#include "tfplus/kv_variable/kernels/hybrid_embedding/storage_config.pb.h"
#include "tfplus/kv_variable/kernels/hybrid_embedding/storage_table.h"
namespace tfplus {
extern GlobalConfigs gConf;
using ::tensorflow::DataType;
using ::tensorflow::mutex;
using tensorflow::mutex_lock;
using ::tensorflow::OpKernelContext;
using ::tensorflow::Status;
using ::tensorflow::string;
using ::tensorflow::Tensor;
using ::tensorflow::TensorShape;
template <typename K, typename V>
class TableManager {
 public:
  using KvMap = IMap<K, EmbeddingValue<V>>;
  using MatrixChip =
      ::Eigen::TensorChippingOp<0l,
                                typename ::tensorflow::TTypes<V, 2>::Matrix>;

  TableManager(StorageOption storage_option, const std::string& variable_name,
               size_t embedding_dim,
               tbb::concurrent_unordered_set<K>* train_delta_list_ptr = nullptr,
               bool auto_evict = true, bool with_ev_table = true)
      : variable_name_(variable_name),
        embedding_dim_(embedding_dim),
        buffer_size_(embedding_dim * sizeof(V)),
        storage_option_(storage_option) {
    ev_table_ =
        MapFactory<K, EmbeddingValue<V>>::CreateMap(MapType(gConf.map_type));
    train_delta_list_ptr_ = train_delta_list_ptr;
    // init zero value
    zero_val_ = reinterpret_cast<V*>(::tensorflow::cpu_allocator()->AllocateRaw(
        ::tensorflow::Allocator::kAllocatorAlignment, buffer_size_));
    typename ::tensorflow::TTypes<V>::Tensor src(zero_val_, embedding_dim_);
    src.setZero();
    auto table = new MemStorageTable<K, V>(-1, embedding_dim_,
                                      ev_table_, variable_name);
    writeable_storage_table_ = table;
    storage_tables_.push_back(table);
  }

  ~TableManager() {
    delete ev_table_;
  }
  KvMap* GetKVMap() { return ev_table_; }

  bool HasMemTable() { return with_ev_table_; }

  inline bool SSDStorageEneabled() {
    return false;
  }

  inline bool MayNeedDeltaInfo() { return need_delta_info_; }

  inline StorageType GetLowestWriteableStorageType() {
    return writeable_storage_table_->GetStorageType();
  }

  void InsertWithFnUnsafe(
      const K& key, std::function<void(EVContext<V>* context)> insert_func,
      EVContext<V>* context) {
    EmbeddingValue<V> ev(nullptr, false, 1, false,
                         GetLowestWriteableStorageType());
    context->UpdateMeta(&ev);
    insert_func(context);
    if (context->Value()) {
      writeable_storage_table_->Put(key, context, true);
    }
    auto it = ev_table_->InsertOrAssignUnsafe(key, std::move(ev));
    context->UpdateMeta(it.first);
  }

  void InsertWithFn(const K& key,
                    std::function<void(EVContext<V>* context)> insert_func) {
    auto lock = GetScopedKeyLock(key, LockType::WRITE_LOCK);
    EVContext<V> context;
    InsertWithFnUnsafe(key, insert_func, &context);
  }

  Status BatchGetWithFn(
      OpKernelContext* ctx, const Tensor& keys,
      std::function<void(const K& key, EVContext<V>* context, size_t row)>
          find_func,
      std::function<void(const K& key, EVContext<V>* context, size_t row)>
          not_find_func) {
    // std::vector<ph::client::SparseTableGetResult> resultVec;
    std::promise<bool> barrier;
    std::future<bool> stat_future = barrier.get_future();
    const auto& keys_flat = keys.flat<K>();
    int keys_size = keys_flat.size();
    Status status = tensorflow::OkStatus();

    std::vector<std::pair<K, size_t>> local_keys;
    std::vector<std::pair<K, size_t>> non_local_keys;
    ClassifyLocalOrRemoteKey(keys, &local_keys, &non_local_keys);

    auto DoWork = [this, &local_keys, &find_func, &not_find_func](int64 start,
                                                                  int64 end) {
      for (int64 index = start; index < end; ++index) {
        auto& key_row = local_keys[index];
        auto& key = key_row.first;
        EVContext<V> context;
        auto lock = GetScopedKeyLock(key, LockType::READ_LOCK);
        GetMetaAndValue(key, &context, true);
        if (context.IsValid()) {
          find_func(key, &context, key_row.second);
        } else {
          not_find_func(key, &context, key_row.second);
        }
      }
    };

    if (ctx != nullptr) {
      auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
      ::tensorflow::Shard(worker_threads.num_threads, worker_threads.workers,
                          local_keys.size(), 5000, DoWork);
    } else {
      DoWork(0, local_keys.size());
    }

    return status;
  }

  void ClassifyLocalOrRemoteKey(
      const Tensor& keys,
      std::vector<std::pair<K, size_t>>* local_keys,
      std::vector<std::pair<K, size_t>>* non_local_keys) {
    const auto& keys_flat = keys.flat<K>();
    for (int index = 0; index < keys_flat.size(); index++) {
      const auto& key = keys_flat(index);
      local_keys->push_back({key, index});
    }
  }

  void FindOrInsertWithDifferentFn(
      const K& key, std::function<void(EVContext<V>* context)> find_func,
      std::function<void(EVContext<V>* context)> insert_func,
      EVContext<V>* context) {
    // locks
    auto lock = GetScopedKeyLock(key, LockType::READ_LOCK);
    GetMetaAndValue(key, context);
    if (context->IsValid()) {
      find_func(context);
    } else {
      if (lock.upgrade_to_writer()) {
        // NOTE: holding the lock all the time
        InsertWithFnUnsafe(key, insert_func, context);
      } else {
        // NOTE: Failed to acquire, released and reacquire the lock
        GetMetaAndValue(key, context, true);
        if (context->IsValid()) {
          find_func(context);
        } else {
          InsertWithFnUnsafe(key, insert_func, context);
        }
      }
    }
  }

  bool FindOrInsertWithFnUnsafe(
      const K& key, std::function<void(EVContext<V>* context)> insert_func,
      EVContext<V>* context) {
    // TODO(jianmu.scj): Consider optimizing to upgrade from read locks to write
    // locks
    GetMetaAndValue(key, context);
    if (context->IsValid()) {
      return true;
    } else {
      InsertWithFnUnsafe(key, insert_func, context);
      return false;
    }
  }

  EmbeddingValue<V>* FindOrNull(const K& key) {
    return ev_table_->FindOrNull(key);
  }

  void GetMetaAndValue(const K& key, EVContext<V>* context,
                       bool local_only = false) {
    if (record_enabled_) {
      ++request_count_;
    }

    if (with_ev_table_) {
      EmbeddingValue<V>* ev = context->Meta();
      if (!ev) {
        ev = ev_table_->FindOrNullUnsafe(key);
        context->UpdateMeta(ev);
      }
      if (ev) {
        auto storage_type = ev->GetStorageType();
        if (ev->InBlacklist()) {
          // assign a temporary zero embedding
          context->InitValue(zero_val_, false);
        } else {
          auto storage = GetStorageWithType(storage_type);
          storage->Get(key, context);
        }
        context->SetValid(true);
      }
    } else {
      storage_tables_[0]->Get(key, context);
      context->SetValid(true);
    }
  }

  bool FindWithFn(const K& key, std::function<void(EVContext<V>* context)> func,
                  EVContext<V>* context) {
    {
      auto lock = GetScopedKeyLock(key, LockType::READ_LOCK);
      GetMetaAndValue(key, context);
      if (context->IsValid()) {
        func(context);
      }
    }
    if (!context->Value()) {
      return false;
    }
    return true;
  }

  void FindWithFnUnsafe(const K& key,
                        std::function<void(EVContext<V>* context)> func,
                        EVContext<V>* context) {
    GetMetaAndValue(key, context);
    if (context->IsValid()) {
      func(context);
    }
  }

  StorageTableInterface<K, V>* GetStorageWithType(StorageType type) {
    return storage_tables_[static_cast<uint8_t>(type)];
  }

  bool UpdateWithFn(const K& key,
                    std::function<void(EVContext<V>* context)> func) {
    auto lock = GetScopedKeyLock(key, LockType::WRITE_LOCK);
    return UpdateWithFnUnsafe(key, func);
  }

  bool UpdateWithFnUnsafe(const K& key,
                          std::function<void(EVContext<V>* context)> func) {
    auto ev = ev_table_->FindOrNullUnsafe(key);
    if (!ev) {
      return false;
    }
    EVContext<V> context(ev);
    return UpdateWithFnUnsafe(key, func, &context);
  }

  bool UpdateWithFn(const K& key,
                    std::function<void(EVContext<V>* context)> func,
                    EVContext<V>* context) {
    auto lock = GetScopedKeyLock(key, LockType::WRITE_LOCK);
    return UpdateWithFnUnsafe(key, func, context);
  }

  bool UpdateWithFnUnsafe(const K& key,
                          std::function<void(EVContext<V>* context)> func,
                          EVContext<V>* context) {
    func(context);
    if (context->Value()) {
      auto storage_type = context->Meta()->GetStorageType();
      auto storage = GetStorageWithType(storage_type);
      storage->Put(key, context);
    }
    return true;
  }

  bool FetchAndUpdateWithFn(const K& key,
                            std::function<void(EVContext<V>* context)> func) {
    auto lock = GetScopedKeyLock(key, LockType::WRITE_LOCK);
    // EmbeddingValue<V> ev(nullptr, false, 1, false);
    EVContext<V> context;
    GetMetaAndValue(key, &context);
    if (!context.IsValid()) {
      return false;
    }
    func(&context);
    auto storage_type = context.Meta()->GetStorageType();
    auto key_storage = GetStorageWithType(storage_type);
    key_storage->Put(key, &context);
    return true;
  }

  void InsertToGivenStorageIndexWithFn(
      const K& key, int storage_index, int64 storage_size,
      std::function<void(EVContext<V>* context)> insert_func) {
    if (storage_index > storage_tables_.size() || storage_index < 0) {
      LOG(FATAL) << "Invalid storage size while insert table " << variable_name_
                 << " with storage index: " << storage_index;
    }
    auto lock = GetScopedKeyLock(key, LockType::WRITE_LOCK);
    EmbeddingValue<V> ev(nullptr, false, 1, false,
                         storage_tables_[storage_index]->GetStorageType());
    EVContext<V> context(&ev);
    insert_func(&context);
    storage_tables_[storage_index]->Put(key, &context, true);
    auto it = ev_table_->InsertOrAssignUnsafe(key, std::move(ev));
    context.UpdateMeta(it.first);
  }

  void MarkBlacklistUnsafe(const K& key, EVContext<V>* context) {
    EmbeddingValue<V>* ev = nullptr;
    if (context) {
      ev = context->Meta();
    }
    if (!ev) {
      ev = ev_table_->FindOrNullUnsafe(key);
    }
    if (!ev) {
      EmbeddingValue<V> new_ev(nullptr, true, 1, false,
                               GetLowestWriteableStorageType());
      ev_table_->InsertOrAssignUnsafe(key, std::move(new_ev));
    } else if (!ev->InBlacklist()) {
      ev->MarkBlacklist();
      ev->SetUnderThreshold(true);
      auto storage_type = ev->GetStorageType();
      auto key_storage = GetStorageWithType(storage_type);
      key_storage->Evict(key);
      if (context != nullptr) {
        context->InitValue(zero_val_, false);
      }
    }
  }

  void RemoveBlacklistUnsafe(const K& key, EVContext<V>* context) {
    // It must be running in a write lock scope
    // Only memory type needs reallocation to replace the temporary buffer
    V* val = reinterpret_cast<V*>(::tensorflow::cpu_allocator()->AllocateRaw(
        ::tensorflow::Allocator::kAllocatorAlignment, buffer_size_));
    typename ::tensorflow::TTypes<V>::Tensor src(val, embedding_dim_);
    src.setZero();
    context->UpdateValue(val, true, buffer_size_);
    auto storage_type = context->Meta()->GetStorageType();
    auto key_storage = GetStorageWithType(storage_type);
    key_storage->Put(key, context);
    context->Meta()->RemoveBlacklist();
    context->Meta()->SetUnderThreshold(true);
  }

  size_t size() const { return ev_table_->size(); }

  size_t size_unsafe() const { return ev_table_->size_unsafe(); }

  void clear() {
    ev_table_->clear();
    for (auto storage : storage_tables_) storage->Clear();
  }

  bool erase(const K& key) { return ev_table_->erase(key); }

  std::string RemoteStorageTableName() const {
    return "";
  }

  bool RemoteStorageEnabled() {
    return false;
  }

  void LockAll() { ev_table_->LockAll(); }

  void ReleaseAll() { ev_table_->ReleaseAll(); }

  bool NeedExplicitLock() {
    if (with_ev_table_) {
      return ev_table_->NeedExplicitLock();
    } else {
      return false;
    }
  }

  void DeleteKey(const K& key) {
    auto lock = GetScopedKeyLock(key, LockType::WRITE_LOCK);
    auto ev = ev_table_->FindOrNullUnsafe(key);
    if (!ev) {
      return;
    }
    EVContext<V> context(ev);
    auto storage_type = ev->GetStorageType();
    auto key_storage = GetStorageWithType(storage_type);
    key_storage->Evict(key);
    ev_table_->erase_unsafe(key);
  }

  // If ForEach need ssd value, please use FetchAndForEachUnsafe
  void ForEach(
      std::function<void(const K& key, const EVContext<V>* context)> func) {
    auto for_each_func_with_context = [this, func](const K& key,
                                                   const EmbeddingValue<V>* v) {
      EVContext<V> context(const_cast<EmbeddingValue<V>*>(v));
      func(key, &context);
    };
    ev_table_->ForEach(for_each_func_with_context);
  }

  void ForEachUnsafe(
      std::function<void(const K& key, const EVContext<V>* context)> func) {
    auto for_each_func_with_context = [this, func](const K& key,
                                                   const EmbeddingValue<V>* v) {
      EVContext<V> context(const_cast<EmbeddingValue<V>*>(v));
      func(key, &context);
    };
    ev_table_->ForEachUnsafe(for_each_func_with_context);
  }

  Status ApplySnapshot() { return ::tensorflow::OkStatus(); }

  Status PrepareSnapshots(const std::vector<std::string>& src_files) {
    return ::tensorflow::OkStatus();
  }

  Status ExportSnapshot(tensorflow::BundleWriter* writer,
                        const TensorShape& value_shape,
                        const std::string& tensor_key) {
    return ::tensorflow::OkStatus();
  }

  Status TriggerTransferSSD(bool with_lock) {
    return ::tensorflow::OkStatus();
  }

  Status InitRemoteTable(std::string remote_table_name) {
    return ::tensorflow::OkStatus();
  }

  void CopyToWritableStorage(const K& key, EVContext<V>* context) {
    return;
  }

  void MayTransferToMEMSSD() {}

  void DisableRecordRequest() { record_enabled_ = false; }

  void EnableRecordRequest() { record_enabled_ = true; }

  const std::vector<StorageTableInterface<K, V>*>* GetStorageTables() {
    return &storage_tables_;
  }

  std::vector<size_t> CountStorageSize() {
    mutex_lock l(mu_);
    std::vector<size_t> storage_size;
    storage_size.push_back(ev_table_->size_unsafe());
    return storage_size;
  }

  mutex* mu() { return &mu_; }

  ScopedSpinLock GetScopedKeyLock(const K& key, LockType lock_type) {
    return ev_table_->GetScopedKeyLock(key, lock_type);
  }

  class ScopedLock {
   public:
    ScopedLock() = delete;
    explicit ScopedLock(TableManager<K, V>* table) : table_(table) {
      if (table_ != nullptr) {
        table_->LockAll();
      }
    }
    ~ScopedLock() {
      if (table_ != nullptr) {
        table_->ReleaseAll();
      }
    }

   private:
    TableManager<K, V>* table_;
  };

  class ScopedDisableRecordRequest {
   public:
    ScopedDisableRecordRequest() = delete;
    explicit ScopedDisableRecordRequest(TableManager<K, V>* table)
        : table_(table) {
      table_->DisableRecordRequest();
      for (auto& storage_table : *table_->GetStorageTables()) {
        storage_table->StopRecordRequest();
      }
    }
    ~ScopedDisableRecordRequest() {
      for (auto& storage_table : *table_->GetStorageTables()) {
        storage_table->StartRecordRequest();
      }
      table_->EnableRecordRequest();
    }

   private:
    TableManager<K, V>* table_;
  };

 private:
  KvMap* ev_table_;
  StorageOption storage_option_;
  std::vector<StorageTableInterface<K, V>*> storage_tables_;
  StorageTableInterface<K, V>* writeable_storage_table_;
  size_t embedding_dim_;
  size_t buffer_size_;
  std::string variable_name_;
  std::atomic<size_t> request_count_{0};
  bool with_ev_table_ = true;
  bool need_delta_info_ = false;
  std::atomic<bool> record_enabled_;
  tbb::concurrent_unordered_set<K>* train_delta_list_ptr_ = nullptr;

  tensorflow::Thread* eviction_thread_ = nullptr;
  tensorflow::condition_variable shutdown_cv_;
  mutex mu_;
  bool shutdown_ TF_GUARDED_BY(mu_) = false;
  V* zero_val_;
};

}  // namespace tfplus
#endif  // TFPLUS_KV_VARIABLE_KERNELS_HYBRID_EMBEDDING_TABLE_MANAGER_H_
