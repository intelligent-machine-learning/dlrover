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

#ifndef TFPLUS_KV_VARIABLE_KERNELS_HASHMAP_H_
#define TFPLUS_KV_VARIABLE_KERNELS_HASHMAP_H_

#include <cstdlib>
#include <functional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "libcuckoo/cuckoohash_map.hh"
#include "sparsehash/dense_hash_map"
#include "src/MurmurHash2.h"
#include "tensorflow/core/platform/logging.h"

#include "tensorflow/core/platform/stacktrace.h"

#include "tensorflow/core/platform/types.h"
#include "tfplus/kv_variable/kernels/mutex.h"
#include "tfplus/kv_variable/kernels/utility.h"

namespace tfplus {

enum MapType {
  UNORDERED_MAP,
  CUCKOO_HASH,
  CONCURRENT_UNORDERED_MAP,
  CONCURRENT_DENSE_HASH_MAP,
  MULTI_LEVEL_MAP
};

enum LockType { WRITE_LOCK, READ_LOCK };

using ::tensorflow::int64;

constexpr size_t HASH_SIZE_DEFAULT = 1031;
constexpr uint64_t MAGIC_SEED = 0x5446534d;

template <class T>
struct murmurhash_a : public std::hash<T> {};

template <>
struct murmurhash_a<int64> {
  std::size_t operator()(const int64& key) const {
    return MurmurHash64A(&key, sizeof(int64), MAGIC_SEED);
  }
};

template <>
struct murmurhash_a<std::string> {
  std::size_t operator()(const std::string& key) const {
    return MurmurHash64A(key.c_str(), key.size(), MAGIC_SEED);
  }
};

template <class T>
struct murmurhash_b : public std::hash<T> {};

template <>
struct murmurhash_b<int64> {
  std::size_t operator()(const int64& key) const {
    return MurmurHash64B(&key, sizeof(int64), MAGIC_SEED);
  }
};

template <>
struct murmurhash_b<std::string> {
  std::size_t operator()(const std::string& key) const {
    return MurmurHash64B(key.c_str(), key.size(), MAGIC_SEED);
  }
};

template <class K, class V>
class IMap {
 public:
  virtual ~IMap() {}
  virtual V* FindOrNull(const K& key) = 0;

  virtual V* FindOrNullUnsafe(const K& key) { return FindOrNull(key); }
  /*
  Search key, apply func(V* val), then return true, else
  return false. Func
  */
  virtual bool FindWithFn(const K& key, std::function<void(V* val)> func) = 0;

  virtual bool FindWithFnUnsafe(const K& key,
                                std::function<void(V* val)> func) {
    return FindWithFn(key, func);
  }

  virtual V* FindOrInsertWithFn(const K& key,
                                std::function<V(const K& key)> func) = 0;

  virtual bool FindOrInsertWithDifferentFn(
      const K& key, std::function<void(V* val)> find_func,
      std::function<V(const K& key)> insert_func) = 0;

  /*
   Insert the given key-value pair into the map. Returns true if and
   only if the key from the given pair doesn't previously exist. Otherwise, the
   value in the map is replaced with the value from the given pair.
   Notice that V must implement move assignment operation.
   */

  virtual bool InsertOrAssign(const K& key, V&& val) = 0;

  virtual std::pair<V*, bool> InsertOrAssignUnsafe(const K& key, V&& val) {
    bool succ = InsertOrAssign(key, std::move(val));
    V* v = FindOrNull(key);
    return {v, succ};
  }

  /*
    If find key, apply func(V* val), then return true, else
    return false.
  */
  virtual bool UpdateWithFn(const K& key, std::function<void(V* val)> func) = 0;
  virtual size_t size() const = 0;
  virtual size_t size_unsafe() const = 0;
  virtual void clear() = 0;
  virtual bool erase(const K& key) = 0;
  virtual bool erase_unsafe(const K& key) { return erase(key); }
  virtual void ForEach(
      std::function<void(const K& key, const V* val)> func) = 0;
  virtual void ForEachUnsafe(
      std::function<void(const K& key, const V* val)> func) = 0;
  virtual bool NeedExplicitLock() = 0;

  virtual void LockAll() = 0;
  virtual void ReleaseAll() = 0;
  virtual spin_rw_mutex* LockKey(const K& key, LockType lock_type) {
    return nullptr;
  }

  virtual ScopedSpinLock GetScopedKeyLock(const K& key, LockType lock_type) {
    return ScopedSpinLock();
  }

  class ScopedLock {
   public:
    ScopedLock() = delete;
    explicit ScopedLock(IMap* table) : table_(table) {
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
    IMap* table_;
  };
};

template <class K, class V>
class UnorderedMap : public IMap<K, V> {
 public:
  V* FindOrNull(const K& key) override {
    auto it = table_.find(key);
    if (it == table_.end()) {
      return nullptr;
    }
    return &it->second;
  }

  V* FindOrNullUnsafe(const K& key) override { return FindOrNull(key); }

  bool FindWithFn(const K& key, std::function<void(V* val)> func) override {
    V* val = FindOrNull(key);
    if (val == nullptr) {
      return false;
    }
    func(val);
    return true;
  }

  bool InsertOrAssign(const K& key, V&& val) override {
    table_[key] = std::move(val);
    return true;
  }

  bool UpdateWithFn(const K& key, std::function<void(V* val)> func) override {
    V* val = FindOrNull(key);
    if (val == nullptr) {
      return false;
    }
    func(val);
    return true;
  }

  size_t size() const override { return table_.size(); }

  size_t size_unsafe() const override { return table_.size(); }

  void clear() override { table_.clear(); }

  bool erase(const K& key) override { return table_.erase(key) != 0; }

  void ForEach(std::function<void(const K& key, const V* val)> func) override {
    for (auto it = table_.begin(); it != table_.end(); ++it) {
      func(it->first, &it->second);
    }
  }

  void ForEachUnsafe(
      std::function<void(const K& key, const V* val)> func) override {
    ForEach(func);
  }

  V* FindOrInsertWithFn(const K& key, std::function<V(const K& key)> func) {
    V* val = FindOrNull(key);
    if (val == nullptr) {
      V new_val = func(key);
      table_[key] = std::move(new_val);
      val = FindOrNull(key);
    }
    return val;
  }

  bool FindOrInsertWithDifferentFn(const K& key,
                                   std::function<void(V* val)> find_func,
                                   std::function<V(const K& key)> insert_func) {
    V* val = FindOrNull(key);
    if (val == nullptr) {
      V new_val = insert_func(key);
      table_[key] = std::move(new_val);
      return false;
    }

    find_func(val);
    return true;
  }

  bool NeedExplicitLock() override { return true; }

  void LockAll() {}

  void ReleaseAll() {}

 private:
  std::unordered_map<K, V, murmurhash_a<K>> table_;
};

template <class K, class V>
class CuckooHashMap : public IMap<K, V> {
 public:
  V* FindOrNull(const K& key) override { return table_.find_or_null(key); }

  V* FindOrNullUnsafe(const K& key) override { return FindOrNull(key); }

  bool FindWithFn(const K& key, std::function<void(V* val)> func) override {
    return table_.find_fn_ptr(key, func);
  }

  V* FindOrInsertWithFn(const K& key, std::function<V(const K& key)> func) {
    return table_.find_or_insert_fn(key, func);
  }

  bool FindOrInsertWithDifferentFn(const K& key,
                                   std::function<void(V* val)> find_func,
                                   std::function<V(const K& key)> insert_func) {
    return table_.find_or_insert_fn(key, find_func, insert_func);
  }

  bool InsertOrAssign(const K& key, V&& val) override {
    return table_.insert_or_update(key, std::move(val));
    // return true;
  }

  bool UpdateWithFn(const K& key, std::function<void(V* val)> func) override {
    return table_.update_fn_ptr(key, func);
  }

  size_t size() const override { return table_.size(); }

  size_t size_unsafe() const override { return table_.size(); }

  void clear() override { table_.clear(); }

  bool erase(const K& key) override { return table_.erase(key); }

  void ForEach(std::function<void(const K& key, const V* val)> func) override {
    auto lt = table_.lock_table();
    for (const auto& iter : lt) {
      func(iter.first, &iter.second);
    }
  }

  void ForEachUnsafe(
      std::function<void(const K& key, const V* val)> func) override {
    auto& lt = locked_tables_[0];
    for (const auto& iter : lt) {
      func(iter.first, &iter.second);
    }
  }

  bool NeedExplicitLock() override { return false; }

  void LockAll() {
    mu_.lock();
    auto lt = table_.lock_table();
    locked_tables_.emplace_back(std::move(lt));
  }

  void ReleaseAll() {
    locked_tables_.clear();
    mu_.unlock();
  }

 private:
  libcuckoo::cuckoohash_map<K, V, murmurhash_a<K>> table_;
  std::vector<
      typename libcuckoo::cuckoohash_map<K, V, murmurhash_a<K>>::locked_table>
      locked_tables_;
  mutable spin_rw_mutex mu_;
};

template <class K, class V, class F = murmurhash_b<K>>
class ConcurrentUnorderedMap : public IMap<K, V> {
 public:
  /*hash table*/
  typedef typename std::unordered_map<K, V, murmurhash_a<K>> hash_segment;

  V* FindOrNull(const K& key) override {
    size_t segment_id = hash_id(key);
    tfplus_shared_spin_lock r_lock(table_[segment_id].mu);
    auto it = table_[segment_id].map.find(key);
    if (it == table_[segment_id].map.end()) {
      return nullptr;
    }
    return &it->second;
  }

  V* FindOrNullUnsafe(const K& key) override {
    size_t segment_id = hash_id(key);
    auto it = table_[segment_id].map.find(key);
    if (it == table_[segment_id].map.end()) {
      return nullptr;
    }
    return &it->second;
  }

  V* FindOrInsertWithFn(const K& key, std::function<V(const K& key)> func) {
    size_t segment_id = hash_id(key);
    tfplus_spin_lock w_lock(table_[segment_id].mu);

    auto it = table_[segment_id].map.find(key);
    if (it == table_[segment_id].map.end()) {
      V new_val = func(key);
      table_[segment_id].map[key] = std::move(new_val);
      V* val = &table_[segment_id].map.find(key)->second;
      return val;
    }

    return &it->second;
  }

  bool FindOrInsertWithDifferentFn(const K& key,
                                   std::function<void(V* val)> find_func,
                                   std::function<V(const K& key)> insert_func) {
    size_t segment_id = hash_id(key);
    tfplus_spin_lock w_lock(table_[segment_id].mu);

    auto it = table_[segment_id].map.find(key);
    if (it == table_[segment_id].map.end()) {
      table_[segment_id].map[key] = std::move(insert_func(key));
      return false;
    }

    find_func(&it->second);
    return true;
  }

  bool FindWithFn(const K& key, std::function<void(V* val)> func) override {
    V* val = nullptr;
    size_t segment_id = hash_id(key);
    tfplus_shared_spin_lock r_lock(table_[segment_id].mu);
    auto it = table_[segment_id].map.find(key);
    if (it != table_[segment_id].map.end()) {
      val = &it->second;
      func(val);
      return true;
    }
    return false;
  }

  bool FindWithFnUnsafe(const K& key,
                        std::function<void(V* val)> func) override {
    V* val = nullptr;
    size_t segment_id = hash_id(key);
    auto it = table_[segment_id].map.find(key);
    if (it != table_[segment_id].map.end()) {
      val = &it->second;
      func(val);
      return true;
    }
    return false;
  }

  bool InsertOrAssign(const K& key, V&& val) override {
    size_t segment_id = hash_id(key);
    tfplus_spin_lock w_lock(table_[segment_id].mu);
    table_[segment_id].map[key] = std::move(val);
    return true;
  }

  std::pair<V*, bool> InsertOrAssignUnsafe(const K& key,
                                                   V&& val) override {
    size_t segment_id = hash_id(key);
    auto it = table_[segment_id].map.insert_or_assign(key, std::move(val));
    return {&it.first->second, it.second};
    // return {nullptr, false};
  }

  bool UpdateWithFn(const K& key, std::function<void(V* val)> func) override {
    V* val = nullptr;
    size_t segment_id = hash_id(key);
    tfplus_spin_lock w_lock(table_[segment_id].mu);
    auto it = table_[segment_id].map.find(key);
    if (it != table_[segment_id].map.end()) {
      val = &it->second;
      func(val);
      return true;
    }
    return false;
  }

  size_t size() const override {
    size_t total_size = 0;
    for (size_t segment_id = 0; segment_id < HASH_SIZE_DEFAULT; segment_id++) {
      tfplus_shared_spin_lock r_lock(table_[segment_id].mu);
      total_size += table_[segment_id].map.size();
    }
    return total_size;
  }

  size_t size_unsafe() const override {
    size_t total_size = 0;
    for (size_t segment_id = 0; segment_id < HASH_SIZE_DEFAULT; segment_id++) {
      total_size += table_[segment_id].map.size();
    }
    return total_size;
  }

  void clear() override {
    for (size_t segment_id = 0; segment_id < HASH_SIZE_DEFAULT; segment_id++) {
      auto& mu = table_[segment_id].mu;
      mu.lock();
      table_[segment_id].map.clear();
      mu.unlock();
    }
  }

  bool erase(const K& key) override {
    size_t segment_id = hash_id(key);
    tfplus_spin_lock w_lock(table_[segment_id].mu);
    return table_[segment_id].map.erase(key) != 0;
  }

  bool erase_unsafe(const K& key) override {
    size_t segment_id = hash_id(key);
    return table_[segment_id].map.erase(key) != 0;
  }

  void ForEach(std::function<void(const K& key, const V* val)> func) override {
    for (size_t segment_id = 0; segment_id < HASH_SIZE_DEFAULT; segment_id++) {
      auto& seg = table_[segment_id];
      tfplus_spin_lock w_lock(seg.mu);
      for (auto it = seg.map.begin(); it != seg.map.end(); ++it) {
        func(it->first, &it->second);
      }
    }
  }

  void ForEachUnsafe(
      std::function<void(const K& key, const V* val)> func) override {
    for (size_t segment_id = 0; segment_id < HASH_SIZE_DEFAULT; segment_id++) {
      auto& seg = table_[segment_id];
      for (auto it = seg.map.begin(); it != seg.map.end(); ++it) {
        func(it->first, &it->second);
      }
    }
  }

  bool NeedExplicitLock() override { return false; }

  void LockAll() {
    for (size_t segment_id = 0; segment_id < HASH_SIZE_DEFAULT; segment_id++) {
      table_[segment_id].mu.lock();
    }
  }

  void ReleaseAll() {
    for (size_t segment_id = 0; segment_id < HASH_SIZE_DEFAULT; segment_id++) {
      table_[segment_id].mu.unlock();
    }
  }

  spin_rw_mutex* LockKey(const K& key, LockType lock_type) override {
    size_t segment_id = hash_id(key);
    auto& mu = table_[segment_id].mu;
    if (lock_type == LockType::READ_LOCK) {
      mu.lock_read();
    } else {
      mu.lock();
    }
    return &mu;
  }

  ScopedSpinLock GetScopedKeyLock(const K& key, LockType lock_type) override {
    size_t segment_id = hash_id(key);
    return ScopedSpinLock(table_[segment_id].mu,
                          lock_type == LockType::WRITE_LOCK);
  }

 private:
  size_t hash_id(const K& key) { return hash_fn_(key) % HASH_SIZE_DEFAULT; }

  struct concurrent_hash_map {
    hash_segment map;
    mutable spin_rw_mutex mu;
  };

  concurrent_hash_map table_[HASH_SIZE_DEFAULT];
  F hash_fn_;
};

template <class K>
struct dense_hash_sepecial_key {
  K empty_key() { return -1; }

  K deleted_key() { return -2; }
};

template <>
struct dense_hash_sepecial_key<std::basic_string<char>> {
  std::basic_string<char> empty_key() { return "\x01"; }

  std::basic_string<char> deleted_key() { return "\x02"; }
};

template <class K, class V, class F = murmurhash_b<K>>
class ConcurrentDenseHashMap : public IMap<K, V> {
 public:
  ConcurrentDenseHashMap() {
    for (size_t i = 0; i < HASH_SIZE_DEFAULT; i++) {
      table_[i].map.max_load_factor(0.8);
      dense_hash_sepecial_key<K> keys;
      table_[i].map.set_empty_key(keys.empty_key());
      table_[i].map.set_deleted_key(keys.deleted_key());
    }
  }

  V* FindOrNull(const K& key) override {
    size_t segment_id = hash_id(key);
    tfplus_shared_spin_lock r_lock(table_[segment_id].mu);
    auto it = table_[segment_id].map.find(key);
    if (it == table_[segment_id].map.end()) {
      return nullptr;
    }
    return &it->second;
  }

  V* FindOrNullUnsafe(const K& key) override {
    size_t segment_id = hash_id(key);
    auto it = table_[segment_id].map.find(key);
    if (it == table_[segment_id].map.end()) {
      return nullptr;
    }
    return &it->second;
  }

  V* FindOrInsertWithFn(const K& key, std::function<V(const K& key)> func) {
    size_t segment_id = hash_id(key);
    tfplus_spin_lock w_lock(table_[segment_id].mu);

    auto it = table_[segment_id].map.find(key);
    if (it == table_[segment_id].map.end()) {
      V new_val = func(key);
      table_[segment_id].map[key] = std::move(new_val);
      V* val = &table_[segment_id].map.find(key)->second;
      return val;
    }

    return &it->second;
  }

  bool FindOrInsertWithDifferentFn(const K& key,
                                   std::function<void(V* val)> find_func,
                                   std::function<V(const K& key)> insert_func) {
    size_t segment_id = hash_id(key);
    tfplus_spin_lock w_lock(table_[segment_id].mu);

    auto it = table_[segment_id].map.find(key);
    if (it == table_[segment_id].map.end()) {
      table_[segment_id].map[key] = std::move(insert_func(key));
      return false;
    }

    find_func(&it->second);
    return true;
  }

  bool FindWithFn(const K& key, std::function<void(V* val)> func) override {
    V* val = nullptr;
    size_t segment_id = hash_id(key);
    tfplus_shared_spin_lock r_lock(table_[segment_id].mu);
    auto it = table_[segment_id].map.find(key);
    if (it != table_[segment_id].map.end()) {
      val = &it->second;
      func(val);
      return true;
    }
    return false;
  }

  bool FindWithFnUnsafe(const K& key,
                        std::function<void(V* val)> func) override {
    V* val = nullptr;
    size_t segment_id = hash_id(key);
    auto it = table_[segment_id].map.find(key);
    if (it != table_[segment_id].map.end()) {
      val = &it->second;
      func(val);
      return true;
    }
    return false;
  }

  bool InsertOrAssign(const K& key, V&& val) override {
    size_t segment_id = hash_id(key);
    tfplus_spin_lock w_lock(table_[segment_id].mu);
    table_[segment_id].map[key] = std::move(val);
    return true;
  }

  std::pair<V*, bool> InsertOrAssignUnsafe(const K& key,
                                                   V&& val) override {
    size_t segment_id = hash_id(key);
    table_[segment_id].map[key] = std::move(val);
    V* v = FindOrNull(key);
    return {v, true};
  }

  bool UpdateWithFn(const K& key, std::function<void(V* val)> func) override {
    V* val = nullptr;
    size_t segment_id = hash_id(key);
    tfplus_spin_lock w_lock(table_[segment_id].mu);
    auto it = table_[segment_id].map.find(key);
    if (it != table_[segment_id].map.end()) {
      val = &it->second;
      func(val);
      return true;
    }
    return false;
  }

  size_t size() const override {
    size_t total_size = 0;
    for (size_t segment_id = 0; segment_id < HASH_SIZE_DEFAULT; segment_id++) {
      tfplus_shared_spin_lock r_lock(table_[segment_id].mu);
      total_size += table_[segment_id].map.size();
    }
    return total_size;
  }

  size_t size_unsafe() const override {
    size_t total_size = 0;
    for (size_t segment_id = 0; segment_id < HASH_SIZE_DEFAULT; segment_id++) {
      total_size += table_[segment_id].map.size();
    }
    return total_size;
  }

  void clear() override {
    for (size_t segment_id = 0; segment_id < HASH_SIZE_DEFAULT; segment_id++) {
      auto& mu = table_[segment_id].mu;
      mu.lock();
      table_[segment_id].map.clear();
      mu.unlock();
    }
  }

  bool erase(const K& key) override {
    size_t segment_id = hash_id(key);
    tfplus_spin_lock w_lock(table_[segment_id].mu);
    return table_[segment_id].map.erase(key) != 0;
  }

  bool erase_unsafe(const K& key) override {
    size_t segment_id = hash_id(key);
    return table_[segment_id].map.erase(key) != 0;
  }

  void ForEach(std::function<void(const K& key, const V* val)> func) override {
    for (size_t segment_id = 0; segment_id < HASH_SIZE_DEFAULT; segment_id++) {
      auto& seg = table_[segment_id];
      tfplus_spin_lock w_lock(seg.mu);
      for (auto it = seg.map.begin(); it != seg.map.end(); ++it) {
        func(it->first, &it->second);
      }
    }
  }

  void ForEachUnsafe(
      std::function<void(const K& key, const V* val)> func) override {
    for (size_t segment_id = 0; segment_id < HASH_SIZE_DEFAULT; segment_id++) {
      auto& seg = table_[segment_id];
      for (auto it = seg.map.begin(); it != seg.map.end(); ++it) {
        func(it->first, &it->second);
      }
    }
  }

  void LockAll() {
    for (size_t segment_id = 0; segment_id < HASH_SIZE_DEFAULT; segment_id++) {
      table_[segment_id].mu.lock();
    }
  }

  void ReleaseAll() {
    for (size_t segment_id = 0; segment_id < HASH_SIZE_DEFAULT; segment_id++) {
      table_[segment_id].mu.unlock();
    }
  }

  spin_rw_mutex* LockKey(const K& key, LockType lock_type) override {
    size_t segment_id = hash_id(key);
    auto& mu = table_[segment_id].mu;
    if (lock_type == LockType::READ_LOCK) {
      mu.lock_read();
    } else {
      mu.lock();
    }
    return &mu;
  }

  ScopedSpinLock GetScopedKeyLock(const K& key, LockType lock_type) override {
    size_t segment_id = hash_id(key);
    return ScopedSpinLock(table_[segment_id].mu,
                          lock_type == LockType::WRITE_LOCK);
  }

  bool NeedExplicitLock() override { return false; }

 private:
  size_t hash_id(const K& key) { return hash_fn_(key) % HASH_SIZE_DEFAULT; }

  struct concurrent_dense_hash_map {
    google::dense_hash_map<K, V, murmurhash_a<K>> map;
    mutable spin_rw_mutex mu;
  };

  concurrent_dense_hash_map table_[HASH_SIZE_DEFAULT];
  F hash_fn_;
};

template <class K, class V, class F = murmurhash_b<K>>
class MultiLevelHashMap : public IMap<K, V> {
 public:
  ~MultiLevelHashMap() {
    for (auto&& p : tables_) delete p;
    tables_.clear();
    table_names_.clear();
  }
  MultiLevelHashMap() {
    MapType map_type = MapType(GetEnvVar<int>(
        "INNER_MULTI_LEVEL_MAP", MapType::CONCURRENT_UNORDERED_MAP));
    switch (map_type) {
      case UNORDERED_MAP:
        hashtable_creator_ = []() { return new UnorderedMap<K, V>(); };
      case CUCKOO_HASH:
        hashtable_creator_ = []() { return new CuckooHashMap<K, V>(); };
      case CONCURRENT_UNORDERED_MAP:
        hashtable_creator_ = []() {
          VLOG(0) << "new ConcurrentUnorderedMap";
          return new ConcurrentUnorderedMap<K, V>();
        };
      case CONCURRENT_DENSE_HASH_MAP:
        hashtable_creator_ = []() {
          return new ConcurrentDenseHashMap<K, V>();
        };
      default:
        hashtable_creator_ = []() {
          return new ConcurrentUnorderedMap<K, V>();
        };
    }
    map_type_ = map_type;
  }

  void AppendNewSubHash(const std::string& variable_name) {
    if (table_names_index_.find(variable_name) == table_names_index_.end()) {
      table_names_index_[variable_name] = tables_.size();
      tables_.push_back(hashtable_creator_());
      table_names_.push_back(variable_name);
      LOG(INFO) << "Append new kv table"
                << " name: " << variable_name << " size: " << tables_.size();
    } else {
      LOG(INFO) << " name: " << variable_name << " is in table, skip ";
    }
  }

  IMap<K, V>* GetInnerTableForMultiHash(const std::string& variable_name) {
    // TODO(zhangji.zhang) add bound check
    int index = table_names_index_[variable_name];
    return tables_[index];
  }
  V* FindOrNull(const K& key) override {
    const struct encode_id id = get_shard_id(key);
    return get_table(id)->FindOrNull(id.lo);
  }

  V* FindOrNullUnsafe(const K& key) override {
    const struct encode_id id = get_shard_id(key);
    return get_table(id)->FindOrNullUnsafe(id.lo);
  }

  V* FindOrInsertWithFn(const K& key, std::function<V(const K& key)> func) {
    const struct encode_id id = get_shard_id(key);
    return get_table(id)->FindOrInsertWithFn(id.lo, func);
  }

  bool FindOrInsertWithDifferentFn(const K& key,
                                   std::function<void(V* val)> find_func,
                                   std::function<V(const K& key)> insert_func) {
    // read lock here, write lock when do export
    tfplus_shared_spin_lock l(mu_);
    const struct encode_id id = get_shard_id(key);
    return get_table(id)->FindOrInsertWithDifferentFn(id.lo, find_func,
                                                      insert_func);
  }

  bool FindWithFn(const K& key, std::function<void(V* val)> func) override {
    // read lock here, write lock when do export
    tfplus_shared_spin_lock l(mu_);
    const struct encode_id id = get_shard_id(key);
    return get_table(id)->FindWithFn(id.lo, func);
  }

  bool InsertOrAssign(const K& key, V&& val) override {
    const struct encode_id id = get_shard_id(key);
    return get_table(id)->InsertOrAssign(id.lo, std::move(val));
  }

  bool UpdateWithFn(const K& key, std::function<void(V* val)> func) override {
    const struct encode_id id = get_shard_id(key);
    return get_table(id)->UpdateWithFn(id.lo, func);
  }

  size_t size() const override { return size_unsafe(); }

  size_t size_unsafe() const override {
    size_t total_size = 0;
    for (auto&& each_kv : tables_) total_size += each_kv->size_unsafe();
    return total_size;
  }

  void clear() override {
    for (auto&& each_kv : tables_) each_kv->clear();
  }

  bool erase(const K& key) override {
    const struct encode_id id = get_shard_id(key);
    return get_table(id)->erase(id.lo) != 0;
  }

  void ForEach(std::function<void(const K& key, const V* val)> func) override {
    ForEachUnsafe(func);
  }

  void ForEachUnsafe(
      std::function<void(const K& key, const V* val)> func) override {
    for (auto&& each_kv : tables_) each_kv->ForEachUnsafe(func);
  }

  void LockAll() {
    for (auto&& each : tables_) each->LockAll();
  }

  void ReleaseAll() {
    for (auto&& each : tables_) each->ReleaseAll();
  }

  bool NeedExplicitLock() override {
    switch (map_type_) {
      case UNORDERED_MAP:
        return true;
      case CUCKOO_HASH:
        return false;
      case CONCURRENT_UNORDERED_MAP:
        return false;
      case CONCURRENT_DENSE_HASH_MAP:
        return false;
      default:
        return true;
    }
  }

  void LockSelf() { mu_.lock(); }

  void UnlockSelf() { mu_.unlock(); }

  std::vector<std::string>& TableNames() { return table_names_; }

  void SetThisVariableName(const std::string& name) {
    this_variable_name_ = name;
  }

  class MultiLevelHashMapUpdateLock {
   public:
    MultiLevelHashMapUpdateLock() = delete;
    explicit MultiLevelHashMapUpdateLock(MultiLevelHashMap* table)
        : table_(table) {
      if (table_ != nullptr) {
        table_->LockSelf();
      }
    }
    ~MultiLevelHashMapUpdateLock() {
      if (table_ != nullptr) {
        table_->UnlockSelf();
      }
    }

   private:
    MultiLevelHashMap* table_;
  };

 private:
  struct encode_id {
    const int64_t hi;
    const K lo;
  };

  inline const encode_id do_get_shard_id_(const std::string& key) {
    int64_t hash_key = hash_fn_(key);
    int64_t hi = hash_key >> KEY_LENGTH;
    const int64_t lo_int64 = hash_key & LOW_MASK;
    const std::string lo = std::to_string(lo_int64);
    VLOG(1) << "key: " << key << " hi: " << hi << " low: " << lo;
    return encode_id{hi, lo};
  }

  template <class KK>
  inline const encode_id do_get_shard_id_(const KK& key) {
    int64_t hi = key >> KEY_LENGTH;
    const KK lo = key & LOW_MASK;
    VLOG(1) << "key: " << key << " hi: " << hi << " low: " << lo;
    return encode_id{hi, lo};
  }

  inline const encode_id get_shard_id(const K& key) {
    return do_get_shard_id_(key);
  }

  IMap<K, V>* get_table(const K& key) { return get_table(get_shard_id(key)); }

  IMap<K, V>* get_table(encode_id id) {
    if (id.hi >= tables_.size()) {
      LOG(FATAL) << "index of id is greater than size of table"
                 << " index: " << id.hi << "\n"
                 << "table size: " << tables_.size() << "\n"
                 << " variable is: " << this_variable_name_ << "\n"
                 << " stack is: " << tensorflow::CurrentStackTrace();
      std::abort();
    } else {
      return tables_[id.hi];
    }
  }

  // 52 is count of low bits for encoder
  // refers https://yuque.antfin-inc.com/mobius/planning/emmgft
  constexpr static int64_t KEY_LENGTH = 52;
  // only works for int64
  constexpr static int64_t LOW_MASK = (1LL << KEY_LENGTH) - 1;
  // record all hash tables
  std::vector<IMap<K, V>*> tables_;
  std::vector<std::string> table_names_;
  // mapping name of variable to index
  std::unordered_map<std::string, int> table_names_index_;
  std::function<IMap<K, V>*()> hashtable_creator_;
  MapType map_type_;
  std::string this_variable_name_;
  mutable spin_rw_mutex mu_;
  // Due to eflops is synchronized, export and FindOrInsert should run
  // in serial, we use write lock to block FindOrInsert
  F hash_fn_;
};

template <class K, class V>
class MapFactory {
 public:
  MapFactory() = delete;
  ~MapFactory() {}
  static IMap<K, V>* CreateMap(const MapType& map_type) {
    switch (map_type) {
      case UNORDERED_MAP:
        return new UnorderedMap<K, V>();
      case CUCKOO_HASH:
        return new CuckooHashMap<K, V>();
      case CONCURRENT_UNORDERED_MAP:
        return new ConcurrentUnorderedMap<K, V>();
      case CONCURRENT_DENSE_HASH_MAP:
        return new ConcurrentDenseHashMap<K, V>();
      case MULTI_LEVEL_MAP:
        return new MultiLevelHashMap<K, V>();
      default:
        return new ConcurrentUnorderedMap<K, V>();
    }
  }
};
}  // namespace tfplus

#endif  // TFPLUS_KV_VARIABLE_KERNELS_HASHMAP_H_
