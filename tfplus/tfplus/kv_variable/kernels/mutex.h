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

#ifndef TFPLUS_KV_VARIABLE_KERNELS_MUTEX_H_
#define TFPLUS_KV_VARIABLE_KERNELS_MUTEX_H_
#include "tbb/spin_rw_mutex.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tfplus {
using spin_rw_mutex = ::tbb::spin_rw_mutex;
using tf_mutex = ::tensorflow::mutex;
class ScopedSpinLock : public spin_rw_mutex::scoped_lock {
 public:
  spin_rw_mutex* mu() { return mutex; }
  bool IsWriter() { return is_writer; }

  ScopedSpinLock() : spin_rw_mutex::scoped_lock() {}

  explicit ScopedSpinLock(spin_rw_mutex& m, bool write = true)  // NOLINT
      : spin_rw_mutex::scoped_lock(m, write) {}

  ScopedSpinLock(ScopedSpinLock&& other) {
    mutex = other.mutex;
    is_writer = other.is_writer;
    other.mutex = nullptr;
    other.is_writer = false;
  }
};

class TF_SCOPED_LOCKABLE tfplus_mutex_lock {
 public:
  explicit tfplus_mutex_lock(tf_mutex& mu, bool lockable = true)  // NOLINT
      TF_EXCLUSIVE_LOCK_FUNCTION(mu)
      : mu_(&mu), lockable_(lockable) {
    if (lockable_) mu_->lock();
  }

  tfplus_mutex_lock(tf_mutex& mu, bool lockable,  // NOLINT
                    std::try_to_lock_t)           // NOLINT
      TF_EXCLUSIVE_LOCK_FUNCTION(mu)
      : mu_(&mu), lockable_(lockable) {
    if (lockable && !mu.try_lock()) {
      mu_ = nullptr;
    }
  }

  tfplus_mutex_lock(tfplus_mutex_lock&& ml) noexcept
      TF_EXCLUSIVE_LOCK_FUNCTION(ml.mu_)
      : mu_(ml.mu_), lockable_(ml.lockable_) {
    ml.mu_ = nullptr;
    ml.lockable_ = false;
  }
  ~tfplus_mutex_lock() TF_UNLOCK_FUNCTION() {
    if (lockable_ && mu_ != nullptr) {
      mu_->unlock();
      lockable_ = false;
    }
  }
  tf_mutex* mutex() { return mu_; }

  operator bool() const { return lockable_ && mu_ != nullptr; }

 private:
  tf_mutex* mu_;
  bool lockable_;
};

class TF_SCOPED_LOCKABLE tfplus_shared_lock {
 public:
  explicit tfplus_shared_lock(tf_mutex& mu,          // NOLINT
                              bool lockable = true)  // NOLINT
      TF_SHARED_LOCK_FUNCTION(mu)
      : mu_(&mu), lockable_(lockable) {
    if (lockable_) mu_->lock_shared();
  }

  tfplus_shared_lock(tf_mutex& mu, bool lockable,  // NOLINT
                     std::try_to_lock_t)           // NOLINT
      TF_SHARED_LOCK_FUNCTION(mu)
      : mu_(&mu), lockable_(lockable) {
    if (lockable && !mu.try_lock_shared()) {
      mu_ = nullptr;
    }
  }

  // Manually nulls out the source to prevent double-free.
  // (std::move does not null the source pointer by default.)
  tfplus_shared_lock(tfplus_shared_lock&& ml) noexcept
      TF_SHARED_LOCK_FUNCTION(ml.mu_)
      : mu_(ml.mu_), lockable_(ml.lockable_) {
    ml.mu_ = nullptr;
    ml.lockable_ = false;
  }
  ~tfplus_shared_lock() TF_UNLOCK_FUNCTION() {
    if (lockable_ && mu_ != nullptr) {
      mu_->unlock_shared();
    }
  }
  tf_mutex* mutex() { return mu_; }

  operator bool() const { return lockable_ && mu_ != nullptr; }

 private:
  tf_mutex* mu_;
  bool lockable_;
};

class TF_SCOPED_LOCKABLE tfplus_shared_spin_lock {
 public:
  tfplus_shared_spin_lock(spin_rw_mutex& mu,     // NOLINT
                          bool lockable = true)  // NOLINT
      TF_SHARED_LOCK_FUNCTION(mu)
      : mu_(&mu), lockable_(lockable) {
    if (lockable_) {
      mu_->lock_read();
    }
  }

  tfplus_shared_spin_lock(spin_rw_mutex& mu, bool lockable,  // NOLINT
                          std::try_to_lock_t)                // NOLINT
      TF_SHARED_LOCK_FUNCTION(mu)
      : mu_(&mu), lockable_(lockable) {
    if (lockable_ && !mu_->try_lock_read()) {
      mu_ = nullptr;
    }
  }

  // Manually nulls out the source to prevent double-free.
  // (std::move does not null the source pointer by default.)
  tfplus_shared_spin_lock(tfplus_shared_spin_lock&& ml) noexcept
      TF_SHARED_LOCK_FUNCTION(ml.mu_)
      : mu_(ml.mu_), lockable_(ml.lockable_) {
    ml.mu_ = nullptr;
    ml.lockable_ = false;
  }

  ~tfplus_shared_spin_lock() TF_UNLOCK_FUNCTION() {
    if (lockable_ && mu_) mu_->unlock();
  }

  spin_rw_mutex* mutex() { return mu_; }

  operator bool() const { return lockable_ && mu_ != nullptr; }

 private:
  spin_rw_mutex* mu_;
  bool lockable_;
};  // NOLINT

class TF_SCOPED_LOCKABLE tfplus_spin_lock {
 public:
  tfplus_spin_lock(spin_rw_mutex& mu, bool lockable = true)  // NOLINT
      TF_EXCLUSIVE_LOCK_FUNCTION(mu)
      : mu_(&mu), lockable_(lockable) {
    if (lockable_) {
      mu_->lock();
    }
  }

  tfplus_spin_lock(spin_rw_mutex& mu, bool lockable,  // NOLINT
                   std::try_to_lock_t)                // NOLINT
      TF_EXCLUSIVE_LOCK_FUNCTION(mu)
      : mu_(&mu), lockable_(lockable) {
    if (lockable_ && !mu_->try_lock()) {
      mu_ = nullptr;
    }
  }

  // Manually nulls out the source to prevent double-free.
  // (std::move does not null the source pointer by default.)
  tfplus_spin_lock(tfplus_spin_lock&& ml) noexcept
      TF_EXCLUSIVE_LOCK_FUNCTION(ml.mu_)
      : mu_(ml.mu_), lockable_(ml.lockable_) {
    ml.mu_ = nullptr;
    ml.lockable_ = false;
  }

  ~tfplus_spin_lock() TF_UNLOCK_FUNCTION() {
    if (lockable_ && mu_) mu_->unlock();
  }
  spin_rw_mutex* mutex() { return mu_; }

  operator bool() const { return lockable_ && mu_ != nullptr; }

 private:
  spin_rw_mutex* mu_;
  bool lockable_;
};  // NOLINT

}  // namespace tfplus
#endif  // TFPLUS_KV_VARIABLE_KERNELS_MUTEX_H_
