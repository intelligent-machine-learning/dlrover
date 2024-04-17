#pragma once
#include <butil/logging.h>

#include <array>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace atorch {
namespace util {
std::string execShellCommand(const char* cmd);
std::vector<std::string> split(const std::string& str,
                               const std::string& delimiter);

template <typename T>
class BlockingDeque {
  /* Blocking deque, this is use for queuing XpuTimer object, backgroud thread
   * get object from the queue and parse the duration and name, then pushing to
   * prometheus.
   */
 private:
  std::deque<T*> deque_;
  mutable std::mutex mutex_;
  std::condition_variable cond_var_;

 public:
  void push(T* valuePtr) {
    std::lock_guard<std::mutex> lock(mutex_);
    deque_.push_back(valuePtr);
    cond_var_.notify_one();
    // cond_var_.notify_all(); // maybe, check thread nums.
  }

  T* pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    cond_var_.wait(lock, [this] { return !deque_.empty(); });
    T* value_ptr = deque_.front();
    if (!value_ptr->isReady()) return nullptr;
    deque_.pop_front();
    return value_ptr;
  }
};

template <typename T>
class TimerPool {
  /* This is a deque for pooling XpuTimer object.
   */
 public:
  TimerPool() = default;

  TimerPool(const TimerPool&) = delete;
  TimerPool& operator=(const TimerPool&) = delete;

  T* getObject() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (pool_.empty()) {
      return new T();
    } else {
      T* obj = pool_.front();
      pool_.pop_front();
      return obj;
    }
  }

  // Return an object to the pool
  void returnObject(T* obj) {
    if (obj) {
      std::lock_guard<std::mutex> lock(mutex_);
      pool_.push_back(obj);
    }
  }

 private:
  std::deque<T*> pool_;
  std::mutex mutex_;
};

}  // namespace util
}  // namespace atorch
