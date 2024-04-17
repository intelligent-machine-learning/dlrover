#pragma once
#include <brpc/server.h>
#include <prometheus/counter.h>
#include <prometheus/exposer.h>
#include <prometheus/gauge.h>
#include <prometheus/registry.h>
#include <pthread.h>

#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <utility>

#include "xpu_timer/common/util.h"

namespace atorch {
using namespace util;

struct BvarPrometheus {
  /* This class bind bvar and prometheus together.
   * Bvar is for calculation for metrics and pushing to brpc server for better
   * UI. Prometheus service is for pulling metrics from remote server.
   */
  static constexpr int qps = 0;
  static constexpr int avg_latency = 1;
  static constexpr int max_latency = 2;
  static constexpr int p99_latency = 3;
  static constexpr int p9999_latency = 4;
  static constexpr int cap = 5;
  static std::string pod_name_;
  static std::string ip_;
  static std::string rank_;
  static std::string local_rank_;
  static std::string job_name_;
  static std::string metrics_name[];
  static std::shared_ptr<prometheus::Registry> registry_;
  static prometheus::Exposer* exposer_;
  static std::chrono::seconds push_interval_;

  std::unique_ptr<bvar::LatencyRecorder> bv_;
  prometheus::Gauge*
      gauge_[cap];  // not owned, prometheus use unique_ptr to
                    // manage the gauge objects, so we only ref it.
  std::chrono::steady_clock::time_point start_;

  BvarPrometheus(){};
  BvarPrometheus(BvarPrometheus&& other) noexcept {
    bv_ = std::move(other.bv_);
    start_ = std::move(other.start_);
    for (int i = 0; i < BvarPrometheus::cap; i++) gauge_[i] = other.gauge_[i];
  };
  BvarPrometheus(const std::string& name, const std::string& type,
                 const std::string& flop) noexcept;
  // start brpc and prometheus server.
  // get label from env var.
  static void setUp(int port);
  // push metrics into bvar, weite metrics to gauge. Limiting every write in
  // push_interval_ seconds.
  void pushMetrics(uint64_t dur_in_us);
};

template <typename T>
class GpuTimerManager {
  /*
   * TODO check the overhead of mutex.
   * since the critical section is tiny(only take fron deque, deque will not
   * resize in std's impl) but we do not know how many thread will fail into
   * critical section, if have lots of contention, maybe we have to use
   * coroutine for queue execution or use global tid map to manage each thread.

                            working flow

                                │
          ┌─────────────────┐   │
          │  pooling queue  ◄───┼──────────┐
          └────────┬────────┘   │          │
                   │1 get obj   │          │
          ┌────────▼────────┐   │          │
          │  get event      │   │          │7  return obj
          └────────┬────────┘   │  ┌───────┴────────┐
                   │2           │  │  compute time  │
          ┌────────▼────────┐   │  │  pushing data  │
          │  record event   │   │  └───────▲────────┘
          └────────┬────────┘   │          │
                   │3           │          │
          ┌────────▼────────┐   │          │
          │ lanunch kernel  │   │          │
          └────────┬────────┘   │          │6
                   │4           │          │
          ┌────────▼────────┐  5│  ┌───────┴────────┐
          │  record event   ├───┼──►  workig queue  │
          └─────────────────┘   │  └────────────────┘
                                │
              main thread       │     backgroud
                                       thread

   */
 public:
  GpuTimerManager(const GpuTimerManager&) = delete;
  GpuTimerManager& operator=(const GpuTimerManager&) = delete;

  static GpuTimerManager& getInstance() {
    std::call_once(init_flag_, &GpuTimerManager::initSingleton);
    return *instance_;
  }

  T* getEvent() { return event_pool_.getObject(); }

  void recordEvent(T* event) {
    event->endRecord();
    working_queue_.push(event);
  }

 private:
  inline static GpuTimerManager* instance_ = nullptr;
  inline static std::once_flag init_flag_;
  BlockingDeque<T> working_queue_;
  TimerPool<T> event_pool_;
  std::unordered_map<std::string, BvarPrometheus> metrics_;

  std::atomic<bool> should_run_{false};
  std::thread event_poller_;

  GpuTimerManager() = default;
  ~GpuTimerManager() { stopWork(); }

  BvarPrometheus& getMetrics(T* work_item) {
    const std::string name = work_item->getName();
    auto it = metrics_.find(name);
    if (it == metrics_.end()) {
      const std::string type = work_item->getType();
      auto result = metrics_.emplace(
          std::piecewise_construct, std::forward_as_tuple(name),
          std::forward_as_tuple(name, work_item->getType(),
                                work_item->getFlop()));
      if (!result.second) LOG(FATAL) << "insert error";
      return result.first->second;
    }
    return it->second;
  }

  static void initSingleton() {
    instance_ = new GpuTimerManager;
    instance_->startWork();
    std::atexit([] { delete instance_; });
  }

  void stopWork() {
    should_run_.store(false);
    if (event_poller_.joinable()) {
      event_poller_.join();
    }
    metrics_.clear();
  }
  void doWork() {
    while (should_run_.load()) {
      std::this_thread::sleep_for(std::chrono::microseconds(100));
      T* work_item = working_queue_.pop();
      if (!work_item) continue;
      BvarPrometheus& metrics = getMetrics(work_item);
      metrics.pushMetrics(work_item->getDuration());
      event_pool_.returnObject(work_item);
    }
  }

  void startWork() {
    const char* local_rank_env = std::getenv("LOCAL_RANK");
    const char* base_port_env = std::getenv("XPU_TIMER_BASEPORT");
    const char* open_brpc_env = std::getenv("XPU_TIMER_OPENBRPC");

    int port_offset;
    int base_port;
    int open_broc = open_brpc_env == nullptr ? 0 : std::atoi(open_brpc_env);
    if (local_rank_env != nullptr) {
      port_offset = std::atoi(local_rank_env);
    } else {
      port_offset = 0;
      LOG(INFO) << "LOCAL_RANK is not set use 0";
    }
    base_port = base_port_env == nullptr ? 28888 : std::atoi(base_port_env);
    if (open_broc)
      brpc::StartDummyServerAt(base_port + port_offset);
    else
      LOG(INFO) << "Brpc server not start";
    BvarPrometheus::setUp(base_port + port_offset + 10000);
    should_run_.store(true);
    T::doPrepare();
#ifdef _GNU_SOURCE
    // pthread_setname_np is not posix
    event_poller_ = std::thread(&GpuTimerManager::doWork, this);
    auto handle = event_poller_.native_handle();
    pthread_setname_np(handle, "xpu_timer_event_poller");
#endif
  }
};

}  // namespace atorch
