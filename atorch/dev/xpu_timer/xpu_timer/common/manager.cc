#include "xpu_timer/common/manager.h"

namespace atorch {

std::string BvarPrometheus::metrics_name[BvarPrometheus::cap] = {
    "_qps", "_avg_latency", "_max_latency", "_p99_latency", "_p9999_latency"};
std::shared_ptr<prometheus::Registry> BvarPrometheus::registry_ = nullptr;
prometheus::Exposer* BvarPrometheus::exposer_ = nullptr;
std::chrono::seconds BvarPrometheus::push_interval_(0);

std::string BvarPrometheus::pod_name_("");
std::string BvarPrometheus::ip_("");
std::string BvarPrometheus::rank_("");
std::string BvarPrometheus::local_rank_("");
std::string BvarPrometheus::job_name_("");

BvarPrometheus::BvarPrometheus(const std::string& name, const std::string& type,
                               const std::string& flop) noexcept {
  for (int i = 0; i < BvarPrometheus::cap; i++) {
    LOG(INFO) << "Register prometheus metrics" << name + metrics_name[i];
    auto& gauge = prometheus::BuildGauge()
                      .Name(name + metrics_name[i])
                      .Register(*registry_);
    auto& t = gauge.Add({{"pod_name", pod_name_},
                         {"job_name", job_name_},
                         {"ip", ip_},
                         {"rank", rank_},
                         {"local_rank", local_rank_},
                         {"flop", flop},
                         {"type", type}});
    gauge_[i] = &t;
  }
  bv_ = std::make_unique<bvar::LatencyRecorder>(name);
  start_ = std::chrono::steady_clock::now();
}

void BvarPrometheus::pushMetrics(uint64_t dur_in_us) {
  *bv_ << dur_in_us;
  if (std::chrono::steady_clock::now() - start_ <
      BvarPrometheus::push_interval_)
    return;
  start_ = std::chrono::steady_clock::now();
  gauge_[qps]->Set(bv_->qps());
  gauge_[avg_latency]->Set(bv_->latency());
  gauge_[max_latency]->Set(bv_->max_latency());
  gauge_[p99_latency]->Set(bv_->latency_percentile(0.99));
  gauge_[p9999_latency]->Set(bv_->latency_percentile(0.9999));
}

void BvarPrometheus::setUp(int port) {
  const char* rank = std::getenv("RANK");
  const char* local_rank = std::getenv("LOCAL_RANK");
  const char* ip = std::getenv("POD_IP");
  const char* job_name = std::getenv("ENV_ARGO_WORKFLOW_NAME");
  const char* pod_name = std::getenv("POD_NAME");
  const char* interval = std::getenv("XPU_TIMER_PROMETHEUS_UPDATE_INTERVAL");
  const std::string unknown = "unknown";
  pod_name_ = pod_name ? std::string(pod_name) : unknown;
  ip_ = ip ? std::string(ip) : unknown;
  rank_ = rank ? std::string(rank) : unknown;
  local_rank_ = local_rank ? std::string(local_rank) : unknown;
  job_name_ = job_name ? std::string(job_name) : unknown;
  push_interval_ = interval ? std::chrono::seconds(std::atoi(interval))
                            : std::chrono::seconds(5);
  std::ostringstream oss;
  oss << "0.0.0.0:" << port;
  exposer_ = new prometheus::Exposer(oss.str());
  registry_ = std::make_shared<prometheus::Registry>();
  exposer_->RegisterCollectable(registry_);
  LOG(INFO) << "Start prometheus at 0.0.0.0:" << port;
}

}  // namespace atorch
