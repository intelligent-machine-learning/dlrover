#include "xpu_timer/python/py_tracing_manager.h"

#include <cstring>
#include <thread>

#include "xpu_timer/python/py_tracing_data.h"

namespace xpu_timer {
namespace py_tracing_manager {

PyTracingManager& PyTracingManager::getInstance() {
  std::call_once(init_flag_, &PyTracingManager::initSingleton);
  return *instance_;
}

void PyTracingManager::initSingleton() { instance_ = new PyTracingManager(); }

XpuTimerPyTracingDataArray* PyTracingManager::getEmptyPyTracingDataArray(
    int name) {
  auto& item = pool_[name];
  XpuTimerPyTracingDataArray* data = item.empty_pool.getObject();
  std::memset(data, 0, sizeof(XpuTimerPyTracingDataArray));
  return data;
}
void PyTracingManager::returnPyTracingDataArray(
    XpuTimerPyTracingDataArray* array, int type, int name) {
  if (!array) return;

  int pool_queue_size;
  auto& item = pool_[name];
  if (type == PY_TRACING_READY_POOL)
    item.ready_pool.returnObject(array, &pool_queue_size);
  else if (type == PY_TRACING_EMPTY_POOL)
    item.empty_pool.returnObject(array, &pool_queue_size);
}

XpuTimerPyTracingDataArray* PyTracingManager::getPyTracingDataArray(int name) {
  auto& item = pool_[name];
  return item.ready_pool.getObject<false>();
}
}  // namespace py_tracing_manager
}  // namespace xpu_timer
