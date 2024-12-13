#pragma once
#include <butil/logging.h>

#include "xpu_timer/common/util.h"

#define XLOG(level)                                                            \
  LOG(level) << ::xpu_timer::util::config::GlobalConfig::rank_str

namespace xpu_timer {
void setLoggingPath(bool is_brpc_server);
}
