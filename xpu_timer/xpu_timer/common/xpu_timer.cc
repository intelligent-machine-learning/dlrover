#include "xpu_timer/common/xpu_timer.h"

#include <dlfcn.h>

#include "xpu_timer/common/logging.h"

namespace xpu_timer {
LibraryLoader::LibraryLoader(const std::string &library_path)
    : handle_(nullptr), can_use_(false), library_path_(library_path) {
  LoadLibrary();
}

void LibraryLoader::LoadLibrary() {
  handle_ = dlopen(library_path_.c_str(), RTLD_LAZY);
  if (!handle_) {
    XLOG(WARNING) << "Open " << library_path_ << " error";
    can_use_ = false;
    return;
  }
}
} // namespace xpu_timer
