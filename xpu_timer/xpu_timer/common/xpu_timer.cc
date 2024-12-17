// Copyright 2024 The DLRover Authors. All rights reserved.
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

#include "xpu_timer/common/xpu_timer.h"

#include <dlfcn.h>

#include "xpu_timer/common/logging.h"

namespace xpu_timer {
LibraryLoader::LibraryLoader(const std::string& library_path)
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
}  // namespace xpu_timer
