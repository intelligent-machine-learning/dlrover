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

#pragma once
#define EXPOSE_API __attribute__((visibility("default")))

#define STRINGIFY(x) #x
#define CUDA_SYMBOL_STRING(x) STRINGIFY(x)

#define SETUP_DLSYM(fn_name)                                        \
  if (__builtin_expect(!(orig_##fn_name), 0)) {                     \
    orig_##fn_name =                                                \
        (fn_name##Fn)dlsym(RTLD_NEXT, CUDA_SYMBOL_STRING(fn_name)); \
    if (!orig_##fn_name) {                                          \
      LOG(ERROR) << "Get origin " << CUDA_SYMBOL_STRING(fn_name)    \
                 << " failed!";                                     \
      std::exit(1);                                                 \
    }                                                               \
  }

#define SETUP_DLSYM_WITH_DLOPEN(fn_name, lib_name, DLSYM)                   \
  if (__builtin_expect((!orig_##fn_name), 0)) {                             \
    ::xpu_timer::util::config::setUpDlopenLibrary();                        \
    const std::string& lib_path =                                           \
        ::xpu_timer::util::EnvVarRegistry::getLibPath(lib_name);            \
    void* handle = dlopen(lib_path.c_str(), RTLD_LAZY);                     \
    if (!handle) {                                                          \
      LOG(ERROR) << "Can't open lib " << lib_path                           \
                 << ", please check if the path exists."                    \
                 << " You can set the lib path via env "                    \
                    "XPU_TIMER_"                                            \
                 << lib_name << "_LIB_PATH.";                               \
    } else {                                                                \
      orig_##fn_name =                                                      \
          (fn_name##Fn)DLSYM(handle, CUDA_SYMBOL_STRING(fn_name));          \
      LOG(INFO) << "Get symbol " << CUDA_SYMBOL_STRING(fn_name) << " from " \
                << lib_path;                                                \
    }                                                                       \
    dlclose(handle);                                                        \
    if (!orig_##fn_name) {                                                  \
      LOG(ERROR) << "Get origin " << CUDA_SYMBOL_STRING(fn_name)            \
                 << " failed!";                                             \
      std::exit(1);                                                         \
    }                                                                       \
  }

#define SETUP_DLSYM_WITH_CUBLASLT(fn_name) \
  SETUP_DLSYM_WITH_DLOPEN(fn_name, "CUBLASLT", dlsym)

#define SETUP_DLSYM_WITH_CUBLAS(fn_name) \
  SETUP_DLSYM_WITH_DLOPEN(fn_name, "CUBLAS", dlsym)

#define SETUP_DLSYM_WITH_NCCL(fn_name) \
  SETUP_DLSYM_WITH_DLOPEN(fn_name, "NCCL", dlsym)

#define SETUP_DLSYM_WITH_HPU_OPAPI(fn_name) \
  SETUP_DLSYM_WITH_DLOPEN(fn_name, "HPU_OPAPI", real_dlsym)

#define SETUP_DLSYM_WITH_HPU_HCCL(fn_name) \
  SETUP_DLSYM_WITH_DLOPEN(fn_name, "HPU_HCCL", dlsym)

#define SETUP_DLSYM_WITH_HPU_HCCL_REAL(fn_name) \
  SETUP_DLSYM_WITH_DLOPEN(fn_name, "HPU_HCCL", real_dlsym)

#define SETUP_SYMBOL_FOR_LOAD_LIBRARY(handle, symbol, func_ptr, func_type, \
                                      msg)                                 \
  do {                                                                     \
    func_ptr = (func_type)dlsym(handle, symbol);                           \
    const char* dlsym_error = dlerror();                                   \
    if (dlsym_error) {                                                     \
      XLOG(WARNING) << "Load fn `" << symbol << "` error in " << msg       \
                    << dlsym_error;                                        \
      can_use_ = false;                                                    \
      return;                                                              \
    }                                                                      \
  } while (0)
