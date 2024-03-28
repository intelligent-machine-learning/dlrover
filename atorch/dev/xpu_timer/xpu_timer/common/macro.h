#pragma once
#define EXPOSE_API __attribute__((visibility("default")))

#define STRINGIFY(x) #x
#define CUDA_SYMBOL_STRING(x) STRINGIFY(x)

#define SETUP_DLSYM(fn_name)                                        \
  if (!orig_##fn_name) {                                            \
    orig_##fn_name =                                                \
        (fn_name##Fn)dlsym(RTLD_NEXT, CUDA_SYMBOL_STRING(fn_name)); \
  } else {                                                          \
  }
