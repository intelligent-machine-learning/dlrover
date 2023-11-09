licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")

cuda_library(
  copts = [
    "-O3",
    "-std=c++17",
    "-fPIC",
    "-D_GLIBCXX_USE_CXX11_ABI=0",
  ],
  name = "tf_flash_attn",
  hdrs = [
    "kernels/flash_attention.h",
  ],
  srcs = [
    "ops/flash_attention_ops.cc",
    "kernels/flash_attention_fwd_kernel.cc",
    "kernels/flash_attention_bwd_kernel.cc",
  ],
  deps = [
    "@local_config_tf//:libtensorflow_framework",
    "@local_config_tf//:tf_header_lib",
    "@flash_attn//:flash_attn"
  ],
)

cc_binary(
    name = "python/ops/_flash_attention.so",
    copts = [
        "-std=c++17",
    ],
    linkshared = 1,
    deps = [
      ":tf_flash_attn",
      "@local_config_tf//:libtensorflow_framework",
      "@local_config_tf//:tf_header_lib",
    ],
)

