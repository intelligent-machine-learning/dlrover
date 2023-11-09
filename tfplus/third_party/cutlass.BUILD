licenses(["notice"]) # # BSD 3-Clause

package(default_visibility = ["//visibility:public"])

load("@local_config_cuda//cuda:build_defs.bzl", "cuda_header_library")


cuda_header_library(
  name = "cutlass",
  hdrs = glob([
    "include/cutlass/**",
  ]),
  includes = [
    "include",
  ],
  strip_include_prefix = "include",
  visibility = ["//visibility:public"],
)