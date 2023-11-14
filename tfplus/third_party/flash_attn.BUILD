licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library", "if_cuda")

cuda_library(
  name = "flash_attn",
  srcs = if_cuda([
    "csrc/flash_attn/src/fmha_fwd_hdim32.cu.cc",
    "csrc/flash_attn/src/fmha_fwd_hdim64.cu.cc",
    "csrc/flash_attn/src/fmha_fwd_hdim128.cu.cc",
    "csrc/flash_attn/src/fmha_bwd_hdim32.cu.cc",
    "csrc/flash_attn/src/fmha_bwd_hdim64.cu.cc",
    "csrc/flash_attn/src/fmha_bwd_hdim128.cu.cc",
  ]),
  hdrs = if_cuda([
    "csrc/flash_attn/src/fmha/gemm.h",
    "csrc/flash_attn/src/fmha/gmem_tile.h",
    "csrc/flash_attn/src/fmha/kernel_traits.h",
    "csrc/flash_attn/src/fmha/mask.h",
    "csrc/flash_attn/src/fmha/utils.h",
    "csrc/flash_attn/src/fmha/smem_tile.h",
    "csrc/flash_attn/src/fmha/softmax.h",
    "csrc/flash_attn/src/fmha_utils.h",
    "csrc/flash_attn/src/fmha.h",
    "csrc/flash_attn/src/fmha_fwd_launch_template.h",
    "csrc/flash_attn/src/fmha_bwd_launch_template.h",
    "csrc/flash_attn/src/static_switch.h",
    "csrc/flash_attn/src/fmha_dgrad_kernel_1xN_loop.h",
    "csrc/flash_attn/src/fmha_fprop_kernel_1xN.h",
    "csrc/flash_attn/src/fmha_kernel.h",
    "csrc/flash_attn/src/philox.cuh",
  ]),
  copts = [
    "-O3",
    "-std=c++17",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--use_fast_math",
    "--ptxas-options=-v",
    "-lineinfo"
  ],
  deps = if_cuda(["@cutlass//:cutlass",]),
  strip_include_prefix = "csrc/flash_attn/src",
)

