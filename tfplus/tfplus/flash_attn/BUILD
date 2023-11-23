licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library", "if_cuda", "cuda_cc_library")
# load("//tensorflow:tensorflow.bzl", "tf_cc_binary", "tf_custom_op_library")

cc_binary(
    name = "python/ops/_flash_attention.so",
    srcs = if_cuda([
        "kernels/flash_attention.h",
        "ops/flash_attention_ops.cc",
        "kernels/flash_attention_fwd_kernel.cc",
        "kernels/flash_attention_bwd_kernel.cc",
    ]),
    copts = [
        "-x", "cuda", 
        "--cuda-gpu-arch=sm_75",
        "--cuda-gpu-arch=sm_80",
        "--cuda-gpu-arch=sm_90",
        "-DGOOGLE_CUDA=1",
        "-nvcc_options=relaxed-constexpr", "-nvcc_options=ftz=true",
        "-O3",
        "-std=c++17",
        "-fPIC",
        "-D_GLIBCXX_USE_CXX11_ABI=1",
    ],
    linkshared = 1,
    deps = [
      "@flash_attn//:flash_attn",
      "@local_config_tf//:libtensorflow_framework",
      "@local_config_tf//:tf_header_lib",
    ],
)
