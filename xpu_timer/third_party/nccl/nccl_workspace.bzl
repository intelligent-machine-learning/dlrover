load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@xpu_timer_cfg//:xpu_config.bzl", "NCCL_VERSION")

def nccl_workspace():
    # only for nccl 2.18.5, if you change version, you should generate the headers of `devcomm`
    http_archive(
        name = "nccl",
        build_file = "//third_party/nccl:nccl.BUILD",
        strip_prefix = "nccl-{}".format(NCCL_VERSION),
        urls = ["https://github.com/NVIDIA/nccl/archive/refs/tags/v{}.tar.gz".format(NCCL_VERSION)],
    )
