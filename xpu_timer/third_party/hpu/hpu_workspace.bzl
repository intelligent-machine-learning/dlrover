load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//:workspace.bzl", "ali_code_repository", "dynamic_local_repository")

def hpu_workspace():
    dynamic_local_repository(
        name = "hpu",
        include_default_path = "/usr/local/Ascend/ascend-toolkit/latest",
        build_file = "//third_party/hpu:hpu.BUILD",
        include = "ASCEND_TOOLKIT_HOME",
    )
