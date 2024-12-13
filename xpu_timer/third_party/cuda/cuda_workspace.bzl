load("//:workspace.bzl", "dynamic_local_repository")

def cuda_workspace():
    dynamic_local_repository(
        name = "cuda",
        include_default_path = "/usr/local/cuda/",
        build_file = "//third_party/cuda:cuda.BUILD",
        include = "CUDA_HOME",
    )
