def cuda_workspace():
    native.new_local_repository(
        name = "cuda",
        path = "/usr/local/cuda/",
        build_file = "//third_party/cuda:cuda.BUILD",
    )

