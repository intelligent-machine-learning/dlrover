package(
    default_visibility=["//visibility:public"]
)

cc_library(
    name = "nccl_headers",
    srcs = glob(["nccl/*/src/include/**/*.h", "nccl/*/src/include/**/*.hpp",
                 "nccl/*/*.h", "nccl/*/*.hpp"]),
    deps = [
        "@cuda//:cuda_headers",
    ],
    linkopts = ["-ldl"],
    includes = ["nccl"],
)

cc_library(
    name = "nccl_h",
    hdrs = ["nccl.h"],
    deps = [
        "@cuda//:cuda_headers",
    ],
    includes = ["."]
    
)


cc_library(
    name = "nccl_parser",
    srcs = glob(["nccl*.cc"]),
    copts = [
        "-fPIC",
    ],
    linkopts = [
        "-shared",
        "-Wl,--version-script=$(location :nccl.lds)",
        "-fuse-ld=bfd",
        "-L/usr/local/cuda/lib64",
        "-lcudart",
    ],
    deps = [":nccl.lds", "@cuda//:cuda_headers", ":nccl_headers"],
    # force to keep all syms, if not, linker will remove useless symbols.
    alwayslink = True,
)
