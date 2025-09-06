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


# The current version of XCCL uses a self-defined XCCL_TRACE_ID, which results in a different offset compared to the original NCCL,
# refer to: https://code.alipay.com/xccl/nccl/blob/dev/src/include/comm.h#L18
cc_library(
    name = "xccl_parser",
    srcs = glob(["xccl*.cc"]),
    copts = [
        "-fPIC",
        "-Iexternal/nccl_multi_version/nccl/NCCL_2.21.5/",
        "-Iexternal/nccl_multi_version/nccl/NCCL_2.21.5/src/include",
    ],
    linkopts = [
        "-shared",
        # By default, Bazel uses the `-fuse-ld=gold` option. This causes the linker to strip versioned symbols like `func@V2.21` to `func`,
        # making it difficult to distinguish between different versions of the same function.
        # To avoid this issue, we change the linker to `bfd`.
        "-fuse-ld=bfd",
        "-L/usr/local/cuda/lib64",
        "-lcudart",
    ],
    deps = ["@cuda//:cuda_headers", ":nccl_headers", ],
    # force to keep all syms, if not, linker will remove useless symbols.
    alwayslink = True,
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
    deps = [":nccl.lds", "@cuda//:cuda_headers", ":nccl_headers", ":xccl_parser"],
    # force to keep all syms, if not, linker will remove useless symbols.
    alwayslink = True,
)
