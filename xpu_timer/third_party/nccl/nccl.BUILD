package(
    default_visibility=["//visibility:public"]
)

genrule(
    name = "nccl_h",
    srcs = glob(["**"]),
    outs = ["src/include/nccl.h"],
    cmd = "(cd $$(dirname $(location src/Makefile)) && BUILDDIR=`pwd` make `pwd`/include/nccl.h && cd - && cp $$(dirname $(location src/Makefile))/include/nccl.h $(@D)/nccl.h)"
)

filegroup(
    name = "nccl_source",
    srcs = glob(["src/include/**/*.h", "src/include/**/*.hpp"]),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "nccl_headers",
    hdrs = [":nccl_h", ":nccl_source"],
    deps = [
        "@cuda//:cuda_headers",
    ],
    linkopts = ["-ldl"],
    includes = ["src/include"],
)
