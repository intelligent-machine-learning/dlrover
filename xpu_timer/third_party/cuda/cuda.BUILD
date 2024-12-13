package(
    default_visibility=["//visibility:public"]
)

cc_library(
    name = "cuda_nv",
    hdrs = glob(["include/nv/*"]),
    includes = ["include"],
)

cc_library(
    name = "cuda_nv_detail",
    hdrs = glob(["include/nv/detail/*"]),
    includes = ["include/nv/target"],
    deps = [":cuda_nv"],
)

cc_library(
    name = "cuda_headers",
    hdrs = glob(
        include=["include/**/*.h", "include/**/*.hpp"],
        exclude=["include/**/nccl*.h"]
    ),
    includes = ["include"],
    deps = [":cuda_nv", ":cuda_nv_detail"]
)
