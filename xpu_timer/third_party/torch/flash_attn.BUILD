package(
    default_visibility=["//visibility:public"]
)

cc_library(
	name = "flash_attn_headers",
    hdrs = ["csrc/flash_attn/src/flash.h"],
    includes = ["csrc/flash_attn/src/"],
    deps = [
        "@torch//:torch_headers",
        "@cuda//:cuda_headers",
    ]
)
