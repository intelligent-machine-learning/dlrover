package(
    default_visibility=["//visibility:public"]
)


cc_library(
    name = "hpu_headers",
    hdrs = glob(
        include=["include/**/*.h", "include/**/*.hpp"],
    ),
    includes = ["include", "include/aclnn"],
	
)

cc_import(
    name = "opapi",
    shared_library = "lib64/libopapi.so",
    includes = ["include"],
    alwayslink = 1,
)


cc_import(
    name = "ascendcl",
    shared_library = "lib64/libascendcl.so",
    includes = ["include"],
    alwayslink = 1,
)

cc_import(
    name = "hccl",
    shared_library = "lib64/libhccl.so",
    alwayslink = 1,
)
