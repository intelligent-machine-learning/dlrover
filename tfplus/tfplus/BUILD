licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])


cc_binary(
    name = "libtfplus.so",
    copts = [
        "-std=c++17",
        "-DNDEBUG",
    ],
    linkopts = [
      '-static-libstdc++',
    ],
    linkshared = 1,
    deps = [
        "//tfplus/kv_variable:python/ops/_kv_variable_ops.so",
    ],
)

cc_binary(
    name = "libtfplus_opdef.so",
    copts = [
        "-std=c++17",
        "-DNDEBUG",
    ],
    linkshared = 1,
    deps = [
        "//tfplus/kv_variable:kv_variable_opdef",
    ],
)
