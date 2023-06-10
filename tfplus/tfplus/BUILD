package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0
# https://docs.bazel.build/versions/master/be/c-cpp.html#cc_library
cc_library(
    name = "demo_lib",
    hdrs = [
        "cc/demo.h",
    ],
    srcs = [
        "cc/demo.cc",
    ],
    copts = [
        "-std=c++11",
        "-DNDEBUG",
    ],
    deps = [
        # ":demo",
    ],
)

cc_binary(
    name = "python/_demo.so",
    copts = [
        "-std=c++11",
        "-DNDEBUG",
    ],
    srcs = [
        "cc/demo.h",
        "cc/demo.cc",
    ],
    linkshared = True,
    deps = [
        # ":demo_lib",
    ],
)

cc_test(
    name = "cc/demo_test",
    size = "small",
    srcs = [
        "cc/demo_test.cc"
    ],
    copts = [
        "-std=c++11",
        "-DNDEBUG",
    ],
    deps = [
        ":demo_lib",
        "@com_google_googletest//:gtest_main",
    ],
)