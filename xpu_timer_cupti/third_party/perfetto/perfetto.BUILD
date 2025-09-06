package(default_visibility = ["//visibility:public"])
load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")


py_proto_library(
    name = "proto",
    srcs = ["protos/perfetto/trace/perfetto_trace.proto"],
    deps = [
        "@com_google_protobuf//:protobuf_python",
    ],
)
