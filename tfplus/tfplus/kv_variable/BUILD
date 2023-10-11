licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load(
    "@com_google_protobuf//:protobuf.bzl",
    "cc_proto_library",
    "proto_gen",
    "py_proto_library",
)

proto_library(
    name = "checkpoint_state_ext_proto",
    srcs = ["python/training/checkpoint_state_extend.proto"],
)

py_proto_library(
    name = "checkpoint_state_extend_py_pb2",
    srcs = ["python/training/checkpoint_state_extend.proto"],
)

proto_library(
    name = "storage_config_proto",
    srcs = ["kernels/hybrid_embedding/storage_config.proto"],
    visibility = ["//visibility:public"],
)

cc_proto_library(
    name = "storage_config_proto_cc",
    # srcs = glob(["kernels/hybrid_embedding/*.proto"]),
    srcs = ["kernels/hybrid_embedding/storage_config.proto"],
    include = "kernels/hybrid_embedding"
    # deps = [":storage_config_proto"]
)

py_proto_library(
    name = "storage_config_proto_py",
    srcs = ["kernels/hybrid_embedding/storage_config.proto"],
)

cc_library(
    name = "kv_variable_lib",
    hdrs = [
        "kernels/hashmap.h",
        "kernels/embedding_value.h",
        "kernels/hybrid_embedding/embedding_context.h",
        "kernels/hybrid_embedding/table_manager.h",
        "kernels/hybrid_embedding/storage_table.h",
        "kernels/kv_variable_interface.h",
        "kernels/kv_variable.h",
        "kernels/dynamic_restore.hpp",
        "kernels/dynamic_save.hpp",
        "kernels/mutex.h",
        "kernels/utility.h",
        "kernels/kv_variable_cwise_op.h",
        "utils/utils.h",
        "utils/progress_bar.h",
        "kernels/naming.h",
    ],
    srcs = [
       "kernels/utility.cc",
       "kernels/kv_variable_ops.cc",
       "kernels/training_ops.cc",
       "ops/kv_variable_ops.cc",
       "ops/training_ops.cc",
       "utils/utils.cc",
       "utils/progress_bar.cc",
       "kernels/naming.cc",
    ],
    linkstatic = 1,
    copts = [
        "-std=c++17",
        "-fPIC",
        "-DNDEBUG",
        "-D_GLIBCXX_USE_CXX11_ABI=0",
        "-D__STDC_FORMAT_MACROS",
    ],
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
        "@tbb",
        "@libcuckoo",
        "@sparsehash",
        "@murmurhash",
        "@farmhash",
        ":storage_config_proto_cc",
    ],
)

cc_binary(
    name = "python/ops/_kv_variable_ops.so",
    srcs = [
        "kernels/hashmap.h",
        "kernels/embedding_value.h",
        "kernels/hybrid_embedding/embedding_context.h",
        "kernels/hybrid_embedding/table_manager.h",
        "kernels/hybrid_embedding/storage_table.h",
        "kernels/kv_variable_interface.h",
        "kernels/kv_variable.h",
        "kernels/dynamic_restore.hpp",
        "kernels/dynamic_save.hpp",
        "kernels/mutex.h",
        "kernels/utility.h",
        "kernels/kv_variable_cwise_op.h",
        "utils/utils.h",
        "utils/progress_bar.h",
        "kernels/naming.h",
        "kernels/utility.cc",
        "kernels/kv_variable_ops.cc",
        "kernels/training_ops.cc",
        "ops/kv_variable_ops.cc",
        "ops/training_ops.cc",
        "utils/utils.cc",
        "utils/progress_bar.cc",
        "kernels/naming.cc",
    ],
    linkshared = 1,
    copts = [
        "-std=c++17",
        "-pthread",
        "-D_GLIBCXX_USE_CXX11_ABI=1",
    ],
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
        "@tbb",
        "@libcuckoo",
        "@sparsehash",
        "@murmurhash",
        "@farmhash",
        ":storage_config_proto_cc",
    ],
)
cc_library(
    name = "kv_variable_opdef",
    srcs = [
        "ops/kv_variable_ops.cc",
        "ops/training_ops.cc",
    ],
    copts = [
        "-std=c++17",
        "-DNDEBUG",
    ],
    deps = [
        "@local_config_tf//:tf_header_lib",
    ],
)



cc_test(
    name = "kv_variable_test",
    size = "small",
    srcs = [
        "kernels/kv_variable_test.cc",
    ],
    linkopts = ["-lbz2", "-llzma"],
    copts = [
        "-std=c++17",
    ],
    deps = [
        ":kv_variable_lib",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
        "@com_google_googletest//:gtest_main",
    ],
)


cc_library(
    name = "kv_variable_interface",
    hdrs = [
        "kernels/kv_variable_interface.h",
        "kernels/tensor_bundle.h",
        "kernels/naming.h",
        "kernels/mutex.h",
    ],
    copts = [
        "-std=c++17",
        "-DNDEBUG",
    ],
    deps = [
        "@local_config_tf//:tf_header_lib",
        "@local_config_tf//:libtensorflow_framework",
        "@tbb",
    ],
)