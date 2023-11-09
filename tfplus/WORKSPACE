load("//tf:tf_configure.bzl", "tf_configure")
load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//third_party/gpus:cuda_configure.bzl", "cuda_configure")

tf_configure(name = "local_config_tf")
cuda_configure(name = "local_config_cuda")

tf_http_archive(
    name = "com_google_googletest",
    sha256 = "bc1cc26d1120f5a7e9eb450751c0b24160734e46a02823a573f3c6b6c0a574a7",
    strip_prefix = "googletest-e2c06aa2497e330bab1c1a03d02f7c5096eb5b0b",
    urls = tf_mirror_urls("https://github.com/google/googletest/archive/e2c06aa2497e330bab1c1a03d02f7c5096eb5b0b.zip"),
)

http_archive(
    name = "tbb",
    build_file = "//third_party:tbb.BUILD",
    sha256 = "e75fafb171fcd392fdedac14f1a6d6c6211230c6a38169a0ec279ea0d80b8a22",
    strip_prefix = "oneTBB-2019_U1",
    urls = [
        "https://github.com/01org/tbb/archive/2019_U1.zip",
    ],
)


http_archive(
    name = "libcuckoo",
    build_file = "//third_party:libcuckoo.BUILD",
    patch_args = ["-p1"],
    patches = [
        "//third_party:cuckoohash_map.patch",
    ],
    sha256 = "7238436b7346a0edf4ce57c12f43f71af5347b8b15f9bf2f0e24bfdca6225fc5",
    strip_prefix = "libcuckoo-0.3",
    urls = [
        "https://github.com/efficient/libcuckoo/archive/v0.3.zip"],
)

http_archive(
    name = "sparsehash",
    build_file = "//third_party:sparsehash.BUILD",
    sha256 = "d4a43cad1e27646ff0ef3a8ce3e18540dbcb1fdec6cc1d1cb9b5095a9ca2a755",
    strip_prefix = "sparsehash-c11-2.11.1",
    urls = [
        "https://github.com/sparsehash/sparsehash-c11/archive/v2.11.1.tar.gz"],
)

http_archive(
    name = "murmurhash",
    build_file = "//third_party:murmurhash.BUILD",
    sha256 = "19a7ccc176ca4185db94047de6847d8a0332e8f4c14e8e88b9048f74bdafe879",
    strip_prefix = "smhasher-master",
    urls = [
        "https://github.com/aappleby/smhasher/archive/master.zip"],
)


http_archive(
    name = "farmhash",
    sha256 = "6560547c63e4af82b0f202cb710ceabb3f21347a4b996db565a411da5b17aba0",
    build_file = "//third_party:farmhash.BUILD",
    strip_prefix = "farmhash-816a4ae622e964763ca0862d9dbd19324a1eaf45",
    urls = [
        "https://mirror.bazel.build/github.com/google/farmhash/archive/816a4ae622e964763ca0862d9dbd19324a1eaf45.tar.gz",
        "https://github.com/google/farmhash/archive/816a4ae622e964763ca0862d9dbd19324a1eaf45.tar.gz",
    ],
)


tf_http_archive(
    name = "com_google_protobuf",
    patch_file = ["//third_party:protobuf/protobuf.patch"],
    sha256 = "f66073dee0bc159157b0bd7f502d7d1ee0bc76b3c1eac9836927511bdc4b3fc1",
    strip_prefix = "protobuf-3.21.9",
    system_build_file = "//third_party/protobuf:protobuf.BUILD",
    system_link_files = {
        "//third_party/protobuf/protobuf.bzl": "protobuf.bzl",
        "//third_party/protobuf/protobuf_deps.bzl": "protobuf_deps.bzl",
    },
    urls = tf_mirror_urls("https://github.com/protocolbuffers/protobuf/archive/v3.21.9.zip"),
)

tf_http_archive(
    name = "rules_python",
    sha256 = "aa96a691d3a8177f3215b14b0edc9641787abaaa30363a080165d06ab65e1161",
    urls = tf_mirror_urls("https://github.com/bazelbuild/rules_python/releases/download/0.0.1/rules_python-0.0.1.tar.gz"),
)

http_archive(
    name = "rules_pkg",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_pkg/releases/download/0.7.1/rules_pkg-0.7.1.tar.gz",
        "https://github.com/bazelbuild/rules_pkg/releases/download/0.7.1/rules_pkg-0.7.1.tar.gz",
    ],
    sha256 = "451e08a4d78988c06fa3f9306ec813b836b1d076d0f055595444ba4ff22b867f",
)

http_archive(
    name = "bazel_skylib",
    sha256 = "74d544d96f4a5bb630d465ca8bbcfe231e3594e5aae57e1edbf17a6eb3ca2506",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/bazel-skylib/releases/download/1.3.0/bazel-skylib-1.3.0.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.3.0/bazel-skylib-1.3.0.tar.gz",
    ],
)

tf_http_archive(
    name = "zlib",
    build_file = "//third_party:zlib.BUILD",
    sha256 = "b3a24de97a8fdbc835b9833169501030b8977031bcb54b3b3ac13740f846ab30",
    strip_prefix = "zlib-1.2.13",
    system_build_file = "//third_party/zlib.BUILD",
    urls = tf_mirror_urls("https://zlib.net/zlib-1.2.13.tar.gz"),
)

http_archive(
    name = "cutlass",
    urls = ["https://github.com/NVIDIA/cutlass/archive/319a389f42b776fae5701afcb943fc03be5b5c25.zip"],
    build_file = "//third_party:cutlass.BUILD",
    strip_prefix = "cutlass-319a389f42b776fae5701afcb943fc03be5b5c25",
)

http_archive(
    name = "flash_attn",
    urls = ["https://github.com/Dao-AILab/flash-attention/archive/9818f85fee29ac6b60c9214bce841f8109a18b1b.zip"],  # v1.0.4
    build_file = "//third_party:flash_attn.BUILD",
    sha256 = "15f29a1095600ba2a3af688fa96a0a48635edb90fffec56c6eb7c48a4a322d2b",
    strip_prefix = "flash-attention-9818f85fee29ac6b60c9214bce841f8109a18b1b",
    patches = [
        "//third_party:flash_attn.patch",
    ],
    patch_args = ["-p1"],
)
