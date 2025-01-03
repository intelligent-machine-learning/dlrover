# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//:workspace.bzl", "dynamic_local_repository")

BAZEL_SKYLIB_VERSION = "1.1.1"  # 2021-09-27T17:33:49Z

BAZEL_SKYLIB_SHA256 = "c6966ec828da198c5d9adbaa94c05e3a1c7f21bd012a0b29ba8ddbccb2c93b0d"

def brpc_workspace():
    http_archive(
        name = "bazel_skylib",
        sha256 = BAZEL_SKYLIB_SHA256,
        urls = [
            "https://github.com/bazelbuild/bazel-skylib/releases/download/{version}/bazel-skylib-{version}.tar.gz".format(version = BAZEL_SKYLIB_VERSION),
            "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/{version}/bazel-skylib-{version}.tar.gz".format(version = BAZEL_SKYLIB_VERSION),
        ],
    )

    http_archive(
        name = "com_google_protobuf",  # 2021-10-29T00:04:02Z
        sha256 = "3bd7828aa5af4b13b99c191e8b1e884ebfa9ad371b0ce264605d347f135d2568",
        strip_prefix = "protobuf-3.19.4",
        urls = ["https://github.com/protocolbuffers/protobuf/archive/refs/tags/v3.19.4.tar.gz"],
    )

    http_archive(
        name = "com_github_google_leveldb",
        build_file = "//third_party/brpc:leveldb.BUILD",
        strip_prefix = "leveldb-a53934a3ae1244679f812d998a4f16f2c7f309a6",
        url = "https://github.com/google/leveldb/archive/a53934a3ae1244679f812d998a4f16f2c7f309a6.tar.gz",
        sha256 = "3912ac36dbb264a62797d68687711c8024919640d89b6733f9342ada1d16cda1"
    )

    http_archive(
        name = "com_github_madler_zlib",  # 2017-01-15T17:57:23Z
        build_file = "//third_party/brpc:zlib.BUILD",
        sha256 = "c3e5e9fdd5004dcb542feda5ee4f0ff0744628baf8ed2dd5d66f8ca1197cb1a1",
        strip_prefix = "zlib-1.2.11",
        urls = [
            "https://downloads.sourceforge.net/project/libpng/zlib/1.2.11/zlib-1.2.11.tar.gz",
            "https://zlib.net/fossils/zlib-1.2.11.tar.gz",
        ],
    )

    dynamic_local_repository(
        name = "openssl",
        include_default_path = "/opt/conda",
        build_file = "//third_party/brpc:openssl.BUILD",
        include = "OPENSSL_HOME",
        lib = "OPENSSL_HOME",
    )

    http_archive(
        name = "com_github_gflags_gflags",
        strip_prefix = "gflags-46f73f88b18aee341538c0dfc22b1710a6abedef",
        url = "https://github.com/gflags/gflags/archive/46f73f88b18aee341538c0dfc22b1710a6abedef.tar.gz",
        sha256 = "a8263376b409900dd46830e4e34803a170484707327854cc252fc5865275a57d",
    )

    http_archive(
        name = "apache_brpc",
        strip_prefix = "brpc-1.8.0",
        url = "https://github.com/apache/brpc/archive/refs/tags/1.8.0.tar.gz",
        patch_args = ["-p1"],
        patches = [
            "//third_party/brpc:brpc.patch",
        ],
        sha256 = "13ffb2f1f57c679379a20367c744b3e597614a793ec036cd7580aae90798019d",
    )
