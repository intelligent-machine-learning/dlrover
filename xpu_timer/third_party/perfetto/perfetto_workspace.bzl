load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")

def perfetto_workspace():
    http_archive(
        name = "com_google_perfetto",
        urls = ["https://github.com/google/perfetto/archive/refs/tags/v45.0.tar.gz"],
        strip_prefix = "perfetto-45.0",
        build_file = "//third_party/perfetto:perfetto.BUILD",
        sha256 = "dcb815fb54370fa20a657552288016cb66e7a98237c1a1d47e7645a4325ac75e"
    )
