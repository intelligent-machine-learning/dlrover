def linkstatic_cc_test(name, srcs, deps = [], **kwargs):
    native.cc_test(
        name = name,
        srcs = srcs,
        deps = deps + ["@bazel_tools//tools/cpp/runfiles"],
        linkstatic = True,
        **kwargs
    )
