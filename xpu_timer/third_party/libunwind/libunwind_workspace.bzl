# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modifications made by [xpu_timer authors], [2024].
# Description of modifications:
# - Change the target to linunwind-x86

def _configure(ctx):
    bash_exe = ctx.os.environ["BAZEL_SH"] if "BAZEL_SH" in ctx.os.environ else "bash"
    ctx.report_progress("Running configure script...")

    # Run configure script and move host-specific directory to a well-known
    # location.
    ctx.execute(
        [bash_exe, "-c", """./configure --disable-dependency-tracking {args};
                            mv $(. config.guess) configure-bazel-gen || true
                         """.format(args = " ".join(ctx.attr.configure_args))],
        quiet = ctx.attr.quiet,
    )

def _buildfile(ctx):
    bash_exe = ctx.os.environ["BAZEL_SH"] if "BAZEL_SH" in ctx.os.environ else "bash"
    if ctx.attr.build_file:
        ctx.execute([bash_exe, "-c", "rm -f BUILD BUILD.bazel"])
        ctx.symlink(ctx.attr.build_file, "BUILD.bazel")
    elif ctx.attr.build_file_content:
        ctx.execute([bash_exe, "-c", "rm -f BUILD.bazel"])
        ctx.file("BUILD.bazel", ctx.attr.build_file_content)

def _autotools_repository_impl(ctx):
    if ctx.attr.build_file and ctx.attr.build_file_content:
        ctx.fail("Only one of build_file and build_file_content can be provided.")
    ctx.download_and_extract(
        ctx.attr.urls,
        "",  # output
        ctx.attr.sha256,
        "",  # type
        ctx.attr.strip_prefix,
    )
    _configure(ctx)
    _buildfile(ctx)

autotools_repository = repository_rule(
    attrs = {
        "urls": attr.string_list(
            mandatory = True,
            allow_empty = False,
        ),
        "sha256": attr.string(),
        "strip_prefix": attr.string(),
        "build_file": attr.label(
            mandatory = True,
            allow_single_file = [".BUILD"],
        ),
        "build_file_content": attr.string(),
        "configure_args": attr.string_list(
            allow_empty = True,
        ),
        "quiet": attr.bool(
            default = True,
        ),
    },
    implementation = _autotools_repository_impl,
)


def libunwind_workspace():
    autotools_repository(
       name = "org_gnu_libunwind",
       build_file = "//third_party/libunwind:libunwind.BUILD",
       configure_args = [
           "--disable-documentation",
           "--disable-minidebuginfo",
           "--disable-shared",
       ],
       #sha256 = "4a6aec666991fb45d0889c44aede8ad6eb108071c3554fcdff671f9c94794976",  # 2024-02-22
       strip_prefix = "libunwind-1.8.1",
       urls = ["https://github.com/libunwind/libunwind/releases/download/v1.8.1/libunwind-1.8.1.tar.gz"],
    )
