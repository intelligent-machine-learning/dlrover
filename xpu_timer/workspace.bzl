# Copyright 2024 The DLRover Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@xpu_timer_cfg//:xpu_config.bzl", "XPU_TIMER_CONFIG")

def _link(repository_ctx, src, dst):
    symlink_target = repository_ctx.path(dst)
    if symlink_target.exists:
        repository_ctx.execute(["rm", symlink_target])
    repository_ctx.symlink(repository_ctx.path(src), symlink_target)

def _dynamic_local_repository_impl(repository_ctx):
    include_path = repository_ctx.os.environ.get(repository_ctx.attr.include, repository_ctx.attr.include_default_path)
    lib_path = repository_ctx.os.environ.get(repository_ctx.attr.lib, repository_ctx.attr.lib_default_path)
    if include_path:
        _link(repository_ctx, include_path, "")
    if lib_path:
        lib_name = lib_path.split("/")[-1]
        _link(repository_ctx, lib_path, lib_name)
    if repository_ctx.attr.template:
        repository_ctx.template(
            "BUILD",
            repository_ctx.path(repository_ctx.attr.build_file),
            {"%{SHARED_LIB}": lib_name},
        )
    else:
        repository_ctx.file("BUILD", repository_ctx.read(repository_ctx.attr.build_file))

dynamic_local_repository = repository_rule(
    implementation = _dynamic_local_repository_impl,
    local = True,
    attrs = {
        "include": attr.string(),
        "include_default_path": attr.string(),
        "lib": attr.string(),
        "lib_default_path": attr.string(),
        "build_file": attr.label(allow_single_file = True),
        "template": attr.bool(default = False),
    },
)


def xpu_cc_library(name, srcs = [], hdrs = [], copts = [], deps = [], deps_on_platform = True, **kwargs):
    if name != "platform" and deps_on_platform:
        deps = deps + ["//xpu_timer/common:platform"]

    native.cc_library(
        name = name,
        srcs = srcs,
        hdrs = hdrs,
        copts = copts + XPU_TIMER_CONFIG.copt,
        deps = deps + XPU_TIMER_CONFIG.deps,
        **kwargs
    )

def xpu_cc_binary(name, srcs, copts = [], deps = [], **kwargs):
    native.cc_binary(
        name = name,
        srcs = srcs,
        copts = copts + XPU_TIMER_CONFIG.copt,
        deps = deps + XPU_TIMER_CONFIG.deps,
        **kwargs
    )

def _cp_xpu_build_wheel_files_impl(ctx):
    inputs = list(ctx.files.srcs)
    append_bin = list(ctx.files.append_bin)
    append_py = list(ctx.files.append_py)

    outputs = []
    for src in ctx.files.srcs:
        output_file = ctx.actions.declare_file("/".join(src.path.split("/")[1:]))
        outputs.append(output_file)

    commands = ["cp -r py_xpu_timer/*   {bin_dir}/{package_name}".format(bin_dir = ctx.bin_dir.path, package_name = ctx.label.package)]

    for src in ctx.files.append_bin:
        output_file = ctx.actions.declare_file("bin/{dst}".format(dst = src.basename))
        commands.append("rm -f {dst}  && cp -f {src} {dst}".format(
            src = src.path,
            dst = output_file.path,
        ))
        outputs.append(output_file)

    for src in ctx.files.append_py:
        output_file = ctx.actions.declare_file("py_xpu_timer/{dst}".format(dst = src.basename))
        commands.append("rm -rf {dst} && cp -f {src} {dst}".format(
            src = src.path,
            dst = output_file.path,
        ))
        outputs.append(output_file)

    command = " && ".join(commands)

    inputs += append_bin
    inputs += append_py
    ctx.actions.run_shell(
        inputs = inputs,
        outputs = outputs,
        command = command,
    )

    return DefaultInfo(files = depset(outputs))

cp_xpu_build_wheel_files = rule(
    implementation = _cp_xpu_build_wheel_files_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = True, mandatory = True),
        "append_bin": attr.label_list(allow_files = True, mandatory = True),
        "append_py": attr.label_list(allow_files = True, mandatory = True),
        "dest": attr.string(mandatory = True),
    },
)

def _nccl_code_repo_impl(ctx):
    urls = ctx.attr.urls
    sha256s = ctx.attr.sha256s
    nccl_number = len(urls)

    version_script_template = ctx.read(ctx.attr.version_script_file)

    version_script_file_content = ""
    pre_version = ""
    for i in range(nccl_number):
        nccl_version = ctx.attr.version_tags[i]
        nccl_version_path = ctx.path("nccl/{}".format(nccl_version))
        ctx.download_and_extract(
            url = urls[i],
            sha256 = sha256s[i],
            output = nccl_version_path,
        )
        cmd_mv = ["sh", "-c", "cd {nccl_version_path} && find . -maxdepth 1 -type d ! -name '.' ! -name '.*' -exec sh -c 'mv $0/* .' {{}} \\;".format(nccl_version_path = nccl_version_path)]
        ctx.execute(cmd_mv)

        src_template_file = ctx.read(ctx.attr.src_template_file)
        src_file = src_template_file.replace("VERSION_TAG", nccl_version)
        src_file = src_file.replace("FUNCTION_TAG", nccl_version.replace(".", "_").replace("-", "_"))

        # generate version script file
        version_script_file_content += "\n"
        version_script_file_content = version_script_file_content + nccl_version
        version_script_file_content = version_script_file_content + version_script_template
        version_script_file_content = version_script_file_content + pre_version + ";"
        pre_version = nccl_version
        ctx.file("nccl_parser_{}.cc".format(nccl_version), content = src_file, executable = False)

        cmd_make = ["sh", "-c", "cd {nccl_version_path} && cd $(dirname src/Makefile) && BUILDDIR=`pwd` make `pwd`/include/nccl.h && cp {nccl_version_path}/src/include/nccl.h {nccl_version_path}/nccl.h".format(nccl_version_path = nccl_version_path)]
        ctx.execute(cmd_make)

    ctx.file("nccl.lds".format(nccl_version), content = version_script_file_content, executable = False)
    ctx.template("BUILD", ctx.path(ctx.attr.build_file))
    _link(ctx, "nccl/{}/src/include/nccl.h".format(ctx.attr.version_tags[0]), "nccl.h")

nccl_code_repository = repository_rule(
    implementation = _nccl_code_repo_impl,
    attrs = {
        "urls": attr.string_list(),
        "sha256s": attr.string_list(),
        "version_tags": attr.string_list(),
        "build_file": attr.label(allow_single_file = True),
        "src_template_file": attr.label(allow_single_file = True),
        "version_script_file": attr.label(allow_single_file = True),
    },
)
